"""
utils_align.py  —  Rigid 3D image registration utilities (PyTorch / GPU)
=========================================================================

Physical coordinate convention
-------------------------------
  p_a = (i_a  −  (N_a − 1) / 2)  ×  v_a

  The physical origin is always the geometric centre of the volume,
  for both even and odd voxel counts:
      odd  N=5: voxels at  −2v, −v, 0, +v, +2v
      even N=4: voxels at  −1.5v, −0.5v, +0.5v, +1.5v

  Rotations are around that centre by default.
  Translations t are in mm relative to it.

Parameter vector convention
----------------------------
  params = [tz, ty, tx, ωz, ωy, ωx]     shape (6,)
    • tz, ty, tx  : translation in mm, along axes (D, H, W)
    • ωz, ωy, ωx  : Rodrigues rotation vector in radians

  The rigid transform maps physical coords as:
      p_out  →  p_in = R(ω) @ p_out + t

Public API
----------
  warp(image, params, voxel_size)                          → (D, H, W)
  invert_params(params)                                    → (6,)
  init_from_center_of_mass(fixed, moving, voxel_size)      → (6,)
  register(fixed, moving, voxel_size, cost_fn, …)          → (6,)
  register_multiscale(fixed, moving, voxel_size, cost_fn)  → (6,)
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn.functional as TF
from torch import Tensor


def ssd(fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
    return ((fixed - warped) ** 2).sum()


def ncc(fixed, warped):  # normalised cross-correlation
    f = fixed - fixed.mean()
    w = warped - warped.mean()
    return -(f * w).sum() / (f.norm() * w.norm() + 1e-8)


# ── Internal: rotation ────────────────────────────────────────────────────────


def _rotvec_to_matrix(rv: Tensor) -> Tensor:
    """Rodrigues' formula: rv (3,) → R (3,3). Fully differentiable."""
    theta = rv.norm().clamp(min=1e-8)
    k = rv / theta
    kz, ky, kx = k[0], k[1], k[2]
    O = torch.zeros_like(kz)
    K = torch.stack(
        [
            torch.stack([O, -kx, ky]),
            torch.stack([kx, O, -kz]),
            torch.stack([-ky, kz, O]),
        ]
    )
    I = torch.eye(3, device=rv.device, dtype=rv.dtype)
    return I + theta.sin() * K + (1 - theta.cos()) * (K @ K)


# ── Internal: sampling grid ───────────────────────────────────────────────────


def _make_sampling_grid(
    R: Tensor,
    t: Tensor,
    out_shape: Tuple[int, int, int],
    out_vox: Tuple[float, float, float],
    in_shape: Tuple[int, int, int],
    in_vox: Tuple[float, float, float],
) -> Tensor:
    """
    Build a (1, Nz, Ny, Nx, 3) sampling grid for TF.grid_sample.

    For each output voxel i_out:
      1. Centred physical coord:   p_out_a = (i_out_a − (N_out_a−1)/2) · v_out_a
      2. Rigid transform:          p_in = R @ p_out + t
      3. Normalised coord of M:    n_a   = 2 · p_in_a / ((N_in_a−1) · v_in_a)
      4. Flip (z,y,x) → (x,y,z)  for grid_sample.
    """
    oNz, oNy, oNx = out_shape
    ovz, ovy, ovx = out_vox
    iNz, iNy, iNx = in_shape
    ivz, ivy, ivx = in_vox

    z_mm = (torch.arange(oNz, device=R.device, dtype=R.dtype) - (oNz - 1) / 2) * ovz
    y_mm = (torch.arange(oNy, device=R.device, dtype=R.dtype) - (oNy - 1) / 2) * ovy
    x_mm = (torch.arange(oNx, device=R.device, dtype=R.dtype) - (oNx - 1) / 2) * ovx
    gz, gy, gx = torch.meshgrid(z_mm, y_mm, x_mm, indexing="ij")

    p_out = torch.stack([gz, gy, gx], dim=-1)  # (Nz, Ny, Nx, 3) mm
    p_in = p_out @ R.T + t  # rigid transform

    extent = torch.tensor(
        [(iNz - 1) * ivz, (iNy - 1) * ivy, (iNx - 1) * ivx],
        device=R.device,
        dtype=R.dtype,
    )
    n_dhw = 2.0 * p_in / extent  # centred: no offset term
    n_xyz = n_dhw[..., [2, 1, 0]]  # flip to (x,y,z) for grid_sample
    return n_xyz.unsqueeze(0)  # (1, Nz, Ny, Nx, 3)


# ── Internal: Gaussian smoothing ──────────────────────────────────────────────


def _gaussian_kernel_1d(sigma: float, device, dtype) -> Tensor:
    """1-D Gaussian kernel, truncated at ±3σ."""
    radius = max(1, int(3.0 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _smooth3d(vol5d: Tensor, sigmas: Tuple[float, float, float]) -> Tensor:
    """
    Separable 3D Gaussian smoothing of a (1, 1, D, H, W) tensor.
    sigmas = (σz, σy, σx) in voxels. Axes with σ < 0.5 are skipped.
    Uses replicate padding to avoid border artefacts.
    """
    out = vol5d
    for ax, sigma in enumerate(sigmas):
        if sigma < 0.5:
            continue
        k = _gaussian_kernel_1d(sigma, vol5d.device, vol5d.dtype)
        r = len(k) // 2
        # kernel shape: (1,1,k,1,1) for ax=0 (D), (1,1,1,k,1) for ax=1 (H), …
        kshape = [1, 1, 1, 1, 1]
        kshape[2 + ax] = len(k)
        k5d = k.view(kshape)
        # F.pad order (innermost dim first): W, H, D → axis 2 maps to pad index 2*(2-ax)
        pad = [0] * 6
        pad_idx = 2 * (2 - ax)
        pad[pad_idx] = r
        pad[pad_idx + 1] = r
        out = TF.conv3d(TF.pad(out, pad, mode="replicate"), k5d)
    return out


def _smooth_and_downsample(
    vol: Tensor,
    scale: float,
    voxel_size: Tuple[float, float, float],
    sigma_vox: Tuple[float, float, float] | None = None,
) -> Tuple[Tensor, Tuple[float, float, float], Tuple[int, int, int]]:
    """
    Smooth then downsample a (D, H, W) volume.

    Anti-aliasing sigma defaults to 0.5 / scale per axis (in voxels),
    which rolls off frequencies that would alias after downsampling
    while still broadening the cost-function basin at coarse scales.

    Returns (downsampled_volume, effective_voxel_size, new_shape).
    The effective voxel size is computed exactly from the shape change
    assuming align_corners=True throughout.
    """
    if sigma_vox is None:
        s = max(0.5 / scale, 0.5)  # at least 0.5 even at full scale
        sigma_vox = (s, s, s)

    vol5d = vol[None, None]  # (1,1,D,H,W)
    smooth = _smooth3d(vol5d, sigma_vox)
    down = TF.interpolate(
        smooth,
        scale_factor=scale,
        mode="trilinear",
        align_corners=True,
    )
    vol_ds = down[0, 0]  # (D', H', W')

    old_shape = tuple(vol.shape)  # (Nz, Ny, Nx)
    new_shape = tuple(vol_ds.shape)  # (Nz', Ny', Nx')

    # Exact voxel size: align_corners keeps corner-voxel centres fixed,
    # so (N_old−1)·v_old = (N_new−1)·v_new
    eff_vox = tuple(
        (No - 1) / (Nn - 1) * v for No, Nn, v in zip(old_shape, new_shape, voxel_size)
    )
    return vol_ds, eff_vox, new_shape


# ── Public API ────────────────────────────────────────────────────────────────


def invert_params(params: Tensor) -> Tensor:
    """
    Return parameters of the inverse transform T^{-1}.
    Uses the identity R(ω)^T = R(−ω) to avoid matrix inversion.
    """
    rv = params[3:]
    t = params[:3]
    R_inv = _rotvec_to_matrix(-rv)
    return torch.cat([-(R_inv @ t), -rv])


def warp(
    image: Tensor,
    params: Tensor,
    voxel_size: Tuple[float, float, float],
    *,
    padding_mode: str = "border",
) -> Tensor:
    """
    Apply a rigid transform to a 3D image.

    Args:
        image:        (D, H, W) float tensor.
        params:       (6,) [tz, ty, tx, ωz, ωy, ωx].
                      Translations in mm; angles in radians.
        voxel_size:   (vz, vy, vx) in mm.
        padding_mode: 'border' | 'zeros' | 'reflection'.

    Returns:
        Warped image (D, H, W), same device as input.
    """
    shape = tuple(image.shape)
    R = _rotvec_to_matrix(params[3:])
    grid = _make_sampling_grid(R, params[:3], shape, voxel_size, shape, voxel_size)
    out = TF.grid_sample(
        image[None, None],
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    return out[0, 0]


def init_from_center_of_mass(
    fixed: Tensor,
    moving: Tensor,
    voxel_size: Tuple[float, float, float],
) -> Tensor:
    """
    Return (6,) params whose translation aligns the intensity-weighted
    centres of mass (COM) of fixed and moving, with zero rotation.

    COM is computed in physical (mm) space using the centred coordinate
    convention.  Effective even when there is a large initial offset.

    Note: assumes non-negative image intensities.  If your images can be
    negative, clip or shift them before calling this function.
    """
    device, dtype = fixed.device, fixed.dtype
    Nz, Ny, Nx = fixed.shape
    vz, vy, vx = voxel_size

    z_mm = (torch.arange(Nz, device=device, dtype=dtype) - (Nz - 1) / 2) * vz
    y_mm = (torch.arange(Ny, device=device, dtype=dtype) - (Ny - 1) / 2) * vy
    x_mm = (torch.arange(Nx, device=device, dtype=dtype) - (Nx - 1) / 2) * vx
    gz, gy, gx = torch.meshgrid(z_mm, y_mm, x_mm, indexing="ij")  # each (D,H,W)

    def _com(img: Tensor) -> Tensor:
        w = img.clamp(min=0)  # guard against small negatives
        tot = w.sum().clamp(min=1e-12)
        cz = (w * gz).sum() / tot
        cy = (w * gy).sum() / tot
        cx = (w * gx).sum() / tot
        return torch.stack([cz, cy, cx])

    t_init = _com(fixed) - _com(moving)
    return torch.cat([t_init, torch.zeros(3, device=device, dtype=dtype)])


def register(
    fixed: Tensor,
    moving: Tensor,
    voxel_size: Tuple[float, float, float],
    cost_fn: Callable[[Tensor, Tensor], Tensor] = ncc,
    *,
    init_params: Tensor | None = None,
    n_steps: int = 25,
    max_iter: int = 20,
    history_size: int = 10,
    tol: float = 1e-8,
    verbose: bool = True,
) -> Tensor:
    """
    Single-scale rigid registration using L-BFGS.

    Args:
        fixed:       (D, H, W) reference image.
        moving:      (D, H, W) image to align to fixed.
        voxel_size:  (vz, vy, vx) in mm.
        cost_fn:     metric(fixed, warped) → scalar.  Both args are (D, H, W).
                     Example: lambda f, w: ((f - w) ** 2).sum()
        init_params: (6,) initial guess. Defaults to zeros.
        n_steps:     L-BFGS outer iterations.
        max_iter:    max line-search steps per outer iteration.
        history_size: L-BFGS curvature-pair history.
        tol:         relative loss change for early stopping.
        verbose:     print loss each step.

    Returns:
        params (6,) on the same device as fixed.
        Call warp(moving, params, voxel_size) to obtain the aligned image.
    """
    shape = tuple(fixed.shape)
    device, dtype = fixed.device, fixed.dtype
    moving5d = moving[None, None]

    def _warp_moving(p: Tensor) -> Tensor:
        R = _rotvec_to_matrix(p[3:])
        grid = _make_sampling_grid(R, p[:3], shape, voxel_size, shape, voxel_size)
        return TF.grid_sample(
            moving5d,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, 0]

    params = (
        init_params.clone().detach().to(device=device, dtype=dtype)
        if init_params is not None
        else torch.zeros(6, device=device, dtype=dtype)
    ).requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [params],
        lr=1.0,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    loss_prev = None
    for step in range(n_steps):

        def closure():
            optimizer.zero_grad()
            loss = cost_fn(fixed, _warp_moving(params))
            loss.backward()
            return loss

        loss_val = optimizer.step(closure)
        loss_now = loss_val.item()

        if verbose:
            print(f"    step {step + 1:2d}  loss = {loss_now:.4e}")

        if loss_prev is not None and abs(loss_prev - loss_now) < tol * abs(loss_prev):
            if verbose:
                print(f"    Converged at step {step + 1}")
            break
        loss_prev = loss_now

    if verbose:
        print(
            f"    final parameters: t = [{params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f}] mm, rot = [{params[3]:.4f}, {params[4]:.4f}, {params[5]:.4f}] rad"
        )

    return params.detach()
