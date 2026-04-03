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
  params = (tz, ty, tx, ωz, ωy, ωx)     length-6 tuple of floats
    • tz, ty, tx  : translation in mm, along axes (D, H, W)
    • ωz, ωy, ωx  : Rodrigues rotation vector in radians

  The rigid transform maps physical coords as:
      p_out  →  p_in = R(ω) @ p_out + t

Public API
----------
  warp(image, params, voxel_size)                          → (D, H, W) Tensor
  invert_params(params)                                    → tuple[float, ...]
  register(fixed, moving, voxel_size, cost_fn, …)          → tuple[float, ...]

  All image inputs accept torch.Tensor or np.ndarray.
  All params inputs/outputs are tuples of 6 floats.
"""

from __future__ import annotations

from typing import Callable, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as TF
from torch import Tensor

# Type aliases
Params = Tuple[float, float, float, float, float, float]
ParamsLike = Union[Params, np.ndarray, Tensor]
ImageLike = Union[Tensor, np.ndarray]
VoxelSize = Union[Tuple[float, float, float], np.ndarray]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_float_tensor(
    x: ImageLike,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Convert numpy array or tensor to a float tensor on *device*."""
    if isinstance(x, np.ndarray):
        return torch.tensor(x, device=device, dtype=dtype)
    return x.to(device=device, dtype=dtype)


def _image_device(x: ImageLike) -> torch.device:
    """Return the device for an image input (auto-detect for numpy)."""
    if isinstance(x, np.ndarray):
        return _default_device()
    return x.device


def _params_to_tensor(
    params: ParamsLike,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Convert any params representation to a (6,) tensor on *device*."""
    if isinstance(params, Tensor):
        return params.detach().to(device=device, dtype=dtype)
    return torch.tensor(list(params), device=device, dtype=dtype)


def _tensor_to_params(t: Tensor) -> Params:
    """Convert a (6,) tensor to a tuple of 6 plain floats."""
    v = t.detach().cpu().tolist()
    return (v[0], v[1], v[2], v[3], v[4], v[5])


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


# ── Public API ────────────────────────────────────────────────────────────────


def invert_params(params: ParamsLike) -> Params:
    """
    Return parameters of the inverse transform T^{-1}.
    Uses the identity R(ω)^T = R(−ω) to avoid matrix inversion.

    Args:
        params: length-6 tuple/numpy array [tz, ty, tx, ωz, ωy, ωx].

    Returns:
        Inverse parameters as a tuple of 6 floats.
    """
    device = _default_device()
    p = _params_to_tensor(params, device)
    rv = p[3:]
    t = p[:3]
    R_inv = _rotvec_to_matrix(-rv)
    return _tensor_to_params(torch.cat([-(R_inv @ t), -rv]))


def warp(
    image: ImageLike,
    params: ParamsLike,
    voxel_size: VoxelSize,
    *,
    padding_mode: str = "border",
) -> Tensor | np.ndarray:
    """
    Apply a rigid transform to a 3D image.

    Args:
        image:        (D, H, W) float tensor or numpy array.
                      If numpy, the device is chosen automatically
                      (CUDA if available, otherwise CPU).
        params:       length-6 tuple/numpy array [tz, ty, tx, ωz, ωy, ωx].
                      Translations in mm; angles in radians.
        voxel_size:   (vz, vy, vx) in mm.
        padding_mode: 'border' | 'zeros' | 'reflection'.

    Returns:
        Warped image (D, H, W). Returns a numpy array if *image* was a numpy
        array, otherwise a tensor on the same device as *image*.
    """
    return_numpy = isinstance(image, np.ndarray)
    device = _image_device(image)
    img = _to_float_tensor(image, device)
    p = _params_to_tensor(params, device, img.dtype)
    vox = tuple(float(v) for v in voxel_size)

    shape = tuple(img.shape)
    R = _rotvec_to_matrix(p[3:])
    grid = _make_sampling_grid(R, p[:3], shape, vox, shape, vox)
    out = TF.grid_sample(
        img[None, None],
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    result = out[0, 0]
    if return_numpy:
        return result.cpu().numpy()
    return result


def register(
    fixed: ImageLike,
    moving: ImageLike,
    voxel_size: VoxelSize,
    cost_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    init_params: ParamsLike | None = None,
    n_steps: int = 25,
    max_iter: int = 20,
    history_size: int = 10,
    tol: float = 1e-8,
    verbose: bool = True,
) -> Params:
    """
    Single-scale rigid registration using L-BFGS.

    Args:
        fixed:       (D, H, W) reference image (tensor or numpy array).
        moving:      (D, H, W) image to align to fixed (tensor or numpy array).
                     When numpy arrays are given the device is chosen
                     automatically (CUDA if available, otherwise CPU).
                     When tensors are given their device is used as-is.
        voxel_size:  (vz, vy, vx) in mm — tuple or numpy array.
        cost_fn:     metric(fixed, warped) → scalar.  Both args are (D, H, W).
        init_params: length-6 initial guess (tuple, numpy array, or tensor).
                     Defaults to zeros.
        n_steps:     L-BFGS outer iterations.
        max_iter:    max line-search steps per outer iteration.
        history_size: L-BFGS curvature-pair history.
        tol:         relative loss change for early stopping.
        verbose:     print loss each step.

    Returns:
        params as a tuple of 6 floats (tz, ty, tx, ωz, ωy, ωx).
        Call warp(moving, params, voxel_size) to obtain the aligned image.
    """
    device = _image_device(fixed)
    f = _to_float_tensor(fixed, device)
    m = _to_float_tensor(moving, device)
    dtype = f.dtype
    vox = tuple(float(v) for v in voxel_size)
    shape = tuple(f.shape)
    moving5d = m[None, None]

    def _warp_moving(p: Tensor) -> Tensor:
        R = _rotvec_to_matrix(p[3:])
        grid = _make_sampling_grid(R, p[:3], shape, vox, shape, vox)
        return TF.grid_sample(
            moving5d,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, 0]

    if init_params is not None:
        p0 = _params_to_tensor(init_params, device, dtype)
    else:
        p0 = torch.zeros(6, device=device, dtype=dtype)

    params = p0.clone().detach().requires_grad_(True)

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
            loss = cost_fn(f, _warp_moving(params))
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

    result = _tensor_to_params(params)
    if verbose:
        print(
            f"    final parameters: t = [{result[0]:.2f}, {result[1]:.2f}, {result[2]:.2f}] mm,"
            f" rot = [{result[3]:.4f}, {result[4]:.4f}, {result[5]:.4f}] rad"
        )

    return result


def transform_plane(
    n: np.ndarray,  # (3,) normal vector in source space
    d: float,  # signed distance from the physical origin
    params: tuple[float, ...],  # 6 floats (tz, ty, tx, ωz, ωy, ωx)
) -> tuple[np.ndarray, float]:
    """
    Transform a plane  x·n = d  by the rigid transform T(x) = R @ x + t.

    Args:
        n:      (3,) normal vector (need not be unit length).
        d:      Signed distance from the physical origin.
        params: (tz, ty, tx, ωz, ωy, ωx) — same convention as warp().

    Returns:
        (n', d') defining the transformed plane  p·n' = d'.
        n' has the same norm as n.

    To go the other way (inverse transform) pass invert_params(params).
    """
    p = torch.tensor(params, dtype=torch.float64)
    R = _rotvec_to_matrix(p[3:])
    t = p[:3]

    n_ = R @ torch.tensor(n, dtype=torch.float64)
    d_ = torch.tensor(d, dtype=torch.float64) + n_ @ t

    return n_.numpy(), float(d_)
