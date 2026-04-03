import os

# Enable scipy's experimental array API support so that scipy.ndimage functions
# dispatch correctly for NumPy, CuPy, and PyTorch CPU arrays uniformly.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import argparse
import json
from pathlib import Path

import numpy as np
import array_api_compat
import array_api_compat.cupy as xp
import nibabel as nib
from scipy.ndimage import (
    find_objects,
    gaussian_filter,
    affine_transform,
)
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation


def to_numpy(arr):
    """Return a NumPy array regardless of whether *arr* is NumPy or CuPy."""
    if hasattr(arr, "get"):  # CuPy ndarray
        return arr.get()
    return np.asarray(arr)


def apply_world_transform(
    image: xp.ndarray, M: xp.ndarray, voxsize: xp.ndarray, order: int = 1
) -> xp.ndarray:
    """Resample image under 4x4 world-space rigid matrix M (rotation around image center)."""
    R, t = M[:3, :3], M[:3, 3]
    D_inv = xp.diag(1.0 / voxsize)
    c = xp.array(image.shape) / 2.0
    mat = D_inv @ R.T @ xp.diag(voxsize)
    offset = c - mat @ c - D_inv @ (R.T @ t)
    return affine_transform(
        image, mat, offset=offset, order=order, mode="constant", cval=0.0
    )


def transform_plane(M: xp.ndarray, normal: xp.ndarray, offset: float):
    """Transform plane (normal · x = offset) in world coords. Returns (new_normal, new_offset)."""
    n_out = M[:3, :3] @ normal
    return n_out, offset + n_out @ M[:3, 3]


def _residuals(params, moving, fixed, voxsize):
    # params is always a numpy array (Rotation requires numpy)
    R = Rotation.from_rotvec(params[3:]).as_matrix()
    M = xp.eye(4)
    M[:3, :3] = xp.asarray(R)
    M[:3, 3] = xp.asarray(params[:3])
    return (apply_world_transform(moving, M, voxsize, order=1) - fixed).ravel()


def _lm(fun, x0, args=(), ftol=1e-10, xtol=1e-10, gtol=1e-10, max_iter=200, lam=1e-8):
    """Levenberg-Marquardt solver whose residuals may be xp (CuPy/NumPy) arrays.

    Only the 6 parameters (x) stay as numpy — Rotation.from_rotvec requires it.
    All Jacobian accumulation and the 6×6 linear solve run on the device via ns.linalg.solve.
    """
    x = to_numpy(x0).astype(np.float64)
    n = len(x)

    f = fun(x, *args)
    ns = array_api_compat.get_namespace(f)  # cupy or numpy namespace

    cost = float(ns.sum(f * f))
    # λ is dimensionless here because the damping term is λ·diag(J^T J).
    # The Moré τ·max(diag(J^T J)) formula applies only to identity-damped LM (λI).
    # For scaled damping, the standard initialisation is simply λ = τ = 1e-3.
    eps = np.sqrt(np.finfo(np.float64).eps)  # ~1.49e-8

    delta_np = np.zeros(n)  # last accepted step (used for xtol check)

    for i_outer in range(max_iter):
        print(i_outer, cost, lam)
        # Forward-difference Jacobian: stack columns into shape (n, N) on the device
        J_cols = []
        for i in range(n):
            h = eps * max(abs(float(x[i])), 1.0)
            xh = x.copy()
            xh[i] += h
            J_cols.append((fun(xh, *args) - f) / h)

        J = ns.stack(J_cols, axis=0)  # (n, N)  — J^T in the usual convention
        JtJ = J @ J.T  # (n, n)  on device
        Jtf = J @ f  # (n,)    on device

        # Gradient convergence
        if float(ns.max(ns.abs(Jtf))) < gtol:
            break

        diag_JtJ = ns.diag(JtJ)

        # Damping loop: try step, adjust λ based on gain ratio
        cost_prev = cost
        step_accepted = False
        while True:
            A = JtJ + lam * ns.diag(diag_JtJ)
            delta_xp = ns.linalg.solve(A, -Jtf)
            delta_np = to_numpy(delta_xp)

            x_new = x + delta_np
            f_new = fun(x_new, *args)
            cost_new = float(ns.sum(f_new * f_new))

            predicted = float(-2.0 * (Jtf @ delta_xp) - delta_xp @ JtJ @ delta_xp)
            rho = (cost - cost_new) / predicted if predicted > 0 else -1.0

            if rho > 0:  # accept
                x, f, cost = x_new, f_new, cost_new
                lam = max(lam / max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), 1e-10)
                step_accepted = True
                break
            else:  # reject, increase damping
                lam = min(lam * 10.0, 1e10)
                if lam >= 1e10:
                    break  # no step improves cost even at max damping → converged

        if not step_accepted:
            break  # inner loop exhausted damping range without progress
        if abs(cost_prev - cost) < ftol * (1.0 + cost_prev):
            break  # relative cost change below threshold
        if np.max(np.abs(delta_np)) < xtol * (np.max(np.abs(x)) + xtol):
            break

    return x


# %%
# input parameters
parser = argparse.ArgumentParser()
parser.add_argument("fname", type=Path, help="path to the NIfTI file")
parser.add_argument(
    "--nbins", type=int, default=256, help="number of bins for rebinning"
)
parser.add_argument(
    "--cube_diam_mm", type=float, default=14.96, help="cube diameter in mm"
)
parser.add_argument(
    "--edge", type=str, default="xs", help="edge direction (xs, xl, ys, yl, zs, zl)"
)
parser.add_argument(
    "--r1", type=float, default=10.0, help="outer ROI radius in mm along edge direction"
)
parser.add_argument(
    "--r2", type=float, default=4.0, help="inner ROI radius in mm perpendicular to edge"
)
parser.add_argument(
    "--lm_lam", type=float, default=1e-10, help="lambda for LM algorithm for alignment"
)
parser.add_argument(
    "--interp_z",
    action="store_true",
    default=False,
    help="interpolate by factor 2 in z after cropping (linear, half-voxel shift)",
)
parser.add_argument(
    "--interp_x",
    action="store_true",
    default=False,
    help="interpolate by factor 2 in x after cropping (linear, half-voxel shift)",
)
parser.add_argument(
    "--interp_y",
    action="store_true",
    default=False,
    help="interpolate by factor 2 in y after cropping (linear, half-voxel shift)",
)
args = parser.parse_args()

fname: Path = args.fname
data_dir: Path = fname.parent
nbins: int = args.nbins
cube_diam_mm: float = args.cube_diam_mm
edge: str = args.edge
r1: float = args.r1
r2: float = args.r2
# lam: float = args.lam
interp_z: bool = args.interp_z
interp_x: bool = args.interp_x
interp_y: bool = args.interp_y
lm_lam: float = args.lm_lam

inner_cube_diam_mm = 29.3
offset_scalar_mm = 6.45
################################################################################
# %%
# READ THE DATA FROM NIFTI, orient to RAS

nii = nib.as_closest_canonical(nib.load(fname))
pet = xp.asarray(nii.get_fdata())
voxsize = xp.array(nii.header["pixdim"][1:4])  # small metadata array, kept as numpy


################################################################################
# %%
# crop pet_vol to bounding box of non-zero values

slices_file = fname.parent / (fname.stem + "_slices.json")
# if slices_file.exists():
if False:
    with open(slices_file, "r") as f:
        slices_json = json.load(f)
    slices = tuple(slice(s["start"], s["stop"]) for s in slices_json)
else:
    pet_sm = gaussian_filter(pet, 4.0 / (2.35 * voxsize))
    slices = find_objects((pet_sm > 0.05 * pet_sm.max()).astype(xp.int32))[0]
    del pet_sm

    # pad slices by 5 voxels in each direction
    pad = 10
    slices = tuple(
        slice(max(0, s.start - pad), min(pet.shape[i], s.stop + pad))
        for i, s in enumerate(slices)
    )

    # write slices to json for later use
    with open(slices_file, "w") as f:
        json.dump([{"start": s.start, "stop": s.stop} for s in slices], f)

pet_full = pet.copy()
pet = pet[slices]

if interp_x:
    # Interpolate by factor 2 in x using half-voxel-shifted sampling.
    nx = pet.shape[0]
    pet_interp = xp.empty((2 * nx - 1, pet.shape[1], pet.shape[2]), dtype=pet.dtype)
    pet_interp[0::2, :, :] = pet  # original planes
    pet_interp[1::2, :, :] = 0.5 * (pet[:-1, :, :] + pet[1:, :, :])  # midpoints
    pet = pet_interp
    voxsize = voxsize.copy()
    voxsize[0] /= 2

if interp_y:
    # Interpolate by factor 2 in y using half-voxel-shifted sampling.
    ny = pet.shape[1]
    pet_interp = xp.empty((pet.shape[0], 2 * ny - 1, pet.shape[2]), dtype=pet.dtype)
    pet_interp[:, 0::2, :] = pet  # original planes
    pet_interp[:, 1::2, :] = 0.5 * (pet[:, :-1, :] + pet[:, 1:, :])  # midpoints
    pet = pet_interp
    voxsize = voxsize.copy()
    voxsize[1] /= 2

if interp_z:
    # Interpolate by factor 2 in z using half-voxel-shifted sampling.
    # New sampling positions in original voxel coords: 0, 0.5, 1, 1.5, ...
    # Even new indices map to original planes (no interpolation needed).
    # Odd new indices are the average of two neighboring original planes.
    nz = pet.shape[2]
    pet_interp = xp.empty((pet.shape[0], pet.shape[1], 2 * nz - 1), dtype=pet.dtype)
    pet_interp[:, :, 0::2] = pet  # original planes
    pet_interp[:, :, 1::2] = 0.5 * (pet[:, :, :-1] + pet[:, :, 1:])  # midpoints
    pet = pet_interp
    voxsize = voxsize.copy()
    voxsize[2] /= 2

# ensure odd size in all dimensions by padding with a single zero if needed
pad_width = [(0, s % 2 == 0) for s in pet.shape]
if any(p[1] for p in pad_width):
    pet = xp.pad(pet, pad_width)

################################################################################
# %%
# get the 3 dominant intensity values in the pet image, which should correspond to the two cubes and the overlap
counts, bin_edges = xp.histogram(pet[pet > 0.1 * pet.max()].ravel(), bins=200)
peaks, _ = find_peaks(counts, distance=5, prominence=200)
intensities = sorted(bin_edges[peaks])

################################################################################
# %%
# center of the mass of the image in voxel coordinates
center = xp.array(pet.shape) // 2
# world coordinates, such that center voxel is at (0m, 0m, 0m)
# to go from voxel index to world coordinate: world = (index - center) * voxsize
# to go from world coordinate to voxel index: index = world / voxsize + center
center_mm = (center - center) * voxsize

# this is the diameter of a the single "filled" cube
inner_cube_diam_vox = xp.round(inner_cube_diam_mm / voxsize).astype(int)
# the center offset of the two cubes, before adding them (half the distance between the centers of the two cubes)
offset_mm = offset_scalar_mm * xp.array([-1.0, 1.0, 1.0])
offset_vox = xp.round(offset_mm / voxsize).astype(int)
# offset_vox[0] *= -1

mask1 = xp.zeros_like(pet)
mask2 = xp.zeros_like(pet)
mask = xp.zeros_like(pet)

# first cube mask
c1 = center - offset_vox
c1_mm = center_mm - offset_mm

start1 = c1 - inner_cube_diam_vox // 2
start1_mm = c1_mm - inner_cube_diam_mm / 2

end1 = start1 + inner_cube_diam_vox
end1_mm = start1_mm + inner_cube_diam_mm

mask1[start1[0] : end1[0], start1[1] : end1[1], start1[2] : end1[2]] += (
    1.0 * intensities[0]
)

# second cube mask
c2 = center + offset_vox
start2 = c2 - inner_cube_diam_vox // 2
end2 = start2 + inner_cube_diam_vox
mask2[start2[0] : end2[0], start2[1] : end2[1], start2[2] : end2[2]] += (
    1.0 * intensities[1]
)

mask = mask1 + mask2

################################################################################
# %%
# align pet to mask using rigid registration (least squares / LM)


scale = float(mask.max())
# least_squares is CPU-only; convert to numpy for the registration solve
pet_scaled = pet / scale
mask_scaled = mask / scale

reg_params = _lm(
    _residuals,
    np.zeros(6),
    args=(pet_scaled, mask_scaled, voxsize),
    ftol=1e-10,
    xtol=1e-10,
    lam=lm_lam,
)

R_reg = Rotation.from_rotvec(reg_params[3:]).as_matrix()
M_reg = xp.eye(4)
M_reg[:3, :3] = xp.asarray(R_reg)
M_reg[:3, 3] = xp.asarray(reg_params[:3])
M_reg_inv = xp.linalg.inv(M_reg)

pet_aligned = apply_world_transform(pet, M_reg, voxsize, order=1)

################################################################################
# %%
# world coordinates of the start and end of the edge between the two cubes
start_mm = center_mm - inner_cube_diam_mm / 2 + xp.abs(offset_mm)
end_mm = center_mm + inner_cube_diam_mm / 2 - xp.abs(offset_mm)

# voxel coordinates of the start and end of the edge between the two cubes (float)
start = start_mm / voxsize + center
end = end_mm / voxsize + center
