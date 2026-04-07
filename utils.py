import numpy as np
from scipy.ndimage import find_objects, gaussian_filter
from scipy.signal import find_peaks

from utils_align import register, warp, invert_params, transform_plane
from metrics import ncc


def make_two_cubes(
    d: float,
    o: float,
    shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    val1: float = 1.0,
    val2: float = 1.0,
    offset_signs=(-1, 1, 1),
) -> np.ndarray:
    """
    Create a float voxel volume containing two axis-aligned cubes.

    Parameters
    ----------
    d          : side length of each cube (same in all directions) [mm]
    o          : offset magnitude; cube centers at (+o,+o,+o) and (-o,-o,-o) [mm]
    shape      : (nx, ny, nz) – number of voxels along each axis
    voxel_size : (vx, vy, vz) – physical size of one voxel [mm]
    val1       : intensity value for the cube centred at (+o,+o,+o)
    val2       : intensity value for the cube centred at (-o,-o,-o)

    Returns
    -------
    volume : np.ndarray, dtype float32, shape (nx, ny, nz)
        Voxel values are val1, val2, or val1+val2 in the overlap region.
    """
    nx, ny, nz = shape
    vx, vy, vz = voxel_size
    half_d = d / 2.0

    # World coordinates of every voxel centre; volume centre = world (0,0,0)
    xs = (np.arange(nx) - (nx - 1) / 2.0) * vx
    ys = (np.arange(ny) - (ny - 1) / 2.0) * vy
    zs = (np.arange(nz) - (nz - 1) / 2.0) * vz

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    def cube_mask(cx, cy, cz) -> np.ndarray:
        return (
            (np.abs(X - cx) <= half_d)
            & (np.abs(Y - cy) <= half_d)
            & (np.abs(Z - cz) <= half_d)
        )

    o0 = offset_signs[0] * o
    o1 = offset_signs[1] * o
    o2 = offset_signs[2] * o

    volume = (
        cube_mask(-o0, -o1, -o2).astype(np.float32) * val1
        + cube_mask(o0, o1, o2).astype(np.float32) * val2
    )

    return volume


def rebin_xy(x, y, nbins=256, x_min=None, x_max=None):
    x = np.asarray(x)
    y = np.asarray(y)

    if x_min is None:
        x_min = np.nanmin(x)
    if x_max is None:
        x_max = np.nanmax(x)

    # Bin edges and centers
    edges = np.linspace(x_min, x_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Assign each x to a bin index in [0, nbins-1]
    idx = np.digitize(x, edges) - 1
    idx = np.clip(idx, 0, nbins - 1)

    # Aggregate y per bin
    sums = np.bincount(idx, weights=y, minlength=nbins)
    counts = np.bincount(idx, minlength=nbins)

    yb = sums / np.maximum(counts, 1)

    # Keep only bins that actually have data
    mask = counts > 0
    return centers[mask], yb[mask]


def find_planes(pet_vol, voxsize, inner_cube_diam_mm: float, offset_scalar_mm: float):
    # crop pet_vol to bounding box of non-zero values
    pet_sm = gaussian_filter(pet_vol, 4.0 / (2.35 * voxsize))
    slices = find_objects((pet_sm > 0.05 * pet_sm.max()).astype(int))[0]
    del pet_sm

    # pad slices by 5 voxels in each direction
    pad = 10
    slices = tuple(
        slice(max(0, s.start - pad), min(pet_vol.shape[i], s.stop + pad))
        for i, s in enumerate(slices)
    )

    pet = pet_vol[slices]

    #############################################################################
    ## Interpolate by factor 2 in x using half-voxel-shifted sampling.
    # nx = pet.shape[0]
    # pet_interp = np.empty((2 * nx - 1, pet.shape[1], pet.shape[2]), dtype=pet.dtype)
    # pet_interp[0::2, :, :] = pet  # original planes
    # pet_interp[1::2, :, :] = 0.5 * (pet[:-1, :, :] + pet[1:, :, :])  # midpoints
    # pet = pet_interp

    ## Interpolate by factor 2 in y using half-voxel-shifted sampling.
    # ny = pet.shape[1]
    # pet_interp = np.empty((pet.shape[0], 2 * ny - 1, pet.shape[2]), dtype=pet.dtype)
    # pet_interp[:, 0::2, :] = pet  # original planes
    # pet_interp[:, 1::2, :] = 0.5 * (pet[:, :-1, :] + pet[:, 1:, :])  # midpoints
    # pet = pet_interp

    ## Interpolate by factor 2 in z using half-voxel-shifted sampling.
    ## New sampling positions in original voxel coords: 0, 0.5, 1, 1.5, ...
    ## Even new indices map to original planes (no interpolation needed).
    ## Odd new indices are the average of two neighboring original planes.
    # nz = pet.shape[2]
    # pet_interp = np.empty((pet.shape[0], pet.shape[1], 2 * nz - 1), dtype=pet.dtype)
    # pet_interp[:, :, 0::2] = pet  # original planes
    # pet_interp[:, :, 1::2] = 0.5 * (pet[:, :, :-1] + pet[:, :, 1:])  # midpoints
    # pet = pet_interp

    # voxsize = voxsize.copy()
    # voxsize /= 2.0
    #############################################################################

    # get the 3 dominant intensity values in the pet image, which should correspond to the two cubes and the overlap
    counts, bin_edges = np.histogram(pet[pet > 0.1 * pet.max()].ravel(), bins=200)
    peaks, _ = find_peaks(counts, distance=5, prominence=200)
    intensities = sorted(bin_edges[peaks])

    # rigidly coregister the pet to the reference image

    ref_img = make_two_cubes(
        inner_cube_diam_mm,
        offset_scalar_mm,
        pet.shape,
        voxsize,
        intensities[0],
        intensities[1],
    )
    coreg_params = register(ref_img, pet, voxsize, ncc)
    pet_coreg = warp(pet, coreg_params, voxsize)  # (D, H, W) tensor

    inv_coreg_params = invert_params(coreg_params)
    mask_coreg = warp(ref_img, inv_coreg_params, voxsize)

    # setup the start and end coordinates of the inner overlapping cube in world coordinates (of the cropped image)
    center_mm = np.zeros(3)
    offset_mm = offset_scalar_mm * np.array([-1.0, 1.0, 1.0])

    # world coordinates of the start and end of the edge between the two cubes
    start_mm = center_mm - inner_cube_diam_mm / 2 + np.abs(offset_mm)
    end_mm = center_mm + inner_cube_diam_mm / 2 - np.abs(offset_mm)

    ################################################################################
    # %%
    edge = "xe"

    if edge.startswith("x"):
        if edge == "xs":
            n_coreg = np.asarray([1.0, 0.0, 0.0])  # normal vector along x-axis
            p_coreg = np.asarray([start_mm[0], 0, 0])
        elif edge == "xe":
            n_coreg = np.asarray([-1.0, 0.0, 0.0])  # normal vector along x-axis
            p_coreg = np.asarray([end_mm[0], 0, 0])
        else:
            raise NotImplementedError("invalid edge mode")
    elif edge.startswith("y"):
        if edge == "ys":
            n_coreg = np.asarray([0.0, 1.0, 0.0])  # normal vector along y-axis
            p_coreg = np.asarray([0, start_mm[1], 0])
        elif edge == "ye":
            n_coreg = np.asarray([0.0, -1.0, 0.0])  # normal vector along y-axis
            p_coreg = np.asarray([0, end_mm[1], 0])
        else:
            raise NotImplementedError("invalid edge mode")
    elif edge.startswith("z"):
        if edge == "zs":
            n_coreg = np.asarray([0.0, 0.0, 1.0])  # normal vector along z-axis
            p_coreg = np.asarray([0, 0, start_mm[2]])
        elif edge == "ze":
            n_coreg = np.asarray([0.0, 0.0, -1.0])  # normal vector along z-axis
            p_coreg = np.asarray([0, 0, end_mm[2]])
        else:
            raise NotImplementedError("invalid edge mode")
    else:
        raise NotImplementedError("invalid edge mode")

    # transform the plane parameters (normal vector and in plane point) to the original space
    p, n = transform_plane(p_coreg, n_coreg, coreg_params)

    # shift the origin vector of the plane center back to the uncropped image space
    center_shift_mm = np.zeros(3)

    center_shift_mm[0] = (
        0.5 * (slices[0].start - pet_vol.shape[0] + slices[0].stop) * voxsize[0]
    )
    center_shift_mm[1] = (
        0.5 * (slices[1].start - pet_vol.shape[1] + slices[1].stop) * voxsize[1]
    )
    center_shift_mm[2] = (
        0.5 * (slices[2].start - pet_vol.shape[2] + slices[2].stop) * voxsize[2]
    )

    p[0] += center_shift_mm[0]
    p[1] += center_shift_mm[1]
    p[2] += center_shift_mm[2]

    return p, n
