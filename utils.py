import numpy as np
from scipy.ndimage import find_objects, gaussian_filter
from scipy.signal import find_peaks

from utils_align import register, transform_plane
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


def find_planes(
    pet_vol, voxsize, inner_cube_diam_mm: float, offset_scalar_mm: float
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    # crop pet_vol to bounding box of non-zero values
    pet_sm = gaussian_filter(pet_vol, 4.0 / (2.35 * voxsize))
    slices = find_objects((pet_sm > 0.05 * pet_sm.max()).astype(int))[0]
    del pet_sm

    # pad slices by 10 voxels in each direction
    pad = 10
    slices = tuple(
        slice(max(0, s.start - pad), min(pet_vol.shape[i], s.stop + pad))
        for i, s in enumerate(slices)
    )

    pet = pet_vol[slices]

    ############################################################################
    # Interpolate by factor 4 in x.
    # New sampling positions in original voxel coords: 0, 0.25, 0.5, 0.75, 1, ...
    # Every 4th new index maps to an original plane; the 3 in-between indices are
    # linearly interpolated at 1/4, 1/2, and 3/4 between neighboring original planes.
    nx = pet.shape[0]
    pet_interp = np.empty((4 * nx - 3, pet.shape[1], pet.shape[2]), dtype=pet.dtype)
    pet_interp[0::4, :, :] = pet  # original planes
    pet_interp[1::4, :, :] = 0.75 * pet[:-1, :, :] + 0.25 * pet[1:, :, :]  # 1/4 point
    pet_interp[2::4, :, :] = 0.50 * pet[:-1, :, :] + 0.50 * pet[1:, :, :]  # midpoint
    pet_interp[3::4, :, :] = 0.25 * pet[:-1, :, :] + 0.75 * pet[1:, :, :]  # 3/4 point
    pet = pet_interp

    # Interpolate by factor 4 in y.
    ny = pet.shape[1]
    pet_interp = np.empty((pet.shape[0], 4 * ny - 3, pet.shape[2]), dtype=pet.dtype)
    pet_interp[:, 0::4, :] = pet
    pet_interp[:, 1::4, :] = 0.75 * pet[:, :-1, :] + 0.25 * pet[:, 1:, :]
    pet_interp[:, 2::4, :] = 0.50 * pet[:, :-1, :] + 0.50 * pet[:, 1:, :]
    pet_interp[:, 3::4, :] = 0.25 * pet[:, :-1, :] + 0.75 * pet[:, 1:, :]
    pet = pet_interp

    # Interpolate by factor 4 in z.
    nz = pet.shape[2]
    pet_interp = np.empty((pet.shape[0], pet.shape[1], 4 * nz - 3), dtype=pet.dtype)
    pet_interp[:, :, 0::4] = pet
    pet_interp[:, :, 1::4] = 0.75 * pet[:, :, :-1] + 0.25 * pet[:, :, 1:]
    pet_interp[:, :, 2::4] = 0.50 * pet[:, :, :-1] + 0.50 * pet[:, :, 1:]
    pet_interp[:, :, 3::4] = 0.25 * pet[:, :, :-1] + 0.75 * pet[:, :, 1:]
    pet = pet_interp

    voxsize_interp = voxsize / 4.0
    #############################################################################

    # get the 3 dominant intensity values in the pet image, which should correspond to the two cubes and the overlap
    counts, bin_edges = np.histogram(pet[pet > 0.1 * pet.max()].ravel(), bins=200)
    peaks, _ = find_peaks(counts / counts.max(), distance=20, prominence=0.05)
    intensities = sorted(bin_edges[peaks])

    # rigidly coregister the pet to the reference image

    ref_img = make_two_cubes(
        inner_cube_diam_mm,
        offset_scalar_mm,
        pet.shape,
        voxsize_interp,
        intensities[0],
        intensities[1],
    )
    coreg_params, final_loss = register(ref_img, pet, voxsize_interp, ncc)

    # setup the start and end coordinates of the inner overlapping cube in world coordinates (of the cropped image)
    offset_mm = offset_scalar_mm * np.array([-1.0, 1.0, 1.0])

    # world coordinates of the start and end of the edge between the two cubes
    start_mm = -inner_cube_diam_mm / 2 + np.abs(offset_mm)
    end_mm = inner_cube_diam_mm / 2 - np.abs(offset_mm)

    #################
    # from utils_align import warp
    # import pymirc.viewer as pv

    # pet_coreg = warp(pet, coreg_params, voxsize_interp)  # (D, H, W) tensor
    # vi = pv.ThreeAxisViewer(
    #    pet_coreg - ref_img,
    #    voxsize=voxsize_interp,
    #    imshow_kwargs=dict(cmap="bwr", vmin=-pet.max(), vmax=pet.max()),
    # )
    # breakpoint()

    ################################################################################
    # %%
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

    # setup plane origin and normal vector in the coregistered space, which is the same as the reference image space
    vectors = dict()

    vectors["x0"] = (
        np.asarray([-0.5 * inner_cube_diam_mm - offset_scalar_mm, 0, 0]),
        np.asarray([1.0, 0.0, 0.0]),
    )
    vectors["x1"] = (
        np.asarray([0.5 * inner_cube_diam_mm + offset_scalar_mm, 0, 0]),
        np.asarray([-1.0, 0.0, 0.0]),
    )

    vectors["xs"] = (np.asarray([start_mm[0], 0, 0]), np.asarray([1.0, 0.0, 0.0]))

    vectors["xe"] = (
        np.asarray([end_mm[0], 0, 0]),
        np.asarray([-1.0, 0.0, 0.0]),
    )

    vectors["y0"] = (
        np.asarray([0, -0.5 * inner_cube_diam_mm - offset_scalar_mm, 0]),
        np.asarray([0.0, 1.0, 0.0]),
    )
    vectors["y1"] = (
        np.asarray([0, 0.5 * inner_cube_diam_mm + offset_scalar_mm, 0]),
        np.asarray([0.0, -1.0, 0.0]),
    )

    vectors["ys"] = (
        np.asarray([0, start_mm[1], 0]),
        np.asarray([0.0, 1.0, 0.0]),
    )

    vectors["ye"] = (
        np.asarray([0, end_mm[1], 0]),
        np.asarray([0.0, -1.0, 0.0]),
    )

    vectors["z0"] = (
        np.asarray([0, 0, -0.5 * inner_cube_diam_mm - offset_scalar_mm]),
        np.asarray([0, 0.0, 1.0]),
    )
    vectors["z1"] = (
        np.asarray([0, 0, 0.5 * inner_cube_diam_mm + offset_scalar_mm]),
        np.asarray([0.0, 0.0, -1.0]),
    )

    vectors["zs"] = (
        np.asarray([0, 0, start_mm[2]]),
        np.asarray([0.0, 0.0, 1.0]),
    )

    vectors["ze"] = (
        np.asarray([0, 0, end_mm[2]]),
        np.asarray([0.0, 0.0, -1.0]),
    )

    # transform vectors to tilted space

    vectors_tilted = dict()

    for key, val in vectors.items():
        p_tilted, n_tilted = transform_plane(val[0], val[1], coreg_params)
        p_tilted += center_shift_mm
        vectors_tilted[key] = (
            [float(x) for x in p_tilted],
            [float(x) for x in n_tilted],
        )

    return vectors_tilted, final_loss
