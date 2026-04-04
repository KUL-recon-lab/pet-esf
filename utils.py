import numpy as np


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
