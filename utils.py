import numpy as np


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
