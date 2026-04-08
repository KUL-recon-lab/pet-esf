import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import find_objects
from scipy.signal import find_peaks

from utils import find_planes
from vis_pet import show_pet_cuts


def save_plane_vectors(path, d):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def load_plane_vectors(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: tuple(v) for k, v in raw.items()}


def find_roots(x, y):
    signs = np.sign(y)
    crossings = np.where(np.diff(signs) != 0)[0]  # indices where sign changes
    roots = []
    for i in crossings:
        # linear interpolation between x[i] and x[i+1]
        roots.append(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return np.array(roots)


# %%
# input parameters
parser = argparse.ArgumentParser()
parser.add_argument("fname", type=Path, help="path to the NIfTI file")
parser.add_argument(
    "--r1", type=float, default=8.0, help="outer ROI radius in mm along edge direction"
)
parser.add_argument(
    "--r2", type=float, default=2.5, help="inner ROI radius in mm perpendicular to edge"
)
parser.add_argument(
    "--pvf",
    type=Path,
    default=None,
    help="file containing the plane vectors in json format",
)
parser.add_argument(
    "--vis",
    action="store_true",
    help="open interactive 3-axis PET viewer with plane overlay",
)

args = parser.parse_args()

fname: Path = args.fname
data_dir: Path = fname.parent
r1: float = args.r1
r2: float = args.r2
pvf: Path | None = args.pvf
vis: bool = args.vis


inner_cube_diam_mm = 29.3
offset_scalar_mm = 6.45
################################################################################
# %%
# READ THE DATA FROM NIFTI, orient to RAS

nii = nib.as_closest_canonical(nib.load(fname))
pet = nii.get_fdata().copy()
voxsize = nii.header["pixdim"][1:4]

center_mm = (np.array(pet.shape) / 2 - 0.5) * voxsize

# setup X,Y,Z coordinates in world coordinates, such that the in plane point it as (0, 0, 0)mm
x = np.arange(pet.shape[0]) * voxsize[0] - center_mm[0]
y = np.arange(pet.shape[1]) * voxsize[1] - center_mm[1]
z = np.arange(pet.shape[2]) * voxsize[2] - center_mm[2]

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# pad slices by 10 voxels in each direction
slices = find_objects((pet > 0.05 * pet.max()).astype(int))[0]
pad = 10
slices = tuple(
    slice(max(0, s.start - pad), min(pet.shape[i], s.stop + pad))
    for i, s in enumerate(slices)
)
X = X[slices]
Y = Y[slices]
Z = Z[slices]

# 1-D world coordinate arrays for the cropped volume (used by the viewer)
x_crop = x[slices[0]]
y_crop = y[slices[1]]
z_crop = z[slices[2]]

V = np.stack([X, Y, Z], axis=-1)
################################################################################
# %%
if pvf is not None and pvf.exists():
    tilted_plane_vectors = load_plane_vectors(pvf)
    print(f"Loaded plane vectors from {pvf}")
else:
    tilted_plane_vectors = find_planes(
        pet, voxsize, inner_cube_diam_mm, offset_scalar_mm
    )
    save_plane_vectors("plane_vectors.json", tilted_plane_vectors)
    print(f"Calculated plane vectors and saved to plane_vectors.json")

################################################################################
# %%
# calculate the distance from the edge

roi_masks = dict()

fig, ax = plt.subplots(3, 6, figsize=(18, 9), layout="constrained")

for ie, edge in enumerate(["xs", "xe", "ys", "ye", "zs", "ze"]):
    p, n = tilted_plane_vectors[edge]
    # get the signed distance of all voxels from the plane
    Vp = V - p
    signed_plane_dist = Vp @ n
    # calculate the perpendicular distance to ray formed by plane origin and normal vector
    perp_dist = np.linalg.norm(Vp - signed_plane_dist[..., None] * n, axis=-1)

    roi_mask = (np.abs(signed_plane_dist) <= r1) * (perp_dist <= r2)
    roi_masks[edge] = roi_mask

    sd = signed_plane_dist[roi_mask].ravel()
    signal = pet[slices][roi_mask].ravel()

    #############################################################################
    # bin the ESF around sample clusters and fit a smoothing spline

    inds = np.argsort(sd)
    sd = sd[inds]
    signal = signal[inds]
    signal /= signal.max()

    gaps = np.diff(sd)

    # threshold: gaps much larger than the within-cluster spread
    # gap_threshold = np.percentile(gaps, 75) * 2.0  # tune if needed
    gap_threshold = np.percentile(gaps, 95) * 3.0  # tune if needed
    split_indices = np.where(gaps > gap_threshold)[0] + 1

    cluster_ids = np.zeros(len(sd), dtype=int)
    cluster_ids[split_indices] = 1
    cluster_ids = np.cumsum(cluster_ids)

    # for each cluster: mean distance and mean signal
    bin_centers = np.array(
        [sd[cluster_ids == c].mean() for c in np.unique(cluster_ids)]
    )
    esf_binned = np.array(
        [signal[cluster_ids == c].mean() for c in np.unique(cluster_ids)]
    )
    esf_std = np.array([signal[cluster_ids == c].std() for c in np.unique(cluster_ids)])

    spline = make_smoothing_spline(bin_centers, esf_binned)

    #############################################################################

    ss, dss = np.linspace(-r1, r1, 2048, retstep=True, endpoint=False)

    esf = spline(ss)
    lsf = spline.derivative(1)(ss)
    nlsf = lsf / lsf.max()

    contrast = esf[-1] / esf[0]

    roots_half = find_roots(ss, nlsf - 0.5)
    if len(roots_half) < 2:
        raise ValueError(f"Could not find two roots for half maximum for edge {edge}")
    if len(roots_half) > 2:
        # take the two roots that are closest to the center (0 mm)
        roots_half = sorted(sorted(roots_half, key=lambda x: abs(x))[:2])
    fwhm = roots_half[1] - roots_half[0]

    roots_tenth = find_roots(ss, nlsf - 0.1)
    if len(roots_tenth) < 2:
        raise ValueError(f"Could not find two roots for tenth maximum for edge {edge}")
    if len(roots_tenth) > 2:
        # take the two roots that are closest to the center (0 mm)
        roots_tenth = sorted(sorted(roots_tenth, key=lambda x: abs(x))[:2])

    fwtm = roots_tenth[1] - roots_tenth[0]

    print(
        f"{data_dir.stem}; {fname.name}; {edge}; contrast: {contrast:.1f}; FWHM: {fwhm:.2f} mm; FWTM: {fwtm:.2f} mm"
    )

    # calculate the MTF
    peak_idx = np.argmax(nlsf)
    lsf_centered = np.roll(nlsf, -peak_idx)

    mtf = np.abs(np.fft.rfft(lsf_centered))
    spatial_freq = np.fft.rfftfreq(len(lsf_centered), dss)  # dss unchanged
    mtf /= mtf[0]

    # %%
    # visualize the results

    ax[0, ie].plot(sd, signal, ".", markersize=3.0, color="C0")
    ax[0, ie].plot(bin_centers, esf_binned, "x", markersize=5.0, color="k")
    ax[0, ie].plot(ss, esf, "-", color="C1")
    ax[1, ie].plot(ss, nlsf, "-")
    ax[2, ie].plot(spatial_freq, mtf, ".-")

    # indicate FWHM in normalized coordinates (0 to 1)
    xmin_ax, xmax_ax = ax[1, ie].get_xlim()
    ax[1, ie].axhline(
        0.5,
        color="gray",
        ls="--",
        xmin=(roots_half[0] - xmin_ax) / (xmax_ax - xmin_ax),
        xmax=(roots_half[-1] - xmin_ax) / (xmax_ax - xmin_ax),
    )
    ax[1, ie].text(
        0.5 * roots_half[1],
        0.5,
        f"FWHM: {fwhm:.2f} mm",
        ha="left",
        va="bottom",
        fontsize="x-small",
    )
    ax[1, ie].axhline(
        0.1,
        color="gray",
        ls="--",
        xmin=(roots_tenth[0] - xmin_ax) / (xmax_ax - xmin_ax),
        xmax=(roots_tenth[-1] - xmin_ax) / (xmax_ax - xmin_ax),
    )
    ax[1, ie].text(
        0.5 * roots_tenth[1],
        0.1,
        f"FWTM: {fwtm:.2f} mm",
        ha="left",
        va="bottom",
        fontsize="x-small",
    )

    ax[0, ie].set_title(f"{edge} contrast: {contrast:.1f}", fontsize="medium")

ax[0, 0].set_ylabel(f"ESF", fontsize="medium")
ax[1, 0].set_ylabel(f"norm. LSF", fontsize="medium")
ax[2, 0].set_ylabel(f"norm. MTF", fontsize="medium")

for axx in ax.ravel():
    axx.grid(ls=":")
for axx in ax[0, :]:
    axx.set_xlabel("distance (mm)")
    axx.set_ylim(-0.05, 1.1)
for axx in ax[1, :]:
    axx.set_xlabel("distance (mm)")
for axx in ax[2, :]:
    axx.set_xlim(0, 0.7)
    axx.set_xlabel("spatial freq. (1/mm)")

fig.suptitle(str(fname), fontsize="small")

if vis:
    counts, bin_edges = np.histogram(
        pet[slices][pet[slices] > 0.1 * pet[slices].max()].ravel(), bins=200
    )
    peaks, _ = find_peaks(counts, distance=5, prominence=200)
    intensities = sorted(bin_edges[peaks])

    _fig_vis, _ax_vis, _widgets_vis = show_pet_cuts(
        pet[slices],
        x_crop,
        y_crop,
        z_crop,
        tilted_plane_vectors=tilted_plane_vectors,
        roi_masks=roi_masks,
        vmax=1.2 * intensities[-1],
        fig_title=str(fname),
    )

plt.show()
