import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
import pymirc.viewer as pv

from utils import rebin_xy, find_planes


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
    "--nbins", type=int, default=128, help="number of bins for rebinning"
)
parser.add_argument(
    "--edge",
    type=str,
    default="xs",
    help="edge direction",
    choices=["xs", "xe", "ys", "ye", "zs", "ze"],
)
parser.add_argument(
    "--r1", type=float, default=8.0, help="outer ROI radius in mm along edge direction"
)
parser.add_argument(
    "--r2", type=float, default=2.5, help="inner ROI radius in mm perpendicular to edge"
)
parser.add_argument(
    "--lam", type=float, default=0.2, help="smoothing spline regularization parameter"
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
edge: str = args.edge
r1: float = args.r1
r2: float = args.r2
lam: float = args.lam
interp_z: bool = args.interp_z
interp_x: bool = args.interp_x
interp_y: bool = args.interp_y

inner_cube_diam_mm = 29.2
offset_scalar_mm = 6.6
################################################################################
# %%
# READ THE DATA FROM NIFTI, orient to RAS

nii = nib.as_closest_canonical(nib.load(fname))
pet = nii.get_fdata().copy()
voxsize = nii.header["pixdim"][1:4]
full_img_shape = np.array(pet.shape)


################################################################################
# %%
tilted_plane_vectors = find_planes(pet, voxsize, inner_cube_diam_mm, offset_scalar_mm)
p, n = tilted_plane_vectors[edge]

################################################################################
# %%
# calculate the distance from the edge

# assert wheter pet.shape is full_img_shape
assert np.all(
    np.array(pet.shape) == full_img_shape
), f"pet shape {pet.shape} does not match full image shape {full_img_shape}"

center_mm = (full_img_shape / 2 - 0.5) * voxsize

# setup X,Y,Z coordinates in world coordinates, such that the in plane point it as (0, 0, 0)mm
x = np.arange(pet.shape[0]) * voxsize[0] - center_mm[0]
y = np.arange(pet.shape[1]) * voxsize[1] - center_mm[1]
z = np.arange(pet.shape[2]) * voxsize[2] - center_mm[2]
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# get the signed distance of all voxels from the plane
V = np.stack([X, Y, Z], axis=-1) - p
signed_plane_dist = V @ n
# calculate the perpendicular distance to ray formed by plane origin and normal vector
perp_dist = np.linalg.norm(V - signed_plane_dist[..., None] * n, axis=-1)

roi_mask = (np.abs(signed_plane_dist) <= r1) * (perp_dist <= r2)

sd = signed_plane_dist[roi_mask].ravel()
signal = pet[roi_mask].ravel()

#################################################################################
# %%
# fit a smoothing spline to the data and calculate the ESF, LSF, and FWHM

inds = np.argsort(sd)
sd = sd[inds]
signal = signal[inds]
signal /= signal[np.argmax(sd)]

sd_binned, signal_binned = rebin_xy(sd, signal, nbins=nbins)

spline = make_smoothing_spline(sd_binned, signal_binned, lam=lam)

ss = np.linspace(sd.min(), sd.max(), 1000)

esf = spline(ss)
lsf = spline.derivative(1)(ss)
nlsf = lsf / lsf.max()

contrast = esf[-1] / esf[0]

roots_half = find_roots(ss, nlsf - 0.5)
fwhm = roots_half[1] - roots_half[0]

roots_tenth = find_roots(ss, nlsf - 0.1)
fwtm = roots_tenth[1] - roots_tenth[0]

print(
    f"{data_dir.stem}; {fname.name}; {edge}; contrast: {contrast:.1f}; FWHM: {fwhm:.2f} mm; FWTM: {fwtm:.2f} mm"
)

################################################################################
# %%
# visualize the results

fig, ax = plt.subplots(1, 2, figsize=(12, 6), layout="constrained", sharex=True)
ax[0].plot(sd, signal, ".", markersize=0.2, color="k")
ax[0].plot(sd_binned, signal_binned, ".", markersize=3.0, color="C0")
ax[0].plot(ss, esf, "-", color="C1")
# ax[0].set_ylim(-0.05, 1.05 * esf.max())
ax[1].plot(ss, nlsf, "-")

# indicate FWHM in normalized coordinates (0 to 1)
xmin_ax, xmax_ax = ax[1].get_xlim()
ax[1].axhline(
    0.5,
    color="gray",
    ls="--",
    xmin=(roots_half[0] - xmin_ax) / (xmax_ax - xmin_ax),
    xmax=(roots_half[-1] - xmin_ax) / (xmax_ax - xmin_ax),
)
ax[1].axhline(
    0.1,
    color="gray",
    ls="--",
    xmin=(roots_tenth[0] - xmin_ax) / (xmax_ax - xmin_ax),
    xmax=(roots_tenth[-1] - xmin_ax) / (xmax_ax - xmin_ax),
)

for axx in ax:
    axx.grid(ls=":")
    axx.set_xlabel("Distance from edge (mm)")

ax[0].set_title("ESF", fontsize="medium")
ax[1].set_title("normalized LSF", fontsize="medium")

fig.show()
