import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline, InterpolatedUnivariateSpline
from scipy.ndimage import find_objects, center_of_mass, gaussian_filter
from scipy.signal import find_peaks
import pymirc.viewer as pv
import pymirc.image_operations as pymi

from utils import rebin_xy

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
    "--lam", type=float, default=0.5, help="smoothing spline regularization parameter"
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
lam: float = args.lam
interp_z: bool = args.interp_z
interp_x: bool = args.interp_x
interp_y: bool = args.interp_y

################################################################################
# %%
# READ THE DATA FROM NIFTI, orient to RAS

nii = nib.as_closest_canonical(nib.load(fname))
pet = nii.get_fdata()
voxsize = nii.header["pixdim"][1:4]


################################################################################
# %%
# crop pet_vol to bounding box of non-zero values

slices_file = fname.parent / (fname.stem + "_slices.json")
if slices_file.exists():
    with open(slices_file, "r") as f:
        slices_json = json.load(f)
    slices = tuple(slice(s["start"], s["stop"]) for s in slices_json)
else:
    pet_sm = gaussian_filter(pet, 4.0 / (2.35 * voxsize))
    slices = find_objects((pet_sm > 0.05 * pet_sm.max()).astype(int))[0]
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
    pet_interp = np.empty((2 * nx - 1, pet.shape[1], pet.shape[2]), dtype=pet.dtype)
    pet_interp[0::2, :, :] = pet  # original planes
    pet_interp[1::2, :, :] = 0.5 * (pet[:-1, :, :] + pet[1:, :, :])  # midpoints
    pet = pet_interp
    voxsize = voxsize.copy()
    voxsize[0] /= 2

if interp_y:
    # Interpolate by factor 2 in y using half-voxel-shifted sampling.
    ny = pet.shape[1]
    pet_interp = np.empty((pet.shape[0], 2 * ny - 1, pet.shape[2]), dtype=pet.dtype)
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
    pet_interp = np.empty((pet.shape[0], pet.shape[1], 2 * nz - 1), dtype=pet.dtype)
    pet_interp[:, :, 0::2] = pet  # original planes
    pet_interp[:, :, 1::2] = 0.5 * (pet[:, :, :-1] + pet[:, :, 1:])  # midpoints
    pet = pet_interp
    voxsize = voxsize.copy()
    voxsize[2] /= 2

################################################################################
# %%
# get the 3 dominant intensity values in the pet image, which should correspond to the two cubes and the overlap
counts, bin_edges = np.histogram(pet[pet > 0.1 * pet.max()].ravel(), bins=200)
peaks, _ = find_peaks(counts, distance=5, prominence=200)
intensities = sorted(bin_edges[peaks])

################################################################################
# %%
center = np.array(center_of_mass(pet)).astype(int)

# this is the diameter of a the single "filled" cube
inner_cube_diam_mm = 29.6  # (50 - 6) * 2 / 3  # ~30 mm
inner_cube_diam_vox = np.round(inner_cube_diam_mm / voxsize).astype(int)
# the center offset of the two cubes, before adding them
offset_vox = np.round(6.45 / voxsize).astype(int)
offset_vox[0] *= -1

mask = np.zeros_like(pet)
for i_s, sign in enumerate((-1, +1)):
    c = center + sign * offset_vox
    start2 = c - inner_cube_diam_vox // 2
    end2 = start2 + inner_cube_diam_vox
    mask[start2[0] : end2[0], start2[1] : end2[1], start2[2] : end2[2]] += (
        1.0 * intensities[i_s]
    )

# mask *= float(pet.sum() / mask.sum())

ddx = (inner_cube_diam_vox // 2) - 2 * offset_vox * np.sign(offset_vox)
start = center - np.sign(offset_vox) * (offset_vox) - ddx
end = center + np.sign(offset_vox) * (offset_vox) + ddx


################################################################################
# %%
# rigidly coregister the pet to the reference image

coreg_file = fname.parent / (fname.stem + "_coreg.txt")

if coreg_file.exists():
    coreg_aff = np.loadtxt(coreg_file)
else:
    _, coreg_aff, coreg_params = pymi.rigid_registration(
        pet,
        mask,
        nii.affine,
        nii.affine,
        metric=lambda x, y: np.sum((x - y) ** 2),
        opts=dict(ftol=1e-4, xtol=1e-4, maxiter=1000, disp=False),
    )

    np.savetxt(coreg_file, coreg_aff)

pet_coreg = pymi.aff_transform(pet, coreg_aff, pet.shape)
mask_coreg = pymi.aff_transform(mask, np.linalg.inv(coreg_aff), pet.shape)

# fig, ax = plt.subplots()
# ax.plot(pet_coreg[:, 81, 69] / 1.9e6)
# ax.plot(mask[:, 81, 69] / 1.9e6)
# ax.grid(ls=":")
# fig.show()
################################################################################
# %%
# create a ROI image indicating the region of interest used for LSF
roi_img = np.zeros_like(pet)

if edge.startswith("x"):
    if edge == "xs":
        start_x = int(start[0] - r1 / voxsize[0])
        end_x = int(start[0] + r1 / voxsize[0])
        ps = np.array([start[0], (end[1] + start[1]) // 2, (end[2] + start[2]) // 2])
        ns = np.array([1, 0, 0])
    else:
        start_x = int(end[0] - r1 / voxsize[0])
        end_x = int(end[0] + r1 / voxsize[0])
        ps = np.array([end[0], (end[1] + start[1]) // 2, (end[2] + start[2]) // 2])
        ns = np.array([-1, 0, 0])

    start_y = int(0.5 * (start[1] + end[1]) - r2 / voxsize[1])
    end_y = int(0.5 * (start[1] + end[1]) + r2 / voxsize[1])
    start_z = int(0.5 * (start[2] + end[2]) - r2 / voxsize[2])
    end_z = int(0.5 * (start[2] + end[2]) + r2 / voxsize[2])
elif edge.startswith("y"):
    if edge == "ys":
        start_y = int(start[1] - r1 / voxsize[1])
        end_y = int(start[1] + r1 / voxsize[1])
        ps = np.array([(end[0] + start[0]) // 2, start[1], (end[2] + start[2]) // 2])
        ns = np.array([0, 1, 0])
    else:
        start_y = int(end[1] - r1 / voxsize[1])
        end_y = int(end[1] + r1 / voxsize[1])
        ps = np.array([(end[0] + start[0]) // 2, end[1], (end[2] + start[2]) // 2])
        ns = np.array([0, -1, 0])

    start_x = int(0.5 * (start[0] + end[0]) - r2 / voxsize[0])
    end_x = int(0.5 * (start[0] + end[0]) + r2 / voxsize[0])
    start_z = int(0.5 * (start[2] + end[2]) - r2 / voxsize[2])
    end_z = int(0.5 * (start[2] + end[2]) + r2 / voxsize[2])
elif edge.startswith("z"):
    if edge == "zs":
        start_z = int(start[2] - r1 / voxsize[2])
        end_z = int(start[2] + r1 / voxsize[2])
        ps = np.array([(end[0] + start[0]) // 2, (end[1] + start[1]) // 2, start[2]])
        ns = np.array([0, 0, 1])
    else:
        start_z = int(end[2] - 6 / voxsize[2])
        end_z = int(end[2] + 6 / voxsize[2])
        ps = np.array([(end[0] + start[0]) // 2, (end[1] + start[1]) // 2, end[2]])
        ns = np.array([0, 0, -1])

    start_x = int(0.5 * (start[0] + end[0]) - r2 / voxsize[0])
    end_x = int(0.5 * (start[0] + end[0]) + r2 / voxsize[0])
    start_y = int(0.5 * (start[1] + end[1]) - r2 / voxsize[1])
    end_y = int(0.5 * (start[1] + end[1]) + r2 / voxsize[1])
else:
    raise NotImplementedError("invalid edge mode")

start_x = max(0, start_x)
start_y = max(0, start_y)
start_z = max(0, start_z)
end_x = min(pet.shape[0], end_x)
end_y = min(pet.shape[1], end_y)
end_z = min(pet.shape[2], end_z)

roi_img[start_x:end_x, start_y:end_y, start_z:end_z] = 1
################################################################################
# %%
# calculate the distance from the edge

# transform plane origin and normal vector to the tilted space
p = (coreg_aff @ np.array([ps[0], ps[1], ps[2], 1]))[:3]
n = coreg_aff[:3, :3] @ ns


# setup X,Y,Z coordinates in voxel space
x = np.arange(pet.shape[0])
y = np.arange(pet.shape[1])
z = np.arange(pet.shape[2])
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")


# Calculate distance from all points to the plane
# Plane equation: n · (point - p) = 0, or equivalently n · point - n · p = 0
# Distance from point to plane: (x-p) dot n / ||n||, our n is already normalized

signed_distance = (
    (X - p[0]) * n[0] * voxsize[0]
    + (Y - p[1]) * n[1] * voxsize[1]
    + (Z - p[2]) * n[2] * voxsize[2]
)

################################################################################
# %%

# transform the roi_img to the tilted space
roi_img_coreg = pymi.aff_transform(
    roi_img, np.linalg.inv(coreg_aff), roi_img.shape
).astype(bool)

sd = signed_distance[roi_img_coreg].ravel()
signal = pet[roi_img_coreg].ravel()

################################################################################
# %%
# fit a smoothing spline to the data and calculate the ESF, LSF, and FWHM

inds = np.argsort(sd)
sd = sd[inds]
signal = signal[inds]
signal /= signal[np.argmax(sd)]

sd_binned, signal_binned = rebin_xy(sd, signal, nbins=nbins)

# spline = make_smoothing_spline(sd, signal, lam=15.0)
spline = make_smoothing_spline(sd_binned, signal_binned, lam=lam)

ss = np.linspace(sd.min(), sd.max(), 1000)

esf = spline(ss)
lsf = spline.derivative(1)(ss)
nlsf = lsf / lsf.max()

contrast = esf[-1] / esf[0]

roots_half = InterpolatedUnivariateSpline(ss, lsf - 0.5 * lsf.max()).roots()
fwhm = roots_half[-1] - roots_half[0]

roots_tenth = InterpolatedUnivariateSpline(ss, lsf - 0.1 * lsf.max()).roots()
fwtm = roots_tenth[-1] - roots_tenth[0]

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
ax[0].set_ylim(-0.05, 1.05 * esf.max())
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

# show the aligned images with the ROI
# the viewer expects the images in LPS orientation, so we flip the images accordingly
vi = pv.ThreeAxisViewer(
    [
        np.flip(x, (0, 1))
        for x in [
            pet_coreg,
            mask / mask.max() + 2.0 * roi_img,
            pet,
            mask_coreg / mask_coreg.max() + 2.0 * roi_img_coreg,
        ]
    ],
    voxsize=voxsize,
)
