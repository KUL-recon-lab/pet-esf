import numpy as np
import nibabel as nib
import pymirc.viewer as pv
import pymirc.image_operations as pymi
import matplotlib.pyplot as plt
import json

from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import find_objects, center_of_mass, gaussian_filter

from pathlib import Path
from utils import rebin_xy

# %%
# input parameters

fname: Path = Path("smallest_voxels_first_9min_30it_0310.nii")
# fname: Path = Path("smallest_voxels_first_9min_7it_nopsf_0309.nii")
lam = 0.3
nbins = 256
cube_diam_mm = 35.0
edge = "xs"

################################################################################
# %%
# READ THE DATA FROM NIFTI

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

################################################################################
# %%
# generate a reference image (mask) that we can use for registration
cube_diam_vox = np.round(cube_diam_mm / voxsize).astype(int)

mask = np.zeros_like(pet)

# fill the center of the mask with a cube of diameter 30mm
center = np.array(center_of_mass(pet)).astype(int)
start = center - cube_diam_vox // 2
end = start + cube_diam_vox
mask[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = 1.0

mask *= float(pet.sum() / mask.sum())


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

################################################################################
# %%
# create a ROI image indicating the region of interest used for LSF
roi_img = np.zeros_like(pet)

r1 = 8.0
r2 = 8.0

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
# extract the signal and distance values for the voxels in the ROI, sort by distance, and normalize the signal

roi_img[start_x:end_x, start_y:end_y, start_z:end_z] = 1

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
signal /= signal.max()

sd_binned, signal_binned = rebin_xy(sd, signal, nbins=nbins)

# spline = make_smoothing_spline(sd, signal, lam=15.0)
spline = make_smoothing_spline(sd_binned, signal_binned, lam=lam)

esf = spline(sd_binned)
lsf = spline.derivative(1)(sd_binned)
nlsf = lsf / lsf.max()

i2 = np.where(nlsf > 0.5)[0]
fwhm = sd_binned[i2[-1]] - sd_binned[i2[0]]
print(f"FWHM: {fwhm:.2f} mm")

i10 = np.where(nlsf > 0.1)[0]
fwtm = sd_binned[i10[-1]] - sd_binned[i10[0]]
print(f"FWTM: {fwtm:.2f} mm")

################################################################################
# %%
# visualize the results

fig, ax = plt.subplots(1, 2, figsize=(12, 6), layout="constrained", sharex=True)
ax[0].plot(sd, signal, ".", markersize=0.2, color="k")
ax[0].plot(sd_binned, signal_binned, ".", markersize=3.0, color="C0")
ax[0].plot(sd_binned, esf, "-", color="C1")
ax[1].plot(sd_binned, nlsf, "-")

# indicate FWHM in normalized coordinates (0 to 1)
xmin_ax, xmax_ax = ax[1].get_xlim()
ax[1].axhline(
    0.5,
    color="gray",
    ls="--",
    xmin=(sd_binned[i2[0]] - xmin_ax) / (xmax_ax - xmin_ax),
    xmax=(sd_binned[i2[-1]] - xmin_ax) / (xmax_ax - xmin_ax),
)
ax[1].axhline(
    0.1,
    color="gray",
    ls="--",
    xmin=(sd_binned[i10[0]] - xmin_ax) / (xmax_ax - xmin_ax),
    xmax=(sd_binned[i10[-1]] - xmin_ax) / (xmax_ax - xmin_ax),
)

for axx in ax:
    axx.grid(ls=":")
    axx.set_xlabel("Distance from edge (mm)")

ax[0].set_title("ESF", fontsize="medium")
ax[1].set_title("normalized LSF", fontsize="medium")

fig.show()

vi = pv.ThreeAxisViewer([pet_coreg, mask / mask.max() + 2 * roi_img], voxsize=voxsize)
