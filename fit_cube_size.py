import argparse
from pathlib import Path

import nibabel as nib
from scipy.optimize import minimize

from utils import find_planes


# %%
# input parameters
parser = argparse.ArgumentParser()
parser.add_argument("fname", type=Path, help="path to the NIfTI file")


args = parser.parse_args()
fname: Path = args.fname
################################################################################
# %%
# READ THE DATA FROM NIFTI, orient to RAS

nii = nib.as_closest_canonical(nib.load(fname))
pet = nii.get_fdata().copy()
voxsize = nii.header["pixdim"][1:4]

################################################################################
# %%


def cost(x):
    _, reg_loss = find_planes(pet, voxsize, x[0], x[1])
    return reg_loss


# x = [ 2.934e+01  6.554e+00]
x0 = [29.5, 6.5]
res = minimize(cost, x0, method="Nelder-Mead")
