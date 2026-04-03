"""
main.py  —  Rigid 3D registration demo
=======================================
Uses register_multiscale to robustly handle large initial misalignments.
"""

import numpy as np
import torch
from utils_align import register, warp

# ── 1.  Volume geometry ───────────────────────────────────────────────────────
device = torch.device("cuda")
dtype = torch.float32

F_np = np.load("data/mask.npy")  # fixed image (reference)
shape = F_np.shape
F = torch.tensor(F_np, device=device, dtype=dtype)

voxel_size = tuple(np.load("data/voxsize.npy"))

# ── 3.  Ground-truth transform and moving image ───────────────────────────────
# Large misalignment: 10 voxels translation + moderate rotation
M_np = np.load("data/pet.npy")  # moving image (to be aligned)
M = torch.tensor(M_np, device=device, dtype=dtype)


# ── 4.  Cost function ─────────────────────────────────────────────────────────

p_found = register(
    F,
    M,
    voxel_size,
)

# ── 7.  Aligned volume and quality metrics ────────────────────────────────────
M_aligned = warp(M, p_found, voxel_size)
M_aligned_np = M_aligned.cpu().numpy()

rmse_before = (F - M).pow(2).mean().sqrt().item()
rmse_after = (F - M_aligned).pow(2).mean().sqrt().item()
print(f"\nRMSE before : {rmse_before:.5f}")
print(f"RMSE after  : {rmse_after:.5f}")
