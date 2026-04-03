"""
main_align.py  —  Rigid 3D registration demo
=============================================
Uses register to align a moving PET image to a fixed mask.
All alignment parameters are tuples of 6 floats.
Device selection (CPU / CUDA) is handled automatically inside register/warp.
"""

import numpy as np
from utils_align import register, warp

# ── 1.  Load volumes ──────────────────────────────────────────────────────────

F_np = np.load("data/mask.npy")  # fixed image (reference),  shape (D, H, W)
M_np = np.load("data/pet.npy")  # moving image (to align),  shape (D, H, W)
voxel_size = tuple(np.load("data/voxsize.npy"))  # (vz, vy, vx) in mm

# ── 2.  Register ──────────────────────────────────────────────────────────────
# register accepts numpy arrays directly and picks CUDA if available.
# Returns a tuple of 6 floats: (tz, ty, tx, ωz, ωy, ωx).

p_found = register(F_np, M_np, voxel_size)

# ── 3.  Apply transform and evaluate ─────────────────────────────────────────

M_aligned = warp(M_np, p_found, voxel_size)  # (D, H, W) tensor
M_aligned_np = M_aligned.cpu().numpy()

rmse_before = np.sqrt(np.mean((F_np - M_np) ** 2))
rmse_after = np.sqrt(np.mean((F_np - M_aligned_np) ** 2))
print(f"\nRMSE before : {rmse_before:.5f}")
print(f"RMSE after  : {rmse_after:.5f}")
