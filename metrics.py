"""
metrics.py  —  Differentiable registration cost functions
==========================================================

All functions have the signature:

    cost_fn(fixed: Tensor, warped: Tensor) -> scalar Tensor

where fixed and warped are (D, H, W) float tensors on the same device.
They are designed to be minimized (NCC and NMI are returned negated).

Available metrics
-----------------
  ssd(f, w)                  — Sum of Squared Differences
  ssd_linear_corrected(f, w) — SSD after fitting optimal affine intensity map
  ncc(f, w)                  — Normalised Cross-Correlation
  nmi(...)                   — factory → Normalised Mutual Information

Choosing a metric
-----------------
  Same modality, same reconstruction pipeline  →  ssd
  Same modality, different intensity scaling   →  ssd_linear_corrected
  Same modality, unknown offset + scale        →  ncc
  Different modalities (PET/CT, PET/MR, …)    →  nmi()
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


# ── Sum of Squared Differences ────────────────────────────────────────────────


def ssd(fixed: Tensor, warped: Tensor) -> Tensor:
    """
    Sum of Squared Differences.

    Assumes F(x) ≈ M(T(x)) exactly.  Sensitive to any global intensity
    scale or offset difference between the two images.
    """
    return ((fixed - warped) ** 2).sum()


def ncc(fixed: Tensor, warped: Tensor) -> Tensor:
    """
    Negative Normalised Cross-Correlation.
    Returns a value in [−1, 0]; minimising gives maximum correlation.
    """
    f = fixed - fixed.mean()
    w = warped - warped.mean()
    num = (f * w).sum()
    denom = (f.norm() * w.norm()).clamp(min=1e-8)
    return -(num / denom)


def nmi(
    *,
    n_bins: int = 32,
    n_samples: int = 50_000,
    sigma_ratio: float = 1.0,
) -> Callable[[Tensor, Tensor], Tensor]:
    """
    Factory returning a differentiable Negative Normalised Mutual Information
    cost function.

        NMI(F, M) = ( H(F) + H(M) ) / H(F, M)    ∈ [1, 2]

    The joint histogram is estimated via Gaussian Parzen windows, making
    the result differentiable w.r.t. warped image intensities.

    Sampling
    --------
    For computational efficiency only a strided subset of voxels is used.
    Stride-based (deterministic) sampling ensures the same voxels are
    evaluated at every closure call within an L-BFGS line search, which
    is required for the quasi-Newton update to be consistent.

    Args:
        n_bins:       Histogram bins per marginal.  32–64 is typical.
        n_samples:    Voxels to sample.  50 k gives stable estimates and
                      runs in < 5 ms on a modern GPU.
        sigma_ratio:  Gaussian bandwidth = sigma_ratio / n_bins.
                      Larger → smoother histogram, wider capture range.
                      Smaller → sharper, noisier gradient.
                      Typical range: [0.5, 2.0].

    Returns:
        cost_fn(fixed, warped) → scalar (negative NMI, to be minimized).

    Example:
        >>> from metrics import nmi
        >>> cost = nmi(n_bins=64)
        >>> p = register_multiscale(F, M, voxel_size, cost)
    """
    sigma = sigma_ratio / n_bins

    def _cost(fixed: Tensor, warped: Tensor) -> Tensor:

        # ── 1. Subsample (stride-based, deterministic) ────────────────────
        N = fixed.numel()
        stride = max(1, N // n_samples)
        f = fixed.ravel()[::stride][:n_samples]
        w = warped.ravel()[::stride][:n_samples]

        # ── 2. Robust normalisation to [0, 1] per image ───────────────────
        # Percentile bounds are detached so gradients only flow through the
        # centred / scaled values, not through the bounds themselves.
        with torch.no_grad():
            f_lo = torch.quantile(f, 0.01)
            f_hi = torch.quantile(f, 0.99)
            w_lo = torch.quantile(w, 0.01)
            w_hi = torch.quantile(w, 0.99)

        f_n = ((f - f_lo) / (f_hi - f_lo + 1e-8)).clamp(0.0, 1.0)
        w_n = ((w - w_lo) / (w_hi - w_lo + 1e-8)).clamp(0.0, 1.0)

        # ── 3. Parzen-window joint histogram ──────────────────────────────
        # bins ∈ [0, 1],  shape (n_bins,)
        bins = torch.linspace(0.0, 1.0, n_bins, device=fixed.device, dtype=fixed.dtype)

        # Wf[i, k] = exp(−(f_i − b_k)² / 2σ²),  shape (n_samples, n_bins)
        # Ww[i, l] = exp(−(w_i − b_l)² / 2σ²),  shape (n_samples, n_bins)
        Wf = torch.exp(-0.5 * ((f_n[:, None] - bins[None, :]) / sigma) ** 2)
        Ww = torch.exp(-0.5 * ((w_n[:, None] - bins[None, :]) / sigma) ** 2)

        # Joint histogram: H[k, l] = Σ_i Wf[i,k]·Ww[i,l]
        # Computed as a single matrix multiply: (n_bins, n_samples) @ (n_samples, n_bins)
        H = Wf.T @ Ww  # (n_bins, n_bins)
        H = H / H.sum()  # normalise to joint probability

        # ── 4. Marginal histograms ─────────────────────────────────────────
        H_f = H.sum(dim=1)  # (n_bins,)  marginal over warped bins
        H_w = H.sum(dim=0)  # (n_bins,)  marginal over fixed  bins

        # ── 5. Entropies ──────────────────────────────────────────────────
        eps = 1e-10
        ent_joint = -(H * (H + eps).log()).sum()
        ent_f = -(H_f * (H_f + eps).log()).sum()
        ent_w = -(H_w * (H_w + eps).log()).sum()

        # ── 6. NMI ∈ [1, 2], return negative to minimise ─────────────────
        return -(ent_f + ent_w) / (ent_joint + eps)

    return _cost
