"""
Anomaly scoring — pixel-space (no latent fusion).
===================================================
Operates on numpy arrays in the normalised [-1, 1] pixel space.
No latent residual or fusion — the anomaly map is purely based on
high-resolution pixel residuals, preserving sharp edges.

Key features:
* Modality-weighted residuals (FLAIR + T1ce dominate).
* Contrast-Enhancement (CE = t1c - t1n) residual channel.
* Brain mask with erosion to suppress edge artifacts.
* DICE computation for segmentation quality.
"""

from __future__ import annotations

import numpy as np

import config as C


def brain_mask_2d(orig_norm_np: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """(C,H,W) → (H,W) binary brain mask with erosion to suppress edge noise."""
    mask = (np.abs(orig_norm_np).max(axis=0) > eps).astype(np.float32)
    import scipy.ndimage
    return scipy.ndimage.binary_erosion(mask, iterations=4).astype(np.float32)


def _channel_weights() -> np.ndarray:
    """Per-residual-channel weights: modality weights (+ CE weight if enabled)."""
    w = list(C.MODALITY_WEIGHTS)
    if C.USE_CE_CHANNEL:
        w = w + [C.CE_WEIGHT]
    w = np.asarray(w, dtype=np.float32)
    return w / w.sum()


def residual_stack(orig_norm_np: np.ndarray, recon_np: np.ndarray) -> np.ndarray:
    """Per-channel |orig − recon|, plus the CE residual channel if enabled.

    Returns ``(K, H, W)`` where K = N_CHANNELS (+1 for CE).
    """
    diff = np.abs(orig_norm_np - recon_np)                 # (C, H, W)
    if C.USE_CE_CHANNEL:
        ce_o = orig_norm_np[1] - orig_norm_np[0]
        ce_r = recon_np[1] - recon_np[0]
        ce = np.abs(ce_o - ce_r)[None]                     # (1, H, W)
        diff = np.concatenate([diff, ce], axis=0)
    return diff


def pixel_residual_2d(orig_norm_np: np.ndarray, recon_np: np.ndarray,
                      m_baseline: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """Calibrated, modality-weighted, brain-masked pixel anomaly map (H,W).

    ``Σ_k w_k · clip(|orig−recon|_k − M_baseline_k, 0)`` over the brain.
    """
    stack = residual_stack(orig_norm_np, recon_np)         # (K, H, W)
    stack = np.clip(stack - m_baseline, 0, None)
    w = _channel_weights()[:, None, None]                  # (K,1,1)
    return (stack * w).sum(axis=0) * brain_mask


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred * gt).sum()
    return float(2 * inter / (pred.sum() + gt.sum() + 1e-8))
