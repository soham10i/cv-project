"""
Anomaly scoring: residual stacks, modality weighting, fusion, DICE.
====================================================================
Operates on numpy arrays in the VAE's normalised [-1, 1] pixel space, shared by
calibration (healthy) and evaluation (anomalous) for distribution-matched scores.

Key improvement over a flat channel-mean residual: modalities are weighted by
clinical salience (FLAIR edema + T1ce enhancement dominate), and an explicit
contrast-enhancement (CE = t1c − t1n) residual channel is appended — the CE
signal that the old 3-channel pipeline could not see at all.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import config as C


def brain_mask_2d(orig_norm_np: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """(C,H,W) → (H,W) binary brain mask (any channel above eps)."""
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

    Returns ``(K, H, W)`` where K = N_CHANNELS (+1 for CE).  Raw (no baseline
    subtraction) so calibration can accumulate it into M_baseline.
    """
    diff = np.abs(orig_norm_np - recon_np)                 # (C, H, W)
    if C.USE_CE_CHANNEL:
        # MODALITIES = [t1n, t1c, t2w, t2f] → CE = t1c − t1n  (index 1 − index 0).
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


def latent_residual_2d(z0: torch.Tensor, z_denoised: torch.Tensor,
                       target_size: int = C.TARGET_SIZE) -> np.ndarray:
    """|z_test − z_healthy| averaged over channels, upsampled to pixel space."""
    m = torch.abs(z0 - z_denoised).mean(dim=1, keepdim=True)   # (B,1,32,32)
    m = F.interpolate(m, size=(target_size, target_size),
                      mode="bilinear", align_corners=False)
    return m[0, 0].cpu().numpy()


def fuse_maps(m_pixel: np.ndarray, m_latent: np.ndarray,
              pixel_scale: float, latent_scale: float,
              alpha: float = C.LATENT_FUSION_ALPHA) -> np.ndarray:
    """Dual-space fusion standardised by healthy scales → single threshold valid."""
    p = m_pixel / (pixel_scale + 1e-8)
    l = m_latent / (latent_scale + 1e-8)
    return p + alpha * l


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred * gt).sum()
    return float(2 * inter / (pred.sum() + gt.sum() + 1e-8))
