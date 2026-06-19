"""
Residual computation: brain mask + modality-weighted reconstruction error.
==========================================================================

The anomaly signal is the reconstruction residual |orig - healthy_recon|,
weighted per modality by clinical salience and (optionally) augmented with a
contrast-enhancement (CE = t1c - t1n) residual channel. CE is a primary marker
of active tumour and is invisible to pipelines that drop the native T1.

Refs:
  * Baur et al., 2021, "Autoencoders for Unsupervised Anomaly Segmentation in
    Brain MRI" (Medical Image Analysis) — residual scoring for UAD.
  * Menze et al., 2015, "The Multimodal Brain Tumor Image Segmentation Benchmark
    (BRATS)" (IEEE TMI) — modality roles.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage

from ..config import CONFIG


def brain_mask_2d(image: np.ndarray, eps: float = 0.05) -> np.ndarray:
    """(C, H, W) -> (H, W) binary brain mask.

    A voxel is foreground if any channel exceeds the background level (-1).
    A small erosion removes the thin skull-strip rim; kept at 2 px so peripheral
    lesions near the cortex are not eroded away.
    """
    fg = ((image > (-1.0 + eps)).any(axis=0)).astype(np.float32)
    fg = scipy.ndimage.binary_fill_holes(fg).astype(np.float32)
    if CONFIG.scoring.brain_mask_erosion > 0:
        fg = scipy.ndimage.binary_erosion(
            fg, iterations=CONFIG.scoring.brain_mask_erosion
        ).astype(np.float32)
    return fg


def _channel_weights() -> np.ndarray:
    """Normalized per-residual-channel weights (modalities [+ CE])."""
    w = list(CONFIG.scoring.modality_weights)
    if CONFIG.scoring.use_ce_channel:
        w.append(CONFIG.scoring.ce_weight)
    arr = np.asarray(w, dtype=np.float32)
    return arr / arr.sum()


def residual_stack(orig: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """Per-channel |orig - recon| plus the CE residual channel if enabled.

    Returns ``(K, H, W)`` with K = n_channels (+1 for CE). No weighting yet, so
    callers can inspect raw per-modality residuals (used by XAI attribution).
    """
    diff = np.abs(orig - recon)
    if CONFIG.scoring.use_ce_channel:
        # MODALITIES = [t1n, t1c, t2w, t2f] -> CE = t1c - t1n (idx 1 - idx 0).
        ce = np.abs((orig[1] - orig[0]) - (recon[1] - recon[0]))[None]
        diff = np.concatenate([diff, ce], axis=0)
    return diff


def weighted_residual(orig: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Modality-weighted, brain-masked scalar residual map (H, W)."""
    stack = residual_stack(orig, recon)
    w = _channel_weights()[:, None, None]
    return (stack * w).sum(axis=0) * mask
