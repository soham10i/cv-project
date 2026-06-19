"""
Robust MRI intensity normalisation.
====================================
BraTS modalities have arbitrary, scanner-dependent intensity scales, so each
volume is z-scored over its brain (non-zero) voxels.  Crucially we then clip to
robust *percentiles* rather than a hard ±3σ: edema / FLAIR hyperintensity and
enhancing tumour routinely exceed 3σ, and clipping them away would erase the
exact signal the anomaly detector needs.  Background stays exactly 0.
"""

from __future__ import annotations

import numpy as np

from config import NORM_PCT_LOW, NORM_PCT_HIGH, VAE_INPUT_CLIP


def zscore_volume(volume_3d: np.ndarray) -> np.ndarray:
    """Z-score a 3D volume using statistics pooled over brain (non-zero) voxels.

    Volume-level (not per-slice) normalisation is the BraTS standard and
    preserves the relative intensity gradient along the Z-axis.  Background
    voxels (originally 0) remain 0.
    """
    out = np.zeros_like(volume_3d, dtype=np.float32)
    brain = volume_3d > 0
    if not np.any(brain):
        return out
    vox = volume_3d[brain].astype(np.float32)
    mu, sd = vox.mean(), vox.std()
    if sd < 1e-8:
        return out
    out[brain] = (vox - mu) / sd
    return out


def robust_clip(volume_3d: np.ndarray,
                pct_low: float = NORM_PCT_LOW,
                pct_high: float = NORM_PCT_HIGH) -> np.ndarray:
    """Clip a z-scored volume to robust percentiles of its brain voxels.

    Unlike a fixed ±3σ clip this keeps genuine hyperintensities (only the most
    extreme <0.5% outliers, typically motion / coil artefacts, are clamped).
    Background (exactly 0) is left untouched.
    """
    brain = volume_3d != 0
    if not np.any(brain):
        return volume_3d.astype(np.float32)
    vox = volume_3d[brain]
    lo = np.percentile(vox, pct_low)
    hi = np.percentile(vox, pct_high)
    out = volume_3d.astype(np.float32).copy()
    out[brain] = np.clip(vox, lo, hi)
    return out


def normalize_volume(volume_3d: np.ndarray) -> np.ndarray:
    """Full per-volume normalisation: z-score then robust percentile clip."""
    return robust_clip(zscore_volume(volume_3d))


def to_model_range(slice_4ch: np.ndarray, clip: float = VAE_INPUT_CLIP) -> np.ndarray:
    """Map a normalised slice onto the model's ~[-1, 1] range for the VAE (numpy).

    Divides by a fixed scale (default 5σ, generous enough to retain
    hyperintensity) so the codec sees a bounded input. Used before vae.encode.
    """
    return np.clip(slice_4ch, -clip, clip) / clip


def normalize_for_vae(x, clip: float = VAE_INPUT_CLIP):
    """Torch version of ``to_model_range`` — differentiable, for training/inference."""
    import torch
    return torch.clamp(x, -clip, clip) / clip
