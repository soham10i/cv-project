"""
Intensity normalization for multimodal MRI.
===========================================

Robust per-volume normalization: clip to [p_low, p_high] percentiles of the
*foreground* (non-zero) voxels, then scale to [-1, 1]. This preserves
hyperintense lesion signal that hard +/- 3-sigma clipping would truncate.

Ref: Reinhold et al., 2019, "Evaluating the Impact of Intensity Normalization
on MR Image Synthesis" (SPIE Medical Imaging).
"""

from __future__ import annotations

import numpy as np

from ..config import CONFIG


def robust_normalize_volume(
    volume: np.ndarray,
    p_low: float = CONFIG.data.norm_pct_low,
    p_high: float = CONFIG.data.norm_pct_high,
) -> np.ndarray:
    """Normalize a single-modality 3D volume to [-1, 1].

    Percentiles are computed over foreground (non-zero) voxels only, so the
    large black background does not dominate the statistics.
    """
    fg = volume[volume > 0]
    if fg.size == 0:
        return np.zeros_like(volume, dtype=np.float32)

    lo, hi = np.percentile(fg, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    clipped = np.clip(volume, lo, hi)
    # Map [lo, hi] -> [-1, 1]; background (originally 0) maps below -1 then is
    # re-zeroed so it stays a clean, constant background.
    scaled = 2.0 * (clipped - lo) / (hi - lo) - 1.0
    scaled[volume == 0] = -1.0
    return scaled.astype(np.float32)


def stack_modalities(per_modality: dict[str, np.ndarray], order) -> np.ndarray:
    """Stack normalized per-modality volumes into (C, D, H, W) in channel order."""
    return np.stack([per_modality[m] for m in order], axis=0).astype(np.float32)
