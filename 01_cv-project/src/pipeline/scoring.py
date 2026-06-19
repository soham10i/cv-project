"""
Anomaly scoring: brain masking, residual computation, fusion, and DICE.

These functions operate on numpy arrays in the normalised [-1, 1] pixel
space.  They are shared by calibration (on healthy data) and evaluation
(on anomalous data) to guarantee distribution-matched scoring.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core import constants as C


def brain_mask_2d(orig_norm_np: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Compute a binary brain mask from a normalised slice.

    Parameters
    ----------
    orig_norm_np : (3, 256, 256) normalised slice

    Returns
    -------
    (256, 256) float32 array with 1.0 for brain voxels, 0.0 for background.
    """
    return (np.abs(orig_norm_np).max(axis=0) > eps).astype(np.float32)


def latent_residual_2d(
    z0: torch.Tensor, z_denoised: torch.Tensor,
    target_size: int = C.TARGET_SIZE,
) -> np.ndarray:
    """|z_test − z_healthy| averaged over channels, upsampled to pixel space.

    Returns a ``(target_size, target_size)`` numpy array.
    """
    m = torch.abs(z0 - z_denoised).mean(dim=1, keepdim=True)  # (B,1,32,32)
    m = F.interpolate(
        m, size=(target_size, target_size),
        mode="bilinear", align_corners=False,
    )
    return m[0, 0].cpu().numpy()


def pixel_residual_2d(
    orig_norm_np: np.ndarray, recon_np: np.ndarray,
    m_baseline: np.ndarray, brain_mask: np.ndarray,
) -> np.ndarray:
    """Calibrated, brain-masked pixel residual.

    ``clip( mean_channels(|orig − recon| − M_baseline), 0 ) · brain_mask``
    """
    diff = np.abs(orig_norm_np - recon_np) - m_baseline  # (3, 256, 256)
    diff = np.clip(diff, 0, None)
    return diff.mean(axis=0) * brain_mask  # (256, 256)


def fuse_maps(
    m_pixel: np.ndarray, m_latent: np.ndarray,
    pixel_scale: float, latent_scale: float,
    alpha: float = C.LATENT_FUSION_ALPHA,
) -> np.ndarray:
    """Dual-space fusion standardised by healthy scales.

    ``m_pixel / scale_p + α · m_latent / scale_l``

    Standardisation keeps a single global threshold valid across images.
    """
    p = m_pixel / (pixel_scale + 1e-8)
    l = m_latent / (latent_scale + 1e-8)
    return p + alpha * l


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute the Sørensen–Dice coefficient between two binary masks."""
    inter = (pred * gt).sum()
    return float(2 * inter / (pred.sum() + gt.sum() + 1e-8))
