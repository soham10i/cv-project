"""
Patch extraction and Gaussian fusion utilities.
===============================================

The MPDF model operates on overlapping patches, not whole slices. This module
owns both directions of the transform:

  * ``iter_patch_coords`` / ``extract_patches``  — slice  -> patches  (training & test)
  * ``GaussianPatchFuser``                       — patch scores -> slice score

Why overlapping patches with Gaussian-weighted fusion:
  - Turns a handful of slices into hundreds of thousands of training samples.
  - Every interior pixel is covered by many patches; averaging their scores with
    a centre-weighted Gaussian removes the brain-boundary edge artefacts that
    dominate whole-image reconstruction UAD.
Ref: Behrendt et al., 2023, "Patched Diffusion Models for Unsupervised Anomaly
Detection in Brain MRI" (MIDL 2023).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ..config import CONFIG


def iter_patch_coords(
    image_size: int, patch_size: int, stride: int
) -> Iterator[tuple[int, int]]:
    """Yield (top, left) coordinates of patches tiling an image with overlap.

    The last row/column is clamped so patches always stay in-bounds and the
    image edge is fully covered even when (image_size - patch_size) % stride != 0.
    """
    last = image_size - patch_size
    tops = list(range(0, last + 1, stride))
    if tops[-1] != last:
        tops.append(last)
    for top in tops:
        for left in tops:
            yield top, left


def extract_patches(
    image: np.ndarray, patch_size: int, stride: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Extract all patches from a (C, H, W) image.

    Returns ``(patches[N, C, p, p], coords[N])``.
    """
    _, h, w = image.shape
    assert h == w, "Square slices expected."
    coords = list(iter_patch_coords(h, patch_size, stride))
    patches = np.stack(
        [image[:, t : t + patch_size, l : l + patch_size] for (t, l) in coords]
    )
    return patches, coords


def patch_foreground_fraction(patch: np.ndarray) -> float:
    """Fraction of voxels above background (-1) in any channel."""
    return float((patch > -0.99).any(axis=0).mean())


def _gaussian_window(size: int, sigma: float) -> np.ndarray:
    """2D Gaussian centred on the patch, normalized to peak 1.0."""
    ax = np.arange(size) - (size - 1) / 2.0
    g1 = np.exp(-(ax**2) / (2 * sigma**2))
    g2 = np.outer(g1, g1)
    return (g2 / g2.max()).astype(np.float32)


class GaussianPatchFuser:
    """Stitch per-patch score maps back into a full slice.

    Each patch contributes its score weighted by a centred Gaussian, so the
    confident centre of a patch dominates over its context-starved edges. The
    final score at a pixel is the weighted average over every patch covering it.
    """

    def __init__(
        self,
        image_size: int = CONFIG.data.target_size,
        patch_size: int = CONFIG.patch.patch_size,
        sigma: float = CONFIG.patch.fusion_sigma,
    ) -> None:
        self.image_size = image_size
        self.patch_size = patch_size
        self.window = _gaussian_window(patch_size, sigma)

    def fuse(
        self, patch_scores: np.ndarray, coords: list[tuple[int, int]]
    ) -> np.ndarray:
        """Combine ``patch_scores[N, p, p]`` at ``coords`` into an (H, W) map."""
        acc = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        wsum = np.zeros_like(acc)
        for score, (t, l) in zip(patch_scores, coords):
            acc[t : t + self.patch_size, l : l + self.patch_size] += score * self.window
            wsum[t : t + self.patch_size, l : l + self.patch_size] += self.window
        return acc / np.clip(wsum, 1e-8, None)
