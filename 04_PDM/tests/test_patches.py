"""Unit tests for patch extraction and Gaussian fusion (no GPU/data needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.patches import GaussianPatchFuser, extract_patches, iter_patch_coords


def test_patch_coords_cover_edges():
    coords = list(iter_patch_coords(256, 96, 16))
    tops = sorted({t for t, _ in coords})
    assert tops[0] == 0
    assert tops[-1] == 256 - 96  # last patch flush with the edge


def test_extract_patches_shape():
    img = np.random.randn(4, 256, 256).astype(np.float32)
    patches, coords = extract_patches(img, 96, 16)
    assert patches.shape[1:] == (4, 96, 96)
    assert len(coords) == len(patches)


def test_fusion_reconstructs_constant():
    """Fusing constant-valued patch scores must return that constant on brain."""
    fuser = GaussianPatchFuser(image_size=256, patch_size=96, sigma=32.0)
    _, coords = extract_patches(np.zeros((4, 256, 256), np.float32), 96, 16)
    scores = np.ones((len(coords), 96, 96), np.float32) * 0.7
    fused = fuser.fuse(scores, coords)
    # Interior should be ~0.7 (weights cancel in the average).
    assert np.allclose(fused[128, 128], 0.7, atol=1e-4)
