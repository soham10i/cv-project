"""Unit tests for the noise strategies (CPU-only)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.noise.factory import build_noise_strategy


def test_gaussian_shape_and_stats():
    noise = build_noise_strategy("gaussian")
    x = noise.sample((2, 4, 96, 96), torch.device("cpu"))
    assert x.shape == (2, 4, 96, 96)
    assert abs(x.mean().item()) < 0.1


def test_simplex_unit_variance_and_correlation():
    noise = build_noise_strategy("simplex")
    x = noise.sample((2, 4, 96, 96), torch.device("cpu"))
    assert x.shape == (2, 4, 96, 96)
    # Renormalised to ~unit variance per sample.
    assert 0.5 < x.std().item() < 2.0
    # Spatial correlation: neighbour differences smaller than white noise.
    diff = (x[..., 1:, :] - x[..., :-1, :]).abs().mean().item()
    assert diff < 1.0  # white Gaussian would be ~1.13
