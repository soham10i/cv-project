"""
Isotropic Gaussian noise strategy (the standard DDPM noise; ablation baseline).
===============================================================================

Ref: Ho et al., 2020, "Denoising Diffusion Probabilistic Models" (NeurIPS 2020).
"""

from __future__ import annotations

import torch

from .base import NoiseStrategy


class GaussianNoise(NoiseStrategy):
    """Standard white Gaussian noise N(0, I)."""

    name = "gaussian"

    def sample(self, shape, device, generator=None):
        return torch.randn(shape, device=device, generator=generator)
