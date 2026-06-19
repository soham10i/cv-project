"""
Simplex / fractal noise strategy.
=================================

Spatially-correlated noise that targets a lesion-scale spatial frequency, so
lesions "look like noise" to the denoiser and get in-painted with healthy tissue
at test time. This is the key ingredient that makes diffusion UAD sensitive to
focal pathology.

Ref: Wyatt et al., 2022, "AnoDDPM: Anomaly Detection with Denoising Diffusion
Probabilistic Models using Simplex Noise" (CVPR Workshops 2022).

Implementation note
-------------------
True OpenSimplex noise is expensive to sample per training step. We use the
standard, fast *fractal noise* approximation used by most AnoDDPM
re-implementations: sum several octaves of upsampled low-resolution Gaussian
noise. Each octave is band-limited (smooth, correlated); summing octaves with
decaying amplitude yields multi-scale, simplex-like noise. The result is
renormalised to unit variance so it plugs into the standard DDPM closed form.
If exact simplex is required, install ``opensimplex`` and swap ``_octave``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import CONFIG
from .base import NoiseStrategy


class SimplexNoise(NoiseStrategy):
    """Fractal (multi-octave) approximation of simplex noise."""

    name = "simplex"

    def __init__(
        self,
        octaves: int = CONFIG.noise.simplex_octaves,
        base_frequency: int = CONFIG.noise.simplex_base_frequency,
        persistence: float = CONFIG.noise.simplex_persistence,
        lacunarity: float = CONFIG.noise.simplex_lacunarity,
    ) -> None:
        self.octaves = octaves
        self.base_frequency = base_frequency
        self.persistence = persistence
        self.lacunarity = lacunarity

    def _octave(
        self,
        shape: tuple[int, ...],
        freq: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """One band-limited octave: low-res Gaussian noise upsampled to ``shape``."""
        b, c, h, w = shape
        gh = max(1, min(h, freq))
        gw = max(1, min(w, freq))
        low = torch.randn(b, c, gh, gw, device=device, generator=generator)
        return F.interpolate(low, size=(h, w), mode="bicubic", align_corners=False)

    def sample(self, shape, device, generator=None):
        if len(shape) != 4:
            # Fall back to white noise for non-image tensors.
            return torch.randn(shape, device=device, generator=generator)

        noise = torch.zeros(shape, device=device)
        amplitude = 1.0
        freq = self.base_frequency
        for _ in range(self.octaves):
            noise += amplitude * self._octave(shape, int(freq), device, generator)
            amplitude *= self.persistence
            freq *= self.lacunarity

        # Renormalise to ~unit variance so the DDPM SNR schedule is unchanged.
        std = noise.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return noise / std
