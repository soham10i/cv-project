"""
Noise strategy interface (Strategy design pattern).
===================================================

The forward diffusion process needs a source of noise. We abstract it behind a
``NoiseStrategy`` so the rest of the pipeline (trainer, scorer) is agnostic to
whether the noise is isotropic Gaussian or spatially-correlated simplex. Swapping
strategies is a one-line config change (``CONFIG.noise.strategy``).

Reference for the Strategy pattern: Gamma et al., 1994, "Design Patterns".
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class NoiseStrategy(ABC):
    """Abstract base for noise generators used in the forward process."""

    name: str = "base"

    @abstractmethod
    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return a noise tensor of ``shape`` on ``device`` with unit variance.

        The tensor MUST be (approximately) zero-mean / unit-variance per element
        so it is compatible with the standard DDPM ``add_noise`` closed form.
        """
        raise NotImplementedError
