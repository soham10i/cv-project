"""
Noise strategy factory.
=======================

Maps the ``CONFIG.noise.strategy`` string to a concrete ``NoiseStrategy``.
Factory pattern keeps construction logic in one place and makes adding a new
noise type (e.g. exact OpenSimplex) a single registry entry.
"""

from __future__ import annotations

from ..config import CONFIG
from ..utils.exceptions import ConfigError
from .base import NoiseStrategy
from .gaussian import GaussianNoise
from .simplex import SimplexNoise

_REGISTRY = {
    "gaussian": GaussianNoise,
    "simplex": SimplexNoise,
}


def build_noise_strategy(name: str | None = None) -> NoiseStrategy:
    """Construct the configured noise strategy."""
    key = (name or CONFIG.noise.strategy).lower()
    if key not in _REGISTRY:
        raise ConfigError(
            f"Unknown noise strategy '{key}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[key]()
