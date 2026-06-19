"""
Dense CRF boundary refinement (optional post-processing).
=========================================================

The diffusion score map is spatially smooth (good recall, soft boundaries). A
fully-connected CRF sharpens it to anatomical edges using a guidance modality
(T1c) as appearance evidence, improving boundary precision and hence DICE.

Ref: Krähenbühl & Koltun, 2011, "Efficient Inference in Fully Connected CRFs
with Gaussian Edge Potentials" (NeurIPS 2011).

Optional dependency: ``pydensecrf``. If it is not installed, ``refine`` returns
the input unchanged and logs a one-time warning, so the pipeline never breaks.
"""

from __future__ import annotations

import numpy as np

from ..config import CONFIG
from ..utils.logging_utils import get_logger

log = get_logger("pdm.crf")

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    _HAS_CRF = True
except ImportError:  # pragma: no cover
    _HAS_CRF = False

_WARNED = False


def _normalize01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-8)


def refine(score: np.ndarray, guidance: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Refine a continuous anomaly ``score`` (H, W) with a dense CRF.

    ``guidance`` is a single-channel image (e.g. T1c) used as appearance
    evidence; ``mask`` is the brain mask. Returns a refined score in [0, 1].
    Falls back to the unrefined (normalized) score if CRF is unavailable/disabled.
    """
    global _WARNED
    norm = _normalize01(score) * mask
    if not CONFIG.crf.enabled:
        return norm
    if not _HAS_CRF:
        if not _WARNED:
            log.warning("pydensecrf not installed — skipping CRF refinement.")
            _WARNED = True
        return norm

    h, w = norm.shape
    prob = np.clip(norm, 1e-5, 1 - 1e-5)
    probs = np.stack([1 - prob, prob], axis=0)  # (2, H, W): [bg, anomaly]

    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary_from_softmax(probs))
    d.addPairwiseGaussian(
        sxy=CONFIG.crf.gaussian_sxy, compat=CONFIG.crf.gaussian_compat
    )
    guide = (_normalize01(guidance) * 255).astype(np.uint8)
    guide_rgb = np.ascontiguousarray(np.stack([guide] * 3, axis=-1))
    d.addPairwiseBilateral(
        sxy=CONFIG.crf.bilateral_sxy,
        srgb=CONFIG.crf.bilateral_srgb,
        rgbim=guide_rgb,
        compat=CONFIG.crf.bilateral_compat,
    )
    q = d.inference(CONFIG.crf.n_iterations)
    refined = np.asarray(q)[1].reshape(h, w)
    return (refined * mask).astype(np.float32)
