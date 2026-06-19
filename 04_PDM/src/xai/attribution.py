"""
Attribution: which modality and which noise scale drove the detection.
======================================================================

Complements the counterfactual with two clinically-actionable decompositions and
one model-internal view:

  1. Per-modality attribution — split the anomaly score by modality residual.
     Answers "is this lesion visible mainly on FLAIR, or on contrast (CE)?",
     which mirrors how a radiologist reasons about tumour sub-compartments.
     Ref: Menze et al., 2015 (BRATS, IEEE TMI).

  2. Per-scale attribution — split the score by noise level T. Low-T dominance
     => fine/texture anomaly; high-T dominance => large structural anomaly.
     Ref: Bercea et al., 2023 (MICCAI); Wyatt et al., 2022 (AnoDDPM).

  3. Attention rollout — aggregate the UNet self-attention maps to show where the
     model attends. Ref: Abnar & Zuidema, 2020, "Quantifying Attention Flow in
     Transformers" (ACL).
"""

from __future__ import annotations

import numpy as np

from ..config import CONFIG
from ..scoring.multiscale import ScoreResult
from ..scoring.residual import _channel_weights, brain_mask_2d, residual_stack


def modality_attribution(orig: np.ndarray, recon: np.ndarray) -> dict[str, np.ndarray]:
    """Return a per-modality (and CE) weighted residual map.

    Each entry is that channel's contribution to the final score, so summing the
    maps reproduces the weighted residual (up to the brain mask).
    """
    mask = brain_mask_2d(orig)
    stack = residual_stack(orig, recon)
    weights = _channel_weights()
    names = list(CONFIG.data.modalities) + (["CE"] if CONFIG.scoring.use_ce_channel else [])
    return {name: (stack[i] * weights[i] * mask) for i, name in enumerate(names)}


def scale_attribution(result: ScoreResult) -> dict[int, np.ndarray]:
    """Return the per-noise-scale contribution maps (already weighted)."""
    return {
        t: w * result.per_scale[t]
        for t, w in zip(CONFIG.scoring.score_timesteps, CONFIG.scoring.scale_weights)
    }


def dominant_modality(orig: np.ndarray, recon: np.ndarray) -> tuple[str, dict[str, float]]:
    """Identify the modality contributing most total anomaly signal."""
    attrib = modality_attribution(orig, recon)
    totals = {name: float(m.sum()) for name, m in attrib.items()}
    top = max(totals, key=totals.get) if totals else "n/a"
    return top, totals


class AttentionRollout:
    """Capture and aggregate UNet self-attention via forward hooks.

    Registers hooks on attention modules, runs one forward pass, and averages the
    attention magnitudes upsampled to patch resolution. This is a best-effort,
    architecture-agnostic view (diffusers UNet attention internals vary across
    versions); it degrades gracefully to a zero map if no attention is found.
    """

    def __init__(self, unet) -> None:
        self.unet = unet
        self._maps: list[np.ndarray] = []
        self._handles = []

    def _hook(self, _module, _inp, output):
        try:
            out = output[0] if isinstance(output, tuple) else output
            # (B, C, h, w): use channel-mean magnitude as a saliency proxy.
            sal = out.detach().float().abs().mean(dim=1)[0].cpu().numpy()
            self._maps.append(sal)
        except Exception:  # pragma: no cover - defensive across versions
            pass

    def __enter__(self):
        for module in self.unet.modules():
            if module.__class__.__name__.lower().find("attention") >= 0:
                self._handles.append(module.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()

    def rollout(self, patch_size: int) -> np.ndarray:
        """Aggregate captured maps to a (patch_size, patch_size) saliency map."""
        if not self._maps:
            return np.zeros((patch_size, patch_size), np.float32)
        import scipy.ndimage

        acc = np.zeros((patch_size, patch_size), np.float32)
        for m in self._maps:
            zoom = (patch_size / m.shape[0], patch_size / m.shape[1])
            acc += scipy.ndimage.zoom(m, zoom, order=1)
        acc /= len(self._maps)
        return (acc - acc.min()) / (acc.max() - acc.min() + 1e-8)
