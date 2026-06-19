"""
Multi-scale patch reconstruction + fusion -> slice-level anomaly score.
=======================================================================

For each test slice this scorer:
  1. extracts overlapping patches,
  2. at each configured noise level T (small T -> fine/texture lesions, large T
     -> structural lesions), reconstructs every patch toward the healthy
     manifold and computes the modality-weighted residual,
  3. fuses overlapping patch residuals into a full slice with Gaussian weighting,
  4. combines the per-scale slice maps with configured weights.

Returns a structured ``ScoreResult`` carrying the fused score, per-scale maps,
the (centre-scale) reconstruction, and the brain mask — everything downstream
(calibration, evaluation, XAI) needs.

Refs: Wyatt et al., 2022 (AnoDDPM); Behrendt et al., 2023 (Patched DDPM, MIDL);
Bercea et al., 2023 (multi-T scoring, MICCAI).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import CONFIG
from ..data.patches import GaussianPatchFuser, extract_patches
from ..models.diffusion import DiffusionProcess
from ..utils.exceptions import ScoringError
from .residual import brain_mask_2d, weighted_residual


@dataclass
class ScoreResult:
    """All artefacts produced when scoring one slice."""

    score: np.ndarray            # fused multi-scale anomaly map (H, W)
    per_scale: dict[int, np.ndarray]  # T -> fused single-scale map (H, W)
    recon: np.ndarray            # representative healthy reconstruction (C, H, W)
    mask: np.ndarray             # brain mask (H, W)
    orig: np.ndarray             # original slice (C, H, W)


class MultiScaleScorer:
    """Scores whole slices via patch reconstruction at multiple noise scales."""

    def __init__(
        self,
        unet,
        process: DiffusionProcess,
        device: torch.device,
        patch_batch: int = 64,
    ) -> None:
        self.unet = unet
        self.process = process
        self.device = device
        self.patch_batch = patch_batch
        self.fuser = GaussianPatchFuser()
        self.timesteps = CONFIG.scoring.score_timesteps
        self.weights = CONFIG.scoring.scale_weights
        if len(self.timesteps) != len(self.weights):
            raise ScoringError(
                "score_timesteps and scale_weights length mismatch in config."
            )

    @torch.no_grad()
    def _reconstruct_patches(
        self, patches: np.ndarray, t_int: int, generator
    ) -> np.ndarray:
        """Reconstruct a stack of patches at noise level t_int (mini-batched)."""
        recon = np.empty_like(patches)
        for i in range(0, len(patches), self.patch_batch):
            chunk = torch.from_numpy(patches[i : i + self.patch_batch]).float().to(self.device)
            out = self.process.reconstruct(self.unet, chunk, t_int, generator=generator)
            recon[i : i + self.patch_batch] = out.cpu().numpy()
        return recon

    @torch.no_grad()
    def score_slice(self, image: np.ndarray, generator=None) -> ScoreResult:
        """Compute the multi-scale anomaly score for a (C, H, W) slice."""
        mask = brain_mask_2d(image)
        patches, coords = extract_patches(
            image, CONFIG.patch.patch_size, CONFIG.patch.stride
        )

        per_scale: dict[int, np.ndarray] = {}
        recon_repr = None
        repr_t = self.timesteps[len(self.timesteps) // 2]  # centre scale for display

        for t_int in self.timesteps:
            recon_patches = self._reconstruct_patches(patches, t_int, generator)
            # Per-patch residual map (use centre channel-weighted residual).
            patch_maps = np.stack(
                [
                    weighted_residual(
                        patches[k], recon_patches[k], np.ones(patches[k].shape[1:], np.float32)
                    )
                    for k in range(len(patches))
                ]
            )
            per_scale[t_int] = self.fuser.fuse(patch_maps, coords) * mask
            if t_int == repr_t:
                recon_repr = self._stitch_recon(recon_patches, coords, image.shape)

        fused = sum(w * per_scale[t] for w, t in zip(self.weights, self.timesteps))
        fused = fused * mask
        return ScoreResult(
            score=fused.astype(np.float32),
            per_scale={t: m.astype(np.float32) for t, m in per_scale.items()},
            recon=recon_repr.astype(np.float32),
            mask=mask.astype(np.float32),
            orig=image.astype(np.float32),
        )

    def _stitch_recon(self, recon_patches, coords, shape) -> np.ndarray:
        """Reassemble a representative reconstruction image for visualisation."""
        c, h, w = shape
        acc = np.zeros((c, h, w), np.float32)
        cnt = np.zeros((1, h, w), np.float32)
        p = CONFIG.patch.patch_size
        for patch, (t, l) in zip(recon_patches, coords):
            acc[:, t : t + p, l : l + p] += patch
            cnt[:, t : t + p, l : l + p] += 1.0
        return acc / np.clip(cnt, 1e-8, None)
