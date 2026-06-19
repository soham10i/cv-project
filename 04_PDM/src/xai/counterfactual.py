"""
Counterfactual explanation — the primary XAI strategy for diffusion UAD.
=======================================================================

A diffusion UAD model answers a question that is *inherently counterfactual*:
"what would this brain look like if it were healthy?" The healthy reconstruction
is the counterfactual x', and the anomaly map |x - x'| is the counterfactual
difference — the minimal change that moves the input onto the healthy manifold.
This is the most faithful explanation we can give, because it is exactly what the
model computes to make its decision (no post-hoc surrogate).

Refs:
  * Wachter et al., 2017, "Counterfactual Explanations without Opening the Black
    Box" (Harvard JOLT) — counterfactual XAI foundations.
  * Sanchez et al., 2022, "Healthy/Pathological Brain Counterfactuals with
    Diffusion Models" (arXiv:2203.08089).
  * Atad et al., 2022, "CheXplaining in Style" / diffusion counterfactuals for
    medical imaging.

This module produces the "healing trajectory": a sequence of DDIM denoising
frames showing the lesion being progressively in-painted with healthy tissue,
which makes the model's reasoning visible step by step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import CONFIG
from ..models.diffusion import DiffusionProcess


@dataclass
class CounterfactualResult:
    original: np.ndarray          # (C, H, W)
    counterfactual: np.ndarray    # (C, H, W) healthy reconstruction
    difference: np.ndarray        # (C, H, W) signed counterfactual change
    trajectory: list[np.ndarray]  # list of (C, H, W) denoising frames


@torch.no_grad()
def generate_counterfactual(
    unet,
    process: DiffusionProcess,
    patch: np.ndarray,
    device: torch.device,
    t_int: int | None = None,
    generator: torch.Generator | None = None,
) -> CounterfactualResult:
    """Produce a healthy counterfactual + healing trajectory for one patch.

    Operates on a single patch (C, p, p) so the trajectory is crisp; the
    slice-level counterfactual is the stitched reconstruction from the scorer.
    """
    t = t_int or CONFIG.scoring.score_timesteps[len(CONFIG.scoring.score_timesteps) // 2]
    x0 = torch.from_numpy(patch).float().unsqueeze(0).to(device)
    recon, traj = process.reconstruct(
        unet, x0, t, generator=generator, return_trajectory=True
    )
    cf = recon[0].cpu().numpy()
    return CounterfactualResult(
        original=patch,
        counterfactual=cf,
        difference=patch - cf,
        trajectory=[fr[0].numpy() for fr in traj],
    )
