#!/usr/bin/env python
"""
Stage 4 — Generate XAI explanations (counterfactual + attribution).
==================================================================

For a set of test lesion slices, produce:
  * a counterfactual + multi-modality / multi-scale attribution panel, and
  * a counterfactual "healing trajectory" strip.

Usage
-----
    python scripts/04_explain.py --n-cases 12
    python scripts/04_explain.py --smoke
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from _bootstrap import init

from src.config import CONFIG
from src.data.dataset import AnomalousSliceDataset
from src.data.patches import iter_patch_coords
from src.models.diffusion import DiffusionProcess
from src.models.unet import load_unet
from src.noise.factory import build_noise_strategy
from src.scoring.multiscale import MultiScaleScorer
from src.utils.device import describe_device, get_device, set_seed
from src.utils.exceptions import PDMError
from src.xai.counterfactual import generate_counterfactual
from src.xai.visualization import save_explanation_panel, save_trajectory_strip


def _lesion_centred_patch(image: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Pick the patch most overlapping the lesion (for a crisp trajectory)."""
    p = CONFIG.patch.patch_size
    best, best_cov = None, -1.0
    for (t, l) in iter_patch_coords(image.shape[1], p, CONFIG.patch.stride):
        cov = gt[t : t + p, l : l + p].sum()
        if cov > best_cov:
            best_cov, best = cov, image[:, t : t + p, l : l + p]
    return best


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 4 — XAI explanations")
    p.add_argument("--n-cases", type=int, default=CONFIG.xai.n_explained_cases)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    log, _ = init("explain")
    try:
        set_seed(CONFIG.train.seed)
        device = get_device()
        log.info("Device: %s", describe_device(device))

        unet = load_unet(device)
        process = DiffusionProcess(build_noise_strategy())
        scorer = MultiScaleScorer(unet, process, device)

        n_cases = 2 if args.smoke else args.n_cases
        ds = AnomalousSliceDataset(CONFIG.paths.manifests_dir / "test_anom.txt", limit=n_cases)
        out_dir = CONFIG.paths.xai_dir
        log.info("Explaining %d cases -> %s", len(ds), out_dir)

        for i in range(len(ds)):
            image_t, gt_t, name = ds[i]
            image = image_t.numpy()
            gt = (gt_t.numpy() > 0).astype(np.float32)

            result = scorer.score_slice(image)
            top_mod = save_explanation_panel(name, result, gt, out_dir)

            patch = _lesion_centred_patch(image, gt)
            cf = generate_counterfactual(unet, process, patch, device)
            save_trajectory_strip(name, cf, out_dir)
            log.info("[%d/%d] %s | dominant modality: %s", i + 1, len(ds), name, top_mod)

        log.info("XAI panels written to %s", out_dir)
        return 0
    except PDMError as exc:
        log.error("Explanation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
