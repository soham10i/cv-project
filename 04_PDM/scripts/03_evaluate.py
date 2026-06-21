#!/usr/bin/env python
"""
Stage 3 — Evaluate anomaly segmentation on test lesion slices.
=============================================================

Usage
-----
    python scripts/03_evaluate.py                  # all test slices
    python scripts/03_evaluate.py --n-images 100 --max-plots 24
    python scripts/03_evaluate.py --smoke
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import init

from src.config import CONFIG
from src.evaluation.evaluator import evaluate
from src.models.diffusion import DiffusionProcess
from src.models.unet import load_unet
from src.noise.factory import build_noise_strategy
from src.scoring.multiscale import MultiScaleScorer
from src.utils.device import describe_device, get_device, set_seed
from src.utils.exceptions import PDMError


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 3 — evaluation")
    p.add_argument("--n-images", type=int, default=None)
    p.add_argument("--max-plots", type=int, default=CONFIG.eval.max_plots)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    log, _ = init("evaluate")
    try:
        set_seed(CONFIG.train.seed)
        device = get_device()
        log.info("Device: %s", describe_device(device))

        unet = load_unet(device)
        process = DiffusionProcess(build_noise_strategy())
        scorer = MultiScaleScorer(unet, process, device)

        limit = 6 if args.smoke else args.n_images
        summary = evaluate(
            scorer, CONFIG.paths.manifests_dir / "test_anom.txt", device,
            limit=limit, max_plots=args.max_plots,
        )
        log.info("DICE(calibrated)=%.4f  DICE(oracle)=%.4f  AUROC=%.4f",
                 summary["dice_calibrated"], summary["dice_oracle"], summary["auroc_mean"])
        return 0
    except PDMError as exc:
        log.error("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
