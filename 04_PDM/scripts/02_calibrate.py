#!/usr/bin/env python
"""
Stage 2 — Calibrate the anomaly threshold on healthy validation slices.
======================================================================

Usage
-----
    python scripts/02_calibrate.py
    python scripts/02_calibrate.py --max-samples 200
    python scripts/02_calibrate.py --smoke
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import init

from src.calibration.calibrator import calibrate
from src.config import CONFIG
from src.models.diffusion import DiffusionProcess
from src.models.unet import load_unet
from src.noise.factory import build_noise_strategy
from src.scoring.multiscale import MultiScaleScorer
from src.utils.device import describe_device, get_device, set_seed
from src.utils.exceptions import PDMError


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 2 — calibration")
    p.add_argument("--max-samples", type=int, default=CONFIG.calibration.max_samples)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    log, _ = init("calibrate")
    try:
        set_seed(CONFIG.train.seed)
        device = get_device()
        log.info("Device: %s", describe_device(device))

        unet = load_unet(device)
        process = DiffusionProcess(build_noise_strategy())
        scorer = MultiScaleScorer(unet, process, device)

        max_samples = 8 if args.smoke else args.max_samples
        calib = calibrate(
            scorer, CONFIG.paths.manifests_dir / "val_healthy.txt", device,
            max_samples=max_samples,
        )
        log.info("threshold=%.4f (n=%d). Next: scripts/03_evaluate.py",
                 calib["threshold"], calib["n_samples"])
        return 0
    except PDMError as exc:
        log.error("Calibration failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
