#!/usr/bin/env python
"""
Stage 1 — Train the pixel-space patch diffusion model on healthy patches.
========================================================================

Usage
-----
    python scripts/01_train.py
    python scripts/01_train.py --epochs 300 --bs 128
    python scripts/01_train.py --smoke            # 2 epochs, tiny subset

The model trains on healthy patches (manifest `healthy.txt`) and validates on
healthy patches from the val split (`val_healthy.txt`). Best + periodic EMA
checkpoints land under PDM_OUTPUT_ROOT/checkpoints.
"""

from __future__ import annotations

import argparse
import sys

import torch
from torch.utils.data import DataLoader

from _bootstrap import init

from src.config import CONFIG
from src.data.dataset import HealthyPatchDataset
from src.models.diffusion import DiffusionProcess
from src.models.unet import build_unet
from src.noise.factory import build_noise_strategy
from src.training.trainer import Trainer
from src.utils.device import describe_device, get_device, set_seed
from src.utils.exceptions import PDMError


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 1 — diffusion training")
    p.add_argument("--epochs", type=int, default=CONFIG.train.epochs)
    p.add_argument("--bs", type=int, default=CONFIG.train.batch_size)
    p.add_argument("--noise", type=str, default=CONFIG.noise.strategy,
                   choices=["simplex", "gaussian"])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    log, _ = init("train")
    try:
        set_seed(CONFIG.train.seed)
        device = get_device()
        log.info("Device: %s", describe_device(device))

        limit = 8 if args.smoke else None
        epochs = 2 if args.smoke else args.epochs

        train_ds = HealthyPatchDataset(
            CONFIG.paths.manifests_dir / "healthy.txt", augment=True, limit=limit
        )
        val_ds = HealthyPatchDataset(
            CONFIG.paths.manifests_dir / "val_healthy.txt", augment=False,
            limit=4 if args.smoke else 64,
        )
        log.info("Train patches: %d | Val patches: %d", len(train_ds), len(val_ds))

        train_loader = DataLoader(
            train_ds, batch_size=args.bs, shuffle=True, drop_last=True,
            num_workers=CONFIG.train.num_workers, pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.bs, shuffle=False,
            num_workers=max(1, CONFIG.train.num_workers // 2),
        )

        process = DiffusionProcess(build_noise_strategy(args.noise))
        unet = build_unet()
        trainer = Trainer(unet, process, device, epochs=epochs)
        result = trainer.fit(train_loader, val_loader)
        log.info("Best val %.5f @ epoch %d. Next: scripts/02_calibrate.py",
                 result["best_val"], result["best_epoch"])
        return 0
    except PDMError as exc:
        log.error("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
