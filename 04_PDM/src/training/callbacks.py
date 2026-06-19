"""
Training callbacks: early stopping and checkpoint management.
============================================================

Small, single-responsibility helpers kept out of the trainer loop for clarity.
"""

from __future__ import annotations

from pathlib import Path

from ..models.unet import save_unet
from ..utils.logging_utils import get_logger

log = get_logger("pdm.callbacks")


class EarlyStopping:
    """Stop training when validation loss has not improved for `patience` epochs."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, value: float, epoch: int) -> bool:
        """Record a new value; return True if training should stop."""
        if value < self.best:
            self.best, self.best_epoch, self.counter = value, epoch, 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class CheckpointManager:
    """Saves raw + EMA UNet on best validation and on a periodic schedule."""

    def __init__(self, ckpt_dir: Path) -> None:
        self.ckpt_dir = ckpt_dir
        self.best = float("inf")

    def save_best(self, unet, ema, value: float) -> bool:
        if value < self.best:
            self.best = value
            save_unet(unet, self.ckpt_dir / "unet")
            save_unet(ema.state_module(), self.ckpt_dir / "unet_ema")
            log.info("  ↳ new best val %.5f — UNet + EMA saved", value)
            return True
        return False

    def save_periodic(self, ema, epoch: int) -> None:
        save_unet(ema.state_module(), self.ckpt_dir / f"unet_ema_ep{epoch:03d}")
        log.info("  ↳ periodic checkpoint saved @ epoch %d", epoch)
