"""
Shared utilities: seeding, device, checkpointing, brain masks, metrics.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch

import config as C

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────
def set_seed(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# Checkpointing (resume-safe bundles)
# ─────────────────────────────────────────────
def save_checkpoint(path: Path, *, model, optimizer=None, scheduler=None,
                    ema=None, epoch: int = 0, best_metric=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"epoch": epoch, "best_metric": best_metric,
            "model": model.state_dict()}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path: Path, *, model, optimizer=None, scheduler=None,
                    ema=None, device="cpu") -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    return ckpt


def find_resume(ckpt_dir: Path) -> Path | None:
    p = ckpt_dir / "last.pt"
    return p if p.exists() else None


def prune_old(ckpt_dir: Path, keep_last: int = C.CKPT_KEEP_LAST) -> None:
    snaps = sorted(ckpt_dir.glob("ckpt_ep*.pt"))
    for old in snaps[:-keep_last]:
        old.unlink(missing_ok=True)


# ─────────────────────────────────────────────
# Overfitting safeguards
# ─────────────────────────────────────────────
class EarlyStopper:
    """Tracks the best validation metric and signals when to stop.

    ``patience`` epochs without improvement (beyond ``min_delta``) → stop.
    ``patience <= 0`` disables early stopping (only best-tracking remains).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = 0
        self.wait = 0

    def step(self, val_metric: float, epoch: int) -> tuple[bool, bool]:
        """Returns ``(improved, should_stop)``."""
        improved = val_metric < self.best - self.min_delta
        if improved:
            self.best, self.best_epoch, self.wait = val_metric, epoch, 0
        else:
            self.wait += 1
        should_stop = self.patience > 0 and self.wait >= self.patience
        return improved, should_stop


def overfit_gap(train_loss: float, val_loss: float) -> float:
    """Relative generalisation gap (val−train)/train. Large + growing ⇒ overfit."""
    return (val_loss - train_loss) / (abs(train_loss) + 1e-8)


# ─────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────
def brain_mask_2d(slice_norm: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """(C,H,W) → (H,W) binary brain mask (any channel above eps)."""
    return (np.abs(slice_norm).max(axis=0) > eps).astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 2.0) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float(10 * np.log10(data_range ** 2 / (mse + 1e-12)))


def ssim2d(a: np.ndarray, b: np.ndarray, data_range: float = 2.0) -> float:
    try:
        from skimage.metrics import structural_similarity as _ssim
        return float(_ssim(a, b, data_range=data_range))
    except Exception:
        return float("nan")
