"""
Checkpoint / resume utilities  (crash- and preemption-safe)
===========================================================
Colab T4 sessions disconnect and are time-limited, so long VAE / diffusion runs
*must* be resumable.  This module saves a single self-contained bundle —
model + optimizer + LR scheduler + EMA + epoch + best-metric + RNG state — so a
run can continue exactly where it stopped after a reconnect.

Two artefact kinds are written per stage:
  * ``last.pt``         — overwritten every epoch; the resume point.
  * ``ckpt_ep###.pt``   — periodic snapshots (kept: last ``keep_last``) for
                          inspecting / rolling back to a specific epoch.

The *best* model is still exported separately by each trainer via
``save_pretrained`` (a Hugging Face dir) so downstream ``from_pretrained`` keeps
working; these ``.pt`` bundles are purely for training continuation.

Point ``CV_MODEL_DIR`` at Google Drive so the bundles survive a disconnect.
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path, *, model, optimizer=None, scheduler=None, ema=None,
                    epoch: int, best_metric=None, extra: dict | None = None) -> None:
    """Atomically write a resumable training bundle to ``path`` (a .pt file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        # utils.EMA exposes .state_dict() (the shadow) but no loader, so we
        # snapshot the shadow dict directly and restore it by hand on resume.
        "ema": ema.state_dict() if ema is not None else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "extra": extra or {},
    }
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)            # atomic: a crash mid-write can't corrupt last.pt


def load_checkpoint(path, *, model, optimizer=None, scheduler=None, ema=None,
                    device=None, map_location="cpu") -> dict:
    """Restore a bundle written by ``save_checkpoint`` in place.  Returns the
    raw dict so the caller can read ``epoch`` / ``best_metric`` / ``extra``."""
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer") is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler") is not None:
        scheduler.load_state_dict(ck["scheduler"])
    if ema is not None and ck.get("ema") is not None:
        dev = device if device is not None else next(model.parameters()).device
        ema.shadow = {k: v.to(dev) for k, v in ck["ema"].items()}
    # RNG — restore so data shuffling / noise sampling continue the same stream.
    try:
        torch.set_rng_state(ck["torch_rng"].cpu() if hasattr(ck["torch_rng"], "cpu")
                            else ck["torch_rng"])
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        if ck.get("numpy_rng") is not None:
            np.random.set_state(ck["numpy_rng"])
        if ck.get("python_rng") is not None:
            random.setstate(ck["python_rng"])
    except Exception:
        pass    # RNG restore is best-effort; never let it abort a resume
    return ck


def prune_old(dir_path, pattern: str = "ckpt_ep*.pt", keep_last: int = 3) -> None:
    """Keep only the ``keep_last`` most recent periodic snapshots in ``dir_path``."""
    files = sorted(glob.glob(str(Path(dir_path) / pattern)), key=os.path.getmtime)
    for old in files[:-keep_last] if keep_last > 0 else files:
        try:
            os.remove(old)
        except OSError:
            pass


def find_resume(dir_path, name: str = "last.pt") -> Path | None:
    """Return the resume bundle path if it exists, else None."""
    p = Path(dir_path) / name
    return p if p.exists() else None
