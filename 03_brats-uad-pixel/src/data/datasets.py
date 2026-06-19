"""
Manifest-driven PyTorch datasets for BraTS slices.
===================================================
Each slice is stored once under ``slices/``; a *view* is a manifest file listing
slice stems.  These datasets resolve stems → ``.npy`` paths, so train/val/test
separation is guaranteed by the manifests (no patient leakage).

All slices are ``(N_CHANNELS, 256, 256)`` float16 on disk, returned as float32.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config as C

log = logging.getLogger(__name__)


def read_manifest(manifest_path: str | Path, limit: int | None = None) -> list[str]:
    """Return the slice stems listed in a manifest file."""
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}. Run make_slices.py first.")
    with open(p) as f:
        stems = [ln.strip() for ln in f if ln.strip()]
    if not stems:
        raise RuntimeError(f"Manifest empty: {p}")
    return stems[:limit] if limit else stems


class SliceDataset(Dataset):
    """Returns ``(image,)`` float32 tensors of shape ``(N_CHANNELS, 256, 256)``.

    Used for VAE training (vae_train / vae_val) and diffusion training (healthy).
    """

    def __init__(self, manifest_path: str | Path, limit: int | None = None) -> None:
        self.stems = read_manifest(manifest_path, limit)
        log.info("SliceDataset: %d slices from %s",
                 len(self.stems), Path(manifest_path).name)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = np.load(C.SLICES_DIR / f"{self.stems[idx]}.npy").astype(np.float32)
        return torch.from_numpy(arr)


class AnomalousSliceDataset(Dataset):
    """Lesion slices paired with GT masks for evaluation.

    Returns ``(image, mask, stem)`` where image is ``(N_CHANNELS, 256, 256)``
    float32 and mask is ``(256, 256)`` float32 binary.
    """

    def __init__(self, manifest_path: str | Path = C.MANIFEST_TEST_ANOM,
                 limit: int | None = None) -> None:
        stems = read_manifest(manifest_path, limit)
        self.stems = [s for s in stems if (C.MASKS_DIR / f"{s}.npy").exists()]
        n_missing = len(stems) - len(self.stems)
        if n_missing:
            log.warning("Skipping %d test slice(s) with no mask file.", n_missing)
        if not self.stems:
            raise RuntimeError("No usable (slice, mask) pairs for evaluation.")
        log.info("AnomalousSliceDataset: %d lesion slices", len(self.stems))

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img = np.load(C.SLICES_DIR / f"{stem}.npy").astype(np.float32)
        mask = np.load(C.MASKS_DIR / f"{stem}.npy").astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(mask), stem
