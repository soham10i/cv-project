"""
PyTorch datasets for MPDF.
==========================

Two datasets:
  * ``HealthyPatchDataset`` — yields individual healthy patches for training.
    A flat index over (slice, patch-coord) pairs is built once so __getitem__
    is O(1) and DataLoader workers shard cleanly.
  * ``AnomalousSliceDataset`` — yields whole test slices + lesion masks for
    evaluation (patching happens inside the evaluator so fusion can run).

Augmentation (training only): horizontal flip is valid for axial brain MRI due
to approximate bilateral symmetry; a small intensity jitter improves robustness.
Ref: Isensee et al., 2021, "nnU-Net" (Nature Methods) for medical augmentation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import CONFIG
from ..utils.exceptions import DataError
from ..utils.io import read_manifest
from .patches import iter_patch_coords, patch_foreground_fraction


class HealthyPatchDataset(Dataset):
    """Healthy patches for diffusion training.

    Builds a flat (stem, top, left) index across all healthy slices, filtering
    out background-only patches so the model only ever trains on tissue.
    """

    def __init__(
        self,
        manifest_path: Path,
        slices_dir: Path = CONFIG.paths.slices_dir,
        augment: bool = True,
        limit: int | None = None,
    ) -> None:
        self.slices_dir = Path(slices_dir)
        self.augment = augment
        self.patch_size = CONFIG.patch.patch_size
        self.stride = CONFIG.patch.stride

        stems = read_manifest(manifest_path)
        if limit:
            stems = stems[:limit]

        # Build the flat patch index using memory-mapped reads so we only touch
        # each patch region, never the whole slice. Slices are stored float16.
        self.index: list[tuple[str, int, int]] = []
        for stem in stems:
            f = self.slices_dir / f"{stem}.npy"
            if not f.exists():
                continue
            img = np.load(f, mmap_mode="r")
            size = img.shape[-1]
            for (t, l) in iter_patch_coords(size, self.patch_size, self.stride):
                patch = img[:, t : t + self.patch_size, l : l + self.patch_size]
                if patch_foreground_fraction(patch) >= CONFIG.patch.min_patch_foreground:
                    self.index.append((stem, t, l))

        if not self.index:
            raise DataError(
                f"No foreground patches found from manifest {manifest_path}. "
                "Check preprocessing output and patch thresholds."
            )
        # Cache the open memmap handle for the most-recent slice. Even on a cache
        # miss (expected with shuffle) the crop below pages in only the patch.
        self._cache_stem: str | None = None
        self._cache_img: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, stem: str) -> np.ndarray:
        if stem != self._cache_stem:
            self._cache_img = np.load(self.slices_dir / f"{stem}.npy", mmap_mode="r")
            self._cache_stem = stem
        return self._cache_img  # type: ignore[return-value]

    def __getitem__(self, idx: int) -> torch.Tensor:
        stem, t, l = self.index[idx]
        img = self._load(stem)
        # Materialise ONLY the patch region as float32 (img is a float16 memmap).
        patch = np.asarray(
            img[:, t : t + self.patch_size, l : l + self.patch_size], dtype=np.float32
        )

        if self.augment:
            if np.random.rand() < 0.5:
                patch = patch[:, :, ::-1].copy()  # horizontal flip
            if np.random.rand() < 0.3:
                patch = np.clip(patch * (1.0 + np.random.uniform(-0.05, 0.05)), -1, 1)

        return torch.from_numpy(patch).float()


class AnomalousSliceDataset(Dataset):
    """Whole test slices with lesion ground-truth masks (for evaluation)."""

    def __init__(
        self,
        manifest_path: Path,
        slices_dir: Path = CONFIG.paths.slices_dir,
        masks_dir: Path = CONFIG.paths.masks_dir,
        limit: int | None = None,
    ) -> None:
        self.slices_dir = Path(slices_dir)
        self.masks_dir = Path(masks_dir)
        self.stems = read_manifest(manifest_path)
        if limit:
            self.stems = self.stems[:limit]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        stem = self.stems[idx]
        img = np.load(self.slices_dir / f"{stem}.npy").astype(np.float32)
        mask_f = self.masks_dir / f"{stem}.npy"
        mask = (
            np.load(mask_f).astype(np.float32)
            if mask_f.exists()
            else np.zeros((CONFIG.data.target_size, CONFIG.data.target_size), np.float32)
        )
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float(), stem
