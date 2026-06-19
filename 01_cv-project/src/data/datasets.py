"""
Reusable PyTorch datasets for BraTS-PEDs slices.

Both datasets load preprocessed 2D ``.npy`` slices (shape ``(3, 256, 256)``)
produced by ``data.preprocessing``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class HealthySliceDataset(Dataset):
    """Loads preprocessed healthy brain slices as float32 tensors.

    Each ``.npy`` file is expected to have shape ``(3, 256, 256)``
    (three MRI modalities, zero-padded to 256×256).
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npy"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files in {self.data_dir}")
        log.info("Dataset: %d slices from %s", len(self.files), self.data_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = np.load(self.files[idx])
        return torch.from_numpy(arr).float()


class AnomalousSliceDataset(Dataset):
    """Loads anomalous slices paired with ground-truth segmentation masks.

    Only slices whose corresponding mask file exists in ``mask_dir`` are
    included — missing masks are silently skipped with a warning.

    Returns ``(image, mask, stem)`` where:
      * ``image`` — ``(3, 256, 256)`` float32 tensor
      * ``mask``  — ``(256, 256)`` float32 tensor (binary)
      * ``stem``  — filename stem (for logging / panel titles)
    """

    def __init__(self, img_dir: str | Path, mask_dir: str | Path) -> None:
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)

        all_imgs = sorted(self.img_dir.glob("*.npy"))
        self.img_files = [p for p in all_imgs if (self.mask_dir / p.name).exists()]

        n_missing = len(all_imgs) - len(self.img_files)
        if n_missing:
            log.warning("Skipping %d slice(s) with no matching mask.", n_missing)
        if len(self.img_files) == 0:
            raise FileNotFoundError(
                f"No usable .npy/mask pairs in {self.img_dir}"
            )
        log.info("Test set: %d anomalous slices", len(self.img_files))

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img = np.load(self.img_files[idx])
        mask = np.load(self.mask_dir / self.img_files[idx].name)
        return (
            torch.from_numpy(img).float(),
            torch.from_numpy(mask).float(),
            self.img_files[idx].stem,
        )
