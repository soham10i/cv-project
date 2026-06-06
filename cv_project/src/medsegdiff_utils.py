"""
MedSegDiff — shared helpers
============================
Dataset, mask⇄x0 mapping, DDIM mask sampling, and DICE — used by both the
training and evaluation scripts so the diffusion process is identical on both
sides.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import config as C


# ─────────────────────────────────────────────
# Dataset (downsamples stored 256² slices to the training size)
# ─────────────────────────────────────────────
class SegDataset(Dataset):
    """Returns (image (3,s,s) float32, mask (1,s,s) {0,1}) at ``size``."""

    def __init__(self, split: str, size: int = C.SEG_TRAIN_SIZE):
        self.img_dir  = C.SEG_DATA_DIR / split / "images"
        self.mask_dir = C.SEG_DATA_DIR / split / "masks"
        self.files = sorted(self.img_dir.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No slices in {self.img_dir} — run preprocess_seg.py")
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx].name
        img  = torch.from_numpy(np.load(self.img_dir / name).astype(np.float32))   # (3,256,256)
        mask = torch.from_numpy(np.load(self.mask_dir / name).astype(np.float32))[None]  # (1,256,256)
        if self.size != img.shape[-1]:
            img  = F.interpolate(img[None], size=self.size, mode="bilinear", align_corners=False)[0]
            mask = F.interpolate(mask[None], size=self.size, mode="nearest")[0]
        return img, mask


def mask_to_x0(mask: torch.Tensor) -> torch.Tensor:
    """{0,1} → {-1,1} (the diffusion target)."""
    return mask * 2.0 - 1.0


# ─────────────────────────────────────────────
# DDIM sampling: noise → mask, conditioned on the image
# ─────────────────────────────────────────────
@torch.no_grad()
def sample_mask(model, ddim, image: torch.Tensor, steps: int = 50,
                ensemble: int = 1, generator: torch.Generator | None = None):
    """
    Reverse-diffuse a mask from pure noise, conditioned on ``image``.
    Returns a soft probability map (B,1,s,s) in [0,1] (ensemble-averaged).
    """
    ddim.set_timesteps(steps)
    B, _, s, _ = image.shape
    device = image.device
    prob_sum = torch.zeros(B, 1, s, s, device=device)
    for _ in range(max(1, ensemble)):
        x = torch.randn(B, 1, s, s, device=device, generator=generator)
        for t in ddim.timesteps:
            t_b = torch.full((B,), int(t), device=device, dtype=torch.long)
            eps = model(x, t_b, image)
            x = ddim.step(eps, t, x).prev_sample
        prob_sum += ((x.clamp(-1, 1) + 1) / 2)        # x0_pred → [0,1]
    return prob_sum / max(1, ensemble)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Binary DICE.  Empty-GT slice scores 1 if the prediction is also empty."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if gt.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    inter = np.logical_and(pred, gt).sum()
    return float(2 * inter / (pred.sum() + gt.sum() + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if gt.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / (union + eps))
