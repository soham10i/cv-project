"""
Segmentation / detection metrics.
=================================

Pixel-level AUROC and AUPRC (threshold-free ranking quality) plus DICE at three
operating points:
  * calibrated   — the deployable threshold from healthy calibration,
  * best-global  — single best threshold shared across all test slices (ceiling
                   for a deployable system),
  * oracle       — per-slice optimal threshold (theoretical upper bound).

Reporting all three is standard in UAD papers because the calibrated-vs-oracle
gap quantifies how much performance is left on the table by thresholding.
Ref: Zimmerer et al., 2022, "Medical Out-of-Distribution Analysis Challenge"
(IEEE TMI); Baur et al., 2021 (MedIA).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice = 2|P∩G| / (|P| + |G|)."""
    inter = float((pred * gt).sum())
    denom = float(pred.sum() + gt.sum())
    return 2.0 * inter / (denom + 1e-8)


def pixel_auroc(score: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """AUROC over brain voxels; NaN if a class is absent."""
    brain = mask.flatten() > 0
    y = gt.flatten()[brain]
    s = score.flatten()[brain]
    if not (0 < y.sum() < y.size) or s.max() <= s.min():
        return float("nan")
    return float(roc_auc_score(y, s))


def pixel_auprc(score: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Average precision over brain voxels; NaN if no positives."""
    brain = mask.flatten() > 0
    y = gt.flatten()[brain]
    s = score.flatten()[brain]
    if y.sum() == 0 or s.max() <= s.min():
        return float("nan")
    return float(average_precision_score(y, s))


def dice_over_grid(
    score: np.ndarray, gt: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Vectorized DICE at each threshold in ``grid`` -> array of len(grid)."""
    preds = score[None] > grid[:, None, None]
    inter = (preds * gt[None]).sum(axis=(1, 2))
    psum = preds.reshape(len(grid), -1).sum(axis=1)
    return 2.0 * inter / (psum + gt.sum() + 1e-8)
