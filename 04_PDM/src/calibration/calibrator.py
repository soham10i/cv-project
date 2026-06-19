"""
Threshold calibration on healthy validation slices.
===================================================

Runs the full multi-scale scorer on lesion-free validation slices (which should
produce near-zero anomaly scores) and fits the operating threshold as a high
percentile of the pooled healthy brain-voxel scores. Any test voxel above this
threshold is more anomalous than ~95% of healthy tissue.

Statistical note: a stable p-th percentile estimate needs ~ (20 / (1 - p/100))
samples (Wilks, 1941). With p=95 that is ~200 slices — hence the default cap.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import CONFIG
from ..data.dataset import AnomalousSliceDataset
from ..scoring.multiscale import MultiScaleScorer
from ..utils.exceptions import CalibrationError
from ..utils.io import write_json
from ..utils.logging_utils import get_logger

log = get_logger("pdm.calibration")


def calibrate(
    scorer: MultiScaleScorer,
    healthy_manifest,
    device: torch.device,
    max_samples: int = CONFIG.calibration.max_samples,
    percentile: float = CONFIG.calibration.threshold_percentile,
    generator: torch.Generator | None = None,
) -> dict:
    """Fit and persist the anomaly threshold; return the calibration dict."""
    ds = AnomalousSliceDataset(healthy_manifest, limit=max_samples)
    if len(ds) == 0:
        raise CalibrationError("Healthy calibration set is empty.")

    pooled: list[np.ndarray] = []
    for i in range(len(ds)):
        image, _, name = ds[i]
        result = scorer.score_slice(image.numpy(), generator=generator)
        voxels = result.score[result.mask > 0]
        if voxels.size:
            pooled.append(voxels)
        if (i + 1) % 25 == 0:
            log.info("calibrated on %d/%d healthy slices", i + 1, len(ds))

    if not pooled:
        raise CalibrationError("No brain voxels collected during calibration.")

    all_voxels = np.concatenate(pooled)
    threshold = float(np.percentile(all_voxels, percentile))
    calib = {
        "threshold": threshold,
        "percentile": percentile,
        "n_samples": len(pooled),
        "score_timesteps": list(CONFIG.scoring.score_timesteps),
        "scale_weights": list(CONFIG.scoring.scale_weights),
        "noise_strategy": CONFIG.noise.strategy,
        "healthy_score_mean": float(all_voxels.mean()),
        "healthy_score_std": float(all_voxels.std()),
    }
    write_json(CONFIG.paths.calibration_path, calib)
    log.info(
        "Calibration done | threshold=%.4f | healthy mean=%.4f | n=%d -> %s",
        threshold,
        calib["healthy_score_mean"],
        len(pooled),
        CONFIG.paths.calibration_path,
    )
    return calib
