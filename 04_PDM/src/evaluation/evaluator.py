"""
Evaluation loop: score test slices, refine, threshold, compute metrics.
=======================================================================

For each anomalous test slice: run the multi-scale scorer, optionally refine the
score with a dense CRF, threshold at the calibrated operating point, and
accumulate per-image + aggregate AUROC / AUPRC / DICE. Saves a metrics.json and
qualitative panels.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import CONFIG
from ..data.dataset import AnomalousSliceDataset
from ..scoring.crf import refine as crf_refine
from ..scoring.multiscale import MultiScaleScorer
from ..utils.exceptions import EvaluationError
from ..utils.io import read_json, write_json
from ..utils.logging_utils import get_logger
from .metrics import dice_coefficient, dice_over_grid, pixel_auprc, pixel_auroc

log = get_logger("pdm.eval")


def evaluate(
    scorer: MultiScaleScorer,
    test_manifest,
    device: torch.device,
    limit: int | None = None,
    max_plots: int = CONFIG.eval.max_plots,
    generator: torch.Generator | None = None,
) -> dict:
    """Run evaluation over the test manifest and persist metrics + panels."""
    calib = read_json(CONFIG.paths.calibration_path)
    threshold = float(calib["threshold"])
    log.info("Loaded calibrated threshold=%.4f", threshold)

    ds = AnomalousSliceDataset(test_manifest, limit=limit)
    if len(ds) == 0:
        raise EvaluationError("Test set is empty.")

    grid = np.linspace(
        max(threshold * 0.05, 1e-4), threshold * 4.0, CONFIG.eval.n_threshold_grid
    ).astype(np.float32)
    grid_sum = np.zeros(len(grid))
    n_grid = 0

    aurocs, auprcs, dices, oracle_dices, per_image = [], [], [], [], []
    results_dir = CONFIG.paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(ds)):
        image_t, gt_t, name = ds[i]
        image = image_t.numpy()
        gt = (gt_t.numpy() > 0).astype(np.float32)

        result = scorer.score_slice(image, generator=generator)
        guidance = result.orig[CONFIG.crf.guidance_channel]
        refined = crf_refine(result.score, guidance, result.mask)
        # The deployable prediction thresholds the RAW score, because the
        # operating threshold was calibrated in raw-score space (calibrator.py).
        # The CRF map (a [0,1] probability) is shown as a refinement view; to use
        # it for the decision, calibrate a threshold in CRF space (see docs).
        pred = ((result.score > threshold).astype(np.float32)) * result.mask

        auroc = pixel_auroc(result.score, gt, result.mask)
        auprc = pixel_auprc(result.score, gt, result.mask)
        dice = dice_coefficient(pred, gt)

        oracle = float("nan")
        if gt.sum() > 0:
            dgrid = dice_over_grid(result.score, gt, grid)
            oracle = float(dgrid.max())
            grid_sum += dgrid
            n_grid += 1
            oracle_dices.append(oracle)
            if not np.isnan(auprc):
                auprcs.append(auprc)
        if not np.isnan(auroc):
            aurocs.append(auroc)
        dices.append(dice)

        per_image.append(
            {"name": name, "auroc": auroc, "auprc": auprc, "dice": dice, "oracle_dice": oracle}
        )
        log.info(
            "[%d/%d] %s | AUROC %.3f | AUPRC %.3f | DICE %.3f | oracle %.3f",
            i + 1, len(ds), name, auroc, auprc, dice, oracle,
        )
        if i < max_plots:
            _save_panel(name, result, refined, pred, gt, auroc, dice, results_dir)

    gbest_thr = gbest_dice = float("nan")
    if n_grid:
        mean_by_thr = grid_sum / n_grid
        bi = int(np.argmax(mean_by_thr))
        gbest_thr, gbest_dice = float(grid[bi]), float(mean_by_thr[bi])

    summary = {
        "n_images": len(per_image),
        "auroc_mean": _mean(aurocs),
        "auroc_std": float(np.std(aurocs)) if aurocs else float("nan"),
        "auprc_mean": _mean(auprcs),
        "dice_calibrated": _mean(dices),
        "dice_best_global": gbest_dice,
        "best_global_threshold": gbest_thr,
        "dice_oracle": _mean(oracle_dices),
        "calibrated_threshold": threshold,
        "per_image": per_image,
    }
    write_json(results_dir / "metrics.json", summary)
    _log_summary(summary)
    return summary


def _mean(xs) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _log_summary(s: dict) -> None:
    log.info("=" * 60)
    log.info("AGGREGATE over %d images", s["n_images"])
    log.info("  AUROC                 : %.4f ± %.4f", s["auroc_mean"], s["auroc_std"])
    log.info("  AUPRC                 : %.4f", s["auprc_mean"])
    log.info("  DICE @ calibrated thr : %.4f", s["dice_calibrated"])
    log.info("  DICE @ best global    : %.4f (thr=%.4f)", s["dice_best_global"], s["best_global_threshold"])
    log.info("  DICE oracle (ceiling) : %.4f", s["dice_oracle"])
    log.info("=" * 60)


def _save_panel(name, result, refined, pred, gt, auroc, dice, out_dir) -> None:
    """Save a 6-panel qualitative figure for one slice."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 6, figsize=(28, 5))
    ax[0].imshow(result.orig[1], cmap="gray"); ax[0].set_title("Original (T1c)")
    ax[1].imshow(result.recon[1], cmap="gray"); ax[1].set_title("Healthy recon (T1c)")
    im = ax[2].imshow(result.score, cmap="hot"); ax[2].set_title("Anomaly score")
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    ax[3].imshow(refined, cmap="hot"); ax[3].set_title("CRF-refined")
    ax[4].imshow(pred, cmap="gray"); ax[4].set_title("Prediction")
    ax[5].imshow(gt, cmap="gray"); ax[5].set_title("Ground truth")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{name} | AUROC {auroc:.3f} | DICE {dice:.3f}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / f"eval_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
