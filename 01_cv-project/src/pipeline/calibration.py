"""
Healthy-set calibration: baseline residual computation and threshold derivation.

Calibration runs the full DDIM reconstruction pipeline on held-out HEALTHY
slices to establish:

  * ``M_baseline``      — mean |orig − recon| over healthy tissue
  * ``threshold_pixel``  — percentile-based anomaly detection threshold
  * ``pixel_scale`` / ``latent_scale`` — normalisation factors for fusion

Both single-T and multi-T variants are provided.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from core import constants as C
from pipeline.diffusion import (
    make_ddim_scheduler,
    inference_timesteps,
    reconstruct_healthy,
)
from pipeline.scoring import (
    brain_mask_2d,
    latent_residual_2d,
    pixel_residual_2d,
    fuse_maps,
)

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════
# Single-T calibration
# ═════════════════════════════════════════════
@torch.no_grad()
def calibrate_on_healthy(
    vae, unet, val_loader, device, *,
    t_int: int = C.T_INT,
    ddim_steps: int = C.DDIM_STEPS,
    max_samples: int = C.MAX_CAL_SAMPLES,
    percentile: float = C.THRESHOLD_PERCENTILE,
    alpha: float = C.LATENT_FUSION_ALPHA,
    generator: torch.Generator | None = None,
):
    """Run the full DDIM reconstruction pipeline on held-out healthy slices.

    Returns ``(M_baseline, calib_dict)`` where:
      * ``M_baseline`` — mean |orig − recon| over healthy slices ``(3, 256, 256)``
      * ``calib_dict`` — threshold, scales, and metadata
    """
    ddim = make_ddim_scheduler()
    timesteps = inference_timesteps(ddim, t_int, ddim_steps)

    unet.eval()
    vae.eval()

    # ── Pass 1: accumulate M_baseline + cache raw maps ───────────────
    residual_sum = None
    n_samples = 0
    cache = []  # (raw_diff(3,H,W), latent_2d(H,W), brain_mask(H,W))

    for images in val_loader:
        if n_samples >= max_samples:
            break
        images = images.to(device)
        orig_norm, recon, z0, z_den = reconstruct_healthy(
            vae, unet, ddim, images, timesteps, t_int, generator,
        )

        orig_np = orig_norm.cpu().numpy()
        recon_np = recon.cpu().numpy()
        for b in range(orig_np.shape[0]):
            raw_diff = np.abs(orig_np[b] - recon_np[b])
            residual_sum = (
                raw_diff.copy() if residual_sum is None
                else residual_sum + raw_diff
            )
            cache.append((
                raw_diff,
                latent_residual_2d(z0[b:b + 1], z_den[b:b + 1]),
                brain_mask_2d(orig_np[b]),
            ))
            n_samples += 1

    if n_samples == 0:
        raise RuntimeError("Calibration set is empty.")

    m_baseline = (residual_sum / n_samples).astype(np.float32)

    # ── Pass 2 (in-memory): scales, then thresholds ─────────────────
    pixel_maps, latent_maps, masks = [], [], []
    for raw_diff, lat_2d, bmask in cache:
        diff = np.clip(raw_diff - m_baseline, 0, None).mean(axis=0) * bmask
        pixel_maps.append(diff)
        latent_maps.append(lat_2d * bmask)
        masks.append(bmask > 0)

    pixel_scale = float(np.mean([
        m[m > 0].mean() if np.any(m > 0) else 0.0 for m in pixel_maps
    ])) or 1.0
    latent_scale = float(np.mean([
        m[m > 0].mean() if np.any(m > 0) else 0.0 for m in latent_maps
    ])) or 1.0

    # Voxel-level thresholds
    pixel_voxels = np.concatenate([mp[m] for mp, m in zip(pixel_maps, masks)])
    fused_voxels = np.concatenate([
        fuse_maps(mp, ml, pixel_scale, latent_scale, alpha)[m]
        for mp, ml, m in zip(pixel_maps, latent_maps, masks)
    ])

    # Slice-level detection thresholds (per-slice max)
    max_pixel = [m.max() for m in pixel_maps]
    max_fused = [
        fuse_maps(mp, ml, pixel_scale, latent_scale, alpha).max()
        for mp, ml in zip(pixel_maps, latent_maps)
    ]

    calib = {
        "t_int": int(t_int),
        "ddim_steps": int(ddim_steps),
        "n_samples": int(n_samples),
        "percentile": float(percentile),
        "threshold_pixel": float(np.percentile(pixel_voxels, percentile)),
        "threshold_fused": float(np.percentile(fused_voxels, percentile)),
        "threshold_pixel_slicemax": float(np.percentile(max_pixel, percentile)),
        "threshold_fused_slicemax": float(np.percentile(max_fused, percentile)),
        "pixel_scale": pixel_scale,
        "latent_scale": latent_scale,
        "alpha": float(alpha),
    }
    return m_baseline, calib


# ─────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────
def save_calibration(
    m_baseline: np.ndarray, calib: dict,
    baseline_path: Path = C.BASELINE_PATH,
    calib_path: Path = C.CALIBRATION_PATH,
) -> None:
    """Save single-T calibration artefacts to disk."""
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(baseline_path, m_baseline)
    with open(calib_path, "w") as f:
        json.dump(calib, f, indent=2)


def load_calibration(
    baseline_path: Path = C.BASELINE_PATH,
    calib_path: Path = C.CALIBRATION_PATH,
):
    """Load single-T calibration. Returns ``(M_baseline, calib_dict)``."""
    m_baseline = np.load(baseline_path).astype(np.float32)
    calib = None
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)
    return m_baseline, calib


# ═════════════════════════════════════════════
# Multi-timestep calibration & scoring
# ═════════════════════════════════════════════
def aggregate_t_scores(
    score_stack: np.ndarray, mode: str = C.MULTI_T_AGG,
) -> np.ndarray:
    """Aggregate a ``(T, H, W)`` stack of standardised score maps → ``(H, W)``.

    ``mode="mean"`` averages (smoother); ``mode="max"`` takes voxel-wise maximum.
    """
    if mode == "max":
        return score_stack.max(axis=0)
    return score_stack.mean(axis=0)


@torch.no_grad()
def reconstruct_and_score_t(
    vae, unet, ddim, image, t_int, ddim_steps,
    m_baseline, pixel_scale, latent_scale, alpha,
    use_fusion, generator=None,
):
    """Single-T, healthy-scale-standardised anomaly score for one image.

    Returns ``(score_2d, brain_mask_2d, recon_np)``.
    """
    timesteps = inference_timesteps(ddim, t_int, ddim_steps)
    orig_norm, recon, z0, z_den = reconstruct_healthy(
        vae, unet, ddim, image, timesteps, t_int, generator,
    )

    orig_np = orig_norm[0].cpu().numpy()
    recon_np = recon[0].cpu().numpy()
    bmask = brain_mask_2d(orig_np)
    m_pixel = pixel_residual_2d(orig_np, recon_np, m_baseline, bmask)

    if use_fusion:
        m_latent = latent_residual_2d(z0, z_den) * bmask
        score = fuse_maps(m_pixel, m_latent, pixel_scale, latent_scale, alpha)
    else:
        score = m_pixel / (pixel_scale + 1e-8)
    return score, bmask, recon_np


@torch.no_grad()
def score_image_multi_t(
    vae, unet, ddim, image, baselines: dict, calib: dict,
    generator=None,
):
    """Aggregated multi-T anomaly score for one ``(1, 3, H, W)`` image.

    Returns ``(agg_score, brain_mask, recon_repr)``.
    """
    t_list = [int(t) for t in calib["t_list"]]
    use_fusion = bool(calib["use_fusion"])
    alpha = float(calib["alpha"])
    ddim_steps = int(calib["ddim_steps"])
    agg_mode = calib.get("agg", C.MULTI_T_AGG)

    stack, bmask, recon_repr = [], None, None
    repr_idx = len(t_list) // 2
    for i, t in enumerate(t_list):
        scales = calib["per_t"][str(t)]
        score_t, bmask, recon_np = reconstruct_and_score_t(
            vae, unet, ddim, image, t, ddim_steps, baselines[t],
            scales["pixel_scale"], scales["latent_scale"], alpha,
            use_fusion, generator,
        )
        stack.append(score_t)
        if i == repr_idx:
            recon_repr = recon_np

    agg_score = aggregate_t_scores(np.stack(stack, axis=0), agg_mode)
    return agg_score, bmask, recon_repr


@torch.no_grad()
def calibrate_on_healthy_multi_t(
    vae, unet, val_loader, device, *,
    t_list=None, ddim_steps: int = C.DDIM_STEPS,
    max_samples: int = C.MAX_CAL_SAMPLES,
    percentile: float = C.THRESHOLD_PERCENTILE,
    alpha: float = C.LATENT_FUSION_ALPHA,
    agg_mode: str = C.MULTI_T_AGG,
    use_fusion: bool = C.USE_LATENT_FUSION,
    generator: torch.Generator | None = None,
):
    """Multi-T healthy calibration.

    For every T in ``t_list`` computes its own ``M_baseline`` and healthy
    pixel/latent scales, then derives ONE operating threshold from the pooled
    distribution of the aggregated healthy score.

    Returns ``(baselines, calib)`` where ``baselines`` maps ``t_int → (3,H,W)``.
    """
    t_list = [int(t) for t in (t_list if t_list is not None else C.MULTI_T_LIST)]
    ddim = make_ddim_scheduler()
    unet.eval()
    vae.eval()

    # ── Pass 1: per-T M_baseline + cache ─────────────────────────────
    cache: dict[int, list] = {t: [] for t in t_list}
    resid_sum: dict[int, np.ndarray | None] = {t: None for t in t_list}
    n_samples = 0

    for images in val_loader:
        if n_samples >= max_samples:
            break
        images = images.to(device)
        bs = images.shape[0]
        for t in t_list:
            timesteps = inference_timesteps(ddim, t, ddim_steps)
            orig_norm, recon, z0, z_den = reconstruct_healthy(
                vae, unet, ddim, images, timesteps, t, generator,
            )
            orig_np = orig_norm.cpu().numpy()
            recon_np = recon.cpu().numpy()
            for b in range(bs):
                raw_diff = np.abs(orig_np[b] - recon_np[b])
                resid_sum[t] = (
                    raw_diff.copy() if resid_sum[t] is None
                    else resid_sum[t] + raw_diff
                )
                cache[t].append((
                    raw_diff,
                    latent_residual_2d(z0[b:b + 1], z_den[b:b + 1]),
                    brain_mask_2d(orig_np[b]),
                ))
        n_samples += bs

    if n_samples == 0:
        raise RuntimeError("Calibration set is empty.")
    n_cached = len(cache[t_list[0]])
    baselines = {
        t: (resid_sum[t] / n_cached).astype(np.float32) for t in t_list
    }

    # ── Pass 2: per-T standardised maps + scales ────────────────────
    masks = [bm > 0 for (_, _, bm) in cache[t_list[0]]]
    pixel_maps_dict: dict[int, list] = {t: [] for t in t_list}
    latent_maps_dict: dict[int, list] = {t: [] for t in t_list}
    scales: dict[int, tuple] = {}
    for t in t_list:
        for raw_diff, lat_2d, bmask in cache[t]:
            pm = np.clip(raw_diff - baselines[t], 0, None).mean(axis=0) * bmask
            pixel_maps_dict[t].append(pm)
            latent_maps_dict[t].append(lat_2d * bmask)
        ps = float(np.mean([
            m[m > 0].mean() if np.any(m > 0) else 0.0
            for m in pixel_maps_dict[t]
        ])) or 1.0
        ls = float(np.mean([
            m[m > 0].mean() if np.any(m > 0) else 0.0
            for m in latent_maps_dict[t]
        ])) or 1.0
        scales[t] = (ps, ls)

    # ── Pass 3: aggregated healthy score → percentile threshold ─────
    agg_voxels = []
    for i in range(n_cached):
        stack = []
        for t in t_list:
            ps, ls = scales[t]
            if use_fusion:
                s = fuse_maps(
                    pixel_maps_dict[t][i], latent_maps_dict[t][i], ps, ls, alpha,
                )
            else:
                s = pixel_maps_dict[t][i] / (ps + 1e-8)
            stack.append(s)
        agg = aggregate_t_scores(np.stack(stack, axis=0), agg_mode)
        agg_voxels.append(agg[masks[i]])
    agg_voxels_arr = np.concatenate(agg_voxels)

    calib = {
        "multi_t": True,
        "t_list": t_list,
        "ddim_steps": int(ddim_steps),
        "agg": agg_mode,
        "percentile": float(percentile),
        "use_fusion": bool(use_fusion),
        "alpha": float(alpha),
        "n_samples": int(n_cached),
        "threshold": float(np.percentile(agg_voxels_arr, percentile)),
        "per_t": {
            str(t): {"pixel_scale": scales[t][0], "latent_scale": scales[t][1]}
            for t in t_list
        },
    }
    return baselines, calib


def save_calibration_multi_t(
    baselines: dict, calib: dict,
    baseline_path: Path = C.MULTI_BASELINE_PATH,
    calib_path: Path = C.MULTI_CALIBRATION_PATH,
) -> None:
    """Save multi-T calibration artefacts to disk."""
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(baseline_path, **{f"t{t}": b for t, b in baselines.items()})
    with open(calib_path, "w") as f:
        json.dump(calib, f, indent=2)


def load_calibration_multi_t(
    baseline_path: Path = C.MULTI_BASELINE_PATH,
    calib_path: Path = C.MULTI_CALIBRATION_PATH,
):
    """Load multi-T calibration. Returns ``(baselines, calib)`` or ``(None, None)``."""
    if not (Path(baseline_path).exists() and Path(calib_path).exists()):
        return None, None
    data = np.load(baseline_path)
    baselines = {int(k[1:]): data[k].astype(np.float32) for k in data.files}
    with open(calib_path) as f:
        calib = json.load(f)
    return baselines, calib
