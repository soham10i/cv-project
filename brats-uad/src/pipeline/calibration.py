"""
Healthy-set calibration: baseline residual + anomaly threshold derivation.
===========================================================================
Runs the full DDIM partial-noise reconstruction on held-out HEALTHY slices to
establish, per timestep level T:

  * ``M_baseline``  — mean residual_stack over healthy tissue   (K, 256, 256)
  * ``pixel_scale`` / ``latent_scale`` — healthy normalisation factors (fusion)
  * ``threshold``   — percentile of the pooled healthy brain-voxel score

Both single-T and multi-T variants are provided; multi-T computes one baseline
per T in MULTI_T_LIST and a single operating threshold on the aggregated score.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

import config as C
from pipeline.diffusion import make_ddim_scheduler, inference_timesteps, reconstruct_healthy
from pipeline.scoring import (brain_mask_2d, residual_stack, pixel_residual_2d,
                              latent_residual_2d, fuse_maps)

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════
# Single-T
# ═════════════════════════════════════════════
@torch.no_grad()
def calibrate_single_t(vae, unet, val_loader, device, *, t_int=C.T_INT,
                       ddim_steps=C.DDIM_STEPS, max_samples=C.MAX_CAL_SAMPLES,
                       percentile=C.THRESHOLD_PERCENTILE, alpha=C.LATENT_FUSION_ALPHA,
                       use_fusion=C.USE_LATENT_FUSION, generator=None, scaling_factor=None):
    sf = C.load_scaling_factor() if scaling_factor is None else scaling_factor
    ddim = make_ddim_scheduler()
    timesteps = inference_timesteps(ddim, t_int, ddim_steps)
    unet.eval(); vae.eval()

    resid_sum, n = None, 0
    cache = []                                       # (stack(K,H,W), lat(H,W), bmask)
    for images in val_loader:
        if n >= max_samples:
            break
        images = images.to(device)
        orig, recon, z0, zden = reconstruct_healthy(
            vae, unet, ddim, images, timesteps, t_int, generator, sf)
        o, r = orig.cpu().numpy(), recon.cpu().numpy()
        for b in range(o.shape[0]):
            st = residual_stack(o[b], r[b])
            resid_sum = st.copy() if resid_sum is None else resid_sum + st
            cache.append((st, latent_residual_2d(z0[b:b+1], zden[b:b+1]),
                          brain_mask_2d(o[b])))
            n += 1
    if n == 0:
        raise RuntimeError("Calibration set is empty.")
    m_baseline = (resid_sum / n).astype(np.float32)

    pixel_maps, latent_maps, masks = [], [], []
    for st, lat, bm in cache:
        pm = pixel_residual_2d_from_stack(st, m_baseline, bm)
        pixel_maps.append(pm)
        latent_maps.append(lat * bm)
        masks.append(bm > 0)

    pixel_scale = _pos_mean(pixel_maps)
    latent_scale = _pos_mean(latent_maps)

    if use_fusion:
        voxels = np.concatenate([
            fuse_maps(pm, lm, pixel_scale, latent_scale, alpha)[m]
            for pm, lm, m in zip(pixel_maps, latent_maps, masks)])
    else:
        voxels = np.concatenate([pm[m] for pm, m in zip(pixel_maps, masks)])

    calib = {"multi_t": False, "t_int": int(t_int), "ddim_steps": int(ddim_steps),
             "n_samples": int(n), "percentile": float(percentile),
             "use_fusion": bool(use_fusion), "alpha": float(alpha),
             "pixel_scale": pixel_scale, "latent_scale": latent_scale,
             "threshold": float(np.percentile(voxels, percentile)),
             "scaling_factor": float(sf)}
    return m_baseline, calib


# ═════════════════════════════════════════════
# Multi-T
# ═════════════════════════════════════════════
@torch.no_grad()
def calibrate_multi_t(vae, unet, val_loader, device, *, t_list=None,
                      ddim_steps=C.DDIM_STEPS, max_samples=C.MAX_CAL_SAMPLES,
                      percentile=C.THRESHOLD_PERCENTILE, alpha=C.LATENT_FUSION_ALPHA,
                      agg_mode=C.MULTI_T_AGG, use_fusion=C.USE_LATENT_FUSION,
                      generator=None, scaling_factor=None):
    sf = C.load_scaling_factor() if scaling_factor is None else scaling_factor
    t_list = [int(t) for t in (t_list or C.MULTI_T_LIST)]
    ddim = make_ddim_scheduler()
    unet.eval(); vae.eval()

    cache = {t: [] for t in t_list}
    resid_sum = {t: None for t in t_list}
    n = 0
    for images in val_loader:
        if n >= max_samples:
            break
        images = images.to(device)
        bs = images.shape[0]
        for t in t_list:
            ts = inference_timesteps(ddim, t, ddim_steps)
            orig, recon, z0, zden = reconstruct_healthy(
                vae, unet, ddim, images, ts, t, generator, sf)
            o, r = orig.cpu().numpy(), recon.cpu().numpy()
            for b in range(bs):
                st = residual_stack(o[b], r[b])
                resid_sum[t] = st.copy() if resid_sum[t] is None else resid_sum[t] + st
                cache[t].append((st, latent_residual_2d(z0[b:b+1], zden[b:b+1]),
                                 brain_mask_2d(o[b])))
        n += bs
    if n == 0:
        raise RuntimeError("Calibration set is empty.")

    n_cached = len(cache[t_list[0]])
    baselines = {t: (resid_sum[t] / n_cached).astype(np.float32) for t in t_list}
    masks = [bm > 0 for (_, _, bm) in cache[t_list[0]]]

    pixel_maps = {t: [] for t in t_list}
    latent_maps = {t: [] for t in t_list}
    scales = {}
    for t in t_list:
        for st, lat, bm in cache[t]:
            pixel_maps[t].append(pixel_residual_2d_from_stack(st, baselines[t], bm))
            latent_maps[t].append(lat * bm)
        scales[t] = (_pos_mean(pixel_maps[t]), _pos_mean(latent_maps[t]))

    agg_voxels = []
    for i in range(n_cached):
        stack = []
        for t in t_list:
            ps, ls = scales[t]
            s = (fuse_maps(pixel_maps[t][i], latent_maps[t][i], ps, ls, alpha)
                 if use_fusion else pixel_maps[t][i] / (ps + 1e-8))
            stack.append(s)
        agg = _aggregate(np.stack(stack, 0), agg_mode)
        agg_voxels.append(agg[masks[i]])
    agg_voxels = np.concatenate(agg_voxels)

    calib = {"multi_t": True, "t_list": t_list, "ddim_steps": int(ddim_steps),
             "agg": agg_mode, "percentile": float(percentile),
             "use_fusion": bool(use_fusion), "alpha": float(alpha),
             "n_samples": int(n_cached),
             "threshold": float(np.percentile(agg_voxels, percentile)),
             "per_t": {str(t): {"pixel_scale": scales[t][0],
                                "latent_scale": scales[t][1]} for t in t_list},
             "scaling_factor": float(sf)}
    return baselines, calib


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────
def pixel_residual_2d_from_stack(stack, m_baseline, bmask):
    """Modality-weighted pixel map from a pre-computed residual stack."""
    from pipeline.scoring import _channel_weights
    s = np.clip(stack - m_baseline, 0, None)
    w = _channel_weights()[:, None, None]
    return (s * w).sum(axis=0) * bmask


def _pos_mean(maps) -> float:
    return float(np.mean([m[m > 0].mean() if np.any(m > 0) else 0.0
                          for m in maps])) or 1.0


def _aggregate(stack: np.ndarray, mode: str) -> np.ndarray:
    return stack.max(0) if mode == "max" else stack.mean(0)


# ─────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────
def save_single_t(m_baseline, calib,
                  baseline_path=C.BASELINE_PATH, calib_path=C.CALIBRATION_PATH):
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(baseline_path, m_baseline)
    with open(calib_path, "w") as f:
        json.dump(calib, f, indent=2)


def load_single_t(baseline_path=C.BASELINE_PATH, calib_path=C.CALIBRATION_PATH):
    if not (Path(baseline_path).exists() and Path(calib_path).exists()):
        return None, None
    m = np.load(baseline_path).astype(np.float32)
    with open(calib_path) as f:
        return m, json.load(f)


def save_multi_t(baselines, calib,
                 baseline_path=C.MULTI_BASELINE_PATH, calib_path=C.MULTI_CALIBRATION_PATH):
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(baseline_path, **{f"t{t}": b for t, b in baselines.items()})
    with open(calib_path, "w") as f:
        json.dump(calib, f, indent=2)


def load_multi_t(baseline_path=C.MULTI_BASELINE_PATH, calib_path=C.MULTI_CALIBRATION_PATH):
    if not (Path(baseline_path).exists() and Path(calib_path).exists()):
        return None, None
    data = np.load(baseline_path)
    baselines = {int(k[1:]): data[k].astype(np.float32) for k in data.files}
    with open(calib_path) as f:
        return baselines, json.load(f)


# ─────────────────────────────────────────────
# Multi-T scoring (evaluation-time, one image)
# ─────────────────────────────────────────────
@torch.no_grad()
def score_image_multi_t(vae, unet, ddim, image, baselines, calib,
                        generator=None, scaling_factor=None):
    """Aggregated multi-T anomaly score for one (1,C,H,W) image.

    Returns (agg_score(H,W), brain_mask(H,W), recon_repr(C,H,W)).
    """
    sf = C.load_scaling_factor() if scaling_factor is None else scaling_factor
    t_list = [int(t) for t in calib["t_list"]]
    use_fusion = bool(calib["use_fusion"])
    alpha = float(calib["alpha"])
    ddim_steps = int(calib["ddim_steps"])
    agg_mode = calib.get("agg", C.MULTI_T_AGG)

    stack, bmask, recon_repr = [], None, None
    repr_idx = len(t_list) // 2
    for i, t in enumerate(t_list):
        ts = inference_timesteps(ddim, t, ddim_steps)
        orig, recon, z0, zden = reconstruct_healthy(
            vae, unet, ddim, image, ts, t, generator, sf)
        o = orig[0].cpu().numpy(); r = recon[0].cpu().numpy()
        bmask = brain_mask_2d(o)
        pm = pixel_residual_2d(o, r, baselines[t], bmask)
        scales = calib["per_t"][str(t)]
        if use_fusion:
            lm = latent_residual_2d(z0, zden) * bmask
            s = fuse_maps(pm, lm, scales["pixel_scale"], scales["latent_scale"], alpha)
        else:
            s = pm / (scales["pixel_scale"] + 1e-8)
        stack.append(s)
        if i == repr_idx:
            recon_repr = r
    return _aggregate(np.stack(stack, 0), agg_mode), bmask, recon_repr
