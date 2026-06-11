"""
Shared utilities
=================
Everything that used to be copy-pasted across the evaluation scripts now
lives here:  device selection, VAE input normalisation, the SAAM attention
processor + hooks, EMA, the DDIM reconstruction loop, the anomaly-scoring
function, and the healthy-set calibration (baseline residual + threshold).

Keeping a *single* implementation of scoring/reconstruction guarantees that
calibration (during training) and inference (during evaluation) are
distribution-matched — which is what makes the calibrated residual and the
percentile threshold meaningful.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# pyrefly: ignore [missing-import]
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

import config as C

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════
# Device / reproducibility
# ═════════════════════════════════════════════
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_cache() -> None:
    """Free accelerator memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def set_seed(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(device: torch.device, seed: int = C.SEED) -> torch.Generator | None:
    """Device-aware seeded generator (None on backends that don't support it)."""
    try:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return g
    except (RuntimeError, TypeError):
        return None


# ═════════════════════════════════════════════
# VAE input normalisation  (M-1 fix)
# ═════════════════════════════════════════════
def normalize_for_vae(x: torch.Tensor, clip: float = C.VAE_CLIP) -> torch.Tensor:
    """
    Map z-score-normalised slices onto the VAE's expected ~[-1, 1] range.

    Clip to ±clip sigma, then linearly scale that range onto [-1, 1].
    Background (exactly 0) maps to 0.  Applied immediately before every
    ``vae.encode`` so the encoder never sees out-of-distribution inputs,
    and so that residuals are always computed in the same [-1, 1] space.
    """
    return torch.clamp(x, -clip, clip) / clip


# ═════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════
def load_unet(device: torch.device):
    """
    Load the trained UNet for inference, preferring EMA weights.
    Falls back to raw weights; raises if neither exists.
    """
    if (C.UNET_EMA_DIR / "config.json").exists():
        unet_dir = C.UNET_EMA_DIR
        log.info("Loading EMA UNet from '%s'", unet_dir)
    elif (C.UNET_DIR / "config.json").exists():
        unet_dir = C.UNET_DIR
        log.warning("EMA weights not found — loading raw UNet from '%s'", unet_dir)
    else:
        raise FileNotFoundError(
            f"No trained UNet in {C.UNET_EMA_DIR} or {C.UNET_DIR}. Train first.")
    unet = UNet2DModel.from_pretrained(str(unet_dir)).to(device)
    unet.eval()
    unet.requires_grad_(False)
    return unet


# ═════════════════════════════════════════════
# Schedulers
# ═════════════════════════════════════════════
def make_ddpm_scheduler() -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=C.NUM_TRAIN_TIMESTEPS,
        beta_schedule=C.BETA_SCHEDULE,
        beta_start=C.BETA_START,
        beta_end=C.BETA_END,
        prediction_type=C.PREDICTION_TYPE,
    )


def make_ddim_scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=C.NUM_TRAIN_TIMESTEPS,
        beta_schedule=C.BETA_SCHEDULE,
        beta_start=C.BETA_START,
        beta_end=C.BETA_END,
        prediction_type=C.PREDICTION_TYPE,
    )


def inference_timesteps(ddim: DDIMScheduler, t_int: int = C.T_INT,
                        ddim_steps: int = C.DDIM_STEPS) -> torch.Tensor:
    """
    DDIM timesteps from ``t_int`` → 0 for partial-noise reconstruction.

    ``ddim_steps`` is the number of *actual* denoising steps taken inside the
    reconstruction window [0, t_int] — NOT over the full [0, 1000] range.

    The naive ``set_timesteps(ddim_steps)`` spaces steps over all 1000 training
    timesteps and then keeps only those ≤ t_int, so the real step count
    collapses to ``ddim_steps · t_int / 1000`` (e.g. 25 steps @ t_int=150 → 4
    steps — almost no denoising).  Here we scale ``num_inference_steps`` up by
    ``1000 / t_int`` so ~``ddim_steps`` steps land in the window, while the
    per-step spacing (and therefore DDIM's prev_timestep math) stays valid.
    """
    full_steps = max(ddim_steps,
                     round(ddim_steps * C.NUM_TRAIN_TIMESTEPS / max(t_int, 1)))
    ddim.set_timesteps(full_steps)
    all_ts = ddim.timesteps
    return all_ts[all_ts <= t_int]


# ═════════════════════════════════════════════
# Encode / reconstruct
# ═════════════════════════════════════════════
@torch.no_grad()
def encode_to_latents(vae, images: torch.Tensor, sample: bool = True,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """
    (B, 3, 256, 256) images → scaled latents (B, 4, 32, 32).

    ``sample=True``  → stochastic latent (training).
    ``sample=False`` → deterministic posterior mean (calibration / inference),
    which removes a source of run-to-run noise in the baseline.
    """
    dist = vae.encode(normalize_for_vae(images)).latent_dist
    latents = dist.sample(generator=generator) if sample else dist.mean
    return latents * C.SCALING_FACTOR


@torch.no_grad()
def ddim_denoise(unet, ddim: DDIMScheduler, z_noisy: torch.Tensor,
                 timesteps: torch.Tensor) -> torch.Tensor:
    """Run the DDIM reverse loop over ``timesteps`` and return the final latent."""
    for t in timesteps:
        noise_pred = unet(z_noisy, t).sample
        z_noisy = ddim.step(noise_pred, t, z_noisy).prev_sample
    return z_noisy


@torch.no_grad()
def reconstruct_healthy(vae, unet, ddim, images, timesteps, t_int=C.T_INT,
                        generator=None):
    """
    Full inference pipeline on a batch:
      encode → noise@t_int → DDIM denoise → decode.

    Returns
    -------
    orig_norm : (B, 3, 256, 256)  normalised input (the comparison reference)
    recon     : (B, 3, 256, 256)  decoded healthy reconstruction
    z0        : (B, 4, 32, 32)     clean latent
    z_denoised: (B, 4, 32, 32)     denoised latent
    """
    z0 = encode_to_latents(vae, images, sample=False)            # deterministic
    noise = torch.randn(z0.shape, device=z0.device, generator=generator)
    t_tensor = torch.tensor([t_int], device=z0.device, dtype=torch.long)
    z_noisy = ddim.add_noise(z0, noise, t_tensor)
    z_denoised = ddim_denoise(unet, ddim, z_noisy, timesteps)
    recon = vae.decode(z_denoised / C.SCALING_FACTOR).sample
    orig_norm = normalize_for_vae(images)
    return orig_norm, recon, z0, z_denoised


# ═════════════════════════════════════════════
# SAAM — Self-Attention Attribution Maps
# ═════════════════════════════════════════════
class AttnMapStore:
    """
    Drop-in processor replacing AttnProcessor2_0.  Computes scaled-dot-product
    attention manually (instead of F.sdpa) so we can capture the probability
    matrix for spatial attribution.
    """

    def __init__(self):
        self.attn_probs: torch.Tensor | None = None    # (B, seq, seq)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):

        residual = hidden_states
        input_ndim = hidden_states.ndim

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)
        kv_in = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        key   = attn.to_k(kv_in)
        value = attn.to_v(kv_in)

        head_dim = query.shape[-1] // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        scale = head_dim ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = scores.softmax(dim=-1)                    # (B, heads, seq, seq)

        # Head-averaged map stored on CPU
        self.attn_probs = weights.mean(dim=1).detach().cpu()  # (B, seq, seq)

        out = torch.matmul(weights, value)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if getattr(attn, "residual_connection", False):
            out = out + residual
        out = out / getattr(attn, "rescale_output_factor", 1.0)

        return out


def install_attn_hooks(unet) -> dict[str, AttnMapStore]:
    """Replace every Attention processor with an AttnMapStore."""
    stores: dict[str, AttnMapStore] = {}
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            proc = AttnMapStore()
            module.set_processor(proc)
            stores[name] = proc
    log.info("SAAM: hooked %d self-attention layers", len(stores))
    return stores


def restore_default_processors(unet) -> None:
    """Revert all Attention modules to the default fast processor."""
    for _, module in unet.named_modules():
        if isinstance(module, Attention):
            module.set_processor(AttnProcessor2_0())


def aggregate_step_attention(stores: dict[str, AttnMapStore],
                             target_size: int = C.TARGET_SIZE,
                             min_seq_len: int = C.MIN_SEQ_LEN) -> np.ndarray:
    """
    Collapse per-layer attention from one DDIM step into a single
    (target_size, target_size) importance map.

    HALO FIX: only aggregate layers whose seq_len ≥ ``min_seq_len`` (skips the
    blurry low-resolution maps).  ``attn.sum(dim=0)`` = total attention each
    position *receives*; normalised to a probability for a clean,
    scale-stable importance signal.
    """
    heatmap = np.zeros((target_size, target_size), dtype=np.float32)
    count = 0

    for proc in stores.values():
        if proc.attn_probs is None:
            continue

        attn = proc.attn_probs[0]                          # (seq, seq)
        seq_len = attn.shape[0]

        if seq_len < min_seq_len:
            proc.attn_probs = None
            continue

        h = w = int(seq_len ** 0.5)
        if h * w != seq_len:
            proc.attn_probs = None
            continue

        importance = attn.sum(dim=0)                       # (seq,)
        importance = importance / (importance.sum() + 1e-8)
        importance = importance.view(1, 1, h, w).float()
        importance = F.interpolate(
            importance, size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        )
        heatmap += importance.squeeze().numpy()
        count += 1
        proc.attn_probs = None                             # free immediately

    return heatmap / max(count, 1)


# ═════════════════════════════════════════════
# EMA  (ARCH-02)  — version-independent manual EMA
# ═════════════════════════════════════════════
class EMA:
    """Exponential moving average of model parameters/buffers."""

    def __init__(self, model, decay: float = C.EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model) -> None:
        model.load_state_dict(self.shadow)

    def state_dict(self) -> dict:
        return self.shadow


# ═════════════════════════════════════════════
# Brain mask + anomaly scoring
# ═════════════════════════════════════════════
def brain_mask_2d(orig_norm_np: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    (3, 256, 256) normalised slice → (256, 256) {0,1} brain mask.
    Background is exactly 0 in the z-scored data, so any non-zero voxel
    across modalities is brain tissue.
    """
    return (np.abs(orig_norm_np).max(axis=0) > eps).astype(np.float32)


def latent_residual_2d(z0: torch.Tensor, z_denoised: torch.Tensor,
                       target_size: int = C.TARGET_SIZE) -> np.ndarray:
    """|z_test − z_healthy| averaged over channels, upsampled to (256, 256)."""
    m = torch.abs(z0 - z_denoised).mean(dim=1, keepdim=True)   # (B,1,32,32)
    m = F.interpolate(m, size=(target_size, target_size),
                      mode="bilinear", align_corners=False)
    return m[0, 0].cpu().numpy()


def pixel_residual_2d(orig_norm_np: np.ndarray, recon_np: np.ndarray,
                      m_baseline: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """
    Calibrated, brain-masked pixel residual:
        clip( mean_channels(|orig − recon| − M_baseline), 0 ) · brain_mask
    """
    diff = np.abs(orig_norm_np - recon_np) - m_baseline    # (3, 256, 256)
    diff = np.clip(diff, 0, None)
    return diff.mean(axis=0) * brain_mask                  # (256, 256)


def fuse_maps(m_pixel: np.ndarray, m_latent: np.ndarray,
              pixel_scale: float, latent_scale: float,
              alpha: float = C.LATENT_FUSION_ALPHA) -> np.ndarray:
    """
    Dual-space fusion standardised by healthy scales (keeps a single global
    threshold valid across images):  m_pixel/scale_p + α · m_latent/scale_l.
    """
    p = m_pixel / (pixel_scale + 1e-8)
    l = m_latent / (latent_scale + 1e-8)
    return p + alpha * l


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred * gt).sum()
    return float(2 * inter / (pred.sum() + gt.sum() + 1e-8))


# ═════════════════════════════════════════════
# Calibration  (BUG-01 + BUG-03)
# ═════════════════════════════════════════════
@torch.no_grad()
def calibrate_on_healthy(vae, unet, val_loader, device, *,
                         t_int: int = C.T_INT, ddim_steps: int = C.DDIM_STEPS,
                         max_samples: int = C.MAX_CAL_SAMPLES,
                         percentile: float = C.THRESHOLD_PERCENTILE,
                         alpha: float = C.LATENT_FUSION_ALPHA,
                         generator: torch.Generator | None = None):
    """
    Run the *full DDIM reconstruction pipeline* on held-out HEALTHY slices to
    produce a distribution-matched calibration:

      * M_baseline  — mean |orig − recon| over healthy slices (3, 256, 256)
      * threshold_pixel / threshold_fused — ``percentile``-th percentile of
        the per-slice max anomaly score (operational detection threshold)
      * pixel_scale / latent_scale — mean healthy map magnitude (fusion scales)

    Returns ``(M_baseline, calib_dict)``.
    """
    ddim = make_ddim_scheduler()
    timesteps = inference_timesteps(ddim, t_int, ddim_steps)

    unet.eval(); vae.eval()

    # ── Pass 1: accumulate M_baseline + cache raw maps ───────────────────
    residual_sum = None
    n_samples = 0
    cache = []   # (raw_diff(3,H,W), latent_2d(H,W), brain_mask(H,W))

    for images in val_loader:
        if n_samples >= max_samples:
            break
        images = images.to(device)
        orig_norm, recon, z0, z_den = reconstruct_healthy(
            vae, unet, ddim, images, timesteps, t_int, generator)

        orig_np = orig_norm.cpu().numpy()
        recon_np = recon.cpu().numpy()
        for b in range(orig_np.shape[0]):
            raw_diff = np.abs(orig_np[b] - recon_np[b])            # (3,H,W)
            residual_sum = raw_diff.copy() if residual_sum is None else residual_sum + raw_diff
            cache.append((
                raw_diff,
                latent_residual_2d(z0[b:b+1], z_den[b:b+1]),
                brain_mask_2d(orig_np[b]),
            ))
            n_samples += 1

    if n_samples == 0:
        raise RuntimeError("Calibration set is empty.")

    m_baseline = (residual_sum / n_samples).astype(np.float32)     # (3,H,W)

    # ── Pass 2 (in-memory): scales, then thresholds ─────────────────────
    pixel_maps, latent_maps, masks = [], [], []
    for raw_diff, lat_2d, bmask in cache:
        diff = np.clip(raw_diff - m_baseline, 0, None).mean(axis=0) * bmask
        pixel_maps.append(diff)
        latent_maps.append(lat_2d * bmask)
        masks.append(bmask > 0)

    pixel_scale  = float(np.mean([m[m > 0].mean() if np.any(m > 0) else 0.0 for m in pixel_maps]))
    latent_scale = float(np.mean([m[m > 0].mean() if np.any(m > 0) else 0.0 for m in latent_maps]))
    pixel_scale  = pixel_scale  or 1.0
    latent_scale = latent_scale or 1.0

    # Threshold = ``percentile``-th of the POOLED healthy BRAIN-VOXEL score
    # distribution, i.e. a per-voxel false-positive rate of (100-percentile)%.
    # (The previous threshold was the percentile of the per-slice MAX residual —
    # an extreme-value, slice-level detection threshold that sat ~16× too high
    # for pixel segmentation and drove DICE to 0.)  The slice-MAX percentile is
    # still saved separately for slice-level anomaly detection.
    pixel_voxels = np.concatenate([mp[m] for mp, m in zip(pixel_maps, masks)])
    fused_voxels = np.concatenate([
        fuse_maps(mp, ml, pixel_scale, latent_scale, alpha)[m]
        for mp, ml, m in zip(pixel_maps, latent_maps, masks)
    ])

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
        # Voxel-level thresholds (used for segmentation / DICE):
        "threshold_pixel": float(np.percentile(pixel_voxels, percentile)),
        "threshold_fused": float(np.percentile(fused_voxels, percentile)),
        # Slice-level detection thresholds (per-slice max), kept for reference:
        "threshold_pixel_slicemax": float(np.percentile(max_pixel, percentile)),
        "threshold_fused_slicemax": float(np.percentile(max_fused, percentile)),
        "pixel_scale": pixel_scale,
        "latent_scale": latent_scale,
        "alpha": float(alpha),
    }
    return m_baseline, calib


def save_calibration(m_baseline: np.ndarray, calib: dict,
                     baseline_path: Path = C.BASELINE_PATH,
                     calib_path: Path = C.CALIBRATION_PATH) -> None:
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(baseline_path, m_baseline)
    with open(calib_path, "w") as f:
        json.dump(calib, f, indent=2)


def load_calibration(baseline_path: Path = C.BASELINE_PATH,
                     calib_path: Path = C.CALIBRATION_PATH):
    """Returns ``(M_baseline, calib_dict)``; calib is ``None`` if absent."""
    m_baseline = np.load(baseline_path).astype(np.float32)
    calib = None
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)
    return m_baseline, calib


# ═════════════════════════════════════════════
# Multi-timestep residual aggregation
# ═════════════════════════════════════════════
#
# Rationale: lesions of different size/contrast become "reconstructable" (and so
# stop producing a residual) at different noise levels T_int.  Scoring at a
# single T_int therefore misses part of the lesion spectrum.  Here the anomaly
# score is computed at several T_int levels and aggregated.  Every T level has
# its OWN healthy baseline + healthy scale so the per-T maps are standardised
# before aggregation and the percentile threshold stays meaningful — this is
# the same calibration contract as the single-T path, just per level.
#
# Cost: N× the reconstruction passes (one per T) at calibration AND inference.
# The UNet is NOT retrained.

def aggregate_t_scores(score_stack: np.ndarray, mode: str = C.MULTI_T_AGG) -> np.ndarray:
    """Aggregate a ``(T, H, W)`` stack of standardised score maps → ``(H, W)``.

    ``mode="mean"`` averages (smoother, fewer FPs); ``mode="max"`` takes the
    voxel-wise maximum (most sensitive — anomalous at ANY level wins).
    """
    if mode == "max":
        return score_stack.max(axis=0)
    return score_stack.mean(axis=0)


@torch.no_grad()
def reconstruct_and_score_t(vae, unet, ddim, image, t_int, ddim_steps,
                            m_baseline, pixel_scale, latent_scale, alpha,
                            use_fusion, generator=None):
    """Single-T, healthy-scale-standardised anomaly score for one image.

    Parameters mirror one entry of the multi-T calibration.  Returns
    ``(score_2d (H,W), brain_mask_2d (H,W), recon_np (3,H,W))`` — the recon is
    returned so callers can reuse it for visualisation without recomputing.
    """
    timesteps = inference_timesteps(ddim, t_int, ddim_steps)
    orig_norm, recon, z0, z_den = reconstruct_healthy(
        vae, unet, ddim, image, timesteps, t_int, generator)

    orig_np  = orig_norm[0].cpu().numpy()
    recon_np = recon[0].cpu().numpy()
    bmask    = brain_mask_2d(orig_np)
    m_pixel  = pixel_residual_2d(orig_np, recon_np, m_baseline, bmask)

    if use_fusion:
        m_latent = latent_residual_2d(z0, z_den) * bmask
        score = fuse_maps(m_pixel, m_latent, pixel_scale, latent_scale, alpha)
    else:
        # Standardise by the healthy pixel scale so every T contributes on a
        # comparable magnitude before aggregation.
        score = m_pixel / (pixel_scale + 1e-8)
    return score, bmask, recon_np


@torch.no_grad()
def score_image_multi_t(vae, unet, ddim, image, baselines: dict, calib: dict,
                        generator=None):
    """Aggregated multi-T anomaly score for one ``(1, 3, H, W)`` image.

    ``baselines`` maps ``t_int -> M_baseline (3,H,W)``; ``calib`` is the dict
    produced by :func:`calibrate_on_healthy_multi_t`.  Returns
    ``(agg_score (H,W), brain_mask (H,W), recon_repr (3,H,W))`` where the
    representative recon is taken at the middle T (for visualisation only).
    """
    t_list      = [int(t) for t in calib["t_list"]]
    use_fusion  = bool(calib["use_fusion"])
    alpha       = float(calib["alpha"])
    ddim_steps  = int(calib["ddim_steps"])
    agg_mode    = calib.get("agg", C.MULTI_T_AGG)

    stack, bmask, recon_repr = [], None, None
    repr_idx = len(t_list) // 2
    for i, t in enumerate(t_list):
        scales = calib["per_t"][str(t)]
        score_t, bmask, recon_np = reconstruct_and_score_t(
            vae, unet, ddim, image, t, ddim_steps, baselines[t],
            scales["pixel_scale"], scales["latent_scale"], alpha,
            use_fusion, generator)
        stack.append(score_t)
        if i == repr_idx:
            recon_repr = recon_np

    agg_score = aggregate_t_scores(np.stack(stack, axis=0), agg_mode)
    return agg_score, bmask, recon_repr


@torch.no_grad()
def calibrate_on_healthy_multi_t(vae, unet, val_loader, device, *,
                                 t_list=None, ddim_steps: int = C.DDIM_STEPS,
                                 max_samples: int = C.MAX_CAL_SAMPLES,
                                 percentile: float = C.THRESHOLD_PERCENTILE,
                                 alpha: float = C.LATENT_FUSION_ALPHA,
                                 agg_mode: str = C.MULTI_T_AGG,
                                 use_fusion: bool = C.USE_LATENT_FUSION,
                                 generator: torch.Generator | None = None):
    """Multi-T healthy calibration.

    For every T in ``t_list`` computes its own ``M_baseline`` and healthy
    pixel/latent scales, then derives ONE operating threshold from the pooled
    distribution of the *aggregated* healthy score (so the threshold matches the
    aggregated map used at inference).

    Returns ``(baselines, calib)`` where ``baselines`` maps ``t_int -> (3,H,W)``.
    """
    t_list = [int(t) for t in (t_list if t_list is not None else C.MULTI_T_LIST)]
    ddim = make_ddim_scheduler()
    unet.eval(); vae.eval()

    # ── Pass 1: per-T M_baseline + cache (raw_diff, latent, brain_mask) ──
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
                vae, unet, ddim, images, timesteps, t, generator)
            orig_np  = orig_norm.cpu().numpy()
            recon_np = recon.cpu().numpy()
            for b in range(bs):
                raw_diff = np.abs(orig_np[b] - recon_np[b])          # (3,H,W)
                resid_sum[t] = raw_diff.copy() if resid_sum[t] is None \
                    else resid_sum[t] + raw_diff
                cache[t].append((
                    raw_diff,
                    latent_residual_2d(z0[b:b+1], z_den[b:b+1]),
                    brain_mask_2d(orig_np[b]),
                ))
        n_samples += bs

    if n_samples == 0:
        raise RuntimeError("Calibration set is empty.")
    n_cached = len(cache[t_list[0]])
    baselines = {t: (resid_sum[t] / n_cached).astype(np.float32) for t in t_list}

    # ── Pass 2: per-T standardised maps + scales ────────────────────────
    masks = [bm > 0 for (_, _, bm) in cache[t_list[0]]]
    pixel_maps: dict[int, list] = {t: [] for t in t_list}
    latent_maps: dict[int, list] = {t: [] for t in t_list}
    scales: dict[int, tuple] = {}
    for t in t_list:
        for raw_diff, lat_2d, bmask in cache[t]:
            pm = np.clip(raw_diff - baselines[t], 0, None).mean(axis=0) * bmask
            pixel_maps[t].append(pm)
            latent_maps[t].append(lat_2d * bmask)
        ps = float(np.mean([m[m > 0].mean() if np.any(m > 0) else 0.0
                            for m in pixel_maps[t]])) or 1.0
        ls = float(np.mean([m[m > 0].mean() if np.any(m > 0) else 0.0
                            for m in latent_maps[t]])) or 1.0
        scales[t] = (ps, ls)

    # ── Pass 3: aggregated healthy score → percentile threshold ─────────
    agg_voxels = []
    for i in range(n_cached):
        stack = []
        for t in t_list:
            ps, ls = scales[t]
            if use_fusion:
                s = fuse_maps(pixel_maps[t][i], latent_maps[t][i], ps, ls, alpha)
            else:
                s = pixel_maps[t][i] / (ps + 1e-8)
            stack.append(s)
        agg = aggregate_t_scores(np.stack(stack, axis=0), agg_mode)
        agg_voxels.append(agg[masks[i]])
    agg_voxels = np.concatenate(agg_voxels)

    calib = {
        "multi_t": True,
        "t_list": t_list,
        "ddim_steps": int(ddim_steps),
        "agg": agg_mode,
        "percentile": float(percentile),
        "use_fusion": bool(use_fusion),
        "alpha": float(alpha),
        "n_samples": int(n_cached),
        # Operating threshold for the AGGREGATED multi-T score:
        "threshold": float(np.percentile(agg_voxels, percentile)),
        "per_t": {str(t): {"pixel_scale": scales[t][0],
                           "latent_scale": scales[t][1]} for t in t_list},
    }
    return baselines, calib


def save_calibration_multi_t(baselines: dict, calib: dict,
                             baseline_path: Path = C.MULTI_BASELINE_PATH,
                             calib_path: Path = C.MULTI_CALIBRATION_PATH) -> None:
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    # Keys must be strings for np.savez; prefix with 't' to keep them valid.
    np.savez(baseline_path, **{f"t{t}": b for t, b in baselines.items()})
    with open(calib_path, "w") as f:
        json.dump(calib, f, indent=2)


def load_calibration_multi_t(baseline_path: Path = C.MULTI_BASELINE_PATH,
                             calib_path: Path = C.MULTI_CALIBRATION_PATH):
    """Returns ``(baselines {t:int -> (3,H,W)}, calib_dict)`` or ``(None, None)``
    if the multi-T calibration artefacts are absent."""
    if not (Path(baseline_path).exists() and Path(calib_path).exists()):
        return None, None
    data = np.load(baseline_path)
    baselines = {int(k[1:]): data[k].astype(np.float32) for k in data.files}
    with open(calib_path) as f:
        calib = json.load(f)
    return baselines, calib
