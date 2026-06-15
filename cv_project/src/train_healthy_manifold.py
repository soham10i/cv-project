"""
Phase 3 — Train the Healthy Manifold (Latent DDPM)
====================================================
Trains a UNet2DModel to denoise latent representations produced by a frozen
StabilityAI VAE.

Key features
------------
* VAE inputs mapped to [-1, 1] (utils.normalize_for_vae) before encoding.
* Gradient clipping + cosine LR annealing.
* EMA (exponential moving average) weights — used for inference.
* Calibration runs the full DDIM reconstruction pipeline on the *explicit*
  val-healthy dataset (VAL_HEALTHY_DIR), not a random slice from train data,
  producing M_baseline and a percentile threshold (calibration.json).
* Healthy reconstruction monitoring: MAE, PSNR, SSIM logged every cal_every
  epochs to results/train_logs/metrics_epoch_NNN.json.
* XAI trajectory: SAAM attention panels saved each calibration step for a
  fixed set of val-healthy slices → results/trajectory/epoch_NNN_*.png.

Usage
-----
    python src/train_healthy_manifold.py                       # defaults
    python src/train_healthy_manifold.py --epochs 20 --bs 8   # custom
"""

import argparse
import json
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# pyrefly: ignore [missing-import]
from diffusers import AutoencoderKL, UNet2DModel

import config as C
import utils
import ckptkit
from logkit import RunLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Number of fixed val-healthy slices to use for XAI and monitoring
N_MONITOR_SLICES = 8


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class HealthySliceDataset(Dataset):
    """Loads preprocessed (3, 256, 256) .npy slices as float32 tensors."""

    def __init__(self, data_dir):
        self.files = sorted(data_dir.glob("*.npy"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files in {data_dir}")
        log.info("Dataset: %d slices from %s", len(self.files), data_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = np.load(self.files[idx])
        return torch.from_numpy(arr).float()


# ─────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────
def build_models(device):
    vae_src = C.resolve_vae_source()
    tag = " (fine-tuned medical codec)" if vae_src != C.VAE_CKPT else " (pretrained)"
    log.info("Loading VAE from '%s'%s …", vae_src, tag)
    vae = AutoencoderKL.from_pretrained(vae_src)
    vae.eval()
    vae.requires_grad_(False)
    vae = vae.to(device)
    log.info("VAE frozen  (%s params)", f"{sum(p.numel() for p in vae.parameters()):,}")

    log.info("Initialising UNet2DModel from scratch …")
    unet = UNet2DModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256, 384, 512),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
    )
    unet = unet.to(device)
    log.info("UNet ready  (%s trainable params)",
             f"{sum(p.numel() for p in unet.parameters()):,}")
    return vae, unet


# ─────────────────────────────────────────────
# Reconstruction quality metrics
# ─────────────────────────────────────────────
def _ssim_2d(a: np.ndarray, b: np.ndarray, k1: float = 0.01,
             k2: float = 0.03, data_range: float = 2.0) -> float:
    """Simplified single-scale SSIM for (H, W) float arrays."""
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a = ((a - mu_a) ** 2).mean() ** 0.5
    sigma_b = ((b - mu_b) ** 2).mean() ** 0.5
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a ** 2 + sigma_b ** 2 + c2)
    return float(num / (den + 1e-8))


def compute_recon_metrics(orig_np: np.ndarray, recon_np: np.ndarray,
                           brain_mask: np.ndarray) -> dict:
    """
    Compute brain-masked MAE, PSNR, and SSIM between normalised orig and recon.

    Both arrays are (3, H, W); brain_mask is (H, W).
    Returns a dict with float values.
    """
    # Brain-masked mean absolute error (average over channels)
    diff = np.abs(orig_np - recon_np)           # (3, H, W)
    masked_diff = diff * brain_mask[None]
    brain_voxels = brain_mask.sum() * orig_np.shape[0]
    mae = float(masked_diff.sum() / max(brain_voxels, 1))

    # PSNR (data range = 2 since inputs are in [-1, 1])
    mse = float((masked_diff ** 2).sum() / max(brain_voxels, 1))
    psnr = float(10 * np.log10(4.0 / (mse + 1e-8)))  # data_range² = 4

    # SSIM on T2w channel (index 1), brain voxels only
    t2_orig  = orig_np[1]  * brain_mask
    t2_recon = recon_np[1] * brain_mask
    ssim = _ssim_2d(t2_orig, t2_recon)

    return {"mae": mae, "psnr": psnr, "ssim": ssim}


# ─────────────────────────────────────────────
# XAI panel logging
# ─────────────────────────────────────────────
@torch.no_grad()
def log_xai_panels(vae, unet, images: torch.Tensor, epoch: int,
                   out_dir, device, generator=None) -> None:
    """
    Generate SAAM + residual panels for a fixed set of val-healthy slices and
    save them to out_dir/epoch_{epoch:03d}_img{i}.png.

    This creates a temporal trajectory of how the model's attention and
    reconstruction quality evolve as training progresses.
    """
    out_dir = out_dir if isinstance(out_dir, type(C.TRAJ_DIR)) else C.TRAJ_DIR.__class__(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ddim = utils.make_ddim_scheduler()
    timesteps = utils.inference_timesteps(ddim, C.T_INT, C.DDIM_STEPS)

    unet.eval()
    attn_stores = utils.install_attn_hooks(unet)

    for i, img in enumerate(images):
        img_batch = img.unsqueeze(0).to(device)

        z0 = utils.encode_to_latents(vae, img_batch, sample=False, generator=generator)
        noise = torch.randn(z0.shape, device=device, generator=generator)
        t_tensor = torch.tensor([C.T_INT], device=device, dtype=torch.long)
        z_noisy = ddim.add_noise(z0, noise, t_tensor)

        attn_accum = np.zeros((C.TARGET_SIZE, C.TARGET_SIZE), dtype=np.float32)
        n_steps = 0
        for t in timesteps:
            noise_pred = unet(z_noisy, t).sample
            z_noisy = ddim.step(noise_pred, t, z_noisy).prev_sample
            attn_accum += utils.aggregate_step_attention(attn_stores)
            n_steps += 1
        a_total = attn_accum / max(n_steps, 1)

        recon = vae.decode(z_noisy / C.SCALING_FACTOR).sample
        orig_np  = utils.normalize_for_vae(img_batch)[0].cpu().numpy()
        recon_np = recon[0].cpu().numpy()
        bmask    = utils.brain_mask_2d(orig_np)

        # Residual using a zero baseline (no M_baseline yet during early training)
        zero_baseline = np.zeros_like(orig_np)
        m_pixel = utils.pixel_residual_2d(orig_np, recon_np, zero_baseline, bmask)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(orig_np[1], cmap="gray")
        axes[0].set_title(f"Original T2w (ep {epoch})", fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(recon_np[1], cmap="gray")
        axes[1].set_title("Healthy Recon (T2w)", fontweight="bold")
        axes[1].axis("off")

        im2 = axes[2].imshow(m_pixel, cmap="hot")
        axes[2].set_title("Pixel Residual", fontweight="bold")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        im3 = axes[3].imshow(a_total, cmap="inferno")
        axes[3].set_title("SAAM Attention", fontweight="bold")
        axes[3].axis("off")
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig.savefig(out_dir / f"epoch_{epoch:03d}_img{i}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    utils.restore_default_processors(unet)
    log.info("XAI panels saved → %s (epoch %d, %d slices)", out_dir, epoch, len(images))


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train_one_epoch(unet, vae, scheduler, loader, optimizer, ema, device,
                    epoch, total_epochs, grad_accum=1):
    unet.train()
    running_loss = 0.0
    n_batches    = 0
    n_total      = len(loader)
    optimizer.zero_grad()

    for batch_idx, images in enumerate(loader):
        images = images.to(device)

        latents = utils.encode_to_latents(vae, images, sample=True)

        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (bsz,), device=device, dtype=torch.long,
        )

        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        noise_pred = unet(noisy_latents, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)

        # Gradient accumulation → effective batch = bs × grad_accum (T4-friendly).
        (loss / grad_accum).backward()
        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == n_total:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), C.GRAD_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad()
            ema.update(unet)

        running_loss += loss.item()
        n_batches += 1
        avg_so_far = running_loss / n_batches

        log.info("  Epoch %d/%d  |  batch %d/%d  |  loss: %.6f  |  avg: %.6f",
                 epoch, total_epochs, batch_idx + 1, n_total,
                 loss.item(), avg_so_far)

    return running_loss / max(n_batches, 1)


# ─────────────────────────────────────────────
# Validation denoising loss (fixed noise → comparable across epochs)
# ─────────────────────────────────────────────
@torch.no_grad()
def validate_denoising(unet, vae, scheduler, val_loader, device,
                       max_batches: int = 25) -> float:
    """Mean ε-prediction MSE on the val set with a *fixed* noise/timestep stream
    (re-seeded each call), so the curve is comparable epoch-to-epoch and exposes
    the train/val generalization gap the training loss alone can't show."""
    unet.eval()
    g = torch.Generator(device=device).manual_seed(C.SEED)
    total, n = 0.0, 0
    for bi, images in enumerate(val_loader):
        if max_batches is not None and bi >= max_batches:
            break
        images = images.to(device)
        latents = utils.encode_to_latents(vae, images, sample=False)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                  (bsz,), device=device, generator=g, dtype=torch.long)
        noise = torch.randn(latents.shape, device=device, generator=g)
        noisy = scheduler.add_noise(latents, noise, timesteps)
        pred = unet(noisy, timesteps).sample
        total += F.mse_loss(pred, noise).item()
        n += 1
    return total / max(n, 1)


# ─────────────────────────────────────────────
# Calibration on EMA weights
# ─────────────────────────────────────────────
@torch.no_grad()
def calibrate_with_ema(vae, unet, ema, val_loader, device, generator):
    raw_state = {k: v.detach().cpu().clone() for k, v in unet.state_dict().items()}
    ema.copy_to(unet)
    unet.eval()

    m_baseline, calib = utils.calibrate_on_healthy(
        vae, unet, val_loader, device, generator=generator)
    utils.save_calibration(m_baseline, calib)

    log.info("Calibration → M_baseline mean %.6f | thr_pixel %.4f | thr_fused %.4f "
             "| n=%d", m_baseline.mean(), calib["threshold_pixel"],
             calib["threshold_fused"], calib["n_samples"])

    unet.load_state_dict(raw_state)
    return calib, m_baseline


# ─────────────────────────────────────────────
# Reconstruction monitoring
# ─────────────────────────────────────────────
@torch.no_grad()
def monitor_reconstruction(vae, unet, ema, monitor_images: torch.Tensor,
                            device, generator, epoch: int) -> dict:
    """
    Swap in EMA weights, reconstruct a fixed set of val-healthy slices,
    compute brain-masked MAE/PSNR/SSIM, log to train_logs/, restore weights.
    """
    raw_state = {k: v.detach().cpu().clone() for k, v in unet.state_dict().items()}
    ema.copy_to(unet)
    unet.eval()

    ddim = utils.make_ddim_scheduler()
    timesteps = utils.inference_timesteps(ddim, C.T_INT, C.DDIM_STEPS)

    all_metrics = []
    for img in monitor_images:
        img_batch = img.unsqueeze(0).to(device)
        orig_norm, recon, _, _ = utils.reconstruct_healthy(
            vae, unet, ddim, img_batch, timesteps, C.T_INT, generator)
        orig_np  = orig_norm[0].cpu().numpy()
        recon_np = recon[0].cpu().numpy()
        bmask    = utils.brain_mask_2d(orig_np)
        all_metrics.append(compute_recon_metrics(orig_np, recon_np, bmask))

    unet.load_state_dict(raw_state)

    avg = {
        "epoch": epoch,
        "mae":   float(np.mean([m["mae"]  for m in all_metrics])),
        "psnr":  float(np.mean([m["psnr"] for m in all_metrics])),
        "ssim":  float(np.mean([m["ssim"] for m in all_metrics])),
        "n_slices": len(all_metrics),
    }
    log.info("Recon metrics  ep %d | MAE %.4f | PSNR %.2f dB | SSIM %.4f",
             epoch, avg["mae"], avg["psnr"], avg["ssim"])

    C.TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = C.TRAIN_LOG_DIR / f"metrics_epoch_{epoch:03d}.json"
    with open(out_path, "w") as f:
        json.dump(avg, f, indent=2)

    return avg


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    utils.set_seed()
    device = utils.get_device()

    # Unified per-run logging (logs/<stage>_<ts>/): run.log + metrics.csv/jsonl + TB.
    rl = RunLogger("diffusion", config_snapshot=C.snapshot(),
                   use_tensorboard=not args.no_tensorboard)
    fh = logging.FileHandler(rl.run_dir / "run.log")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(fh)                       # mirror module logs into the run folder

    log.info("Device: %s", device)
    generator = utils.make_generator(device)

    # ── Data ─────────────────────────────────────────────────────
    train_ds = HealthySliceDataset(C.HEALTHY_DIR)

    if C.VAL_HEALTHY_DIR.exists() and any(C.VAL_HEALTHY_DIR.glob("*.npy")):
        log.info("Using explicit val-healthy dataset from %s", C.VAL_HEALTHY_DIR)
        val_ds = HealthySliceDataset(C.VAL_HEALTHY_DIR)
    else:
        log.warning(
            "VAL_HEALTHY_DIR (%s) not found or empty — falling back to 10%% random "
            "split of train data. Run make_splits.py + preprocess_to_2d.py "
            "with --healthy-subdir val_healthy for proper patient-level splits.",
            C.VAL_HEALTHY_DIR,
        )
        from torch.utils.data import random_split as _random_split
        n_val   = max(1, int(len(train_ds) * 0.10))
        n_train = len(train_ds) - n_val
        train_ds, val_ds = _random_split(
            train_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(C.SEED),
        )

    log.info("Split: %d train  /  %d val", len(train_ds), len(val_ds))

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    # Fixed slice subset for monitoring and XAI (drawn once, kept constant)
    val_files = list(val_ds.files if hasattr(val_ds, "files") else
                     [val_ds.dataset.files[i] for i in val_ds.indices])
    n_monitor = min(N_MONITOR_SLICES, len(val_files))
    monitor_images = torch.stack([
        torch.from_numpy(np.load(val_files[i])).float()
        for i in range(n_monitor)
    ])
    log.info("Monitor/XAI set: %d val-healthy slices (fixed)", n_monitor)

    # ── Models / optim / scheduler / EMA ─────────────────────────
    vae, unet = build_models(device)
    scheduler  = utils.make_ddpm_scheduler()
    optimizer  = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    lr_sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=C.LR_ETA_MIN)
    ema = utils.EMA(unet, decay=C.EMA_DECAY)

    # ── Resume (Colab/preemption safety) ─────────────────────────────
    # best_loss now tracks the best *validation* denoising loss (lower = better
    # generalization). epochs_no_improve drives early stopping.
    start_epoch, best_loss = 1, float("inf")
    epochs_no_improve = 0
    if args.resume:
        rp = ckptkit.find_resume(C.UNET_CKPT_DIR)
        if rp is not None:
            ck = ckptkit.load_checkpoint(rp, model=unet, optimizer=optimizer,
                                         scheduler=lr_sched, ema=ema, device=device)
            start_epoch = int(ck["epoch"]) + 1
            best_loss = ck.get("best_metric") or float("inf")
            log.info("Resumed from %s → epoch %d (best loss %.6f)",
                     rp, start_epoch, best_loss)
        else:
            log.info("--resume set but no %s/last.pt — starting fresh.", C.UNET_CKPT_DIR)

    log.info("=" * 60)
    log.info("Training: %d epochs | bs %d×accum %d | lr %.1e | EMA %.4f | clip %.1f",
             args.epochs, args.bs, args.grad_accum, args.lr, C.EMA_DECAY,
             C.GRAD_CLIP_NORM)
    log.info("=" * 60)

    # ── Before-training XAI snapshot (random-init UNet) → epoch_000 panels ──
    if start_epoch == 1 and not args.no_xai:
        log.info("Before-training XAI snapshot (random-init UNet) …")
        unet.eval()
        log_xai_panels(vae, unet, monitor_images, 0, C.TRAJ_DIR, device, generator)
        utils.clear_cache()

    for epoch in range(start_epoch, args.epochs + 1):
        log.info("─" * 60)
        log.info("EPOCH %d / %d  (lr=%.2e)", epoch, args.epochs,
                 optimizer.param_groups[0]["lr"])
        t0 = time.time()
        avg_loss = train_one_epoch(
            unet, vae, scheduler, train_loader, optimizer, ema,
            device, epoch, args.epochs, grad_accum=args.grad_accum)
        lr_sched.step()
        elapsed = time.time() - t0
        log.info("EPOCH %d/%d  DONE  |  avg_loss: %.6f  |  lr: %.2e  |  time: %.1fs",
                 epoch, args.epochs, avg_loss, lr_sched.get_last_lr()[0], elapsed)

        # ── Validation denoising loss + unified metric logging ──────────
        val_loss = validate_denoising(unet, vae, scheduler, val_loader, device,
                                      max_batches=args.val_batches)
        log.info("  val denoising loss: %.6f", val_loss)
        rl.log_metrics(epoch, "train", {"loss": avg_loss,
                                        "lr": lr_sched.get_last_lr()[0]})
        rl.log_metrics(epoch, "val", {"denoise_loss": val_loss})

        # ── Best-model selection on VALIDATION loss (generalization, not memorization).
        # Saving on train loss kept overwriting with increasingly overfit weights; the
        # val denoising loss is the honest signal of how well the healthy manifold
        # generalizes.  Both the raw UNet and the EMA weights are snapshotted at each
        # new val-best so evaluation always loads the best-generalizing model.
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            C.UNET_DIR.mkdir(parents=True, exist_ok=True)
            unet.save_pretrained(C.UNET_DIR)
            # also snapshot EMA at the val-best
            raw_state = {k: v.detach().cpu().clone() for k, v in unet.state_dict().items()}
            ema.copy_to(unet)
            C.UNET_EMA_DIR.mkdir(parents=True, exist_ok=True)
            unet.save_pretrained(C.UNET_EMA_DIR)
            unet.load_state_dict(raw_state)
            log.info("  ↳ new best VAL loss %.6f — raw+EMA UNet saved (best-generalizing)",
                     best_loss)
        else:
            epochs_no_improve += 1
            log.info("  (no val improvement for %d epoch(s); best %.6f)",
                     epochs_no_improve, best_loss)

        # ── Early stopping — halt once val loss stops improving (overfitting onset).
        if args.patience > 0 and epochs_no_improve >= args.patience:
            log.info("=" * 60)
            log.info("EARLY STOP at epoch %d — no val improvement for %d epochs "
                     "(best val loss %.6f). Best-generalizing model already saved.",
                     epoch, args.patience, best_loss)
            log.info("=" * 60)
            break

        # ── Periodic calibration + monitoring + XAI ──────────────
        if epoch % args.cal_every == 0 or epoch == args.epochs:
            log.info("Running DDIM calibration on up to %d val samples …",
                     min(len(val_ds), C.MAX_CAL_SAMPLES))
            calib, _ = calibrate_with_ema(vae, unet, ema, val_loader, device, generator)
            utils.clear_cache()

            recon = monitor_reconstruction(vae, unet, ema, monitor_images,
                                           device, generator, epoch)
            rl.log_metrics(epoch, "val", {"recon_mae": recon["mae"],
                                          "recon_psnr": recon["psnr"],
                                          "recon_ssim": recon["ssim"]})
            utils.clear_cache()

            # XAI panels — swap in EMA weights then restore
            raw_state = {k: v.detach().cpu().clone() for k, v in unet.state_dict().items()}
            ema.copy_to(unet)
            unet.eval()
            log_xai_panels(vae, unet, monitor_images, epoch, C.TRAJ_DIR, device, generator)
            unet.load_state_dict(raw_state)
            utils.clear_cache()

        # ── Resumable checkpoint every epoch (model + optim + sched + EMA) ──
        ckptkit.save_checkpoint(C.UNET_CKPT_DIR / "last.pt", model=unet,
                                optimizer=optimizer, scheduler=lr_sched, ema=ema,
                                epoch=epoch, best_metric=best_loss)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckptkit.save_checkpoint(C.UNET_CKPT_DIR / f"ckpt_ep{epoch:03d}.pt",
                                    model=unet, optimizer=optimizer, scheduler=lr_sched,
                                    ema=ema, epoch=epoch, best_metric=best_loss)
            ckptkit.prune_old(C.UNET_CKPT_DIR, keep_last=args.keep_last)
            # Persist the *current* EMA periodically to a sibling dir (disconnect
            # safety) — never to UNET_EMA_DIR, which is reserved for the best-val
            # EMA so evaluation always loads the best-generalizing weights.
            raw_state = {k: v.detach().cpu().clone() for k, v in unet.state_dict().items()}
            ema.copy_to(unet)
            periodic_ema = C.UNET_EMA_DIR.parent / "unet_ema_periodic"
            periodic_ema.mkdir(parents=True, exist_ok=True)
            unet.save_pretrained(periodic_ema)
            unet.load_state_dict(raw_state)
            log.info("  ↳ checkpoint + periodic EMA snapshot @ epoch %d → %s",
                     epoch, C.UNET_CKPT_DIR)

    # ── Final EMA weights → separate dir (do NOT clobber the best-val EMA) ──
    # UNET_EMA_DIR holds the best-generalizing EMA (saved at each val-best above);
    # the final/overfit EMA goes to a sibling dir for optional comparison only.
    ema.copy_to(unet)
    final_ema_dir = C.UNET_EMA_DIR.parent / "unet_ema_final"
    final_ema_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(final_ema_dir)
    log.info("Final EMA weights saved → %s (best-val EMA kept at %s)",
             final_ema_dir, C.UNET_EMA_DIR)

    log.info("=" * 60)
    log.info("DONE  —  best VALIDATION loss: %.6f", best_loss)
    log.info("  Raw UNet : %s", C.UNET_DIR)
    log.info("  EMA UNet : %s   (preferred for evaluation)", C.UNET_EMA_DIR)
    log.info("  Baseline : %s", C.BASELINE_PATH)
    log.info("  Calib    : %s", C.CALIBRATION_PATH)
    log.info("  XAI traj : %s", C.TRAJ_DIR)
    log.info("  Metrics  : %s", C.TRAIN_LOG_DIR)
    log.info("  Run logs : %s", rl.run_dir)
    log.info("=" * 60)
    rl.close()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3 — Latent DDPM training on healthy brain slices",
    )
    parser.add_argument("--epochs",    type=int,   default=30,   help="Training epochs (default: 30)")
    parser.add_argument("--bs",        type=int,   default=8,    help="Batch size (default: 8)")
    parser.add_argument("--lr",        type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--cal-every", type=int,   default=5,    help="Calibration + XAI interval in epochs (default: 5)")
    parser.add_argument("--grad-accum",  type=int, default=1,
                        help="Gradient accumulation steps → effective batch = bs × this.")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers (T4 Colab ≈ 2 vCPUs; default 2).")
    parser.add_argument("--save-every",  type=int, default=C.CKPT_EVERY,
                        help="Epoch-tagged checkpoint + EMA snapshot every N epochs.")
    parser.add_argument("--keep-last",   type=int, default=C.CKPT_KEEP_LAST,
                        help="Retain only the most recent N epoch-tagged checkpoints.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from UNET_CKPT_DIR/last.pt if present (Colab-safe).")
    parser.add_argument("--val-batches", type=int, default=25,
                        help="Batches used for the validation denoising loss (default: 25).")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early-stop after N epochs with no val-loss improvement "
                             "(0 = disabled). Best-generalizing model is always saved.")
    parser.add_argument("--no-xai", action="store_true",
                        help="Skip SAAM/residual XAI panels (incl. the epoch-0 snapshot).")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    main(args)
