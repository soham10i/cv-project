"""
Phase 3 — Train the Healthy Manifold (Latent DDPM)
====================================================
Trains a UNet2DModel to denoise latent representations produced by a frozen
StabilityAI VAE.

Key features
------------
* VAE inputs mapped to [-1, 1] (pipeline.diffusion.normalize_for_vae) before encoding.
* Gradient clipping + cosine LR annealing.
* EMA (exponential moving average) weights — used for inference.
* Calibration runs the full DDIM reconstruction pipeline on the *explicit*
  val-healthy dataset (VAL_HEALTHY_DIR), not a random slice from train data.
* Healthy reconstruction monitoring logged every cal_every epochs.
* XAI trajectory: SAAM attention panels saved each calibration step.

Usage
-----
    python src/train_diffusion.py                       # defaults
    python src/train_diffusion.py --epochs 20 --bs 8   # custom
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import os
# Suppress Flax / Diffusers deprecation noise before any diffusers import
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from core import constants as C
from core import ckptkit
from core.logkit import RunLogger

from data.datasets import HealthySliceDataset
from models.factory import build_vae, build_unet
from models.ema import EMA
from models.attention import install_attn_hooks, aggregate_step_attention, restore_default_processors
from pipeline.diffusion import make_ddpm_scheduler, make_ddim_scheduler, inference_timesteps, encode_to_latents, reconstruct_healthy, normalize_for_vae
from pipeline.scoring import pixel_residual_2d, brain_mask_2d
from pipeline.calibration import calibrate_on_healthy, save_calibration
from pipeline.metrics import compute_recon_metrics


# ── Dual-sink logging: terminal + persistent .log file ───────────────────────
def _setup_logging() -> logging.Logger:
    """Wire root logger to both stdout and a timestamped .log file."""
    log_dir = C.LOG_DIR / "diffusion"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove any existing handlers (avoids duplicate lines on re-import)
    root.handlers.clear()

    # 1. Terminal (stdout)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # 2. File sink
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Silence noisy third-party loggers
    for noisy in ("diffusers", "transformers", "accelerate", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Log file: %s", log_file)
    return logger

log = _setup_logging()

# Number of fixed val-healthy slices to use for XAI and monitoring
N_MONITOR_SLICES = 8


# ─────────────────────────────────────────────
# XAI panel logging
# ─────────────────────────────────────────────
@torch.no_grad()
def log_xai_panels(vae, unet, images: torch.Tensor, epoch: int,
                   out_dir, device, generator=None) -> None:
    """
    Generate SAAM + residual panels for a fixed set of val-healthy slices.
    """
    out_dir = out_dir if isinstance(out_dir, type(C.TRAJ_DIR)) else C.TRAJ_DIR.__class__(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ddim = make_ddim_scheduler()
    timesteps = inference_timesteps(ddim, C.T_INT, C.DDIM_STEPS)

    unet.eval()
    attn_stores = install_attn_hooks(unet)

    for i, img in enumerate(images):
        img_batch = img.unsqueeze(0).to(device)

        z0 = encode_to_latents(vae, img_batch, sample=False, generator=generator)
        noise = torch.randn(z0.shape, device=device, generator=generator)
        t_tensor = torch.tensor([C.T_INT], device=device, dtype=torch.long)
        z_noisy = ddim.add_noise(z0, noise, t_tensor)

        attn_accum = np.zeros((C.TARGET_SIZE, C.TARGET_SIZE), dtype=np.float32)
        n_steps = 0
        for t in timesteps:
            noise_pred = unet(z_noisy, t).sample
            z_noisy = ddim.step(noise_pred, t, z_noisy).prev_sample
            attn_accum += aggregate_step_attention(attn_stores)
            n_steps += 1
        a_total = attn_accum / max(n_steps, 1)

        recon = vae.decode(z_noisy / C.SCALING_FACTOR).sample
        orig_np  = normalize_for_vae(img_batch)[0].cpu().numpy()
        recon_np = recon[0].cpu().numpy()
        bmask    = brain_mask_2d(orig_np)

        # Residual using a zero baseline (no M_baseline yet during early training)
        zero_baseline = np.zeros_like(orig_np)
        m_pixel = pixel_residual_2d(orig_np, recon_np, zero_baseline, bmask)

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

    restore_default_processors(unet)
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

        # Encode in chunks to avoid OOM
        VAE_CHUNK = 32
        latent_chunks = []
        with torch.no_grad():
            for i in range(0, images.shape[0], VAE_CHUNK):
                chunk = images[i : i + VAE_CHUNK]
                latent_chunks.append(encode_to_latents(vae, chunk, sample=True))
        latents = torch.cat(latent_chunks, dim=0)
        del latent_chunks

        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (bsz,), device=device, dtype=torch.long,
        )

        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        noise_pred = unet(noisy_latents, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)

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
# Validation denoising loss
# ─────────────────────────────────────────────
@torch.no_grad()
def validate_denoising(unet, vae, scheduler, val_loader, device,
                       max_batches: int = 25) -> float:
    """Mean ε-prediction MSE on the val set with a *fixed* noise/timestep stream."""
    unet.eval()
    g = torch.Generator(device=device).manual_seed(C.SEED)
    total, n = 0.0, 0
    for bi, images in enumerate(val_loader):
        if max_batches is not None and bi >= max_batches:
            break
        images = images.to(device)
        latents = encode_to_latents(vae, images, sample=False)
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

    m_baseline, calib = calibrate_on_healthy(
        vae, unet, val_loader, device, generator=generator)
    save_calibration(m_baseline, calib)

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
                            device, generator, epoch: int, log_dir: Path) -> dict:
    """Swap in EMA weights, reconstruct, compute metrics, and restore."""
    raw_state = {k: v.detach().cpu().clone() for k, v in unet.state_dict().items()}
    ema.copy_to(unet)
    unet.eval()

    ddim = make_ddim_scheduler()
    timesteps = inference_timesteps(ddim, C.T_INT, C.DDIM_STEPS)

    all_metrics = []
    for img in monitor_images:
        img_batch = img.unsqueeze(0).to(device)
        orig_norm, recon, _, _ = reconstruct_healthy(
            vae, unet, ddim, img_batch, timesteps, C.T_INT, generator)
        orig_np  = orig_norm[0].cpu().numpy()
        recon_np = recon[0].cpu().numpy()
        bmask    = brain_mask_2d(orig_np)
        
        orig_tensor = torch.from_numpy(orig_np).unsqueeze(0).to(device)
        recon_tensor = torch.from_numpy(recon_np).unsqueeze(0).to(device)
        metrics = compute_recon_metrics(orig_tensor, recon_tensor)
        
        all_metrics.append(metrics)

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

    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"metrics_epoch_{epoch:03d}.json"
    import json
    with open(out_path, "w") as f:
        json.dump(avg, f, indent=2)

    return avg


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    # Set seed
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rl = RunLogger("diffusion", config_snapshot=C.snapshot(),
                   use_tensorboard=not args.no_tensorboard)
    fh = logging.FileHandler(rl.run_dir / "run.log")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(fh)

    log.info("Device: %s", device)
    generator = torch.Generator(device=device).manual_seed(C.SEED)

    # ── Data ─────────────────────────────────────────────────────
    train_dir = args.data_dir if args.data_dir else C.HEALTHY_DIR
    log.info("Training data dir: %s", train_dir)
    train_ds = HealthySliceDataset(train_dir)

    if C.VAL_HEALTHY_DIR.exists() and any(C.VAL_HEALTHY_DIR.glob("*.npy")):
        log.info("Using explicit val-healthy dataset from %s", C.VAL_HEALTHY_DIR)
        val_ds = HealthySliceDataset(C.VAL_HEALTHY_DIR)
    else:
        log.warning(
            "VAL_HEALTHY_DIR (%s) not found or empty — falling back to 10%% random "
            "split of train data. Run make_splits.py + preprocess.py "
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

    val_files = list(val_ds.files if hasattr(val_ds, "files") else
                     [val_ds.dataset.files[i] for i in val_ds.indices])
    n_monitor = min(N_MONITOR_SLICES, len(val_files))
    monitor_images = torch.stack([
        torch.from_numpy(np.load(val_files[i])).float()
        for i in range(n_monitor)
    ])
    log.info("Monitor/XAI set: %d val-healthy slices (fixed)", n_monitor)

    # ── Models / optim / scheduler / EMA ─────────────────────────
    vae = build_vae(device)
    unet = build_unet(device)
    scheduler  = make_ddpm_scheduler()
    optimizer  = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    lr_sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=C.LR_ETA_MIN)
    ema = EMA(unet, decay=C.EMA_DECAY)

    # ── Resume ─────────────────────────────────────────────
    start_epoch, best_loss = 1, float("inf")
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

    if start_epoch == 1 and not args.no_xai:
        log.info("Before-training XAI snapshot (random-init UNet) …")
        unet.eval()
        log_xai_panels(vae, unet, monitor_images, 0, C.TRAJ_DIR, device, generator)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    patience_counter = 0
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

        val_loss = validate_denoising(unet, vae, scheduler, val_loader, device,
                                      max_batches=args.val_batches)
        log.info("  val denoising loss: %.6f", val_loss)
        rl.log_metrics(epoch, "train", {"loss": avg_loss,
                                        "lr": lr_sched.get_last_lr()[0]})
        rl.log_metrics(epoch, "val", {"denoise_loss": val_loss})

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            C.UNET_DIR.mkdir(parents=True, exist_ok=True)
            unet.save_pretrained(C.UNET_DIR)
            
            raw_st = {k: v.detach().cpu().clone() for k, v in unet.state_dict().items()}
            ema.copy_to(unet)
            C.UNET_EMA_DIR.mkdir(parents=True, exist_ok=True)
            unet.save_pretrained(C.UNET_EMA_DIR)
            unet.load_state_dict(raw_st)
            
            calibrate_with_ema(vae, unet, ema, val_loader, device, generator)
            log.info("  ↳ new best val_loss %.6f — UNet + EMA + calibration saved",
                     best_loss)
        else:
            patience_counter += 1
            log.info("  ↳ no improvement (%d / %d patience)",
                     patience_counter, args.patience)
            if args.patience > 0 and patience_counter >= args.patience:
                log.info("Early stopping triggered at epoch %d (patience=%d)",
                         epoch, args.patience)
                break
                     
        if epoch % args.cal_every == 0 or epoch == args.epochs:
            log.info("Running DDIM calibration on up to %d val samples …",
                     min(len(val_ds), C.MAX_CAL_SAMPLES))
            calib, _ = calibrate_with_ema(vae, unet, ema, val_loader, device, generator)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            recon = monitor_reconstruction(vae, unet, ema, monitor_images,
                                           device, generator, epoch, C.TRAIN_LOG_DIR)
            rl.log_metrics(epoch, "val", {"recon_mae": recon["mae"],
                                          "recon_psnr": recon["psnr"],
                                          "recon_ssim": recon["ssim"]})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raw_state = {k: v.detach().cpu().clone() for k, v in unet.state_dict().items()}
            ema.copy_to(unet)
            unet.eval()
            log_xai_panels(vae, unet, monitor_images, epoch, C.TRAJ_DIR, device, generator)
            unet.load_state_dict(raw_state)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ckptkit.save_checkpoint(C.UNET_CKPT_DIR / "last.pt", model=unet,
                                optimizer=optimizer, scheduler=lr_sched, ema=ema,
                                epoch=epoch, best_metric=best_loss)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckptkit.save_checkpoint(C.UNET_CKPT_DIR / f"ckpt_ep{epoch:03d}.pt",
                                    model=unet, optimizer=optimizer, scheduler=lr_sched,
                                    ema=ema, epoch=epoch, best_metric=best_loss)
            ckptkit.prune_old(C.UNET_CKPT_DIR, keep_last=args.keep_last)
            
            raw_state = {k: v.detach().cpu().clone() for k, v in unet.state_dict().items()}
            ema.copy_to(unet)
            C.UNET_EMA_DIR.mkdir(parents=True, exist_ok=True)
            unet.save_pretrained(C.UNET_EMA_DIR)
            unet.load_state_dict(raw_state)
            log.info("  ↳ checkpoint + EMA snapshot @ epoch %d → %s",
                     epoch, C.UNET_CKPT_DIR)

        if args.drive_sync_dir and epoch % 5 == 0:
            import subprocess
            log.info("Syncing local logs & checkpoints to Google Drive...")
            try:
                sync_log_dest = Path(args.drive_sync_dir) / "logs"
                sync_log_dest.mkdir(parents=True, exist_ok=True)
                subprocess.run(["rsync", "-a", str(C.TRAIN_LOG_DIR) + "/", str(sync_log_dest) + "/"], check=True)
                
                sync_model_dest = Path(args.drive_sync_dir) / "cv_models"
                sync_model_dest.mkdir(parents=True, exist_ok=True)
                subprocess.run(["rsync", "-a", str(C.UNET_CKPT_DIR) + "/", str(sync_model_dest) + "/"], check=True)
            except Exception as e:
                log.warning("Drive sync failed: %s", e)

    ema.copy_to(unet)
    C.UNET_EMA_DIR.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(C.UNET_EMA_DIR)
    log.info("EMA UNet weights saved → %s", C.UNET_EMA_DIR)

    log.info("=" * 60)
    log.info("DONE  —  best val loss: %.6f", best_loss)
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
    try:
        parser = argparse.ArgumentParser(
            description="Phase 3 — Latent DDPM training on healthy brain slices",
        )
        parser.add_argument("--epochs",    type=int,   default=30,   help="Training epochs (default: 30)")
        parser.add_argument("--bs",        type=int,   default=8,    help="Batch size (default: 8)")
        parser.add_argument("--lr",        type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
        parser.add_argument("--data-dir",  type=Path,  default=None,
                            help="Override training data directory (default: config.HEALTHY_DIR)")
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
        parser.add_argument("--patience",  type=int, default=30,
                            help="Early stopping patience: stop after N epochs with no val improvement (0=disabled).")
        parser.add_argument("--no-xai", action="store_true",
                            help="Skip SAAM/residual XAI panels (incl. the epoch-0 snapshot).")
        parser.add_argument("--no-tensorboard", action="store_true")
        parser.add_argument("--drive-sync-dir", type=str, default=None,
                            help="Google Drive path to automatically rsync logs/models to every 5 epochs.")
        args = parser.parse_args()

        main(args)
    except Exception as e:
        log.exception(f"Fatal error in train_diffusion: {e}")
        sys.exit(1)
