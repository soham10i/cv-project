"""
Phase 3 — Train the Healthy Manifold (Latent DDPM)
====================================================
Trains a UNet2DModel to denoise latent representations produced by a frozen
StabilityAI VAE.  Improvements over the original:

  * VAE inputs are mapped to [-1, 1] (utils.normalize_for_vae) before encoding.
  * Gradient clipping + cosine LR annealing for stable convergence.
  * EMA (exponential moving average) weights — these are what evaluation uses.
  * Calibration runs the FULL DDIM reconstruction pipeline on held-out healthy
    slices (utils.calibrate_on_healthy), producing a distribution-matched
    M_baseline AND an operational percentile threshold (calibration.json).
    This replaces the old encode→decode-only baseline.

Usage
-----
    python src/train_healthy_manifold.py                      # defaults
    python src/train_healthy_manifold.py --epochs 30 --bs 4   # custom
"""

import argparse
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# pyrefly: ignore [missing-import]
from diffusers import AutoencoderKL, UNet2DModel

import config as C
import utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


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
        return torch.from_numpy(arr).float()        # (3, 256, 256), z-scored


# ─────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────
def build_models(device):
    """Instantiate and return (vae, unet)."""
    log.info("Loading VAE from '%s' …", C.VAE_CKPT)
    vae = AutoencoderKL.from_pretrained(C.VAE_CKPT)
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
# Training loop
# ─────────────────────────────────────────────
def train_one_epoch(unet, vae, scheduler, loader, optimizer, ema, device,
                    epoch, total_epochs):
    unet.train()
    running_loss = 0.0
    n_batches    = 0

    for batch_idx, images in enumerate(loader):
        images = images.to(device)                          # (B, 3, 256, 256)

        # 1. Encode → scaled latents (normalize_for_vae applied inside)
        latents = utils.encode_to_latents(vae, images, sample=True)

        # 2. Random timestep per image
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (bsz,), device=device, dtype=torch.long,
        )

        # 3-4. Sample noise and add it
        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # 5-6. Predict noise + MSE loss
        noise_pred = unet(noisy_latents, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)

        # 7. Backprop with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), C.GRAD_CLIP_NORM)
        optimizer.step()
        ema.update(unet)

        running_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % max(1, len(loader) // 4) == 0:
            log.info("  Epoch %d/%d  |  batch %d/%d  |  loss: %.6f",
                     epoch, total_epochs, batch_idx + 1, len(loader), loss.item())

    return running_loss / max(n_batches, 1)


# ─────────────────────────────────────────────
# Calibration on EMA weights
# ─────────────────────────────────────────────
@torch.no_grad()
def calibrate_with_ema(vae, unet, ema, val_loader, device, generator):
    """
    Swap EMA weights into the UNet, run the full DDIM calibration, save
    M_baseline + calibration.json, then restore the live training weights.
    """
    raw_state = {k: v.detach().cpu().clone() for k, v in unet.state_dict().items()}
    ema.copy_to(unet)
    unet.eval()

    m_baseline, calib = utils.calibrate_on_healthy(
        vae, unet, val_loader, device, generator=generator)
    utils.save_calibration(m_baseline, calib)

    log.info("Calibration → M_baseline mean %.6f | thr_pixel %.4f | thr_fused %.4f "
             "| n=%d", m_baseline.mean(), calib["threshold_pixel"],
             calib["threshold_fused"], calib["n_samples"])

    unet.load_state_dict(raw_state)                 # restore live weights
    return calib


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    utils.set_seed()
    device = utils.get_device()
    log.info("Device: %s", device)
    generator = utils.make_generator(device)

    # ── Data ─────────────────────────────────────────────────────
    full_dataset = HealthySliceDataset(C.HEALTHY_DIR)
    n_val   = max(1, int(len(full_dataset) * 0.10))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(C.SEED),
    )
    log.info("Split: %d train  /  %d val", n_train, n_val)

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                              num_workers=0, pin_memory=pin)

    # ── Models / optim / scheduler / EMA ─────────────────────────
    vae, unet = build_models(device)
    scheduler = utils.make_ddpm_scheduler()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=C.LR_ETA_MIN)
    ema = utils.EMA(unet, decay=C.EMA_DECAY)

    # ── Training ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Training: %d epochs | bs %d | lr %.1e | EMA %.4f | clip %.1f",
             args.epochs, args.bs, args.lr, C.EMA_DECAY, C.GRAD_CLIP_NORM)
    log.info("=" * 60)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(
            unet, vae, scheduler, train_loader, optimizer, ema,
            device, epoch, args.epochs)
        lr_sched.step()
        log.info("Epoch %d/%d  —  avg loss: %.6f  —  lr %.2e  —  %.1fs",
                 epoch, args.epochs, avg_loss, lr_sched.get_last_lr()[0],
                 time.time() - t0)

        # ── Best raw checkpoint ──────────────────────────────────
        if avg_loss < best_loss:
            best_loss = avg_loss
            C.UNET_DIR.mkdir(parents=True, exist_ok=True)
            unet.save_pretrained(C.UNET_DIR)
            log.info("  ↳ new best (loss %.6f) — raw UNet saved → %s",
                     best_loss, C.UNET_DIR)

        # ── Periodic calibration on EMA ──────────────────────────
        if epoch % args.cal_every == 0 or epoch == args.epochs:
            log.info("Running DDIM calibration on up to %d val samples …",
                     min(n_val, C.MAX_CAL_SAMPLES))
            calibrate_with_ema(vae, unet, ema, val_loader, device, generator)
            utils.clear_cache()

    # ── Save EMA weights (used for inference) ────────────────────
    ema.copy_to(unet)
    C.UNET_EMA_DIR.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(C.UNET_EMA_DIR)
    log.info("EMA UNet weights saved → %s", C.UNET_EMA_DIR)

    log.info("=" * 60)
    log.info("DONE  —  best training loss: %.6f", best_loss)
    log.info("  Raw UNet : %s", C.UNET_DIR)
    log.info("  EMA UNet : %s   (preferred for evaluation)", C.UNET_EMA_DIR)
    log.info("  Baseline : %s", C.BASELINE_PATH)
    log.info("  Calib    : %s", C.CALIBRATION_PATH)
    log.info("=" * 60)


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
    parser.add_argument("--cal-every", type=int,   default=5,    help="Run calibration every N epochs (default: 5)")
    args = parser.parse_args()

    main(args)
