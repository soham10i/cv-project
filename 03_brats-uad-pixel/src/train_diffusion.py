"""
Stage 2 — Train the healthy-manifold pixel-space DDPM.
======================================================
Trains a UNet2DModel to denoise (4, 256, 256) pixel images DIRECTLY using ONLY
lesion-free, buffer-clean slices from TRAIN patients.  No VAE is involved.

At inference, partially noising a test slice and denoising with this model
projects it back onto the healthy manifold; the residual reveals lesions.

Usage
-----
    python src/train_diffusion.py                  # defaults
    python src/train_diffusion.py --epochs 50 --bs 16
    python src/train_diffusion.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[0]))

import config as C
import utils
from logkit import RunLogger
from data.datasets import SliceDataset
from data.normalization import normalize_for_vae
from models.unet import build_unet
from models.ema import EMA
from pipeline.diffusion import make_ddpm_scheduler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_diffusion")
log.setLevel(logging.INFO)
if not log.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(ch)
log.propagate = False


def train_epoch(unet, sched, loader, opt, ema, device, epoch, total, grad_accum):
    """Train one epoch — pixel-space diffusion (no VAE encoding)."""
    unet.train()
    running, n = 0.0, 0
    nb = len(loader)
    opt.zero_grad()
    for bidx, images in enumerate(loader):
        # Normalise images to [-1, 1] range and feed directly to the UNet.
        images = normalize_for_vae(images.to(device))

        noise = torch.randn_like(images)
        bsz = images.shape[0]
        t = torch.randint(0, sched.config.num_train_timesteps, (bsz,), device=device).long()
        noisy = sched.add_noise(images, noise, t)

        pred = unet(noisy, t).sample
        loss = F.mse_loss(pred, noise)

        loss = loss / grad_accum
        loss.backward()
        if (bidx + 1) % grad_accum == 0 or (bidx + 1) == nb:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), C.GRAD_CLIP_NORM)
            opt.step()
            opt.zero_grad()
            ema.update(unet)

        running += loss.item()
        n += 1
        if (bidx + 1) % max(1, nb // 4) == 0:
            log.info("  ep %d/%d | batch %d/%d | loss %.5f | avg %.5f",
                     epoch, total, bidx + 1, nb, loss.item(), running / n)
    return running / max(n, 1)


@torch.no_grad()
def validate(unet, sched, loader, device, max_batches=25):
    """Validate on held-out healthy slices — pixel-space."""
    unet.eval()
    g = torch.Generator(device=device).manual_seed(C.SEED)
    total, n = 0.0, 0
    for bi, images in enumerate(loader):
        if bi >= max_batches:
            break
        images = normalize_for_vae(images.to(device))
        bsz = images.shape[0]
        t = torch.randint(0, sched.config.num_train_timesteps, (bsz,),
                          device=device, generator=g, dtype=torch.long)
        noise = torch.randn(images.shape, device=device, generator=g)
        noisy = sched.add_noise(images, noise, t)
        pred = unet(noisy, t).sample
        total += F.mse_loss(pred, noise).item()
        n += 1
    return total / max(n, 1)


def save_ema_unet(unet, ema, out_dir):
    """Swap in EMA weights, save_pretrained, restore raw weights."""
    raw = {k: v.detach().cpu().clone() for k, v in unet.state_dict().items()}
    ema.copy_to(unet)
    out_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(out_dir))
    unet.load_state_dict(raw)


def main(args):
    utils.set_seed()
    device = utils.get_device()
    rl = RunLogger("diffusion_pixel", config_snapshot=C.snapshot(),
                   run_name=("diffusion_pixel_smoke" if args.smoke else None),
                   use_tensorboard=not args.no_tensorboard)
    rl.attach(log)
    log.info("Device: %s", device)
    log.info("MODE: PIXEL-SPACE DIFFUSION (no VAE)")

    epochs = 2 if args.smoke else args.epochs
    bs = 4 if args.smoke else args.bs
    tlimit = 32 if args.smoke else None
    vlimit = 16 if args.smoke else None

    train_ds = SliceDataset(C.MANIFEST_HEALTHY, limit=tlimit)
    val_ds = SliceDataset(C.MANIFEST_VAL_HEALTHY, limit=vlimit)
    nw = 0 if args.smoke else args.num_workers
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=False)

    log.info("Building pixel-space UNet...")
    unet = build_unet(device)
    sched = make_ddpm_scheduler()
    log.info("UNet and Scheduler built.")

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                          eta_min=C.LR_ETA_MIN)
    ema = EMA(unet, decay=C.EMA_DECAY)
    stopper = utils.EarlyStopper(patience=(0 if args.smoke else args.patience))

    start_epoch, best = 1, float("inf")
    if args.resume and not args.smoke:
        rp = utils.find_resume(C.UNET_CKPT_DIR)
        if rp:
            ck = utils.load_checkpoint(rp, model=unet, optimizer=opt,
                                       scheduler=lr_sched, ema=ema, device=device)
            start_epoch = int(ck["epoch"]) + 1
            best = ck.get("best_metric") or float("inf")
            stopper.best = best
            log.info("Resumed from %s → epoch %d (best %.5f)", rp, start_epoch, best)

    log.info("=" * 60)
    log.info("Pixel-space diffusion | %d ep | bs %d×accum %d | lr %.1e | "
             "wd %.0e | EMA %.4f | patience %d%s",
             epochs, bs, args.grad_accum, args.lr, args.weight_decay,
             C.EMA_DECAY, args.patience, "  [SMOKE]" if args.smoke else "")
    log.info("=" * 60)

    history = []
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        avg = train_epoch(unet, sched, train_loader, opt, ema, device,
                          epoch, epochs, args.grad_accum)
        lr_sched.step()
        vloss = validate(unet, sched, val_loader, device, args.val_batches)
        history.append(avg)
        gap = utils.overfit_gap(avg, vloss)
        rl.log_metrics(epoch, "train", {"loss": avg, "lr": lr_sched.get_last_lr()[0]})
        rl.log_metrics(epoch, "val", {"denoise_loss": vloss, "overfit_gap": gap})
        log.info("Epoch %d/%d | train %.5f | val %.5f | gap %.1f%% | lr %.2e | %.1fs",
                 epoch, epochs, avg, vloss, 100 * gap, lr_sched.get_last_lr()[0],
                 time.time() - t0)
        if gap > 0.5:
            log.warning("  ⚠ overfitting signal: val/train gap %.0f%% (>50%%).", 100 * gap)

        improved, should_stop = stopper.step(vloss, epoch)
        if improved:
            best = vloss
            C.UNET_DIR.mkdir(parents=True, exist_ok=True)
            unet.save_pretrained(str(C.UNET_DIR))
            save_ema_unet(unet, ema, C.UNET_EMA_DIR)
            log.info("  ↳ new best val %.5f — UNet + EMA saved", best)
        else:
            log.info("  ↳ no improvement (%d/%d patience; best %.5f @ ep %d)",
                     stopper.wait, args.patience, stopper.best, stopper.best_epoch)

        if not args.smoke:
            utils.save_checkpoint(C.UNET_CKPT_DIR / "last.pt", model=unet, optimizer=opt,
                                  scheduler=lr_sched, ema=ema, epoch=epoch, best_metric=best)
            if epoch % C.CKPT_EVERY == 0 or epoch == epochs:
                utils.save_checkpoint(C.UNET_CKPT_DIR / f"ckpt_ep{epoch:03d}.pt",
                                      model=unet, optimizer=opt, scheduler=lr_sched,
                                      ema=ema, epoch=epoch, best_metric=best)
                utils.prune_old(C.UNET_CKPT_DIR)
        utils.clear_cache()

        if should_stop:
            log.info("Early stopping at epoch %d (no val improvement for %d epochs).",
                     epoch, args.patience)
            break

    log.info("=" * 60)
    log.info("DONE — best val %.5f @ epoch %d", best, stopper.best_epoch)
    log.info("  UNet     : %s", C.UNET_DIR)
    log.info("  EMA UNet : %s  (preferred for inference)", C.UNET_EMA_DIR)
    log.info("  Logs/metrics/tensorboard → %s", rl.run_dir)
    log.info("  Next: python src/calibrate.py")
    log.info("=" * 60)

    if args.smoke:
        ok = all(np.isfinite(v) for v in history)
        log.info("SMOKE %s — losses %s", "PASSED ✅" if ok else "FAILED ❌",
                 ["%.5f" % v for v in history])
        C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(C.RESULTS_DIR / "diffusion_smoke.json", "w") as f:
            json.dump({"passed": bool(ok), "history": history}, f, indent=2)

    rl.close()


if __name__ == "__main__":
    try:
        p = argparse.ArgumentParser(description="Stage 2 — pixel-space DDPM on healthy slices")
        p.add_argument("--epochs", type=int, default=C.DIFF_EPOCHS)
        p.add_argument("--bs", type=int, default=C.DIFF_BATCH)
        p.add_argument("--lr", type=float, default=C.DIFF_LR)
        p.add_argument("--grad-accum", type=int, default=1)
        p.add_argument("--val-batches", type=int, default=25)
        p.add_argument("--num-workers", type=int, default=2)
        p.add_argument("--weight-decay", type=float, default=0.0,
                       help="AdamW weight decay (regularisation against overfitting).")
        p.add_argument("--patience", type=int, default=C.DIFF_PATIENCE,
                       help="Early-stop after N epochs without val improvement (0=off).")
        p.add_argument("--no-tensorboard", action="store_true")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--smoke", action="store_true")
        main(p.parse_args())
    except Exception as e:
        logging.exception("Fatal error in train_diffusion: %s", e)
        sys.exit(1)
