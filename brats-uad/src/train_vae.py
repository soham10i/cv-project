"""
Stage 1 — Train the medical KL-VAE from scratch.
=================================================
Trains the 4-channel KLVAE codec on ALL BraTS slices (train+val patients) so it
faithfully reconstructs both healthy tissue and lesions, then becomes the frozen
codec for the latent diffusion stage.

Objective
---------
    L = L1(x, x̂) + λ_msssim·(1 − MS-SSIM(x, x̂)) + λ_kl·KL(q(z|x) ‖ N(0, I))

L1 gives pixel fidelity, MS-SSIM enforces structural/texture fidelity (so lesion
detail is not blurred away), and a tiny KL keeps the latent near N(0, I) — which
is exactly the prior the diffusion model assumes.

Usage
-----
    python src/train_vae.py                    # full training (config defaults)
    python src/train_vae.py --epochs 60 --bs 16
    python src/train_vae.py --smoke            # 2-epoch plumbing test
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
from models.kl_vae import KLVAE
from models.losses import ms_ssim_loss

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_vae")


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────
def vae_loss(model, x_raw):
    x = normalize_for_vae(x_raw)
    recon, post = model(x, sample_posterior=True)
    l1 = F.l1_loss(recon, x)
    msssim = ms_ssim_loss(recon.clamp(-1, 1), x, data_range=2.0)
    kl = post.kl()
    total = l1 + C.VAE_MSSSIM_W * msssim + C.VAE_KL_WEIGHT * kl
    comp = {"loss": total.detach(), "l1": l1.detach(),
            "msssim": msssim.detach(), "kl": kl.detach()}
    return total, comp, recon, x


# ─────────────────────────────────────────────
# Train / validate
# ─────────────────────────────────────────────
def train_epoch(model, loader, opt, device, epoch, total, grad_accum):
    model.train()
    sums = {"loss": 0.0, "l1": 0.0, "msssim": 0.0, "kl": 0.0}
    n = 0
    nb = len(loader)
    opt.zero_grad()
    for bidx, x in enumerate(loader):
        x = x.to(device)
        loss, comp, _, _ = vae_loss(model, x)
        (loss / grad_accum).backward()
        if (bidx + 1) % grad_accum == 0 or (bidx + 1) == nb:
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
            opt.step()
            opt.zero_grad()
        for k in sums:
            sums[k] += float(comp[k])
        n += 1
        if (bidx + 1) % max(1, nb // 4) == 0:
            log.info("  ep %d/%d | batch %d/%d | loss %.4f (l1 %.4f, msssim %.4f, kl %.1f)",
                     epoch, total, bidx + 1, nb, float(comp["loss"]),
                     float(comp["l1"]), float(comp["msssim"]), float(comp["kl"]))
    return {k: sums[k] / max(n, 1) for k in sums}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    sums = {"loss": 0.0, "l1": 0.0, "msssim": 0.0, "kl": 0.0,
            "psnr": 0.0, "ssim": 0.0}
    n_b = n_img = 0
    for x in loader:
        x = x.to(device)
        _, comp, recon, x_norm = vae_loss(model, x)
        for k in ("loss", "l1", "msssim", "kl"):
            sums[k] += float(comp[k])
        n_b += 1
        r = recon.clamp(-1, 1).cpu().numpy()
        o = x_norm.cpu().numpy()
        for i in range(o.shape[0]):
            sums["psnr"] += utils.psnr(o[i], r[i])
            sums["ssim"] += utils.ssim2d(o[i, 1], r[i, 1])     # t1c channel
            n_img += 1
    out = {k: sums[k] / max(n_b, 1) for k in ("loss", "l1", "msssim", "kl")}
    out["psnr"] = sums["psnr"] / max(n_img, 1)
    out["ssim"] = sums["ssim"] / max(n_img, 1)
    return out


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    utils.set_seed()
    device = utils.get_device()
    rl = RunLogger("vae", config_snapshot=C.snapshot(),
                   run_name=("vae_smoke" if args.smoke else None),
                   use_tensorboard=not args.no_tensorboard)
    rl.attach(log)
    log.info("Device: %s", device)

    epochs = 2 if args.smoke else args.epochs
    bs = 4 if args.smoke else args.bs
    tlimit = 32 if args.smoke else None
    vlimit = 16 if args.smoke else None

    train_ds = SliceDataset(C.MANIFEST_VAE_TRAIN, limit=tlimit)
    val_ds = SliceDataset(C.MANIFEST_VAE_VAL, limit=vlimit)
    pin = device.type == "cuda"
    nw = 0 if args.smoke else args.num_workers
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=pin)

    model = KLVAE(in_ch=C.N_CHANNELS, out_ch=C.N_CHANNELS, base_ch=C.VAE_BASE_CH,
                  ch_mult=C.VAE_CH_MULT, num_res=C.VAE_NUM_RES, z_ch=C.LATENT_CH).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("KLVAE params: %s | latent (%d, %d, %d)", f"{n_params:,}",
             C.LATENT_CH, C.LATENT_SIZE, C.LATENT_SIZE)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=C.LR_ETA_MIN)
    stopper = utils.EarlyStopper(patience=(0 if args.smoke else args.patience))

    start_epoch, best_val = 1, float("inf")
    if args.resume and not args.smoke:
        rp = utils.find_resume(C.VAE_CKPT_DIR)
        if rp:
            ck = utils.load_checkpoint(rp, model=model, optimizer=opt,
                                       scheduler=sched, device=device)
            start_epoch = int(ck["epoch"]) + 1
            best_val = ck.get("best_metric") or float("inf")
            stopper.best = best_val
            log.info("Resumed from %s → epoch %d (best %.4f)", rp, start_epoch, best_val)

    log.info("=" * 60)
    log.info("VAE train | %d ep | bs %d×accum %d | lr %.1e | wd %.0e | λ_msssim %.2f | "
             "λ_kl %.0e | patience %d%s", epochs, bs, args.grad_accum, args.lr,
             args.weight_decay, C.VAE_MSSSIM_W, C.VAE_KL_WEIGHT, args.patience,
             "  [SMOKE]" if args.smoke else "")
    log.info("=" * 60)

    history = []
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, opt, device, epoch, epochs, args.grad_accum)
        sched.step()
        history.append(tr["loss"])
        rl.log_metrics(epoch, "train", {**tr, "lr": sched.get_last_lr()[0]})
        log.info("Epoch %d/%d | TRAIN loss %.4f (l1 %.4f msssim %.4f kl %.1f) | %.1fs",
                 epoch, epochs, tr["loss"], tr["l1"], tr["msssim"], tr["kl"],
                 time.time() - t0)

        if epoch % args.val_every == 0 or epoch == epochs:
            va = validate(model, val_loader, device)
            gap = utils.overfit_gap(tr["loss"], va["loss"])
            rl.log_metrics(epoch, "val", {**va, "overfit_gap": gap})
            log.info("Epoch %d/%d | VAL loss %.4f | PSNR %.2f | SSIM %.4f | gap %.1f%%",
                     epoch, epochs, va["loss"], va["psnr"], va["ssim"], 100 * gap)
            if gap > 0.5:
                log.warning("  ⚠ overfitting signal: val/train gap %.0f%% (>50%%) — "
                            "consider more data / weight decay / fewer epochs.", 100 * gap)

            improved, should_stop = stopper.step(va["loss"], epoch)
            if improved:
                best_val = va["loss"]
                model.save_pretrained(C.VAE_DIR)
                log.info("  ↳ new best val %.4f — VAE saved → %s", best_val, C.VAE_DIR)
            else:
                log.info("  ↳ no improvement (%d/%d patience; best %.4f @ ep %d)",
                         stopper.wait, args.patience, stopper.best, stopper.best_epoch)

        if not args.smoke:
            utils.save_checkpoint(C.VAE_CKPT_DIR / "last.pt", model=model, optimizer=opt,
                                  scheduler=sched, epoch=epoch, best_metric=best_val)
            if epoch % C.CKPT_EVERY == 0 or epoch == epochs:
                utils.save_checkpoint(C.VAE_CKPT_DIR / f"ckpt_ep{epoch:03d}.pt",
                                      model=model, optimizer=opt, scheduler=sched,
                                      epoch=epoch, best_metric=best_val)
                utils.prune_old(C.VAE_CKPT_DIR)
        utils.clear_cache()

        if should_stop:
            log.info("Early stopping at epoch %d (no val improvement for %d epochs).",
                     epoch, args.patience)
            break

    # Always save final weights (even in smoke) so downstream stages can load.
    if not C.VAE_DIR.exists() or args.smoke:
        model.save_pretrained(C.VAE_DIR)
    log.info("=" * 60)
    log.info("DONE — best val %.4f @ epoch %d | VAE → %s", best_val,
             stopper.best_epoch, C.VAE_DIR)
    log.info("  Logs/metrics/tensorboard → %s", rl.run_dir)
    log.info("  Next: python src/compute_scaling_factor.py")
    log.info("=" * 60)

    if args.smoke:
        finite = all(np.isfinite(v) for v in history)
        decreasing = len(history) < 2 or history[-1] <= history[0] * 1.5
        ok = finite and decreasing
        log.info("SMOKE %s — losses %s", "PASSED ✅" if ok else "FAILED ❌",
                 ["%.4f" % v for v in history])
        C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(C.RESULTS_DIR / "vae_smoke.json", "w") as f:
            json.dump({"passed": bool(ok), "history": history}, f, indent=2)

    rl.close()


if __name__ == "__main__":
    try:
        p = argparse.ArgumentParser(description="Stage 1 — train the medical KL-VAE")
        p.add_argument("--epochs", type=int, default=C.VAE_EPOCHS)
        p.add_argument("--bs", type=int, default=C.VAE_BATCH)
        p.add_argument("--lr", type=float, default=C.VAE_LR)
        p.add_argument("--grad-accum", type=int, default=1)
        p.add_argument("--val-every", type=int, default=1)
        p.add_argument("--num-workers", type=int, default=2)
        p.add_argument("--weight-decay", type=float, default=1e-4,
                       help="AdamW weight decay (regularisation against overfitting).")
        p.add_argument("--patience", type=int, default=C.VAE_PATIENCE,
                       help="Early-stop after N epochs without val improvement (0=off).")
        p.add_argument("--no-tensorboard", action="store_true")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--smoke", action="store_true")
        main(p.parse_args())
    except Exception as e:
        logging.exception("Fatal error in train_vae: %s", e)
        sys.exit(1)
