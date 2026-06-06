"""
MedSegDiff Phase 3 — Train conditional diffusion segmentation
==============================================================
Supervised: diffuse the binary tumour mask conditioned on the MRI.
Reuses the generic diffusion utilities (DDPM/DDIM schedulers, EMA, cosine LR,
gradient clipping).  Validation samples masks with the EMA weights and tracks
the best model by validation DICE.

Usage
-----
    python src/train_medsegdiff.py                       # full run
    python src/train_medsegdiff.py --epochs 1 --bs 4     # smoke
"""

import argparse
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

import config as C
import utils
from medsegdiff_model import build_medsegdiff
from medsegdiff_utils import SegDataset, mask_to_x0, sample_mask, dice_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARCH = {"img_size": C.SEG_TRAIN_SIZE, "base": C.SEG_BASE_CH,
        "ch_mult": list(C.SEG_CH_MULT), "attn_res": list(C.SEG_ATTN_RES)}


@torch.no_grad()
def validate_dice(model, ddim, val_loader, device, generator, max_batches, steps):
    """Mean DICE over a val subset, sampled with the current model weights."""
    model.eval()
    dices = []
    for bi, (img, mask) in enumerate(val_loader):
        if bi >= max_batches:
            break
        img = img.to(device)
        prob = sample_mask(model, ddim, img, steps=steps, ensemble=1, generator=generator)
        pred = (prob > 0.5).cpu().numpy()
        gt = mask.numpy() > 0.5
        for b in range(pred.shape[0]):
            dices.append(dice_score(pred[b, 0], gt[b, 0]))
    return float(np.mean(dices)) if dices else 0.0


def main(args):
    utils.set_seed()
    device = utils.get_device()
    generator = utils.make_generator(device)
    log.info("Device: %s | train size %d²", device, C.SEG_TRAIN_SIZE)

    train_ds = SegDataset("train")
    val_ds   = SegDataset("val")
    log.info("Data: %d train / %d val slices", len(train_ds), len(val_ds))
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True,
                              num_workers=args.workers, pin_memory=pin, persistent_workers=args.workers > 0)
    val_loader   = DataLoader(val_ds, batch_size=args.val_bs, shuffle=False,
                              num_workers=args.workers, pin_memory=pin)

    model = build_medsegdiff(img_size=C.SEG_TRAIN_SIZE, base=C.SEG_BASE_CH,
                             ch_mult=C.SEG_CH_MULT, attn_res=C.SEG_ATTN_RES).to(device)
    log.info("MedSegDiff U-Net: %.1fM params", sum(p.numel() for p in model.parameters()) / 1e6)

    ddpm = utils.make_ddpm_scheduler()
    ddim = utils.make_ddim_scheduler()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=C.LR_ETA_MIN)
    ema = utils.EMA(model, decay=C.EMA_DECAY)
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    log.info("AMP fp16: %s", use_amp)

    C.SEG_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("Training: %d epochs | bs %d | lr %.1e | EMA %.4f", args.epochs, args.bs, args.lr, C.EMA_DECAY)
    log.info("=" * 60)

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, n = 0.0, 0
        for img, mask in train_loader:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x0 = mask_to_x0(mask)
            noise = torch.randn_like(x0)
            t = torch.randint(0, ddpm.config.num_train_timesteps, (x0.shape[0],),
                              device=device, dtype=torch.long)
            x_t = ddpm.add_noise(x0, noise, t)

            opt.zero_grad()
            with autocast("cuda", dtype=torch.float16, enabled=use_amp):
                eps_pred = model(x_t, t, img)
                loss = F.mse_loss(eps_pred, noise)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            running += loss.item()
            n += 1
            if n % max(1, len(train_loader) // 4) == 0:
                log.info("  Epoch %d/%d | batch %d/%d | loss %.5f",
                         epoch, args.epochs, n, len(train_loader), loss.item())

        lr_sched.step()
        log.info("Epoch %d/%d — avg loss %.5f — lr %.2e — %.1fs",
                 epoch, args.epochs, running / max(n, 1), lr_sched.get_last_lr()[0], time.time() - t0)

        # ── Validate on EMA weights ──────────────────────────────
        if epoch % args.val_every == 0 or epoch == args.epochs:
            raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            vdice = validate_dice(model, ddim, val_loader, device, generator,
                                  args.val_batches, args.val_steps)
            log.info("  ↳ val DICE (EMA, %d batches): %.4f", args.val_batches, vdice)
            if vdice > best_dice:
                best_dice = vdice
                torch.save({"state_dict": model.state_dict(), "arch": ARCH, "val_dice": vdice},
                           C.SEG_MODEL_DIR / "medsegdiff_ema_best.pt")
                log.info("    new best (DICE %.4f) → %s", vdice, C.SEG_MODEL_DIR / "medsegdiff_ema_best.pt")
            model.load_state_dict(raw)
            utils.clear_cache()

    ema.copy_to(model)
    torch.save({"state_dict": model.state_dict(), "arch": ARCH, "val_dice": best_dice},
               C.SEG_MODEL_DIR / "medsegdiff_ema_last.pt")
    log.info("=" * 60)
    log.info("DONE — best val DICE %.4f", best_dice)
    log.info("  Best : %s", C.SEG_MODEL_DIR / "medsegdiff_ema_best.pt")
    log.info("  Last : %s", C.SEG_MODEL_DIR / "medsegdiff_ema_last.pt")
    log.info("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MedSegDiff Phase 3 — training")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--bs", type=int, default=12)
    p.add_argument("--val-bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--val-batches", type=int, default=4, help="val batches to sample for DICE")
    p.add_argument("--val-steps", type=int, default=25, help="DDIM steps during val sampling")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="fp16 autocast (faster, less memory on CUDA)")
    main(p.parse_args())
