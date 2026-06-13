"""
Stage 1 — Fine-tune the medical VAE (faithful codec)
====================================================
Fine-tunes the pretrained ``stabilityai/sd-vae-ft-mse`` autoencoder on BraTS
slices so it reconstructs *lesion* texture as faithfully as healthy tissue,
while preserving the 4-channel / 32×32 latent geometry — so the downstream
diffusion UNet is reused unchanged.

Objective  (Rombach et al. 2022, LDM; Zhang et al. 2018, LPIPS)
--------------------------------------------------------------
    L = L1(x, x̂)  +  λ_lpips · LPIPS(x, x̂)  +  λ_kl · KL(q(z|x) ‖ N(0, I))

* L1        — pixel fidelity.
* LPIPS     — perceptual fidelity (texture/structure); gracefully disabled if
              the `lpips` package or its weights are unavailable → falls back
              to L1 + KL so the script always runs.
* KL        — keeps the latent close to N(0, I) (tiny weight, 1e-6).

Tracking  (everything to one run folder via logkit.RunLogger)
-------------------------------------------------------------
* Per-epoch TRAIN and VAL: total loss + each component (L1 / LPIPS / KL).
* Per-epoch VAL recon quality: brain-masked MAE, PSNR, SSIM (Wang et al. 2004).
* Per-epoch VAL fidelity: tumour-vs-healthy recon-error ratio — both absolute
  and an intensity-normalised "relative" ratio that de-confounds the fact that
  tumour voxels are simply brighter (the hardened version of the earlier
  vae_fidelity diagnostic).
* Explainability panels every --xai-every epochs: original / reconstruction /
  per-modality error / tumour overlay, on a fixed monitor set.

Usage
-----
    python src/train_vae.py                       # full fine-tune (config defaults)
    python src/train_vae.py --epochs 40 --bs 6
    python src/train_vae.py --smoke               # 2-epoch plumbing test
"""

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# pyrefly: ignore [missing-import]
from diffusers import AutoencoderKL

import config as C
import utils
import ckptkit
from logkit import RunLogger

# ── Optional perceptual loss (graceful) ──────────────────────────────
try:
    import lpips as _lpips_lib  # type: ignore
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False

N_MONITOR_SLICES = 6


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class VAESliceDataset(Dataset):
    """(3, 256, 256) z-scored slices.  When ``mask_dir`` is given, also returns
    a (256, 256) binary lesion mask (zeros when no mask file exists → healthy)."""

    def __init__(self, data_dir, mask_dir=None, limit: int | None = None):
        self.files = sorted(data_dir.glob("*.npy"))
        if limit is not None:
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"No .npy slices in {data_dir}")
        self.mask_dir = mask_dir

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.load(self.files[idx])).float()
        mask = torch.zeros(x.shape[-2:], dtype=torch.float32)
        if self.mask_dir is not None:
            mp = self.mask_dir / self.files[idx].name
            if mp.exists():
                mask = torch.from_numpy((np.load(mp) > 0).astype(np.float32))
        return x, mask, self.files[idx].stem


# ─────────────────────────────────────────────
# Loss + metrics
# ─────────────────────────────────────────────
def vae_forward_loss(vae, x_raw, lpips_fn, kl_w, lpips_w):
    """Encode→sample→decode and return (total_loss, components, recon, x_norm)."""
    x = utils.normalize_for_vae(x_raw)          # → [-1, 1]
    posterior = vae.encode(x).latent_dist
    z = posterior.sample()                      # reparameterised, differentiable
    recon = vae.decode(z).sample

    l1 = F.l1_loss(recon, x)
    kl = posterior.kl().mean()                  # summed over latent dims, mean over batch
    if lpips_fn is not None:
        perc = lpips_fn(recon.clamp(-1, 1), x).mean()
    else:
        perc = torch.zeros((), device=x.device)

    total = l1 + lpips_w * perc + kl_w * kl
    comp = {"l1": l1.detach(), "lpips": perc.detach(), "kl": kl.detach(),
            "loss": total.detach()}
    return total, comp, recon, x


def _ssim(a: np.ndarray, b: np.ndarray, data_range: float = 2.0) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim  # type: ignore
        return float(ssim(a, b, data_range=data_range))
    except Exception:
        mu_a, mu_b = a.mean(), b.mean()
        va, vb = a.var(), b.var()
        vab = ((a - mu_a) * (b - mu_b)).mean()
        c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
        return float(((2 * mu_a * mu_b + c1) * (2 * vab + c2)) /
                     ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2) + 1e-8))


def recon_metrics(orig: np.ndarray, recon: np.ndarray, bmask: np.ndarray) -> dict:
    """Brain-masked MAE / PSNR / SSIM between normalised orig and recon (3,H,W)."""
    diff = np.abs(orig - recon)
    md = diff * bmask[None]
    nvox = max(bmask.sum() * orig.shape[0], 1)
    mae = float(md.sum() / nvox)
    mse = float((md ** 2).sum() / nvox)
    psnr = float(10 * np.log10(4.0 / (mse + 1e-8)))      # data_range² = 4 in [-1,1]
    ssim = _ssim(orig[1] * bmask, recon[1] * bmask)      # T2w channel
    return {"mae": mae, "psnr": psnr, "ssim": ssim}


def fidelity_ratio(orig: np.ndarray, recon: np.ndarray, bmask: np.ndarray,
                   tumour: np.ndarray) -> dict | None:
    """Tumour-vs-healthy recon-error ratio, absolute and intensity-normalised.

    The absolute ratio conflates "tumour reconstructs worse" with "tumour is
    just brighter".  The relative ratio divides the error by the local intensity
    magnitude, isolating *structural* fidelity loss — the question that actually
    decides whether the codec is throwing lesion detail away."""
    tum = (tumour > 0) & (bmask > 0)
    healthy = (bmask > 0) & (~tum)
    if tum.sum() == 0 or healthy.sum() == 0:
        return None
    err = np.abs(orig - recon).mean(axis=0)              # (H,W)
    amag = np.abs(orig).mean(axis=0)
    t_mae, h_mae = float(err[tum].mean()), float(err[healthy].mean())
    t_rel = float((err[tum] / (amag[tum] + 1e-3)).mean())
    h_rel = float((err[healthy] / (amag[healthy] + 1e-3)).mean())
    return {"fid_ratio": t_mae / (h_mae + 1e-8),
            "fid_rel_ratio": t_rel / (h_rel + 1e-8)}


# ─────────────────────────────────────────────
# Train / validate
# ─────────────────────────────────────────────
def train_one_epoch(vae, loader, optimizer, lpips_fn, device, log, epoch, total,
                    grad_accum=1):
    vae.train()
    sums, n = {"loss": 0.0, "l1": 0.0, "lpips": 0.0, "kl": 0.0}, 0
    optimizer.zero_grad()
    n_batches = len(loader)
    for bidx, (x, _mask, _name) in enumerate(loader):
        x = x.to(device)
        total_loss, comp, _, _ = vae_forward_loss(
            vae, x, lpips_fn, C.VAE_KL_WEIGHT, C.VAE_LPIPS_WEIGHT)

        # Gradient accumulation: scale so the effective batch is bs × grad_accum.
        (total_loss / grad_accum).backward()
        if (bidx + 1) % grad_accum == 0 or (bidx + 1) == n_batches:
            torch.nn.utils.clip_grad_norm_(vae.parameters(), C.GRAD_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad()

        for k in sums:
            sums[k] += float(comp[k])
        n += 1
        if (bidx + 1) % max(1, n_batches // 4) == 0:
            log.info("  ep %d/%d | batch %d/%d | loss %.4f (l1 %.4f, lpips %.4f, kl %.1f)",
                     epoch, total, bidx + 1, n_batches, float(comp["loss"]),
                     float(comp["l1"]), float(comp["lpips"]), float(comp["kl"]))
    return {k: sums[k] / max(n, 1) for k in sums}


@torch.no_grad()
def validate(vae, loader, lpips_fn, device) -> dict:
    vae.eval()
    sums = {"loss": 0.0, "l1": 0.0, "lpips": 0.0, "kl": 0.0,
            "mae": 0.0, "psnr": 0.0, "ssim": 0.0}
    n = 0
    fid_ratios, fid_rel = [], []
    for x, mask, _name in loader:
        x = x.to(device)
        _, comp, recon, x_norm = vae_forward_loss(
            vae, x, lpips_fn, C.VAE_KL_WEIGHT, C.VAE_LPIPS_WEIGHT)
        for k in ("loss", "l1", "lpips", "kl"):
            sums[k] += float(comp[k])

        recon = recon.clamp(-1, 1).cpu().numpy()
        x_np = x_norm.cpu().numpy()
        mask_np = mask.numpy()
        for i in range(x_np.shape[0]):
            bmask = utils.brain_mask_2d(x_np[i])
            m = recon_metrics(x_np[i], recon[i], bmask)
            for k in ("mae", "psnr", "ssim"):
                sums[k] += m[k]
            fr = fidelity_ratio(x_np[i], recon[i], bmask, mask_np[i])
            if fr is not None:
                fid_ratios.append(fr["fid_ratio"])
                fid_rel.append(fr["fid_rel_ratio"])
        n += x_np.shape[0]

    nb = max(len(loader), 1)
    out = {"loss": sums["loss"] / nb, "l1": sums["l1"] / nb,
           "lpips": sums["lpips"] / nb, "kl": sums["kl"] / nb,
           "mae": sums["mae"] / max(n, 1), "psnr": sums["psnr"] / max(n, 1),
           "ssim": sums["ssim"] / max(n, 1)}
    if fid_ratios:
        out["fid_ratio"] = float(np.mean(fid_ratios))
        out["fid_rel_ratio"] = float(np.mean(fid_rel))
    return out


# ─────────────────────────────────────────────
# Explainability panels
# ─────────────────────────────────────────────
@torch.no_grad()
def save_xai_panels(vae, monitor, device, out_dir, epoch, log):
    out_dir.mkdir(parents=True, exist_ok=True)
    vae.eval()
    mod_names = C.MODALITIES
    for i, (x, mask, name) in enumerate(monitor):
        xb = x.unsqueeze(0).to(device)
        x_norm = utils.normalize_for_vae(xb)
        recon = vae.decode(vae.encode(x_norm).latent_dist.mean).sample.clamp(-1, 1)
        o = x_norm[0].cpu().numpy()
        r = recon[0].cpu().numpy()
        err = np.abs(o - r).mean(axis=0)
        tum = mask.numpy()

        fig, axes = plt.subplots(1, 5, figsize=(24, 5))
        axes[0].imshow(o[1], cmap="gray"); axes[0].set_title("Original (T2w)")
        axes[1].imshow(r[1], cmap="gray"); axes[1].set_title("VAE recon (T2w)")
        im2 = axes[2].imshow(err, cmap="hot"); axes[2].set_title("|orig − recon|")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        # per-modality error bars
        per_mod = [float(np.abs(o[c] - r[c]).mean()) for c in range(o.shape[0])]
        axes[3].bar(mod_names, per_mod, color="steelblue")
        axes[3].set_title("Per-modality MAE"); axes[3].set_ylim(bottom=0)
        axes[4].imshow(o[1], cmap="gray")
        axes[4].imshow(np.ma.masked_where(tum == 0, tum), cmap="autumn", alpha=0.6)
        axes[4].set_title("Lesion overlay")
        for a in (axes[0], axes[1], axes[2], axes[4]):
            a.axis("off")
        fig.suptitle(f"{name}  |  epoch {epoch}", fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_dir / f"vae_xai_ep{epoch:03d}_img{i}.png", dpi=120,
                    bbox_inches="tight")
        plt.close(fig)
    log.info("XAI panels saved → %s (epoch %d)", out_dir, epoch)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    utils.set_seed()
    device = utils.get_device()

    rl = RunLogger("vae", config_snapshot=C.snapshot(),
                   run_name=("vae_smoke" if args.smoke else None),
                   use_tensorboard=not args.no_tensorboard)
    log = rl.get_logger()
    log.info("Device: %s", device)

    epochs = 2 if args.smoke else args.epochs
    bs = 2 if args.smoke else args.bs
    train_limit = 8 if args.smoke else None
    val_limit = 4 if args.smoke else None

    # ── Data ─────────────────────────────────────────────────────────
    train_ds = VAESliceDataset(C.VAE_TRAIN_DIR, limit=train_limit)
    val_dir = C.VAE_VAL_DIR if (C.VAE_VAL_DIR.exists() and
              any(C.VAE_VAL_DIR.glob("*.npy"))) else C.VAE_TRAIN_DIR
    val_ds = VAESliceDataset(val_dir, mask_dir=C.VAE_VAL_MASKS_DIR, limit=val_limit)
    log.info("Train: %d slices (%s) | Val: %d slices (%s)",
             len(train_ds), C.VAE_TRAIN_DIR.name, len(val_ds), val_dir.name)

    pin = device.type == "cuda"
    nw = 0 if args.smoke else args.num_workers
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=pin)

    # Fixed monitor set — prefer lesion-bearing val slices for a useful overlay.
    monitor = []
    for j in range(len(val_ds)):
        x, mask, name = val_ds[j]
        if mask.sum() > 0:
            monitor.append((x, mask, name))
        if len(monitor) >= N_MONITOR_SLICES:
            break
    if len(monitor) < N_MONITOR_SLICES:
        for j in range(len(val_ds)):
            if len(monitor) >= N_MONITOR_SLICES:
                break
            monitor.append(val_ds[j])
    log.info("Monitor/XAI set: %d slices (fixed)", len(monitor))

    # ── Model ────────────────────────────────────────────────────────
    log.info("Loading VAE '%s' for fine-tuning …", C.VAE_CKPT)
    vae = AutoencoderKL.from_pretrained(C.VAE_CKPT).to(device)
    vae.requires_grad_(True)
    if args.freeze_encoder:
        vae.encoder.requires_grad_(False)
        if hasattr(vae, "quant_conv"):
            vae.quant_conv.requires_grad_(False)
        log.info("Encoder frozen — fine-tuning decoder only.")
    n_train_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    log.info("VAE trainable params: %s", f"{n_train_params:,}")

    lpips_fn = None
    if _HAS_LPIPS:
        try:
            lpips_fn = _lpips_lib.LPIPS(net="alex").to(device)
            lpips_fn.eval()
            for p in lpips_fn.parameters():
                p.requires_grad_(False)
            log.info("LPIPS (AlexNet) perceptual loss enabled (λ=%.3f)",
                     C.VAE_LPIPS_WEIGHT)
        except Exception as e:
            log.warning("LPIPS init failed (%s) — using L1 + KL only.", e)
    else:
        log.warning("`lpips` not installed — using L1 + KL only "
                    "(pip install lpips to enable the perceptual term).")

    optimizer = torch.optim.AdamW(
        [p for p in vae.parameters() if p.requires_grad], lr=args.lr)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=C.LR_ETA_MIN)

    # ── Resume (Colab/preemption safety) ─────────────────────────────
    ckpt_dir = C.VAE_CKPT_DIR
    start_epoch, best_val = 1, float("inf")
    if args.resume and not args.smoke:
        rp = ckptkit.find_resume(ckpt_dir)
        if rp is not None:
            ck = ckptkit.load_checkpoint(rp, model=vae, optimizer=optimizer,
                                         scheduler=lr_sched, device=device)
            start_epoch = int(ck["epoch"]) + 1
            best_val = ck.get("best_metric") or float("inf")
            log.info("Resumed from %s → epoch %d (best val %.4f)",
                     rp, start_epoch, best_val)
        else:
            log.info("--resume set but no %s/last.pt found — starting fresh.", ckpt_dir)

    log.info("=" * 60)
    log.info("VAE fine-tune | %d epochs | bs %d×accum %d | lr %.1e | λ_lpips %.3f | "
             "λ_kl %.0e%s", epochs, bs, args.grad_accum, args.lr, C.VAE_LPIPS_WEIGHT,
             C.VAE_KL_WEIGHT, "  [SMOKE]" if args.smoke else "")
    log.info("=" * 60)

    # ── Loop ─────────────────────────────────────────────────────────
    train_hist = []
    for epoch in range(start_epoch, epochs + 1):
        tr = train_one_epoch(vae, train_loader, optimizer, lpips_fn,
                             device, log, epoch, epochs, grad_accum=args.grad_accum)
        lr_sched.step()
        rl.log_metrics(epoch, "train", {**tr, "lr": lr_sched.get_last_lr()[0]})
        train_hist.append(tr["loss"])
        log.info("Epoch %d/%d | TRAIN loss %.4f (l1 %.4f, lpips %.4f, kl %.1f)",
                 epoch, epochs, tr["loss"], tr["l1"], tr["lpips"], tr["kl"])

        if epoch % args.val_every == 0 or epoch == epochs:
            va = validate(vae, val_loader, lpips_fn, device)
            rl.log_metrics(epoch, "val", va)
            fid = (f" | fid {va['fid_ratio']:.2f} (rel {va['fid_rel_ratio']:.2f})"
                   if "fid_ratio" in va else "")
            log.info("Epoch %d/%d | VAL loss %.4f | MAE %.4f | PSNR %.2f | SSIM %.4f%s",
                     epoch, epochs, va["loss"], va["mae"], va["psnr"], va["ssim"], fid)

            if va["loss"] < best_val:
                best_val = va["loss"]
                C.VAE_FT_DIR.mkdir(parents=True, exist_ok=True)
                vae.save_pretrained(C.VAE_FT_DIR)
                log.info("  ↳ new best val %.4f — VAE saved → %s", best_val, C.VAE_FT_DIR)

        if (epoch % args.xai_every == 0 or epoch == epochs) and not args.no_xai:
            save_xai_panels(vae, monitor, device, rl.run_dir / "xai", epoch, log)

        # ── Checkpoint: resumable last.pt every epoch + periodic snapshots ──
        if not args.smoke:
            ckptkit.save_checkpoint(ckpt_dir / "last.pt", model=vae,
                                    optimizer=optimizer, scheduler=lr_sched,
                                    epoch=epoch, best_metric=best_val)
            if epoch % args.save_every == 0 or epoch == epochs:
                ckptkit.save_checkpoint(ckpt_dir / f"ckpt_ep{epoch:03d}.pt", model=vae,
                                        optimizer=optimizer, scheduler=lr_sched,
                                        epoch=epoch, best_metric=best_val)
                ckptkit.prune_old(ckpt_dir, keep_last=args.keep_last)
                log.info("  ↳ checkpoint @ epoch %d → %s", epoch, ckpt_dir)
        utils.clear_cache()

    log.info("=" * 60)
    log.info("DONE — best val loss %.4f | fine-tuned VAE → %s", best_val, C.VAE_FT_DIR)
    log.info("  Logs/metrics → %s", rl.run_dir)
    log.info("  To use it downstream: set USE_FINETUNED_VAE = True in config.py")
    log.info("=" * 60)

    if args.smoke:
        finite = all(np.isfinite(v) for v in train_hist) and np.isfinite(best_val)
        decreasing = len(train_hist) < 2 or train_hist[-1] <= train_hist[0] * 1.5
        ok = finite and decreasing
        log.info("SMOKE TEST %s — train losses %s, val finite %s",
                 "PASSED ✅" if ok else "FAILED ❌",
                 ["%.4f" % v for v in train_hist], np.isfinite(best_val))
        rl.save_json("smoke_result.json",
                     {"passed": bool(ok), "train_loss_history": train_hist,
                      "best_val_loss": best_val, "lpips_enabled": lpips_fn is not None})

    rl.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1 — fine-tune the medical VAE")
    p.add_argument("--epochs", type=int, default=C.VAE_FT_EPOCHS)
    p.add_argument("--bs", type=int, default=C.VAE_FT_BATCH)
    p.add_argument("--lr", type=float, default=C.VAE_FT_LR)
    p.add_argument("--val-every", type=int, default=C.VAE_FT_VAL_EVERY)
    p.add_argument("--xai-every", type=int, default=C.VAE_FT_XAI_EVERY)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps → effective batch = bs × this.")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers (T4 Colab ≈ 2 vCPUs; default 2).")
    p.add_argument("--save-every", type=int, default=C.CKPT_EVERY,
                   help="Write an epoch-tagged checkpoint every N epochs.")
    p.add_argument("--keep-last", type=int, default=C.CKPT_KEEP_LAST,
                   help="Retain only the most recent N epoch-tagged checkpoints.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from VAE_CKPT_DIR/last.pt if present (Colab-safe).")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Fine-tune the decoder only (keeps the latent space fixed).")
    p.add_argument("--no-xai", action="store_true", help="Skip explainability panels.")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="2-epoch plumbing test on a tiny subset.")
    args = p.parse_args()
    main(args)
