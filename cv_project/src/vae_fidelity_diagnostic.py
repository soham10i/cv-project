"""
VAE Fidelity Diagnostic  (decides the "switch the VAE?" question)
=================================================================
The latent-diffusion pipeline can only localise a lesion if the VAE actually
*preserves* that lesion through a plain encode→decode round-trip.  If the
frozen natural-image VAE blurs tumour texture away, the lesion signal is lost
in the latent **before** the DDPM ever sees it — and no amount of diffusion
tuning can recover it.  In that case a domain-specific (medical) VAE is
justified; otherwise the VAE is *not* the bottleneck and effort is better spent
elsewhere (noise schedule, T_int, capacity).

This script runs **only** the VAE (no diffusion, no calibration) on the
anomalous test slices and measures the reconstruction error *inside* the tumour
mask versus the surrounding healthy brain:

    tumour_ratio = MAE(recon error | tumour voxels)
                   ───────────────────────────────────
                   MAE(recon error | healthy brain voxels)

Interpretation
--------------
  ratio ≈ 1.0   → the VAE reconstructs tumour as faithfully as healthy tissue.
                   The VAE is NOT throwing the lesion signal away → a VAE swap
                   is unlikely to be the highest-value fix.
  ratio ≫ 1.0   → tumour regions reconstruct much worse than healthy tissue →
                   the codec is losing lesion detail → a medical VAE (trained
                   on ALL data so it stays a faithful codec) is justified.

Usage
-----
    python src/vae_fidelity_diagnostic.py                 # all test slices
    python src/vae_fidelity_diagnostic.py --n-images 50   # first 50
    python src/vae_fidelity_diagnostic.py --max-plots 8
"""

import argparse
import json
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
from diffusers import AutoencoderKL

import config as C
import utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Dataset (mirrors evaluate_pipeline's loader)
# ─────────────────────────────────────────────
class AnomalousSliceDataset(torch.utils.data.Dataset):
    """Anomalous slices + matching ground-truth masks (mask must exist)."""

    def __init__(self, img_dir, mask_dir):
        self.mask_dir = mask_dir
        all_imgs = sorted(img_dir.glob("*.npy"))
        self.img_files = [p for p in all_imgs if (mask_dir / p.name).exists()]
        if not self.img_files:
            raise FileNotFoundError(f"No usable .npy/mask pairs in {img_dir}")
        log.info("Test set: %d anomalous slices", len(self.img_files))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img  = np.load(self.img_files[idx])
        mask = np.load(self.mask_dir / self.img_files[idx].name)
        return (
            torch.from_numpy(img).float(),
            torch.from_numpy(mask).float(),
            self.img_files[idx].stem,
        )


@torch.no_grad()
def vae_roundtrip(vae, image: torch.Tensor) -> np.ndarray:
    """Pure encode→decode (deterministic posterior mean) in the VAE's [-1, 1]
    space.  No SCALING_FACTOR — it is a plain autoencoder pass, so the factor
    would only cancel itself.  Returns the recon as (3, H, W) numpy."""
    x = utils.normalize_for_vae(image)
    z = vae.encode(x).latent_dist.mean
    recon = vae.decode(z).sample
    return recon[0].cpu().numpy()


@torch.no_grad()
def main(args):
    utils.set_seed()
    device = utils.get_device()
    log.info("Device: %s", device)

    log.info("Loading VAE from '%s' …", C.VAE_CKPT)
    vae = AutoencoderKL.from_pretrained(C.VAE_CKPT).to(device)
    vae.eval(); vae.requires_grad_(False)

    dataset = AnomalousSliceDataset(C.ANOMALOUS_DIR, C.MASKS_DIR)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    n_total = len(dataset) if args.n_images is None else min(args.n_images, len(dataset))

    out_dir = C.RESULTS_DIR / "vae_fidelity"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("VAE fidelity over %d slices (saving up to %d plots)", n_total, args.max_plots)
    log.info("=" * 60)

    per_image = []
    tumour_maes, healthy_maes, ratios = [], [], []

    for i, (image, gt_mask, name) in enumerate(loader):
        if args.n_images is not None and i >= args.n_images:
            break
        image   = image.to(device)
        name_str = name[0]

        orig_np  = utils.normalize_for_vae(image)[0].cpu().numpy()   # (3,H,W) in [-1,1]
        recon_np = vae_roundtrip(vae, image)                         # (3,H,W)

        # Mean-abs reconstruction error per pixel (averaged over modalities).
        err = np.abs(orig_np - recon_np).mean(axis=0)               # (H,W)

        bmask  = utils.brain_mask_2d(orig_np)                       # (H,W) {0,1}
        tumour = (gt_mask[0].numpy() > 0).astype(np.float32) * bmask
        healthy = bmask * (1.0 - tumour)                            # brain minus tumour

        t_vox = err[tumour > 0]
        h_vox = err[healthy > 0]
        if t_vox.size == 0 or h_vox.size == 0:
            continue

        t_mae = float(t_vox.mean())
        h_mae = float(h_vox.mean())
        ratio = t_mae / (h_mae + 1e-8)

        tumour_maes.append(t_mae)
        healthy_maes.append(h_mae)
        ratios.append(ratio)
        per_image.append({"name": name_str, "tumour_mae": t_mae,
                          "healthy_mae": h_mae, "ratio": ratio})

        log.info("[%d/%d] %s  tumour MAE %.4f | healthy MAE %.4f | ratio %.2f",
                 i + 1, n_total, name_str, t_mae, h_mae, ratio)

        if i < args.max_plots:
            _save_panel(out_dir, name_str, orig_np, recon_np, err, tumour, ratio)

        utils.clear_cache()

    if not ratios:
        log.error("No slices with both tumour and healthy voxels — cannot report.")
        return

    summary = {
        "n_images": len(ratios),
        "vae_ckpt": C.VAE_CKPT,
        "tumour_mae_mean": float(np.mean(tumour_maes)),
        "healthy_mae_mean": float(np.mean(healthy_maes)),
        "ratio_mean": float(np.mean(ratios)),
        "ratio_median": float(np.median(ratios)),
        "ratio_std": float(np.std(ratios)),
        "per_image": per_image,
    }
    with open(out_dir / "vae_fidelity.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("VAE FIDELITY SUMMARY over %d slices", summary["n_images"])
    log.info("  tumour  MAE (mean): %.4f", summary["tumour_mae_mean"])
    log.info("  healthy MAE (mean): %.4f", summary["healthy_mae_mean"])
    log.info("  tumour/healthy ratio: mean %.2f | median %.2f | std %.2f",
             summary["ratio_mean"], summary["ratio_median"], summary["ratio_std"])
    log.info("-" * 60)
    if summary["ratio_mean"] >= 1.5:
        log.info("VERDICT: ratio ≫ 1 → the VAE reconstructs tumour markedly worse "
                 "than healthy tissue.  Lesion detail is being lost in the codec; "
                 "a medical VAE (trained on ALL data) is justified.")
    elif summary["ratio_mean"] <= 1.15:
        log.info("VERDICT: ratio ≈ 1 → the VAE preserves tumour about as well as "
                 "healthy tissue.  The VAE is likely NOT the bottleneck; prioritise "
                 "the noise schedule / T_int / capacity instead.")
    else:
        log.info("VERDICT: borderline ratio → mild tumour-specific reconstruction "
                 "loss.  A medical VAE may help but is not clearly the top fix.")
    log.info("  Report → %s", out_dir / "vae_fidelity.json")
    log.info("=" * 60)


def _save_panel(out_dir, name, orig_np, recon_np, err, tumour, ratio):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(orig_np[1], cmap="gray")
    axes[0].set_title("Original (T2w)", fontweight="bold"); axes[0].axis("off")

    axes[1].imshow(recon_np[1], cmap="gray")
    axes[1].set_title("VAE Round-trip (T2w)", fontweight="bold"); axes[1].axis("off")

    im2 = axes[2].imshow(err, cmap="hot")
    axes[2].set_title("|orig − recon|", fontweight="bold"); axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(tumour, cmap="gray")
    axes[3].set_title("Tumour mask", fontweight="bold"); axes[3].axis("off")

    fig.suptitle(f"{name}   |   tumour/healthy MAE ratio: {ratio:.2f}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / f"vae_fidelity_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE encode→decode fidelity diagnostic")
    parser.add_argument("--n-images",  type=int, default=None, help="Number of test slices (default: all)")
    parser.add_argument("--max-plots", type=int, default=12,   help="Max side-by-side figures to save (default: 12)")
    args = parser.parse_args()
    main(args)
