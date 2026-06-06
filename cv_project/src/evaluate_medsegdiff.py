"""
MedSegDiff Phase 4 — Evaluate conditional diffusion segmentation
=================================================================
Samples a tumour mask from noise (conditioned on the MRI) at the training
resolution, upsamples the prediction to the stored 256² size, and scores DICE
/ IoU against the ground-truth masks on the held-out (patient-disjoint) test
set.

Usage
-----
    python src/evaluate_medsegdiff.py                 # all test slices
    python src/evaluate_medsegdiff.py --n-images 200  # subset
"""

import argparse
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import config as C
import utils
from medsegdiff_model import build_medsegdiff
from medsegdiff_utils import sample_mask, dice_score, iou_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["arch"]
    model = build_medsegdiff(img_size=a["img_size"], base=a["base"],
                             ch_mult=tuple(a["ch_mult"]), attn_res=tuple(a["attn_res"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log.info("Loaded %s (val DICE %.4f, %d² )", ckpt_path.name,
             ckpt.get("val_dice", float("nan")), a["img_size"])
    return model, a["img_size"]


def _overlay(name, img256, gt256, pred256, dice):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(img256[1], cmap="gray"); ax[0].set_title("MRI (T2w)", fontweight="bold")
    ax[1].imshow(gt256, cmap="gray"); ax[1].set_title("Ground Truth", fontweight="bold")
    ax[2].imshow(pred256, cmap="gray"); ax[2].set_title("Prediction", fontweight="bold")
    ax[3].imshow(img256[1], cmap="gray")
    ax[3].contour(gt256, colors="lime", linewidths=1.0)
    ax[3].contour(pred256, colors="red", linewidths=1.0)
    ax[3].set_title("GT (green) vs Pred (red)", fontweight="bold")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{name}   |   DICE {dice:.4f}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(C.SEG_RESULTS_DIR / f"seg_{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate(args):
    utils.set_seed()
    device = utils.get_device()
    generator = utils.make_generator(device)
    C.SEG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = C.SEG_MODEL_DIR / args.ckpt
    model, train_size = load_model(ckpt, device)
    ddim = utils.make_ddim_scheduler()

    img_dir = C.SEG_DATA_DIR / "test" / "images"
    mask_dir = C.SEG_DATA_DIR / "test" / "masks"
    files = sorted(img_dir.glob("*.npy"))
    if args.n_images:
        files = files[:args.n_images]
    log.info("Test slices: %d | ensemble %d | %d DDIM steps", len(files), args.ensemble, args.steps)

    per_image, dices, ious = [], [], []
    dices_pos = []                       # tumour-bearing slices only
    n_plot = 0

    for i in range(0, len(files), args.bs):
        batch = files[i:i + args.bs]
        imgs = np.stack([np.load(f).astype(np.float32) for f in batch])        # (B,3,256,256)
        gts  = np.stack([np.load(mask_dir / f.name) for f in batch])           # (B,256,256)

        img256 = torch.from_numpy(imgs).to(device)
        img_ds = F.interpolate(img256, size=train_size, mode="bilinear", align_corners=False)

        prob = sample_mask(model, ddim, img_ds, steps=args.steps,
                           ensemble=args.ensemble, generator=generator)        # (B,1,s,s)
        prob256 = F.interpolate(prob, size=C.SEG_IMG_SIZE, mode="bilinear", align_corners=False)
        pred256 = (prob256[:, 0] > 0.5).cpu().numpy().astype(np.uint8)

        for b, f in enumerate(batch):
            gt = (gts[b] > 0).astype(np.uint8)
            d = dice_score(pred256[b], gt)
            j = iou_score(pred256[b], gt)
            dices.append(d); ious.append(j)
            if gt.sum() > 0:
                dices_pos.append(d)
            per_image.append({"name": f.stem, "dice": d, "iou": j, "gt_tumor": int(gt.sum())})
            if n_plot < args.max_plots:
                _overlay(f.stem, imgs[b], gt, pred256[b], d)
                n_plot += 1

        if (i // args.bs) % 10 == 0:
            log.info("  %d/%d  running DICE(all) %.4f | DICE(tumour) %.4f",
                     min(i + args.bs, len(files)), len(files),
                     float(np.mean(dices)), float(np.mean(dices_pos)) if dices_pos else 0.0)

    summary = {
        "n_images": len(per_image),
        "n_tumor_slices": len(dices_pos),
        "dice_all_mean": float(np.mean(dices)), "dice_all_std": float(np.std(dices)),
        "dice_tumor_mean": float(np.mean(dices_pos)) if dices_pos else float("nan"),
        "dice_tumor_std": float(np.std(dices_pos)) if dices_pos else float("nan"),
        "iou_mean": float(np.mean(ious)),
        "ensemble": args.ensemble, "steps": args.steps, "train_size": train_size,
        "per_image": per_image,
    }
    with open(C.SEG_RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("MedSegDiff test results over %d slices (%d with tumour):",
             summary["n_images"], summary["n_tumor_slices"])
    log.info("  DICE (tumour slices): %.4f ± %.4f", summary["dice_tumor_mean"], summary["dice_tumor_std"])
    log.info("  DICE (all slices)   : %.4f ± %.4f", summary["dice_all_mean"], summary["dice_all_std"])
    log.info("  IoU  (all slices)   : %.4f", summary["iou_mean"])
    log.info("  Metrics → %s", C.SEG_RESULTS_DIR / "metrics.json")
    log.info("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MedSegDiff Phase 4 — evaluation")
    p.add_argument("--ckpt", type=str, default="medsegdiff_ema_best.pt")
    p.add_argument("--n-images", type=int, default=None)
    p.add_argument("--max-plots", type=int, default=12)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--steps", type=int, default=C.DDIM_STEPS)
    p.add_argument("--ensemble", type=int, default=C.SEG_ENSEMBLE)
    main = evaluate
    main(p.parse_args())
