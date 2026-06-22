"""Generate report figures from real project data (no synthetic/AI content)."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Diffusion training curves from local_logs/.../metrics.csv
# ---------------------------------------------------------------------------
csv_path = ROOT / "local_logs/diffusion_20260618_134532/metrics.csv"
train_loss, val_loss, lr = {}, {}, {}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        ep = int(row["epoch"]); k = row["key"]; v = float(row["value"])
        if row["split"] == "train" and k == "loss":
            train_loss[ep] = v
        elif row["split"] == "train" and k == "lr":
            lr[ep] = v
        elif row["split"] == "val" and k == "denoise_loss":
            val_loss[ep] = v

ep = sorted(train_loss)
tl = [train_loss[e] for e in ep]
vl = [val_loss[e] for e in ep]
lrv = [lr[e] for e in ep]
best_ep = min(val_loss, key=val_loss.get)

fig, ax1 = plt.subplots(figsize=(7.2, 4.3))
ax1.plot(ep, tl, "o-", color="#1f4e79", lw=2, ms=4, label="Train MSE ($\\epsilon$-loss)")
ax1.plot(ep, vl, "s-", color="#c0392b", lw=2, ms=4, label="Val denoise MSE")
ax1.axvline(best_ep, color="grey", ls="--", lw=1.2)
ax1.annotate(f"best val = {val_loss[best_ep]:.4f}\n(epoch {best_ep})",
             xy=(best_ep, val_loss[best_ep]), xytext=(best_ep + 1.5, val_loss[best_ep] + 0.05),
             fontsize=9, arrowprops=dict(arrowstyle="->", color="grey"))
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Noise-prediction MSE")
ax1.set_xticks(ep[::2])
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(ep, lrv, ":", color="#27ae60", lw=1.6, label="Learning rate")
ax2.set_ylabel("Learning rate", color="#27ae60")
ax2.tick_params(axis="y", labelcolor="#27ae60")
l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "fig_training_curves.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("wrote fig_training_curves.png ; best_ep", best_ep, val_loss[best_ep])

# ---------------------------------------------------------------------------
# 2. Qualitative evaluation montage (4 representative axial slices)
# ---------------------------------------------------------------------------
panels = ["z065", "z071", "z075", "z080"]
imgs = []
for z in panels:
    p = ROOT / f"local_logs/evaluation/eval_BraTS-PED-00029-000_{z}.png"
    imgs.append(Image.open(p).convert("RGB"))
w = min(im.width for im in imgs)
imgs = [im.resize((w, int(im.height * w / im.width))) for im in imgs]
H = sum(im.height for im in imgs)
montage = Image.new("RGB", (w, H), "white")
y = 0
for im in imgs:
    montage.paste(im, (0, y)); y += im.height
montage.save(OUT / "fig_eval_montage.png")
print("wrote fig_eval_montage.png", montage.size)
