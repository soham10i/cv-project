#!/usr/bin/env python
"""
Corpus-level failure diagnostics from the evaluate log + GT masks (no model).
=============================================================================

Explains the inter-patient variance and separates threshold-failures from
detection-failures, using only the per-slice numbers in the evaluate log and the
ground-truth masks (cheap; no GPU).

  fig_dice_vs_area.png   (5) per-slice Dice vs. lesion area, coloured by patient
  fig_dice_vs_auroc.png  (6) per-slice Dice vs. AUROC (threshold- vs detection-fail)
  prints a per-patient summary table.

Usage
-----
    python report_pdm/fig_failure_aggregate.py \
        --eval-log report_pdm/figures/evaluate_20260620_073724.log
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CONFIG  # noqa: E402

FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

EVAL_RE = re.compile(
    r"\[\d+/\d+\]\s+(\S+)\s*\|\s*AUROC\s+([\d.]+)\s*\|\s*AUPRC\s+([\d.]+)"
    r"\s*\|\s*DICE\s+([\d.]+)\s*\|\s*oracle\s+([\d.]+)"
)
PATIENT_RE = re.compile(r"(BraTS-PED-\d+)")


def parse_log(path: Path):
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = EVAL_RE.search(line)
        if m:
            rows.append((m[1], float(m[2]), float(m[3]), float(m[4]), float(m[5])))
    return rows


def lesion_area(stem: str, masks_dir: Path) -> float:
    f = masks_dir / f"{stem}.npy"
    if not f.exists():
        return float("nan")
    return float((np.load(f) > 0).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description="Corpus-level PDM failure diagnostics")
    ap.add_argument("--eval-log", required=True)
    ap.add_argument("--masks-dir", default=str(CONFIG.paths.masks_dir))
    args = ap.parse_args()

    rows = parse_log(Path(args.eval_log))
    if not rows:
        raise SystemExit("No per-slice lines parsed from the eval log.")
    masks_dir = Path(args.masks_dir)
    names = [r[0] for r in rows]
    auroc = np.array([r[1] for r in rows]); dice = np.array([r[3] for r in rows])
    oracle = np.array([r[4] for r in rows])
    patients = [PATIENT_RE.search(n).group(1) if PATIENT_RE.search(n) else "?" for n in names]
    areas = np.array([lesion_area(n, masks_dir) for n in names])
    have_area = np.isfinite(areas) & (areas > 0)

    uniq = sorted(set(patients))
    cmap = plt.get_cmap("tab10")
    colors = {p: cmap(i % 10) for i, p in enumerate(uniq)}

    # ---- (5) Dice vs lesion area ----
    if have_area.any():
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        for p in uniq:
            sel = np.array([pp == p for pp in patients]) & have_area
            if sel.any():
                ax.scatter(areas[sel], dice[sel], s=22, alpha=0.7,
                           color=colors[p], label=p.replace("BraTS-PED-", "P"))
        ax.set_xscale("log"); ax.set_xlabel("lesion area (GT voxels, log)")
        ax.set_ylabel("Dice @ calibrated $\\tau$")
        ax.set_title("Dice vs. lesion size"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIGS / "fig_dice_vs_area.png", dpi=160, bbox_inches="tight"); plt.close(fig)
        print("[ok] fig_dice_vs_area.png")

    # ---- (6) Dice vs AUROC (threshold- vs detection-failure) ----
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for p in uniq:
        sel = np.array([pp == p for pp in patients])
        ax.scatter(auroc[sel], dice[sel], s=22, alpha=0.7, color=colors[p],
                   label=p.replace("BraTS-PED-", "P"))
    ax.axvline(0.5, color="grey", ls=":", lw=0.8)
    ax.axvspan(0.8, 1.0, color="green", alpha=0.05)
    ax.text(0.82, ax.get_ylim()[1] * 0.9, "high AUROC,\nlow Dice =\nthreshold/artefact",
            fontsize=8, color="#27632a")
    ax.set_xlabel("slice AUROC"); ax.set_ylabel("Dice @ calibrated $\\tau$")
    ax.set_title("Dice vs. AUROC: threshold- vs detection-failure")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "fig_dice_vs_auroc.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("[ok] fig_dice_vs_auroc.png")

    # ---- per-patient summary ----
    print("\n================ PER-PATIENT (paste-ready) ================")
    print(f"{'patient':18s} {'n':>4} {'AUROC':>7} {'AUPRC':>7} {'Dice':>7} {'oracle':>7}")
    agg = defaultdict(list)
    for r, p in zip(rows, patients):
        agg[p].append(r)
    for p in uniq:
        rr = agg[p]
        a = np.mean([x[1] for x in rr]); pr = np.mean([x[2] for x in rr])
        d = np.mean([x[3] for x in rr]); o = np.mean([x[4] for x in rr])
        print(f"{p:18s} {len(rr):>4} {a:>7.3f} {pr:>7.3f} {d:>7.3f} {o:>7.3f}")
    print(f"{'OVERALL':18s} {len(rows):>4} {auroc.mean():>7.3f} "
          f"{np.mean([r[2] for r in rows]):>7.3f} {dice.mean():>7.3f} {oracle.mean():>7.3f}")
    print("===========================================================")
    print("Figures ->", FIGS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
