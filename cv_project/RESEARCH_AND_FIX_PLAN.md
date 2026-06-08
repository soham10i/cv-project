# Unsupervised Brain-Tumour Anomaly Detection with Latent Diffusion — Deep Research & Fix Plan

**Scope:** what to change to make the BraTS unsupervised-anomaly-detection goal actually *work*, written
as an expert CV / medical-image-processing engineer would approach it: complete pipeline + explainable AI.

**Method:** grounded in the actual code on `claude/modest-heisenberg-7cf6mc` (identical to `version-2`
today), the run logs (`train_full.log`, `eval_full.log`), `results/metrics.json`, and the
`BraTS_Anomaly_Detection_Workflow.pdf` design spec. Cross-checked against the LDM-anomaly literature
(Wolleb 2022, Pinaya 2022, Bercea AutoDDPM 2023, Graham 2023, Behrendt 2024 pDDPM/mDDPM).

---

## 0. TL;DR — the headline

Your pipeline is **80% correct and one calibration bug away from looking broken.**

| Metric (current run, 2000 test slices) | Value | Verdict |
|---|---|---|
| AUROC (pixel, threshold-free) | **0.788 ± 0.150** | ✅ in the published LDM range (0.72–0.82) |
| AUPRC | 0.119 | ⚠️ low — heavy false positives |
| **DICE @ calibrated threshold** | **0.000** | ❌ **broken — this is a bug, not model failure** |
| DICE @ best single global threshold (0.11) | 0.175 | the map *does* carry signal |
| DICE oracle (per-slice best) | 0.206 | ceiling of the current map |

The model learned a usable healthy manifold (AUROC proves the residual ranks tumour > healthy tissue).
**DICE is 0 purely because the threshold is mis-calibrated** — an image-level detection cutoff is being
applied as a per-pixel segmentation cutoff. Fix that first; it costs ~10 lines and turns 0.00 into ~0.20+
with no retraining.

Then a second, larger gap: the model is **massively undertrained vs. its own spec** (30 epochs run vs.
3000 specified), and the **dataset is pediatric (BraTS-PEDs), not the adult BraTS 2021 the PDF describes** —
which changes every downstream assumption. Closing those lifts the ceiling from ~0.20 DICE toward the
0.35–0.55 the design targets.

---

## 1. Root cause of `DICE = 0.0` (fix this first)

**The bug, traced through the code:**

1. `utils.calibrate_on_healthy` builds the operating threshold from the **per-slice maximum** residual on
   *healthy* slices:
   ```python
   # utils.py:469
   max_pixel = [m.max() for m in pixel_maps]          # one scalar per healthy slice
   # utils.py:480
   "threshold_pixel": float(np.percentile(max_pixel, percentile)),   # 99th pct of those maxima → 0.9904
   ```
2. Evaluation applies that scalar as a **per-pixel** segmentation threshold:
   ```python
   # evaluate_pipeline.py:215
   pred_bin = ((score > thr) .astype(np.float32)) * bmask    # thr = 0.9904
   ```
3. Tumour-pixel residuals (after `M_baseline` subtraction, in [-1,1] space, channel-averaged) live around
   **0.11** (that's exactly where `global_best_threshold` landed). Almost nothing exceeds **0.99**, so
   `pred_bin` is empty → **DICE = 0** on every slice.

**Why AUROC is fine but DICE is zero:** AUROC is threshold-free; it only needs tumour pixels to *rank*
above healthy pixels, which they do. DICE needs a *correct cutoff*. The 99th-percentile-of-maxima statistic
answers a different question ("is this slice anomalous at all?" — image-level detection), not "which pixels
are tumour?" (pixel-level segmentation). Two different operating points were conflated.

**Minimal fix (no retraining):** calibrate the pixel threshold from the distribution of **healthy
brain-voxel residual values**, not per-slice maxima.

```python
# in utils.calibrate_on_healthy, replace the max_pixel block:
all_healthy_vox = np.concatenate([m[bm > 0].ravel()
                                  for m, (_, _, bm) in zip(pixel_maps, cache)])
calib["threshold_pixel"] = float(np.percentile(all_healthy_vox, 99.0))  # ~0.10–0.13 expected
# keep the old per-slice-max value too, but name it honestly:
calib["threshold_detect_imagelevel"] = float(np.percentile(max_pixel, percentile))
```

That single change should reproduce ≈ the `global_best_dice` (~0.175) automatically, with the threshold
*derived from healthy data only* (still fully unsupervised — you never touch tumour labels).

**Even better — per-image normalisation before thresholding.** Residual magnitude drifts slice-to-slice
(brain size, slice position, contrast). Standardise each map by its own healthy statistics so one global
threshold is valid everywhere:

```python
# z-score the residual inside the brain, using healthy-calibrated mean/std,
# then threshold in "sigmas above healthy" units (e.g. 2.5–3σ)
score_z = (m_pixel - healthy_mu) / (healthy_sigma + 1e-8)
pred_bin = (score_z > 3.0) * bmask
```

This is the standard trick in Behrendt et al. (pDDPM/mDDPM) and consistently beats a single raw cutoff.

---

## 2. Code vs. design-spec divergences (ranked)

| # | Area | Spec (PDF) | Actual code/run | Impact | Priority |
|---|---|---|---|---|---|
| 1 | Threshold | percentile of healthy residuals | percentile of per-slice **maxima** (`utils.py:469`) | **DICE=0** | **P0** |
| 2 | Training length | 3000 epochs, loss < 0.01 | **30 epochs**, loss **0.073** (`train_full.log`) | undertrained manifold → blurry recon, weak localisation | **P0** |
| 3 | Dataset | BraTS **2021 adult**, 1251 pts, ~20k healthy slices | **BraTS-PEDs** pediatric, 257 pts, **5000** healthy slices capped (`preprocess` default `max_healthy=50`, run used 5000) | pediatric tumours (diffuse midline/DIPG) enhance poorly on T1ce → core modality assumption weakens; far less data | **P0** |
| 4 | T_int | **sweep** {200,300,400,500,600}, pick max-AUROC | hardcoded `T_INT=300`, no sweep script exists | leaving AUROC/DICE on the table | **P1** |
| 5 | EMA decay | 0.9999 | 0.999 (`config.py:89`) | fine for 30 epochs; revisit if epochs ↑ | P2 |
| 6 | UNet channels | (128,256,512,512) | (128,256,**384**,512) (`train_…:82`) | minor capacity diff | P2 |
| 7 | Batch size | 32 | 8 (8 GB GPU) | slower convergence; OK with grad-accum | P2 |
| 8 | XAI | self-attention overlay | implemented but is *correlational*, not OOD evidence | weak as "explainability" claim | **P1** |
| 9 | Modalities | T1ce+T2+FLAIR as 3ch into RGB SD-VAE | same (`config.py:48` `t1c,t2w,t2f`) | SD-VAE is RGB-trained; per-modality fidelity uneven | P1 |

---

## 3. The DICE-vs-AUROC story, in one paragraph (for your defence)

> *"AUROC measures whether the anomaly score separates tumour from healthy tissue — it is threshold-free and
> reached 0.79, on par with published latent-DDPM results. DICE additionally requires committing to an
> operating threshold. Our first run calibrated that threshold from the wrong statistic (the per-image
> maximum residual, which is a detection cutoff), so the segmentation collapsed to empty masks. After
> re-deriving the threshold from the distribution of healthy residual values — still using zero tumour
> labels — DICE recovered to the model's true operating point. The remaining gap to literature DICE is
> closed by training to convergence and sweeping T_int."*

---

## 4. How I'd build it end-to-end (expert pipeline)

### Phase 1 — Data (the part that silently decides everything)
- **Decide the dataset on purpose.** If the academic goal is the PDF's adult-glioma story, get **BraTS 2021
  adult**. If you're committed to **BraTS-PEDs**, rewrite the narrative: pediatric high-grade gliomas are
  often *non-enhancing*, so T1ce is a weaker anomaly channel — lead with **FLAIR + T2** and treat T1ce as
  auxiliary. Don't mix a pediatric dataset with an adult-glioma justification; reviewers will catch it.
- **Skull-strip & register** (HD-BET + rigid to SRI24) if not already done — BraTS volumes are
  co-registered/skull-stripped, but PEDs preprocessing varies. Confirm.
- **N4 bias-field correction** per modality before z-scoring. Bias fields create smooth intensity ramps the
  diffusion model will flag as "anomalies."
- **Foreground z-score** (already done, `preprocess_to_2d.py:42`) — good. Keep background exactly 0.
- **Healthy/anomalous split is currently slice-level within the same patients.** For a clean unsupervised
  claim, split **by patient**: healthy slices for training must come from patients *not* in the test set,
  or at minimum guarantee no leakage of patient-specific anatomy. (The MedSegDiff side already splits by
  patient — mirror that.)
- **Volume context:** 2D loses through-plane continuity. A cheap, high-ROI upgrade is **2.5D** (stack
  z-1,z,z+1 as input context) — Behrendt's pDDPM shows this sharpens localisation.

### Phase 2 — VAE encoding
- Keep frozen `sd-vae-ft-mse` (correct call; training a KL-VAE is a time sink). Two caveats:
  - It's **RGB-trained**; T1ce/T2/FLAIR-as-RGB reconstructs unevenly. **Run a per-modality reconstruction
    sanity check** (`test_vae_reconstruction.py` exists — report PSNR/SSIM per channel). If FLAIR
    reconstructs poorly, that *directly* caps anomaly fidelity.
  - The `0.18215` scaling (`config.py:57`) is correct and applied (`utils.py:156`). ✅
- **Encode with the posterior mean** (`sample=False`) for inference — already done (`utils.py:155`). ✅

### Phase 3 — Diffusion training (the real lever)
- **Train to convergence.** 30 → at least **800–1500 epochs** on this data size, or until val-recon loss
  plateaus. Current loss 0.073 means the noise predictor is still coarse; reconstructions will be blurry
  and the residual map will be dominated by *reconstruction error*, not anomaly. This is the single biggest
  quality lever after the threshold fix.
- Bump **EMA to 0.9999** once epochs are long, restore **(128,256,512,512)**, keep grad-clip 1.0, cosine LR.
- Add **gradient accumulation** to reach effective batch 32 on 8 GB.
- Log **val reconstruction samples + a held-out AUROC every N epochs** (not just train loss) so you stop at
  the right time, not a fixed epoch count.
- Consider **`v_prediction`** + zero-terminal-SNR; it stabilises sample quality and often helps medical
  recon over `epsilon`.

### Phase 4 — Anomaly inference
- **Implement the T_int sweep** (it's specified but missing). Sweep {150,250,350,450,550} on a 100-slice
  *labelled* val subset, pick max pixel-AUROC, freeze it. Expect the optimum to move once the model is
  trained longer.
- **Upgrade the reconstruction strategy** beyond single partial-noise:
  - **AutoDDPM (Bercea 2023):** mask the high-residual region, stitch original back into low-residual
    region, re-noise and re-sample → far fewer false positives at tissue boundaries. This is the biggest
    AUPRC/DICE win available and matches the reference you already cite.
  - **Ensemble over noise seeds** (4–8 samples, average residual): DDIM is deterministic given a seed, but
    averaging over *different* injected-noise seeds reduces residual variance. (The PDF argues DDIM
    determinism removes variance — true for one seed, but seed averaging still helps localisation.)
- **Post-process the residual** before scoring: median filter (3×3) → remove specks; morphological opening;
  drop connected components < N pixels. Tumours are contiguous; single-pixel residuals are noise. This
  alone should lift AUPRC noticeably.

### Phase 5 — Evaluation & XAI (below)

---

## 5. Explainable AI — do it properly

The current SAAM (self-attention overlay, `utils.py:196–324`) is honest but weak: the PDF itself admits it
shows *"spatial consistency, not a proof of OOD detection."* For a medical-imaging defence, add evidence
that is **causal/quantitative**, not just a pretty heatmap:

1. **The residual map IS the primary explanation.** Frame `M(x) = |x − x̂_healthy|` as a
   counterfactual: *"here is what healthy tissue would look like, here is the difference."* That's the most
   defensible XAI in this setup — show original / healthy-reconstruction / residual / GT side by side
   (your `_save_panel` already does this — lead with it).
2. **Per-modality residual decomposition.** Split the residual into T1ce/T2/FLAIR channels and show which
   modality drives the detection. Clinically meaningful (enhancement vs. oedema vs. infiltration).
3. **Calibrated uncertainty.** Residual-variance across noise seeds → an uncertainty map. High score + low
   variance = confident anomaly; high variance = "model unsure." Reviewers love this.
4. **Faithfulness check (quantitative XAI):** deletion/insertion curves — progressively inpaint the
   top-k residual pixels with healthy tissue and show the image-level anomaly score drops monotonically.
   This *proves* the highlighted region is what drives the decision, unlike attention.
5. **Keep attention as a secondary, clearly-labelled view** ("network attention," not "anomaly
   evidence"). Restrict to the 16×16 layer (already done via `MIN_SEQ_LEN=256`, `config.py:84`). ✅

---

## 6. Evaluation protocol fixes

- **Report all three DICE numbers** (calibrated / global-best / oracle) — you already compute them; the
  oracle is the *map ceiling*, the calibrated one is the *honest operating point*. Don't report only one.
- **Add image-level detection metrics** (slice-wise AUROC: "does this slice contain a tumour?") — that's
  the question the per-slice-max threshold *actually* answers, and it'll look strong.
- **Compute DICE only over slices with tumour** (already filtered) but also report **specificity on healthy
  slices** (false-positive rate) — unsupervised methods must not hallucinate tumours in healthy brains.
- **Use connected-component / lesion-wise detection** (detected if IoU>0.1 with a GT lesion), not only
  pixel DICE — standard in BraTS anomaly papers and more forgiving of boundary slop.
- **Bootstrap CIs** on AUROC/DICE over patients (not slices) to avoid optimistic variance.

---

## 7. Concrete action checklist

**P0 — today, no retraining (turns 0.00 into a real number):**
- [ ] Fix `utils.calibrate_on_healthy` threshold (§1): percentile of healthy *voxel residuals*, not maxima.
- [ ] Re-run `evaluate_pipeline.py`; confirm DICE ≈ 0.17–0.21.
- [ ] Add per-image z-score normalisation option and 3×3 median filter on the residual.

**P0 — retrain (lifts the ceiling):**
- [ ] Train 800–1500 epochs (or val-AUROC plateau), EMA 0.9999, channels (128,256,512,512), eff. batch 32.
- [ ] Decide dataset story: commit to BraTS-PEDs (FLAIR-led narrative) **or** switch to BraTS 2021 adult.

**P1 — quality:**
- [ ] Add `sweep_tint.py` (specified, missing); freeze best T_int.
- [ ] Implement AutoDDPM mask-stitch-resample inference.
- [ ] Add connected-component post-processing + lesion-wise detection metric.
- [ ] VAE per-modality reconstruction sanity report.

**P1 — XAI:**
- [ ] Per-modality residual panel + seed-variance uncertainty map.
- [ ] Deletion/insertion faithfulness curve.

**P2:**
- [ ] 2.5D context input; `v_prediction` + zero-terminal-SNR; patient-level healthy/test split audit.

---

## 8. References anchoring these choices
- Wolleb et al., *Diffusion Models for Medical Anomaly Detection*, MICCAI 2022 — healthy-only manifold.
- Pinaya et al., *Brain Imaging Generation with LDMs*, MICCAI-W 2022 — KL-VAE + LDM in latent space.
- Bercea et al., *Mask, Stitch, Re-Sample (AutoDDPM)*, 2023 — partial noising + stitch (your §4 upgrade).
- Behrendt et al., *Patched/Masked DDPM (pDDPM/mDDPM)*, 2023–24 — 2.5D context, per-image normalisation.
- Graham et al., *Denoising Diffusion Models for OOD Detection*, CVPR-W 2023 — multi-T_int scoring.
- Ho 2020 (DDPM), Song 2020 (DDIM) — schedulers.
