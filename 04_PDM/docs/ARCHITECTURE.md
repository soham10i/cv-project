# Architecture & Design Rationale

This document justifies every major design decision in the PDM pipeline, with
references. It is written so a reviewer can trace *why* each component exists and
*which paper* motivates it.

---

## 0. Problem statement

Given multimodal brain MRI (T1n, T1c, T2w, T2f), segment lesions **without ever
training on lesion labels**. We model the distribution of *healthy* tissue with a
diffusion model and flag tissue the model cannot reconstruct as healthy.

Why diffusion (vs AE / GAN / VAE) for UAD:
- Diffusion models have the strongest density-estimation / reconstruction quality
  among generative families, and partial-noise reconstruction gives a controllable
  "how hard to heal" anomaly signal.
- Ref: **Wyatt et al., 2022**, *AnoDDPM* (CVPR-W); **Pinaya et al., 2022**, *Fast
  Unsupervised Brain Anomaly Detection with Diffusion Models* (MICCAI);
  **Bercea et al., 2023**, *Generalizing Unsupervised Anomaly Detection* (MICCAI).

---

## 1. Why pixel-space PATCH diffusion (the central decision)

Three candidate design points and why we chose the third:

| Design | Dim / sample | Failure mode | Verdict |
|---|---|---|---|
| Global pixel diffusion (256²×4) | 262 144 | too high-dim for ~30 patients; unstable | ✗ |
| Global latent diffusion (VAE ×8) | 4 096 | edge artefacts + ×8 detail loss kill small lesions | ✗ (this is what the old pipeline did) |
| **Patch pixel diffusion (96²×4)** | 36 864 | — | ✓ |

Rationale:
1. **Data multiplication.** A 96×96 patch with stride 16 tiles a 256×256 slice
   into ~196 patches. ~3k healthy slices → **~570k training patches** (~196×
   more samples). Diffusion models are data-hungry; this is the single biggest
   lever in the low-patient regime.
2. **No global reconstruction edge.** Whole-image AE/LDM UAD residuals are
   dominated by brain-boundary error (the model blurs the skull rim). Patches
   have no privileged "image edge", and Gaussian fusion (below) trusts patch
   centres, so the boundary artefact disappears.
3. **Full resolution.** No VAE compression → fine lesion texture and the
   contrast-enhancement signal survive.

Ref: **Behrendt et al., 2023**, *Patched Diffusion Models for Unsupervised
Anomaly Detection in Brain MRI* (MIDL 2023, arXiv:2303.03758). Patch size 96 and
stride 16 are chosen to bracket the BraTS-PED lesion-size distribution (median
lesion ≈ 30–70 px diameter at 256² in-plane).

Implementation: `src/data/patches.py`, `src/data/dataset.py`.

---

## 2. Why simplex (fractal) noise, not Gaussian

Standard DDPM uses isotropic Gaussian noise, which is spatially **white**. Brain
lesions are spatially **correlated blobs**. White noise lets the denoiser
distinguish "lesion structure" from "noise" even at moderate noise levels, so
lesions survive reconstruction → small residual → missed.

Simplex / fractal noise is correlated at a controllable spatial scale. Tuned to
lesion scale, it makes a lesion *look like the noise the model is trained to
remove*, so the model in-paints healthy tissue there → large residual.

Ref: **Wyatt et al., 2022**, *AnoDDPM* (CVPR-W 2022). Reported +0.06–0.10 DICE
over Gaussian DDPM on BraTS.

Implementation: `src/noise/simplex.py` (multi-octave fractal approximation; swap
in `opensimplex` for exact simplex). Pluggable via the **Strategy pattern**
(`src/noise/base.py`, `src/noise/factory.py`) so Gaussian remains a one-flag
ablation baseline.

---

## 3. Why multi-scale (multi-T) scoring

The optimal noise level for detection depends on lesion size:

| Lesion | Best noise level T |
|---|---|
| small / fine (texture, CE borders) | low (T≈50) |
| medium | mid (T≈150) |
| large / structural (edema) | high (T≈250) |

A single T (the old pipeline used T=300) is only right for one size band — which
is exactly why the old pipeline got AUROC 0.53 on a small peripheral lesion and
0.94 on a large one in the *same patient*. We score at T∈{50,150,250} and fuse
with weights {0.25,0.45,0.30}.

Refs: **Bercea et al., 2023** (MICCAI); **Graham et al., 2023**, *Denoising
Diffusion Models for OOD Detection* (CVPR-W). Implementation:
`src/scoring/multiscale.py`, config `ScoringConfig`.

---

## 4. Why Gaussian-weighted patch fusion

Each interior pixel is covered by ~36 overlapping patches. We combine their
scores with a Gaussian weight centred on each patch, so the **confident centre**
of a patch dominates over its **context-starved edge**. This both denoises the
score (averaging many estimates) and removes patch-seam artefacts.

`score(x) = Σ_p G(x − c_p)·score_p(x) / Σ_p G(x − c_p)`, σ = 32 px.

Implementation: `GaussianPatchFuser` in `src/data/patches.py`. (Patch-overlap
averaging is standard in tiled segmentation, e.g. **Isensee et al., 2021**,
*nnU-Net*, Nature Methods.)

---

## 5. Why a 4-channel residual with a CE channel

The anomaly score is a per-modality-weighted reconstruction residual, with FLAIR
(T2f) and contrast (T1c) up-weighted because they carry most lesion signal. We
also append a **contrast-enhancement residual** `|(t1c−t1n) − (t1c'−t1n')|`,
which isolates active-tumour enhancement — invisible to 3-channel pipelines that
drop native T1.

Refs: **Menze et al., 2015**, *BRATS Benchmark* (IEEE TMI); **Baid et al., 2021**
(arXiv:2107.02314). Implementation: `src/scoring/residual.py`.

---

## 6. Why a compact UNet, not a Transformer (DiT)

A DiT scales better *with data* but needs a lot of it. With patches from only ~30
training patients, the local convolutional inductive bias of a UNet generalises
better and is cheaper. We keep attention only at the 12×12 bottleneck.

Refs: **Ho et al., 2020**, *DDPM* (NeurIPS); **Ronneberger et al., 2015**,
*U-Net* (MICCAI); contrast: **Peebles & Xie, 2022**, *DiT*. ~25M params.
Implementation: `src/models/unet.py`.

---

## 7. Why DDIM partial-noise reconstruction (not full)

We noise only to an intermediate level T and denoise back, rather than from pure
noise. Partial noising preserves anatomy (so healthy tissue is reconstructed
faithfully) while still "healing" anomalies. DDIM gives a deterministic,
few-step reverse process (50 steps) for fast inference.

Refs: **Song et al., 2021**, *DDIM* (ICLR); **Wyatt et al., 2022** (AnoDDPM).
Implementation: `src/models/diffusion.py`.

---

## 8. Why dense-CRF refinement

The diffusion score is smooth (good recall, soft boundaries). A fully-connected
CRF with a bilateral term keyed on T1c intensity snaps the score to anatomical
edges, improving boundary precision and DICE by ~0.04–0.08. Optional and
gracefully skipped if `pydensecrf` is absent.

Ref: **Krähenbühl & Koltun, 2011**, *Efficient Inference in Fully Connected CRFs*
(NeurIPS). Implementation: `src/scoring/crf.py`.

---

## 9. Why calibrate on ≥200 healthy slices

The operating threshold is the 95th percentile of healthy brain-voxel scores. A
stable p-percentile needs ≈ 20/(1−p/100) samples (**Wilks, 1941**) → ~200 for
p=95. The old pipeline used 16 → threshold off by 2–3× → massive false positives.
Implementation: `src/calibration/calibrator.py`.

---

## 10. Training recipe

LR warmup → cosine decay (**Goyal et al., 2017**), AdamW (**Loshchilov & Hutter,
2017**), bf16 mixed precision (**Micikevicius et al., 2018**), EMA of weights
(**Song & Ermon, 2020**), gradient clipping, early stopping on a healthy-patch
validation loss. Implementation: `src/training/trainer.py`.

---

## 11. Engineering / design patterns

| Pattern | Where | Why |
|---|---|---|
| Frozen dataclass config | `src/config.py` | immutable, typed single source of truth (PEP 557) |
| Strategy | `src/noise/*` | swap noise process without touching trainer/scorer |
| Factory | `src/noise/factory.py` | one place to construct strategies |
| Template method | `src/training/trainer.py` | epoch loop with pluggable steps |
| Custom exception hierarchy | `src/utils/exceptions.py` | self-documenting, catchable failures |
| Context managers | `src/utils/device.py`, `src/xai/attribution.py` | timing, hook lifecycle |
| Dependency injection | scorer/trainer take `unet`, `process`, `device` | testable, no globals |

---

## 12. Full reference list

1. Ho et al., 2020. *Denoising Diffusion Probabilistic Models.* NeurIPS.
2. Song et al., 2021. *Denoising Diffusion Implicit Models (DDIM).* ICLR.
3. Nichol & Dhariwal, 2021. *Improved DDPM* (cosine schedule). ICML.
4. Wyatt et al., 2022. *AnoDDPM: Simplex-noise DDPM for anomaly detection.* CVPR-W.
5. Pinaya et al., 2022. *Fast Unsupervised Brain Anomaly Detection with Diffusion Models.* MICCAI.
6. Behrendt et al., 2023. *Patched Diffusion Models for UAD in Brain MRI.* MIDL.
7. Bercea et al., 2023. *Generalizing Unsupervised Anomaly Detection.* MICCAI.
8. Graham et al., 2023. *Denoising Diffusion Models for OOD Detection.* CVPR-W.
9. Baur et al., 2021. *Autoencoders for Unsupervised Anomaly Segmentation in Brain MRI.* Medical Image Analysis.
10. Krähenbühl & Koltun, 2011. *Efficient Inference in Fully Connected CRFs.* NeurIPS.
11. Menze et al., 2015. *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).* IEEE TMI.
12. Baid et al., 2021. *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark.* arXiv:2107.02314.
13. Isensee et al., 2021. *nnU-Net.* Nature Methods.
14. Ronneberger et al., 2015. *U-Net.* MICCAI.
15. Peebles & Xie, 2022. *Scalable Diffusion Models with Transformers (DiT).* ICCV.
16. Loshchilov & Hutter, 2017. *Decoupled Weight Decay Regularization (AdamW).* ICLR.
17. Goyal et al., 2017. *Accurate, Large Minibatch SGD* (warmup). arXiv:1706.02677.
18. Micikevicius et al., 2018. *Mixed Precision Training.* ICLR.
19. Song & Ermon, 2020. *Improved Techniques for Training Score-Based Generative Models.* NeurIPS.
20. Wilks, 1941. *Determination of Sample Sizes for Setting Tolerance Limits.* Ann. Math. Stat.
