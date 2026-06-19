# Model Architecture — what we use and why it beats the old pipeline

This document explains the two trained models (the **KL-VAE** codec and the
**latent DDPM UNet**), how they fit together, and — point by point — why each
choice is better than the previous `cv-project` implementation. Use it for the
report's *Method* section.

---

## 1. The system at a glance

```
MRI slice (4×256×256)                                   anomaly map (256×256)
        │                                                        ▲
        ▼                                                        │
   ┌─────────┐   z0 (4×32×32)   ┌──────────────────┐   ẑ0   ┌─────────┐
   │ KL-VAE  │ ───────────────▶ │  add noise @ T   │ ────▶  │ KL-VAE  │
   │ encoder │                  │  DDIM denoise    │        │ decoder │
   └─────────┘                  │  (healthy UNet)  │        └─────────┘
                                └──────────────────┘
                                 learns p(healthy latent)
```

- The **VAE** compresses a 4-modality slice into a small `4×32×32` latent and
  back — an ×8 spatial compression that makes diffusion tractable.
- The **UNet DDPM** learns the distribution of **healthy** latents. Partially
  noising a test latent and denoising it "pulls" anomalies toward the healthy
  manifold; the **decoded residual** localizes the lesion.

This is the **Latent Diffusion** UAD paradigm (Rombach et al. 2022; AnoDDPM,
Wyatt et al. 2022; pDDPM/AutoDDPM, Behrendt/Bercea 2023).

---

## 2. Model A — the medical KL-VAE  (`src/models/kl_vae.py`)

A from-scratch continuous (KL-regularized) autoencoder, CompVis/LDM-style but
native to MRI.

| Spec | Value |
|---|---|
| Input / output channels | **4** (`t1n, t1c, t2w, t2f`) |
| Spatial compression | 256 → 32 (×8), 3 downsample stages |
| Latent | `4 × 32 × 32`, diagonal-Gaussian posterior |
| Blocks | ResNet (GroupNorm + SiLU), self-attention at the 32² bottleneck |
| Params | ~18 M |
| Loss | `L1 + 0.5·(1−MS-SSIM) + 1e-6·KL` |

**Components** (mirrors the canonical LDM autoencoder, scaled down):
`Encoder` (conv-in → 4 ResNet down-stages → mid ResNet+Attn → 2·z conv) →
`quant_conv` → `DiagonalGaussian` (reparameterised sample) → `post_quant_conv`
→ `Decoder` (mid ResNet+Attn → 4 ResNet up-stages → conv-out).

### Why it beats the old VAE (Stable Diffusion `sd-vae-ft-mse`)

| | Old: SD `sd-vae-ft-mse` | New: from-scratch KL-VAE |
|---|---|---|
| Domain | Natural RGB photos | **Brain MRI, trained from scratch** |
| Channels | 3 (RGB) → **T1-native dropped** | **4** incl. T1-native (enables CE = t1c−t1n) |
| Inductive bias | Color/texture priors of photographs | None foreign to MRI |
| Latent scale | SD's, fixed | **Measured on our data** (see §4) |
| Lesion fidelity | Blurs unfamiliar texture | Trained on lesions too → faithful codec |

The core defect of the old design: a photograph autoencoder's encoder statistics
and channel coupling are tuned for RGB semantics. Forcing 3 MRI modalities into
RGB channels distorts the latent geometry the diffusion model then has to learn.
Fine-tuning bends it slightly; it does not remove the bias. Training the codec
from scratch on MRI removes it entirely — the single biggest correctness fix.

### Reference docs
- Diffusers `AutoencoderKL` (the production analogue):
  https://huggingface.co/docs/diffusers/api/models/autoencoderkl
- LDM paper (autoencoder + latent diffusion): https://arxiv.org/abs/2112.10752

---

## 3. Model B — the latent DDPM UNet  (`src/models/unet.py`)

We use the **diffusers `UNet2DModel`** operating purely in latent space.

| Spec | Value |
|---|---|
| Input / output | `4 × 32 × 32` (the VAE latent) |
| Channels per level | `(128, 256, 384, 512)` |
| Down / up blocks | `DownBlock2D` + 3× `AttnDownBlock2D` (mirror on the way up) |
| Objective | ε-prediction MSE, DDPM cosine schedule (`squaredcos_cap_v2`), 1000 steps |
| Inference | DDIM partial-noise from `T_int` (50 steps), EMA weights |
| Params | ~90 M |

Because it runs on `32×32` latents (not `256×256` pixels), it is cheap to train
and fits an 8 GB GPU comfortably.

### Why it beats the old diffusion setup

| | Old | New |
|---|---|---|
| Training data | ~**50** healthy slices (hard cap) | **Full healthy pool** (thousands), patient-level split |
| "Healthy" definition | any slice with no lesion *on that slice* | lesion-free **+ ≥3-slice buffer** (less mass-effect leak) |
| Noise calibration | SD `scaling_factor` 0.18215 (wrong scale) | **empirical** `1/std` of our latents |
| Manifold quality | severely underfit | actually learns healthy anatomy |

A DDPM needs many examples to model a manifold; 50 slices cannot define "healthy
brain." This is why the old residuals were noisy. Same UNet architecture, but
fed enough correctly-scaled data, it can finally do its job.

### Reference docs
- Diffusers `UNet2DModel`: https://huggingface.co/docs/diffusers/api/models/unet2d
- DDPM: https://arxiv.org/abs/2006.11239 · DDIM: https://arxiv.org/abs/2010.02502
- AnoDDPM (diffusion UAD): https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/html/Wyatt_AnoDDPM...

---

## 4. The empirical scaling factor (the subtle but critical fix)

Diffusion assumes unit-variance latents. SD hardcodes `0.18215` because that is
`1/std` of **its** latents. Our medical VAE has a different latent std, so after
training we **measure** it (`compute_scaling_factor.py`) and store
`scaling_factor = 1/std`. Using SD's constant would mis-set the SNR at every
timestep, silently degrading both training and the DDIM reconstruction. (In the
smoke run our value was ~1.24 vs SD's 0.18 — an order of magnitude off.)

---

## 5. Scoring head (not a learned model, but part of the method)

The anomaly map is a **modality-weighted, baseline-calibrated residual**
(`src/pipeline/scoring.py`): per-modality `|orig − recon|`, plus an explicit
**CE = t1c − t1n** channel, weighted by clinical salience (FLAIR + CE dominate),
minus a healthy baseline `M_baseline`, optionally fused with a latent-space
residual. The old pipeline averaged 3 channels equally and could not see CE at
all. See [README](../README.md) for the full fix table.

---

## 6. One-line summary for the report

> We replace a natural-image autoencoder and a 50-slice diffusion model with a
> *from-scratch 4-channel medical KL-VAE* and a *latent DDPM trained on the full
> healthy manifold with empirically-calibrated latent scaling* — removing the
> RGB inductive bias, restoring the T1-native/CE signal, and giving the
> diffusion model enough correctly-scaled data to actually learn healthy anatomy.
