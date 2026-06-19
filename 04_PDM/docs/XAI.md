# Explainability (XAI) Strategy

## Why counterfactual explanation is the *right* XAI here

Most XAI methods (saliency, Grad-CAM, SHAP, LIME) are **post-hoc**: they fit a
surrogate to a black box and approximate *why* it decided. A diffusion UAD model
is different — it makes its decision by literally generating a **healthy version
of the input**. So the most faithful explanation is not an approximation at all;
it is the model's own output:

> **Counterfactual:** "Here is what this brain would look like if it were healthy
> (`x'`). The lesion is exactly the minimal change `|x − x'|` needed to make it
> healthy."

This is a *native*, *faithful* counterfactual — the explanation **is** the
computation, not a surrogate of it. For clinical UAD this is the strongest XAI
because a radiologist can directly inspect the generated healthy tissue and the
change map, and judge whether the flagged region is plausible.

Refs:
- **Wachter et al., 2017**, *Counterfactual Explanations without Opening the
  Black Box* (Harvard JOLT) — counterfactual XAI foundations.
- **Sanchez et al., 2022**, *Healthy/Pathological brain counterfactuals with
  diffusion models* (arXiv:2203.08089).
- **Atad et al., 2022** and related work on diffusion counterfactuals for medical
  imaging.

We deliberately did **not** pick Grad-CAM/SHAP as the primary method: they would
explain a *classifier's* logit, but our model is generative and per-pixel, so a
counterfactual is both more faithful and more actionable.

---

## The four explanation views we generate

### 1. Counterfactual difference (primary)
`|x − x'|` per modality. This is the anomaly map itself, shown next to the
generated healthy reconstruction `x'`. Implementation:
`src/xai/counterfactual.py`, `src/xai/visualization.py::save_explanation_panel`.

### 2. Counterfactual "healing trajectory"
The DDIM reverse process is saved as a sequence of frames showing the lesion
being progressively in-painted with healthy tissue. This exposes the model's
reasoning **step by step**, not just the endpoint — useful for sanity-checking
that the model heals the lesion specifically and leaves healthy anatomy intact.
Implementation: `save_trajectory_strip`.

### 3. Per-modality attribution
Decompose the score into each modality's contribution (T1n, T1c, T2w, T2f, and
the CE channel). Answers a clinically meaningful question: *is this lesion driven
by FLAIR hyperintensity (edema) or by contrast enhancement (active tumour)?* This
mirrors how radiologists reason about BraTS tumour sub-compartments.
Ref: **Menze et al., 2015** (BRATS, IEEE TMI). Implementation:
`src/xai/attribution.py::modality_attribution`, `dominant_modality`.

### 4. Per-noise-scale attribution
Decompose the score by noise level T. Low-T dominance ⇒ a fine/textural anomaly;
high-T dominance ⇒ a large structural anomaly. This explains *at what spatial
scale* the evidence was found and validates the multi-scale design.
Refs: **Bercea et al., 2023** (MICCAI); **Wyatt et al., 2022** (AnoDDPM).
Implementation: `scale_attribution`.

### (Optional) Attention rollout
A best-effort aggregation of the UNet self-attention magnitudes, upsampled to
patch resolution, showing where the model attends. A model-internal complement to
the counterfactual. Ref: **Abnar & Zuidema, 2020**, *Quantifying Attention Flow
in Transformers* (ACL). Implementation: `AttentionRollout`.

---

## How to read an XAI panel

`outputs/xai/xai_<slice>.png`:
- **Top row:** original T1c → healthy counterfactual → counterfactual difference
  → ground truth.
- **Bottom row:** per-noise-scale anomaly maps (T=50/150/250) → a bar chart of
  per-modality contribution with the dominant modality named.

`outputs/xai/trajectory_<slice>.png`: input → DDIM healing frames → final healed
difference.

---

## Faithfulness note

Because the counterfactual and the attributions are computed from the **same
forward pass** that produces the score (no surrogate model, no gradient
approximation), these explanations are **faithful by construction** — they cannot
disagree with the model's decision. This is the key advantage of counterfactual
XAI for generative anomaly detectors and the reason it is our primary strategy.
