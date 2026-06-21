"""
Central configuration for the MPDF (Multi-scale Patch Diffusion Fusion) pipeline.
================================================================================

Single source of truth for every constant used across the pipeline. Values are
grouped into frozen dataclasses by concern (data, patches, noise, model,
training, scoring, calibration, evaluation, xai). A single ``CONFIG`` instance
is exported and imported everywhere.

Environment overrides
---------------------
Paths that change between a laptop and a RunPod pod are read from environment
variables so the *same code* runs in both places without edits:

    PDM_DATA_ROOT      raw BraTS dataset root (NIfTI volumes)
    PDM_PROCESSED_ROOT preprocessed slices / patches
    PDM_OUTPUT_ROOT    checkpoints, logs, results

See ``docs/RUNPOD.md`` for how these are set on a pod.

Design note
-----------
Frozen dataclasses give us immutability (no accidental mutation of a global
constant mid-run), type safety, and IDE autocompletion — a cleaner pattern than
a flat module of UPPER_CASE names. Reference for this pattern: Python ``dataclasses``
(PEP 557).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution (environment-aware)
# ─────────────────────────────────────────────────────────────────────────────
def _env_path(var: str, default: str) -> Path:
    """Return ``Path`` from env var ``var`` or ``default`` if unset."""
    return Path(os.environ.get(var, default)).expanduser()


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. Override the roots via environment variables."""

    project_root: Path = _PROJECT_ROOT
    data_root: Path = _env_path("PDM_DATA_ROOT", str(_PROJECT_ROOT / "data" / "raw"))
    processed_root: Path = _env_path(
        "PDM_PROCESSED_ROOT", str(_PROJECT_ROOT / "data" / "processed")
    )
    output_root: Path = _env_path("PDM_OUTPUT_ROOT", str(_PROJECT_ROOT / "outputs"))

    @property
    def slices_dir(self) -> Path:
        return self.processed_root / "slices"

    @property
    def masks_dir(self) -> Path:
        return self.processed_root / "masks"

    @property
    def patches_dir(self) -> Path:
        return self.processed_root / "patches"

    @property
    def manifests_dir(self) -> Path:
        return self.processed_root / "manifests"

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_root / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs"

    @property
    def results_dir(self) -> Path:
        return self.output_root / "results"

    @property
    def xai_dir(self) -> Path:
        return self.output_root / "xai"

    @property
    def calibration_path(self) -> Path:
        return self.output_root / "calibration.json"


@dataclass(frozen=True)
class DataConfig:
    """MRI modality and slicing configuration.

    Four-channel multimodal input is kept (T1n, T1c, T2w, T2f) because the
    contrast-enhancement signal (T1c - T1n) is a primary marker of active
    tumour and is invisible in 3-channel pipelines. Ref: Baid et al., 2021,
    "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark" (arXiv:2107.02314).
    """

    modalities: Tuple[str, ...] = ("t1n", "t1c", "t2w", "t2f")
    n_channels: int = 4
    target_size: int = 256
    # A slice counts as "healthy" only if it is lesion-free AND at least
    # `healthy_buffer` slices away from any lesion (reduces mass-effect leakage).
    healthy_buffer: int = 3
    # Robust intensity normalisation percentiles (avoids hard sigma clipping that
    # truncates hyperintense lesion signal). Ref: Reinhold et al., 2019,
    # "Evaluating the Impability of Intensity Normalization" (SPIE MI).
    norm_pct_low: float = 0.5
    norm_pct_high: float = 99.5
    # Foreground fraction below which a slice is discarded (mostly-empty edges).
    min_foreground_frac: float = 0.02


@dataclass(frozen=True)
class PatchConfig:
    """Patch extraction configuration — the core of the MPDF approach.

    Why patches (vs whole-image or latent diffusion):
      * Each 96x96 patch is its own training sample → ~196 patches/slice turns
        ~3k healthy slices into ~570k training samples (≈196x more data).
      * No global reconstruction means no brain-boundary edge artefacts, which
        dominate the residual in whole-image autoencoder/LDM UAD.
    Ref: Behrendt et al., 2023, "Patched Diffusion Models for Unsupervised
    Anomaly Detection in Brain MRI" (MIDL 2023, arXiv:2303.03758).
    """

    patch_size: int = 96
    stride: int = 16
    # Gaussian fusion weight sigma (px) used when stitching overlapping patch
    # scores back into a full slice — centre voxels of each patch are trusted
    # more than edge voxels.
    fusion_sigma: float = 32.0
    # Discard a patch at training time if its brain-foreground fraction is below
    # this (pure-background patches teach the model nothing).
    min_patch_foreground: float = 0.10


@dataclass(frozen=True)
class NoiseConfig:
    """Forward-process noise configuration.

    Simplex (fractal) noise is spatially correlated at a controllable scale,
    unlike isotropic Gaussian noise which is spatially white. Correlated noise
    at lesion scale makes lesions "look like noise" to the denoiser, so they are
    in-painted with healthy tissue at test time → larger residual.
    Ref: Wyatt et al., 2022, "AnoDDPM: Anomaly Detection with Denoising
    Diffusion Probabilistic Models using Simplex Noise" (CVPR-W 2022).
    """

    # "simplex" (recommended) or "gaussian" (ablation baseline).
    strategy: str = "simplex"
    # Number of fractal octaves summed to approximate simplex noise.
    simplex_octaves: int = 6
    # Base frequency of the coarsest octave (lower = larger correlated blobs).
    simplex_base_frequency: int = 16
    # Persistence (amplitude decay per octave) and lacunarity (frequency growth).
    simplex_persistence: float = 0.8
    simplex_lacunarity: float = 2.0


@dataclass(frozen=True)
class ModelConfig:
    """Pixel-space epsilon-prediction UNet configuration.

    A compact convolutional UNet (not a Transformer) is chosen deliberately:
    with ~570k patches from only ~30 training patients, the local inductive bias
    of convolutions generalises better than a data-hungry DiT. Ref: Ho et al.,
    2020, "Denoising Diffusion Probabilistic Models" (NeurIPS 2020); Peebles &
    Xie, 2022, "Scalable Diffusion Models with Transformers" (DiT) for the
    contrast.
    """

    in_channels: int = 4
    out_channels: int = 4
    base_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    layers_per_block: int = 2
    # Spatial resolution(s) at which to apply self-attention. 96 -> 48 -> 24 ->
    # 12; attention only at the 12x12 bottleneck keeps memory low.
    attention_resolutions: Tuple[int, ...] = (12,)
    num_attention_heads: int = 8
    dropout: float = 0.1
    # Diffusion process.
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"  # cosine; Nichol & Dhariwal, 2021.
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    prediction_type: str = "epsilon"


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation configuration for the diffusion model."""

    epochs: int = 300
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.9999
    # bf16 autocast on Ampere+ (RunPod A100/A40). Falls back to fp32 elsewhere.
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    num_workers: int = 8
    seed: int = 42
    early_stop_patience: int = 6
    ckpt_every: int = 10
    log_every_steps: int = 50
    # Cap on patches sampled per epoch (None = use all). Keeps epochs bounded.
    max_patches_per_epoch: int | None = None


@dataclass(frozen=True)
class ScoringConfig:
    """Multi-scale anomaly scoring configuration.

    Reconstruction error is computed at several noise levels because the optimal
    noise level depends on lesion size: low T for small/fine lesions, high T for
    large structural lesions. Ref: Bercea et al., 2023, "Generalizing
    Unsupervised Anomaly Detection" (MICCAI 2023); Graham et al., 2023,
    "Denoising diffusion models for OOD detection" (CVPR-W 2023).
    """

    # Noise levels (timesteps) at which to score, and their fusion weights.
    score_timesteps: Tuple[int, ...] = (50, 150, 250)
    scale_weights: Tuple[float, ...] = (0.25, 0.45, 0.30)
    # DDIM steps for the reverse (denoising) trajectory at each scale.
    ddim_steps: int = 50
    # Per-modality salience weights for the residual: [t1n, t1c, t2w, t2f].
    # FLAIR (t2f) and contrast (t1c) dominate lesion signal.
    modality_weights: Tuple[float, ...] = (0.5, 1.0, 0.75, 1.5)
    # Append a contrast-enhancement residual channel |(t1c-t1n) recon error|.
    use_ce_channel: bool = True
    ce_weight: float = 1.0
    # Erosion (px) applied to the brain mask. Kept small (2) so peripheral
    # lesions near cortex are not masked away.
    brain_mask_erosion: int = 2


@dataclass(frozen=True)
class CalibrationConfig:
    """Threshold calibration configuration."""

    # Number of healthy validation slices used to fit the operating threshold.
    # Statistically: ~20/alpha samples are needed for a stable percentile
    # estimate (Wilks, 1941). 200 gives a robust 95th-percentile estimate.
    max_samples: int = 200
    threshold_percentile: float = 95.0


@dataclass(frozen=True)
class CRFConfig:
    """Dense CRF boundary-refinement configuration.

    A fully-connected CRF snaps the smooth anomaly score to anatomical edges
    using T1c intensity as guidance. Ref: Krähenbühl & Koltun, 2011, "Efficient
    Inference in Fully Connected CRFs with Gaussian Edge Potentials" (NeurIPS).
    Optional — gracefully skipped if ``pydensecrf`` is not installed.
    """

    enabled: bool = True
    n_iterations: int = 5
    # Pairwise Gaussian (smoothness) and bilateral (appearance) parameters.
    gaussian_sxy: int = 3
    gaussian_compat: int = 3
    bilateral_sxy: int = 40
    bilateral_srgb: int = 5
    bilateral_compat: int = 8
    # Channel index used as appearance guidance (1 == t1c).
    guidance_channel: int = 1


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation configuration."""

    max_plots: int = 24
    # Number of thresholds in the grid used for oracle/best-global DICE.
    n_threshold_grid: int = 60


@dataclass(frozen=True)
class XAIConfig:
    """Explainability configuration.

    Strategy: **counterfactual-first**. The healthy reconstruction is itself a
    counterfactual ("what this brain would look like without disease"); the
    anomaly map is the counterfactual difference. We complement it with
    per-modality attribution, per-noise-scale attribution, and UNet attention
    rollout. See ``docs/XAI.md`` for the full justification and references.
    """

    n_explained_cases: int = 12
    # Generate attention-rollout maps (Abnar & Zuidema, 2020). Adds a forward
    # hook pass; disable for speed.
    attention_rollout: bool = True
    # Number of DDIM frames to save for the counterfactual "healing" trajectory.
    trajectory_frames: int = 6


@dataclass(frozen=True)
class Config:
    """Top-level aggregate config."""

    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    crf: CRFConfig = field(default_factory=CRFConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)


# The single shared instance imported across the codebase.
CONFIG = Config()
