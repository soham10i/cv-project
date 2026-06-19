# Medical Image Anomaly Detection Pipeline

This project implements a state-of-the-art anomaly detection pipeline for medical images (MRI) using Denoising Diffusion Probabilistic Models (DDPM) and Latent Diffusion. The model is trained purely on healthy brain anatomy and flags tumours as deviations from the learned healthy manifold.

## Explainable AI (xAI) Strategy
Our approach to Explainable AI offers three intersecting layers of interpretability:

1. **Pixel-Space Residuals ($M_{pixel}$)**: The slice is mapped into the healthy latent manifold, denoised, and decoded back to pixels. Subtracting this "reconstructed healthy" slice from the original directly isolates anomalous tissue.
2. **Latent-Space Fusion ($M_{latent}$)**: Because decoders can sometimes hallucinate healthy textures, we also measure the distance between latents during the diffusion process, fusing it with the pixel residual to catch hidden structural anomalies.
3. **Self-Attention Attribution Maps (SAAM)**: During the DDIM denoising process, we hook into the self-attention matrices of the UNet. This shows us exactly which spatial regions the UNet is "attending" to as it reconstructs healthy anatomy, providing insight into *why* the model flags an anomaly.

## Setup and Execution (Colab/H100)

### 1. Upload the Project
Zip the entire `cv-project` folder (excluding the virtual environment) and upload it to Google Drive or directly to your Colab instance.

### 2. Prepare the Environment
In your Colab notebook, run the following to extract and install dependencies:
```bash
!unzip cv-project.zip -d /content/cv-project
%cd /content/cv-project
!pip install diffusers transformers accelerate lpips
```

### 3. Run the Pipeline
We have provided a master script that sequentially executes all 6 stages of the pipeline: dataset splitting, preprocessing, VAE fine-tuning, UNet training, calibration, and evaluation.

To run the full pipeline, simply execute:
```bash
!chmod +x run_all.sh
!./run_all.sh
```

*(Note: The `04_train_diffusion.py` script automatically resumes from the last checkpoint if interrupted, making it safe for Colab preemptions.)*

### 4. Viewing Results
After execution completes, you will find your output mapped logically:
- **`models/`**: Saved Checkpoints, VAE fine-tunes, and EMA weights.
- **`logs/`**: All training, evaluation, and system logs.
- **`results/xai_trajectories/`**: High-quality visual explanations and masks generated during calibration and evaluation.
- **`results/evaluation/`**: Final metrics (`metrics.json`) including your AUROC and DICE scores.
