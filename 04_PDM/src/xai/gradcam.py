"""
Grad-CAM for the epsilon-prediction diffusion UNet (model-focus maps).
======================================================================

DEPRECATED as an explainer (retained as a reproducible negative result).
------------------------------------------------------------------------
Grad-CAM is a *classifier*-attribution method: it localises the regions that
raise a single class logit. A diffusion denoiser has no class logit, so the
explained scalar here is a *dense-regression* noise-error summed over the patch,
whose global-average-pooled gradient has no spatial selectivity; the resulting
maps are diffuse and do NOT track the model's localisation (they stay flat even
on slices where the anomaly score clearly localises the lesion). The faithful
explainers for this generative-UAD pipeline are counterfactual generation
(``counterfactual.py``: the healthy reconstruction is the thresholded decision
variable / manifold projection) and additive residual attribution
(``attribution.py``). This module
is kept only to reproduce the "why post-hoc CAM fails here" figure in the report;
it is no longer called by ``scripts/04_explain.py``.


Counterfactual / residual maps answer *what* the model changed; Grad-CAM answers
*where the network looked* to produce that decision. For a denoising UNet there
is no class logit, so we use the diffusion model's own training target as the
explained scalar:

    y(x0, t) = || eps_theta(x_t, t) - eps ||^2 ,   x_t = sqrt(a_bar_t) x0 + sqrt(1 - a_bar_t) eps

i.e. the per-patch noise-prediction error at a fixed scoring timestep t. This is
high exactly where the network *fails* to denoise as if healthy — the anomaly.
Grad-CAM of y w.r.t. a high-resolution decoder feature map A therefore localises
the spatial regions that drive the anomaly response:

    alpha_k = (1 / Z) sum_{i,j} d y / d A^k_{ij}        (global-average-pooled grads)
    L_GradCAM = ReLU( sum_k alpha_k A^k )               (Selvaraju et al., 2017, ICCV)

We compute one CAM per patch and Gaussian-fuse them into a slice-level map with
the same fuser used for the anomaly score, so the Grad-CAM map is spatially
comparable to the score and the ground-truth mask.

Refs:
  * Selvaraju et al., 2017, "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization" (ICCV).
  * Ho et al., 2020 (DDPM) — the epsilon objective used as the explained scalar.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import CONFIG
from ..data.patches import GaussianPatchFuser, extract_patches
from ..models.diffusion import DiffusionProcess
from ..scoring.residual import brain_mask_2d
from ..utils.logging_utils import get_logger

log = get_logger("pdm.gradcam")


def _select_target_module(unet) -> torch.nn.Module:
    """Pick a high-resolution decoder feature map as the Grad-CAM target.

    The last up-block outputs a full-patch-resolution feature map — the sharpest
    localisation. Falls back to the mid (bottleneck) block, then to the whole
    UNet, so this stays robust across diffusers versions.
    """
    if hasattr(unet, "up_blocks") and len(unet.up_blocks) > 0:
        return unet.up_blocks[-1]
    if hasattr(unet, "mid_block") and unet.mid_block is not None:
        return unet.mid_block
    return unet


class DiffusionGradCAM:
    """Grad-CAM explainer hooked onto one decoder feature map of the UNet."""

    def __init__(self, unet, process: DiffusionProcess, target_module=None) -> None:
        self.unet = unet
        self.process = process
        self.target = target_module or _select_target_module(unet)
        self._activation: torch.Tensor | None = None
        self._handle = self.target.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, _module, _inp, output) -> None:
        act = output[0] if isinstance(output, tuple) else output
        # Retain grad on this non-leaf activation so autograd.grad can reach it.
        act.retain_grad()
        self._activation = act

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def _cam_from_batch(self, x0: torch.Tensor, t_int: int) -> np.ndarray:
        """Return per-patch normalised CAMs (B, p, p) for a batch of patches."""
        device = x0.device
        noise = self.process.noise.sample(tuple(x0.shape), device, None)
        t_tensor = torch.full((x0.shape[0],), int(t_int), device=device, dtype=torch.long)
        x_t = self.process.ddpm.add_noise(x0, noise, t_tensor)

        with torch.enable_grad():
            self.unet.zero_grad(set_to_none=True)
            pred = self.unet(x_t, t_tensor).sample
            # Explained scalar: total epsilon-prediction error over the batch.
            y = ((pred - noise) ** 2).sum()
            grads = torch.autograd.grad(y, self._activation, retain_graph=False)[0]

        act = self._activation.detach()                       # (B, K, h, w)
        weights = grads.detach().mean(dim=(2, 3), keepdim=True)  # alpha_k (B, K, 1, 1)
        cam = torch.relu((weights * act).sum(dim=1))          # (B, h, w)
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=(CONFIG.patch.patch_size, CONFIG.patch.patch_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        # Per-patch min-max normalisation.
        flat = cam.flatten(1)
        lo = flat.min(dim=1).values[:, None, None]
        hi = flat.max(dim=1).values[:, None, None]
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam.cpu().numpy()

    def slice_cam(
        self,
        image: np.ndarray,
        t_int: int | None = None,
        patch_batch: int = 16,
    ) -> np.ndarray:
        """Slice-level Grad-CAM (H, W), Gaussian-fused from per-patch CAMs.

        ``t_int`` defaults to the centre scoring timestep. Brain-masked so the
        map is directly comparable to the anomaly score and the GT mask.
        """
        t = t_int or CONFIG.scoring.score_timesteps[len(CONFIG.scoring.score_timesteps) // 2]
        mask = brain_mask_2d(image)
        patches, coords = extract_patches(
            image, CONFIG.patch.patch_size, CONFIG.patch.stride
        )
        device = next(self.unet.parameters()).device
        cams = np.empty((len(patches), CONFIG.patch.patch_size, CONFIG.patch.patch_size),
                        dtype=np.float32)
        for i in range(0, len(patches), patch_batch):
            chunk = torch.from_numpy(patches[i : i + patch_batch]).float().to(device)
            cams[i : i + patch_batch] = self._cam_from_batch(chunk, t)
        fuser = GaussianPatchFuser()
        fused = fuser.fuse(cams, coords) * mask
        m = fused.max()
        return (fused / m).astype(np.float32) if m > 0 else fused.astype(np.float32)
