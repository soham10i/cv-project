"""
Image reconstruction metrics for diffusion validation.

Contains SSIM, MAE, and PSNR computation for comparing the clean reference
image with the DDIM-denoised reconstruction during training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ssim_2d(
    img1: torch.Tensor, img2: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Compute structural similarity index (SSIM) between two batches of images.

    Parameters
    ----------
    img1, img2 : (B, C, H, W) tensors.
    """
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    # Provide a simple 1D gaussian window
    sigma = 1.5
    gauss = torch.tensor([
        float(torch.exp(torch.tensor(-(x - window_size // 2)**2 / (2 * sigma**2))))
        for x in range(window_size)
    ], device=img1.device)
    gauss = gauss / gauss.sum()

    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(img1.shape[1], 1, window_size, window_size)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img1.shape[1])

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean(dim=(1, 2, 3))


def compute_recon_metrics(
    orig: torch.Tensor, recon: torch.Tensor,
) -> dict[str, float]:
    """Compute standard reconstruction metrics for a batch of images.

    Returns a dict containing mean SSIM, MAE, and PSNR.
    """
    bs = orig.shape[0]
    ssim_val = ssim_2d(orig, recon).mean().item()
    mae_val = F.l1_loss(recon, orig).item()
    mse_val = F.mse_loss(recon, orig).item()

    if mse_val == 0:
        psnr_val = 100.0
    else:
        # data range is 2.0 because images are in [-1, 1]
        psnr_val = 20 * torch.log10(torch.tensor(2.0)) - 10 * torch.log10(torch.tensor(mse_val))
        psnr_val = psnr_val.item()

    return {
        "ssim": ssim_val,
        "mae": mae_val,
        "psnr": psnr_val,
    }
