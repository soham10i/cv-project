"""
Diffusion model trainer.
========================

Trains the pixel-space UNet on healthy patches with the epsilon-prediction DDPM
objective. Implements the practical ingredients that the smoke-test predecessor
lacked: linear LR warmup -> cosine decay, mixed-precision (bf16) autocast, EMA,
gradient clipping, periodic checkpoints, and early stopping on a healthy-patch
validation loss.

Refs:
  * Ho et al., 2020 (DDPM objective).
  * Loshchilov & Hutter, 2017, "Decoupled Weight Decay Regularization" (AdamW).
  * Goyal et al., 2017, "Accurate, Large Minibatch SGD" — LR warmup.
  * Micikevicius et al., 2018, "Mixed Precision Training" (ICLR).
"""

from __future__ import annotations

import csv

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from ..config import CONFIG
from ..models.diffusion import DiffusionProcess
from ..models.ema import EMA
from ..utils.device import amp_dtype_from_str
from ..utils.exceptions import TrainingError
from ..utils.logging_utils import get_logger
from .callbacks import CheckpointManager, EarlyStopping

log = get_logger("pdm.trainer")


class Trainer:
    """Owns the optimisation loop for the diffusion UNet."""

    def __init__(
        self,
        unet,
        process: DiffusionProcess,
        device: torch.device,
        epochs: int = CONFIG.train.epochs,
    ) -> None:
        self.unet = unet.to(device)
        self.process = process
        self.device = device
        self.epochs = epochs
        cfg = CONFIG.train

        self.opt = AdamW(
            self.unet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.99)
        )
        warmup = LinearLR(self.opt, start_factor=0.1, total_iters=max(1, cfg.warmup_epochs))
        cosine = CosineAnnealingLR(self.opt, T_max=max(1, epochs - cfg.warmup_epochs), eta_min=1e-6)
        self.sched = SequentialLR(self.opt, [warmup, cosine], milestones=[cfg.warmup_epochs])

        self.ema = EMA(self.unet, cfg.ema_decay)
        self.amp = cfg.use_amp and device.type == "cuda"
        self.amp_dtype = amp_dtype_from_str(cfg.amp_dtype)
        # GradScaler is only needed for fp16; bf16 has fp32 range so no scaling.
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp and self.amp_dtype == torch.float16
        )

        self.ckpt = CheckpointManager(CONFIG.paths.checkpoints_dir)
        self.stopper = EarlyStopping(cfg.early_stop_patience)
        self._metrics_path = CONFIG.paths.logs_dir / "train_metrics.csv"

    # ── one epoch ─────────────────────────────────────────────────────────
    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.unet.train()
        total, n = 0.0, 0
        for bidx, batch in enumerate(loader):
            x0 = batch.to(self.device, non_blocking=True)
            self.opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp):
                loss = self.process.training_loss(self.unet, x0)
            if not torch.isfinite(loss):
                raise TrainingError(f"Non-finite loss at epoch {epoch} batch {bidx}.")

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), CONFIG.train.grad_clip_norm)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.ema.update(self.unet)

            total += loss.item() * x0.size(0)
            n += x0.size(0)
            if bidx % CONFIG.train.log_every_steps == 0:
                log.info("  ep %d | batch %d/%d | loss %.5f", epoch, bidx, len(loader), loss.item())
        return total / max(n, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Validation denoising loss using the EMA weights."""
        ema_model = self.ema.state_module().to(self.device).eval()
        total, n = 0.0, 0
        for batch in loader:
            x0 = batch.to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp):
                loss = self.process.training_loss(ema_model, x0)
            total += loss.item() * x0.size(0)
            n += x0.size(0)
        return total / max(n, 1)

    # ── full fit ──────────────────────────────────────────────────────────
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        CONFIG.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        with self._metrics_path.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

        log.info("Training | %d epochs | bs %d | lr %.1e | noise=%s | amp=%s",
                 self.epochs, CONFIG.train.batch_size, CONFIG.train.lr,
                 self.process.noise.name, self.amp)

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            val_loss = self._validate(val_loader)
            self.sched.step()
            lr = self.opt.param_groups[0]["lr"]
            log.info("Epoch %d/%d | train %.5f | val %.5f | lr %.2e",
                     epoch, self.epochs, train_loss, val_loss, lr)

            with self._metrics_path.open("a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, val_loss, lr])

            self.ckpt.save_best(self.unet, self.ema, val_loss)
            if epoch % CONFIG.train.ckpt_every == 0:
                self.ckpt.save_periodic(self.ema, epoch)
            if self.stopper.step(val_loss, epoch):
                log.info("Early stopping at epoch %d (best %.5f @ %d).",
                         epoch, self.stopper.best, self.stopper.best_epoch)
                break

        log.info("DONE — best val %.5f @ epoch %d", self.stopper.best, self.stopper.best_epoch)
        return {"best_val": self.stopper.best, "best_epoch": self.stopper.best_epoch}
