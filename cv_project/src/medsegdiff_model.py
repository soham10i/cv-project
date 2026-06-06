"""
MedSegDiff Phase 2 — Conditional Diffusion Segmentation U-Net
==============================================================
A supervised diffusion model that denoises a noisy binary mask ``x_t`` (1 ch)
conditioned on the 3-channel MRI ``I``, predicting the noise ε.

Implements the two MedSegDiff innovations (Wu et al., 2023):

  * **FF-Parser** — a learnable *frequency-domain* filter applied to the
    conditional image features at each resolution.  It performs a 2-D FFT,
    re-weights the spectrum with a learnable per-(channel, frequency) map,
    and inverse-FFTs — attenuating high-frequency noise in the condition so it
    doesn't corrupt the mask prediction.

  * **Dynamic conditional encoding** — the (FF-Parsed) image feature is fused
    into the diffusion encoder *dynamically*: modulated by the timestep
    embedding (FiLM) and gated by the current diffusion feature, so the
    conditioning adapts to the noise level at each reverse step.

The image (condition) encoder mirrors the diffusion encoder's channel schedule
so features can be fused stage-for-stage.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Timestep embedding
# ─────────────────────────────────────────────
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, ch), ch)


# ─────────────────────────────────────────────
# Core blocks
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    """Residual block with optional FiLM timestep conditioning."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.in_layers = nn.Sequential(_gn(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.emb = nn.Linear(t_dim, out_ch) if t_dim is not None else None
        self.out_layers = nn.Sequential(
            _gn(out_ch), nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.in_layers(x)
        if self.emb is not None and t_emb is not None:
            h = h + self.emb(t_emb)[:, :, None, None]
        h = self.out_layers(h)
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Self-attention over spatial positions (used at low resolutions)."""

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = _gn(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.heads = heads

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        scale = (C // self.heads) ** -0.5
        attn = torch.softmax((q.transpose(-2, -1) @ k) * scale, dim=-1)   # (B,heads,N,N)
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ─────────────────────────────────────────────
# MedSegDiff modules
# ─────────────────────────────────────────────
class FFParser(nn.Module):
    """Learnable frequency-domain filter (MedSegDiff)."""

    def __init__(self, ch: int, h: int, w: int):
        super().__init__()
        # rfft2 last-dim length is w//2 + 1; init near 1 (identity-ish filter).
        weight = torch.zeros(ch, h, w // 2 + 1, 2)
        weight[..., 0] = 1.0                      # real part ≈ 1 → pass-through start
        weight += torch.randn_like(weight) * 0.02
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x.float(), dim=(-2, -1), norm="ortho")
        x = x * torch.view_as_complex(self.weight)        # broadcast over batch
        x = torch.fft.irfft2(x, s=(H, W), dim=(-2, -1), norm="ortho")
        return x


class DynamicFusion(nn.Module):
    """
    Dynamic conditional encoding: fuse image feature ``c`` into diffusion
    feature ``h``.  ``c`` is FiLM-modulated by the timestep (dynamic w.r.t.
    noise level) and the additive contribution is gated by ``h``.
    """

    def __init__(self, ch: int, t_dim: int):
        super().__init__()
        self.cond_conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.film = nn.Linear(t_dim, ch * 2)
        self.gate = nn.Sequential(nn.Conv2d(ch * 2, ch, 1), nn.Sigmoid())

    def forward(self, h, c, t_emb):
        c = self.cond_conv(c)
        scale, shift = self.film(t_emb)[:, :, None, None].chunk(2, dim=1)
        c = c * (1 + scale) + shift
        g = self.gate(torch.cat([h, c], dim=1))
        return h + g * c


# ─────────────────────────────────────────────
# Conditional image encoder (mirrors diffusion channel schedule)
# ─────────────────────────────────────────────
class CondEncoder(nn.Module):
    """Produces image features at each diffusion-encoder resolution."""

    def __init__(self, cond_ch, base, ch_mult, layers=1):
        super().__init__()
        self.stem = nn.Conv2d(cond_ch, base, 3, padding=1)
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        chs = [base * m for m in ch_mult]
        cur = base
        for i, ch in enumerate(chs):
            blocks = nn.ModuleList([ResBlock(cur if j == 0 else ch, ch) for j in range(layers)])
            self.stages.append(blocks)
            cur = ch
            self.downs.append(Downsample(ch) if i < len(chs) - 1 else nn.Identity())

    def forward(self, img):
        h = self.stem(img)
        feats = []
        for blocks, down in zip(self.stages, self.downs):
            for b in blocks:
                h = b(h)
            feats.append(h)          # feature at this resolution (pre-downsample)
            h = down(h)
        return feats                 # [c_0, c_1, ..., c_{L-1}]


# ─────────────────────────────────────────────
# Main conditional diffusion U-Net
# ─────────────────────────────────────────────
class MedSegDiffUNet(nn.Module):
    def __init__(self, mask_ch=1, cond_ch=3, out_ch=1, base=64,
                 ch_mult=(1, 2, 4, 8), layers_per_block=2, img_size=256,
                 attn_res=(32,), dropout=0.0):
        super().__init__()
        self.t_dim = base * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base, self.t_dim), nn.SiLU(), nn.Linear(self.t_dim, self.t_dim)
        )
        self.base = base
        chs = [base * m for m in ch_mult]
        L = len(chs)

        # Condition encoder (image features per resolution)
        self.cond_enc = CondEncoder(cond_ch, base, ch_mult, layers=1)

        # ── Diffusion encoder ────────────────────────────────────
        self.stem = nn.Conv2d(mask_ch, base, 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        self.ff_parsers = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.downs = nn.ModuleList()

        res = img_size
        cur = base
        skip_chs = []
        for i, ch in enumerate(chs):
            blocks = nn.ModuleList([ResBlock(cur if j == 0 else ch, ch, self.t_dim, dropout)
                                    for j in range(layers_per_block)])
            self.enc_blocks.append(blocks)
            self.enc_attn.append(AttnBlock(ch) if res in attn_res else nn.Identity())
            self.ff_parsers.append(FFParser(ch, res, res))
            self.fusions.append(DynamicFusion(ch, self.t_dim))
            cur = ch
            skip_chs.append(ch)
            if i < L - 1:
                self.downs.append(Downsample(ch))
                res //= 2
            else:
                self.downs.append(nn.Identity())

        # ── Bottleneck ───────────────────────────────────────────
        self.mid1 = ResBlock(cur, cur, self.t_dim, dropout)
        self.mid_attn = AttnBlock(cur)
        self.mid2 = ResBlock(cur, cur, self.t_dim, dropout)

        # ── Decoder ──────────────────────────────────────────────
        self.dec_blocks = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in reversed(range(L)):
            ch = chs[i]
            blocks = nn.ModuleList()
            for j in range(layers_per_block):
                in_c = cur + skip_chs[i] if j == 0 else ch
                blocks.append(ResBlock(in_c, ch, self.t_dim, dropout))
            self.dec_blocks.append(blocks)
            self.dec_attn.append(AttnBlock(ch) if res in attn_res else nn.Identity())
            cur = ch
            if i > 0:
                self.ups.append(Upsample(ch))
                res *= 2
            else:
                self.ups.append(nn.Identity())

        self.out = nn.Sequential(_gn(cur), nn.SiLU(), nn.Conv2d(cur, out_ch, 3, padding=1))

    def forward(self, x_t, t, cond):
        """x_t: (B,1,S,S) noisy mask | t: (B,) | cond: (B,3,S,S) image → ε̂ (B,1,S,S)."""
        t_emb = self.time_mlp(timestep_embedding(t, self.base))
        cond_feats = self.cond_enc(cond)

        h = self.stem(x_t)
        skips = []
        for i, (blocks, attn, ffp, fuse, down) in enumerate(
                zip(self.enc_blocks, self.enc_attn, self.ff_parsers, self.fusions, self.downs)):
            for b in blocks:
                h = b(h, t_emb)
            h = attn(h)
            c = ffp(cond_feats[i])                 # FF-Parser on image feature
            h = fuse(h, c, t_emb)                  # dynamic conditional encoding
            skips.append(h)
            h = down(h)

        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), t_emb)

        for blocks, attn, up in zip(self.dec_blocks, self.dec_attn, self.ups):
            h = torch.cat([h, skips.pop()], dim=1)
            for k, b in enumerate(blocks):
                h = b(h, t_emb)
            h = attn(h)
            h = up(h)

        return self.out(h)


def build_medsegdiff(img_size=256, base=64, ch_mult=(1, 2, 4, 8),
                     layers_per_block=2, attn_res=(32,)):
    return MedSegDiffUNet(img_size=img_size, base=base, ch_mult=ch_mult,
                          layers_per_block=layers_per_block, attn_res=attn_res)
