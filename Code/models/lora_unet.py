# === Cell 13 — Module colorad/models/lora_unet.py ===
"""Compact ADM-style pixel-space U-Net with LoRA attention (RAD).

FIXES:
  - ConditionedAttentionBlock: added self-attention pass *before* cross-attention.
    (Original only had cross-attention; missing self-attn hurt intra-patch coherence.)
  - Attention applied at the two deepest encoder/decoder levels (i >= len(chs)-2)
    instead of only the deepest one, giving the model richer multi-scale conditioning.
  - num_heads now scales with channel count (head_dim ≈ 32) to keep heads sensible
    at lower-channel levels.
"""
import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank update (ΔW = B A)."""

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 16, alpha: int = 16,
                 bias: bool = False, freeze_base: bool = False) -> None:
        super().__init__()
        self.linear  = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A  = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B  = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        if freeze_base:
            self.linear.weight.requires_grad = False
            if bias and self.linear.bias is not None:
                self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


class TimeResBlock(nn.Module):
    """ResBlock with spatially-varying time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_ch: int) -> None:
        super().__init__()
        self.norm1     = nn.GroupNorm(8, in_ch)
        self.conv1     = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Conv2d(time_ch, out_ch, 1)
        self.norm2     = nn.GroupNorm(8, out_ch)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip      = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act       = nn.SiLU()

    def forward(self, x: torch.Tensor, t_spatial: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        t = F.adaptive_avg_pool2d(t_spatial, h.shape[2:])
        h = h + self.time_proj(t)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class ConditionedAttentionBlock(nn.Module):
    """Self-attention THEN cross-attention to context; LoRA on all projections.

    FIX: original only had cross-attention (no self-attention).  Missing
    self-attention meant the model could not reason about spatial relationships
    within the generated patch — a fundamental gap for inpainting coherence.

    FIX: num_heads now derived from dim to keep head_dim ≈ 32 regardless of scale.
    """

    def __init__(self, dim: int, ctx_dim: int,
                 rank: int = 16, alpha: int = 16) -> None:
        super().__init__()
        # pin head_dim ≈ 32 for all channel counts
        self.heads = max(1, dim // 32)
        self.dh    = dim // self.heads

        # Self-attention projections
        self.norm_sa = nn.GroupNorm(8, dim)
        self.sa_q    = LoRALinear(dim, dim, rank, alpha)
        self.sa_k    = LoRALinear(dim, dim, rank, alpha)
        self.sa_v    = LoRALinear(dim, dim, rank, alpha)
        self.sa_out  = LoRALinear(dim, dim, rank, alpha)

        # Cross-attention projections
        self.norm_ca = nn.GroupNorm(8, dim)
        self.to_q    = LoRALinear(dim, dim, rank, alpha)
        self.to_k    = LoRALinear(ctx_dim, dim, rank, alpha)
        self.to_v    = LoRALinear(ctx_dim, dim, rank, alpha)
        self.to_out  = LoRALinear(dim, dim, rank, alpha)

    def _mha(self, q: torch.Tensor, k: torch.Tensor,
             v: torch.Tensor) -> torch.Tensor:
        b, nq, _ = q.shape
        nk = k.shape[1]
        q = q.view(b, nq, self.heads, self.dh).transpose(1, 2)
        k = k.view(b, nk, self.heads, self.dh).transpose(1, 2)
        v = v.view(b, nk, self.heads, self.dh).transpose(1, 2)
        # Uses Flash Attention when available (PyTorch 2.0+)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(b, nq, -1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # ── Self-attention (intra-patch coherence) ──────────────────────────
        xf  = self.norm_sa(x).flatten(2).permute(0, 2, 1)   # (B, HW, C)
        sa  = self._mha(self.sa_q(xf), self.sa_k(xf), self.sa_v(xf))
        x   = x + self.sa_out(sa).permute(0, 2, 1).reshape(b, c, h, w)

        # ── Cross-attention to fused context ────────────────────────────────
        xf  = self.norm_ca(x).flatten(2).permute(0, 2, 1)   # (B, HW, C)
        cf  = context.flatten(2).permute(0, 2, 1)            # (B, cond^2, Cctx)
        ca  = self._mha(self.to_q(xf), self.to_k(cf), self.to_v(cf))
        out = self.to_out(ca)
        return x + out.permute(0, 2, 1).reshape(b, c, h, w)


class LoRAUNet(nn.Module):
    """Compact pixel-space U-Net. Attention at two deepest scales."""

    def __init__(self, lora_rank: int = 16, lora_alpha: int = 16,
                 in_channels: int = 3, model_channels: int = 64,
                 channel_mult=(1, 2, 4), num_res_blocks: int = 2,
                 ctx_dim: int = 256, time_ch: int = 256) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        chs = [model_channels * m for m in channel_mult]

        # ── Encoder ──────────────────────────────────────────────────────────
        self.down_blocks  = nn.ModuleList()
        self.down_attn    = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        prev = model_channels
        self.skip_chs: List[int] = []
        for i, ch in enumerate(chs):
            blocks = nn.ModuleList()
            attns  = nn.ModuleList()
            # FIX: attention at the two deepest levels (i >= len(chs)-2)
            use_attn = (i >= len(chs) - 2)
            for _ in range(num_res_blocks):
                blocks.append(TimeResBlock(prev, ch, time_ch))
                attns.append(
                    ConditionedAttentionBlock(ch, ctx_dim, rank=lora_rank, alpha=lora_alpha)
                    if use_attn else None)
                prev = ch
                self.skip_chs.append(prev)
            self.down_blocks.append(blocks)
            self.down_attn.append(attns)
            self.downsamplers.append(
                nn.Conv2d(ch, ch, 3, stride=2, padding=1) if i < len(chs) - 1
                else nn.Identity())

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.mid_block1 = TimeResBlock(prev, prev, time_ch)
        self.mid_attn   = ConditionedAttentionBlock(prev, ctx_dim,
                              rank=lora_rank, alpha=lora_alpha)
        self.mid_block2 = TimeResBlock(prev, prev, time_ch)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up_blocks  = nn.ModuleList()
        self.up_attn    = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chs))):
            blocks   = nn.ModuleList()
            attns    = nn.ModuleList()
            use_attn = (i >= len(chs) - 2)    # mirror encoder attention depth
            for _ in range(num_res_blocks):
                skip = self.skip_chs.pop()
                blocks.append(TimeResBlock(prev + skip, ch, time_ch))
                attns.append(
                    ConditionedAttentionBlock(ch, ctx_dim, rank=lora_rank, alpha=lora_alpha)
                    if use_attn else None)
                prev = ch
            self.up_blocks.append(blocks)
            self.up_attn.append(attns)
            self.upsamplers.append(
                nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                              nn.Conv2d(ch, ch, 3, padding=1))
                if i > 0 else nn.Identity())

        self.out_norm = nn.GroupNorm(8, prev)
        self.out_conv = nn.Conv2d(prev, in_channels, 3, padding=1)
        self.act      = nn.SiLU()

    def freeze_base(self) -> None:
        """Freeze all weights except LoRA (call after loading pretrained base)."""
        for n, p in self.named_parameters():
            p.requires_grad = ("lora_A" in n) or ("lora_B" in n)

    def forward(self, noisy_image: torch.Tensor, t_spatial: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        h     = self.in_conv(noisy_image)
        skips = []
        for blocks, attns, ds in zip(self.down_blocks, self.down_attn, self.downsamplers):
            for blk, at in zip(blocks, attns):
                h = blk(h, t_spatial)
                if at is not None:
                    h = at(h, context)
                skips.append(h)
            h = ds(h)
        h = self.mid_block1(h, t_spatial)
        h = self.mid_attn(h, context)
        h = self.mid_block2(h, t_spatial)
        for blocks, attns, us in zip(self.up_blocks, self.up_attn, self.upsamplers):
            for blk, at in zip(blocks, attns):
                h = torch.cat([h, skips.pop()], dim=1)
                h = blk(h, t_spatial)
                if at is not None:
                    h = at(h, context)
            h = us(h)
        return self.out_conv(self.act(self.out_norm(h)))
