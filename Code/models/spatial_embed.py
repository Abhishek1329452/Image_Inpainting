# === Cell 09 — Module colorad/models/spatial_embed.py ===
"""Per-pixel noise-level embedding via 1x1 convs (RAD)."""
import math
import torch
import torch.nn as nn


class SpatialNoiseEmbedding(nn.Module):
    """Sinusoidally encode per-pixel alpha_bar, then 1x1-conv to embed_dim.

    FIX vs spec: avoid the (B*H*W, sin_dim) flatten which explodes memory at
    256px; broadcast frequencies over channels directly on (B,1,H,W).
    """

    def __init__(self, embed_dim: int = 256, sin_dim: int = 256) -> None:
        super().__init__()
        self.sin_dim = sin_dim
        self.conv1 = nn.Conv2d(sin_dim, embed_dim * 2, 1)
        self.conv2 = nn.Conv2d(embed_dim * 2, embed_dim, 1)
        self.act = nn.SiLU()
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, a=0.0)
            m.weight.data.mul_(0.02)
            nn.init.zeros_(m.bias)

    def forward(self, alpha_bar: torch.Tensor) -> torch.Tensor:
        """alpha_bar (B,1,H,W) in [0,1] -> (B, embed_dim, H, W)."""
        half = self.sin_dim // 2
        device = alpha_bar.device
        freqs = torch.exp(
            torch.arange(half, device=device) * (-math.log(10000.0) / (half - 1))
        ).view(1, half, 1, 1)
        ang = alpha_bar * freqs * 1000.0
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        return self.conv2(self.act(self.conv1(emb)))