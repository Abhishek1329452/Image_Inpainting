# === Cell 12 — Module colorad/models/embed_fusion.py ===
"""Fuse spatial-noise and MAE conditioning at the cond grid."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingFusion(nn.Module):
    def __init__(self, spatial_dim: int = 256, mae_dim: int = 256,
                 out_dim: int = 256, cond_size: int = 32) -> None:
        super().__init__()
        self.cond_size = cond_size
        self.fusion = nn.Sequential(
            nn.Conv2d(spatial_dim + mae_dim, out_dim, 1),
            nn.GroupNorm(8, out_dim), nn.SiLU())

    def forward(self, spatial_embed: torch.Tensor,
                mae_embed: torch.Tensor) -> torch.Tensor:
        """spatial_embed (B,C,H,W), mae_embed (B,C,cond,cond) -> (B,out,cond,cond)."""
        s = F.adaptive_avg_pool2d(spatial_embed, (self.cond_size, self.cond_size))
        return self.fusion(torch.cat([s, mae_embed], dim=1))