# === Corrected Cell 11 — Module colorad/models/mae_align.py ===
"""MAE alignment module (ASUKA-inspired).

Projects and upsamples frozen ViT patch features from the raw patch grid 
(e.g., 14x14) to the target spatial conditioning resolution (e.g., 32x32).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAEAlignmentModule(nn.Module):
    def __init__(self, mae_dim: int = 768, out_dim: int = 256, cond_size: int = 32) -> None:
        super().__init__()
        self.cond_size = cond_size
        
        self.align_conv = nn.Sequential(
            nn.Conv2d(mae_dim, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.SiLU()
        )

    def forward(self, mae_features: torch.Tensor) -> torch.Tensor:
        """Projects (B, mae_dim, H_patch, W_patch) to (B, out_dim, cond_size, cond_size)."""
        # Upsample patch features (typically 14x14) to the conditioning size (typically 32x32)
        x = F.interpolate(mae_features, size=(self.cond_size, self.cond_size), 
                          mode="bilinear", align_corners=False)
        return self.align_conv(x)