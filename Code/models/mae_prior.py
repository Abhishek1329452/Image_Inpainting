# === Cell 10 — Module colorad/models/mae_prior.py ===
"""Frozen MAE/ViT context prior (ASUKA).

FIX: replaced random.random() with torch.rand(1).item() so that the
ASUKA 50%-misalignment trick is reproducible under torch.manual_seed().
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class MAEPrior(nn.Module):
    """ViT-B/16 features as a context-stable prior. Always frozen."""

    def __init__(self, model_name: str = "vit_base_patch16_224",
                 pretrained: bool = True, freeze: bool = True) -> None:
        super().__init__()
        self.mae        = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.patch_size = 16
        self.grid       = 224 // 16          # 14
        self.num_patches = self.grid ** 2
        self.embed_dim  = self.mae.embed_dim
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(_IMAGENET_STD ).view(1, 3, 1, 1))
        if freeze:
            for p in self.mae.parameters():
                p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, image: torch.Tensor, mask: torch.Tensor,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """image (B,3,H,W) [-1,1]; mask (B,1,H,W). Returns features + patch_mask."""
        img01   = (image * 0.5 + 0.5).clamp(0, 1)
        img224  = F.interpolate(img01, (224, 224), mode="bilinear", align_corners=False)
        mask224 = F.interpolate(mask,  (224, 224), mode="nearest")

        # ASUKA misalignment trick: 50 % of the time feed the full image so the
        # alignment head cannot learn to ignore the prior.
        # FIX: use torch.rand (not random.random) for PyTorch-seed reproducibility.
        if training and torch.rand(1).item() < 0.5:
            inp = img224
        else:
            inp = img224 * (1 - mask224)

        inp          = (inp - self.mean) / self.std
        tokens       = self.mae.forward_features(inp)          # (B, 1+196, C)
        patch_tokens = tokens[:, 1:, :]                        # drop CLS
        b            = image.shape[0]
        feats        = patch_tokens.permute(0, 2, 1).reshape(
                           b, self.embed_dim, self.grid, self.grid)
        patch_mask   = F.avg_pool2d(mask224, kernel_size=16, stride=16)
        patch_mask   = (patch_mask > 0.5).float()              # (B,1,14,14)
        return feats, patch_mask
