# === Cell 16 — Module colorad/losses/losses.py ===
"""COLORAD training losses."""
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


class DiffusionLoss(nn.Module):
    """Masked noise-prediction MSE (RAD)."""

    def forward(self, pred_noise: torch.Tensor, true_noise: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        m = mask.expand_as(pred_noise)
        diff = (pred_noise - true_noise) ** 2
        return (diff * m).sum() / (m.sum() + 1e-8)


class ColorLoss(nn.Module):
    """Chrominance consistency at masked edges (ASUKA G@e signal)."""

    def forward(self, pred_image: torch.Tensor, original_image: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # clamp into (eps, 1) — rgb_to_lab uses x**(1/3) whose gradient is inf at 0
        pred_lab = kornia.color.rgb_to_lab((pred_image * 0.5 + 0.5).clamp(1e-4, 1.0))
        with torch.no_grad():
            orig_lab = kornia.color.rgb_to_lab((original_image * 0.5 + 0.5).clamp(1e-4, 1.0))
        edges = kornia.filters.spatial_gradient(orig_lab[:, :1])     # (B,1,2,H,W)
        edge_mag = edges.norm(dim=2)                                 # (B,1,H,W)
        thr = edge_mag.mean(dim=[2, 3], keepdim=True) * 1.5
        edge_map = (edge_mag > thr).float()
        edge_mask = edge_map * mask
        color_diff = (pred_lab[:, 1:] - orig_lab[:, 1:]) ** 2        # (B,2,H,W)
        em2 = edge_mask.expand_as(color_diff)
        return (color_diff * em2).sum() / (em2.sum() + 1e-8)


class ContextLoss(nn.Module):
    """CLIP semantic consistency of the masked region (frozen CLIP).

    FIX vs spec: do NOT wrap encode_image in no_grad, else no gradient reaches
    the generator. CLIP weights stay frozen via requires_grad=False; gradients
    flow through the (frozen) graph to the input image.
    """

    def __init__(self) -> None:
        super().__init__()
        import open_clip
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai")
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.register_buffer("mean", torch.tensor([0.4815, 0.4578, 0.4082]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.2686, 0.2613, 0.2758]).view(1, 3, 1, 1))

    def _prep(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mv = img.mean(dim=[2, 3], keepdim=True)
        region = img * mask + mv * (1 - mask)
        region = F.interpolate(region, (224, 224), mode="bilinear", align_corners=False)
        region = (region + 1) / 2
        return (region - self.mean) / self.std

    def forward(self, pred_image: torch.Tensor, original_image: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        fp = self.model.encode_image(self._prep(pred_image, mask))
        fo = self.model.encode_image(self._prep(original_image, mask))
        fp = F.normalize(fp, dim=-1)
        fo = F.normalize(fo, dim=-1)
        return 1 - (fp * fo).sum(dim=-1).mean()


def compute_total_loss(pred_noise, true_noise, alpha_bar, pred_image,
                       original_image, mask, phase: int = 1,
                       color_fn=None, context_fn=None
                       ) -> Tuple[torch.Tensor, Dict[str, float]]:
    l_diff = DiffusionLoss()(pred_noise, true_noise, mask)
    if phase == 1:
        return l_diff, {"diffusion": l_diff.item()}
    l_color = (color_fn or ColorLoss())(pred_image, original_image, mask)
    if phase == 2:
        total = l_diff + 0.5 * l_color
        return total, {"diffusion": l_diff.item(), "color": l_color.item()}
    l_ctx = context_fn(pred_image, original_image, mask)
    total = l_diff + 0.5 * l_color + 0.3 * l_ctx
    return total, {"diffusion": l_diff.item(), "color": l_color.item(),
                   "context": l_ctx.item()}