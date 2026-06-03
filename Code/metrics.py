# === Cell 20 — Module colorad/metrics.py ===
"""Evaluation metrics (C@m and G@e implemented; FID/LPIPS/U-IDS via libs)."""
import torch
import torch.nn.functional as F
import kornia


@torch.no_grad()
def clip_at_mask(pred, gt, mask, clip_model, mean, std) -> float:
    """C@m: CLIP cosine sim of masked regions, scaled to [0,100]."""
    def prep(img):
        mv = img.mean(dim=[2, 3], keepdim=True)
        r = img * mask + mv * (1 - mask)
        r = F.interpolate(r, (224, 224), mode="bilinear", align_corners=False)
        return ((r + 1) / 2 - mean) / std
    fp = F.normalize(clip_model.encode_image(prep(pred)), dim=-1)
    fo = F.normalize(clip_model.encode_image(prep(gt)), dim=-1)
    sim = (fp * fo).sum(-1).mean().item()
    return max(0.0, sim) * 100.0


@torch.no_grad()
def gradient_at_edge(pred, gt, mask) -> float:
    """G@e: mean gradient difference at mask-boundary pixels (lower=better)."""
    pl = kornia.color.rgb_to_lab((pred * 0.5 + 0.5).clamp(0, 1))
    gl = kornia.color.rgb_to_lab((gt * 0.5 + 0.5).clamp(0, 1))
    gp = kornia.filters.spatial_gradient(pl).norm(dim=2)
    gg = kornia.filters.spatial_gradient(gl).norm(dim=2)
    edge = (kornia.filters.spatial_gradient(mask).norm(dim=2) > 0).float()
    diff = (gp - gg).abs() * edge
    return (diff.sum() / (edge.sum() + 1e-8)).item()