# === Cell 14 — Module colorad/models/harmonizer.py ===
"""In-model color harmonization decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n1 = nn.GroupNorm(8, out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.act(h + self.skip(x))


class ColorHarmonizationDecoder(nn.Module):
    """Multi-scale color/texture harmonizer with learned warp + alpha blend.

    Note: in pixel space (unlike ASUKA's latent SD) the unmasked region is
    already pristine, so this mainly smooths mask-boundary color/texture rather
    than correcting VAE-induced shifts.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.b1 = nn.Sequential(ResBlock(7, 64), ResBlock(64, 128), ResBlock(128, 128))
        self.b2 = nn.Sequential(ResBlock(7, 64), ResBlock(64, 128))
        self.b3 = nn.Sequential(ResBlock(7, 32), ResBlock(32, 64))
        self.merge = nn.Sequential(nn.Conv2d(128 + 128 + 64, 128, 1), nn.GELU())
        self.warpnet = nn.Sequential(
            nn.Conv2d(128 + 6, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Tanh())
        self.alphanet = nn.Sequential(
            nn.Conv2d(128 + 6, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid())

    @staticmethod
    def _color_stats(original: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lab = kornia.color.rgb_to_lab((original * 0.5 + 0.5).clamp(1e-4, 1.0))
        unmask = 1 - mask
        cnt = unmask.sum(dim=[2, 3]).clamp(min=1e-6)            # (B,1)
        mean = (lab * unmask).sum(dim=[2, 3]) / cnt              # (B,3)
        var = ((lab - mean[:, :, None, None]) ** 2 * unmask).sum(dim=[2, 3]) / cnt
        std = torch.sqrt(var + 1e-6)
        stats = torch.cat([mean, std], dim=-1) / 100.0          # rough normalize
        return stats                                            # (B,6)

    def forward(self, generated_image: torch.Tensor, original_image: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """All tensors in [-1,1] / {0,1}. Returns harmonized (B,3,H,W)."""
        b, _, h, w = generated_image.shape
        stats = self._color_stats(original_image, mask)
        x_in = torch.cat([generated_image, original_image, mask], dim=1)  # (B,7,H,W)

        f1 = self.b1(F.interpolate(x_in, scale_factor=0.25, mode="bilinear", align_corners=False))
        f2 = self.b2(F.interpolate(x_in, scale_factor=0.5, mode="bilinear", align_corners=False))
        f3 = self.b3(x_in)
        f1 = F.interpolate(f1, size=(h, w), mode="bilinear", align_corners=False)
        f2 = F.interpolate(f2, size=(h, w), mode="bilinear", align_corners=False)
        merged = self.merge(torch.cat([f1, f2, f3], dim=1))               # (B,128,H,W)

        color_map = stats.view(b, 6, 1, 1).expand(b, 6, h, w)
        warp_in = torch.cat([merged, color_map], dim=1)                   # (B,134,H,W)

        warp_delta = self.warpnet(warp_in) * 0.1
        warped = generated_image + warp_delta * mask
        alpha = self.alphanet(warp_in)
        harmonized = alpha * warped + (1 - alpha) * generated_image
        harmonized = harmonized * mask + original_image * (1 - mask)
        return harmonized.clamp(-1, 1)