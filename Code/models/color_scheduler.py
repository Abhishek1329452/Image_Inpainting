# === Cell 08 — Module colorad/models/color_scheduler.py ===
"""Color-aware spatially-variant noise scheduler (RAD-inspired).

FIXES:
  - Removed vestigial betas_prime (identical to betas; unmasked pixels are
    paste-backed in add_noise anyway, so a separate schedule is meaningless).
  - color_weight clamped positive via .abs() so high-gradient pixels always
    receive *more* noise (correct RAD semantics; previously could flip sign).
"""
from typing import Tuple, Union
import torch
import torch.nn as nn
import kornia


class ColorAwareNoiseScheduler(nn.Module):
    """Spatially-variant DDPM schedule modulated by local LAB color gradient."""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, color_weight: float = 0.2) -> None:
        super().__init__()
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        abar  = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("abar_table", abar)
        # Learnable; .abs() used at call sites so it stays non-negative
        self.color_weight = nn.Parameter(torch.tensor(float(color_weight)))

    def _color_gradient(self, image: torch.Tensor) -> torch.Tensor:
        """Normalised per-pixel LAB gradient magnitude. image in [-1,1]."""
        img01 = (image * 0.5 + 0.5).clamp(0, 1)
        lab   = kornia.color.rgb_to_lab(img01)
        grad  = kornia.filters.spatial_gradient(lab, mode="sobel")  # (B,3,2,H,W)
        mag   = torch.sqrt(grad[:, :, 0] ** 2 + grad[:, :, 1] ** 2 + 1e-12)
        cg    = mag.mean(dim=1, keepdim=True)
        b     = cg.shape[0]
        cmin  = cg.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        cmax  = cg.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        return (cg - cmin) / (cmax - cmin + 1e-8)

    def forward(self, image: torch.Tensor, mask: torch.Tensor,
                t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Spatial beta_t (B,3,H,W) — unmasked region never used (paste-back)."""
        cg = self._color_gradient(image)
        ci = 1.0 + self.color_weight.abs() * cg          # always >= 1
        if torch.is_tensor(t):
            beta_t = self.betas[t].view(-1, 1, 1, 1)
        else:
            beta_t = self.betas[t]
        return torch.clamp(beta_t * ci, 1e-6, 0.999).repeat(1, 3, 1, 1)

    def get_alpha_bar(self, image: torch.Tensor, mask: torch.Tensor,
                      t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Cumulative ᾱ_t (B,3,H,W). Unmasked pixels → 1 (no noise)."""
        cg = self._color_gradient(image)
        ci = 1.0 + self.color_weight.abs() * cg
        if not torch.is_tensor(t):
            t = torch.full((image.shape[0],), int(t),
                           device=image.device, dtype=torch.long)
        t_f   = t.float().view(-1, 1, 1, 1)
        t_eff = torch.clamp(t_f * ci, 0, self.T - 1)
        lo    = torch.floor(t_eff).long().clamp(0, self.T - 1)
        hi    = torch.clamp(lo + 1, max=self.T - 1)
        frac  = t_eff - lo.float()
        table = self.abar_table.to(image.device)
        ab_m  = table[lo] * (1 - frac) + table[hi] * frac
        return (ab_m * mask + 1.0 * (1 - mask)).repeat(1, 3, 1, 1)

    def add_noise(self, x0: torch.Tensor, mask: torch.Tensor,
                  t: Union[int, torch.Tensor]
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward-noise the masked region only."""
        noise = torch.randn_like(x0)
        beta  = self.forward(x0, mask, t)
        ab    = self.get_alpha_bar(x0, mask, t)
        noisy = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
        noisy = noisy * mask + x0 * (1 - mask)    # pristine unmasked region
        return noisy, noise, beta, ab
