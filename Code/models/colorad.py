# === Cell 15 — Module colorad/models/colorad.py ===
"""COLORAD: Color-Aware Region Diffusion for image inpainting.

ADDED: DDIM inference (ddim=True, default) — deterministic 50-step sampling
  replaces 100-step stochastic DDPM; ~2× faster, comparable quality.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn

from .color_scheduler  import ColorAwareNoiseScheduler
from .spatial_embed    import SpatialNoiseEmbedding
from .mae_prior        import MAEPrior
from .mae_align        import MAEAlignmentModule
from .embed_fusion     import EmbeddingFusion
from .lora_unet        import LoRAUNet
from .harmonizer       import ColorHarmonizationDecoder


class COLORAD(nn.Module):
    def __init__(self, cond_size: int = 32, lora_rank: int = 16,
                 model_channels: int = 64, mae_pretrained: bool = True) -> None:
        super().__init__()
        self.cond_size       = cond_size
        self.noise_scheduler = ColorAwareNoiseScheduler(T=1000)
        self.spatial_embed   = SpatialNoiseEmbedding(embed_dim=256)
        self.mae_prior       = MAEPrior(pretrained=mae_pretrained, freeze=True)
        self.mae_align       = MAEAlignmentModule(
                                   mae_dim=self.mae_prior.embed_dim,
                                   out_dim=256, cond_size=cond_size)
        self.embed_fusion    = EmbeddingFusion(cond_size=cond_size)
        self.unet            = LoRAUNet(lora_rank=lora_rank,
                                        model_channels=model_channels,
                                        ctx_dim=256, time_ch=256)
        self.harmonizer      = ColorHarmonizationDecoder()

    # ── Training forward ─────────────────────────────────────────────────────
    def forward(self, image: torch.Tensor, mask: torch.Tensor,
                t: Optional[torch.Tensor] = None, training: bool = True):
        if not training:
            return self.inpaint(image, mask)
        b = image.shape[0]
        if t is None:
            t = torch.randint(0, self.noise_scheduler.T, (b,), device=image.device)
        noisy, noise, beta, alpha_bar = self.noise_scheduler.add_noise(image, mask, t)
        spatial_emb = self.spatial_embed(alpha_bar[:, :1])
        mae_feats, _ = self.mae_prior(image, mask, training=True)
        mae_emb      = self.mae_align(mae_feats)
        combined     = self.embed_fusion(spatial_emb, mae_emb)
        pred_noise   = self.unet(noisy, spatial_emb, combined)
        return pred_noise, noise, alpha_bar, beta

    # ── Inference ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def inpaint(self, image: torch.Tensor, mask: torch.Tensor,
                num_steps: int = 50, ddim: bool = True) -> torch.Tensor:
        """Reverse diffusion + harmonization.

        Args:
            num_steps: 50 DDIM ≈ quality of 200 DDPM steps.  Use 20 for quick
                       previews and 100 for publication-quality results.
            ddim: True  → deterministic DDIM (η=0), recommended.
                  False → stochastic ancestral DDPM (original behaviour).
        """
        b            = image.shape[0]
        mae_feats, _ = self.mae_prior(image, mask, training=False)
        mae_emb      = self.mae_align(mae_feats)          # precomputed once

        x     = image * (1 - mask) + torch.randn_like(image) * mask
        steps = torch.linspace(self.noise_scheduler.T - 1, 0, num_steps).long()

        for i, t_cur in enumerate(steps):
            t_batch    = torch.full((b,), int(t_cur), device=x.device, dtype=torch.long)
            ab_t       = self.noise_scheduler.get_alpha_bar(image, mask, t_batch)
            spatial_emb = self.spatial_embed(ab_t[:, :1])
            combined    = self.embed_fusion(spatial_emb, mae_emb)
            pred_noise  = self.unet(x, spatial_emb, combined)

            # x0 prediction
            x0 = ((x - torch.sqrt(1 - ab_t) * pred_noise)
                  / torch.sqrt(ab_t).clamp(min=1e-8)).clamp(-1, 1)

            if i < num_steps - 1:
                t_prev  = torch.full((b,), int(steps[i + 1]),
                                     device=x.device, dtype=torch.long)
                ab_prev = self.noise_scheduler.get_alpha_bar(image, mask, t_prev)

                if ddim:
                    # Deterministic DDIM update (η=0)
                    x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * pred_noise
                else:
                    # Stochastic DDPM ancestral sampling
                    alpha_t  = (ab_t / ab_prev).clamp(1e-6, 1.0)
                    beta_eff = (1 - alpha_t).clamp(1e-6, 0.999)
                    coef_x0  = (torch.sqrt(ab_prev) * beta_eff
                                / (1 - ab_t).clamp(min=1e-6))
                    coef_xt  = (torch.sqrt(alpha_t) * (1 - ab_prev)
                                / (1 - ab_t).clamp(min=1e-6))
                    mean     = coef_x0 * x0 + coef_xt * x
                    var      = (beta_eff * (1 - ab_prev)
                                / (1 - ab_t).clamp(min=1e-6))
                    x = mean + torch.sqrt(var.clamp(min=0)) * torch.randn_like(x)
            else:
                x = x0

            x = x * mask + image * (1 - mask)   # preserve unmasked region

        return self.harmonizer(x, image, mask).clamp(-1, 1)
