# === Cell 21 — Module colorad/training/trainer.py ===
"""Phase-aware trainer for COLORAD.

FIXES:
  - Phase 2 LoRA: ALL LoRA params enabled (was: every-10th — too sparse for
    a small model and prevented colour/harmonizer signal reaching the U-Net).
  - Harmonizer gradient: .detach() removed in Phase 2 so the harmonizer
    reconstruction loss can propagate into the U-Net and improve its outputs.
    .detach() is kept in Phase 3 to avoid unstable gradients with the CLIP loss.
"""
from typing import Dict
import torch
from torch.optim.lr_scheduler import (CosineAnnealingLR, StepLR,
                                      LinearLR, SequentialLR)
from ..losses.losses import compute_total_loss, ColorLoss


def _set_lr(opt, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


class COLORADTrainer:
    """Handles phase freezing, optimizer, schedulers, and a single train step."""

    def __init__(self, model, device: str = "cuda", use_amp: bool = True) -> None:
        self.model      = model
        self.core       = model.module if hasattr(model, "module") else model
        self.device     = device
        self.use_amp    = use_amp
        self.optimizer  = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4)
        self.scheduler  = None
        self.scaler     = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.color_fn   = ColorLoss().to(device)
        self.context_fn = None

    def set_phase(self, phase: int) -> None:
        m = self.core
        for p in m.parameters():
            p.requires_grad = False

        if phase == 1:
            # No pretrained base → train full U-Net + conditioning modules.
            for mod in (m.spatial_embed, m.mae_align, m.embed_fusion, m.unet):
                for p in mod.parameters():
                    p.requires_grad = True
            lr = 1e-4
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=80_000, eta_min=1e-6)

        elif phase == 2:
            for mod in (m.spatial_embed, m.mae_align, m.embed_fusion, m.harmonizer):
                for p in mod.parameters():
                    p.requires_grad = True
            m.noise_scheduler.color_weight.requires_grad = True
            # FIX: enable ALL LoRA params (original every-10th was too sparse)
            for n, p in m.unet.named_parameters():
                if "lora" in n:
                    p.requires_grad = True
            lr = 5e-5
            self.scheduler = StepLR(self.optimizer, step_size=10_000, gamma=0.5)

        else:  # phase 3
            for p in m.parameters():
                p.requires_grad = True
            for p in m.mae_prior.parameters():
                p.requires_grad = False         # keep ViT frozen throughout
            if self.context_fn is None:
                from ..losses.losses import ContextLoss
                self.context_fn = ContextLoss().to(self.device)
            lr = 2e-5
            warm = LinearLR(self.optimizer, start_factor=0.1, total_iters=500)
            cos  = CosineAnnealingLR(self.optimizer, T_max=20_000)
            self.scheduler = SequentialLR(
                self.optimizer, [warm, cos], milestones=[500])

        # Rebuild optimizer over currently-trainable params
        self.optimizer = torch.optim.Adam(
            [p for p in m.parameters() if p.requires_grad], lr=lr)
        _set_lr(self.optimizer, lr)

    def train_step(self, image: torch.Tensor, mask: torch.Tensor,
                   phase: int = 1) -> Dict[str, float]:
        image, mask = image.to(self.device), mask.to(self.device)
        b = image.shape[0]
        t = torch.randint(0, self.core.noise_scheduler.T, (b,), device=self.device)

        self.optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=self.use_amp and phase < 3):
            pred_noise, true_noise, alpha_bar, beta = self.model(image, mask, t, True)
            ab_safe = alpha_bar.clamp(min=1e-3)
            x0_hat  = ((image - torch.sqrt((1 - alpha_bar).clamp(min=0)) * pred_noise)
                       / torch.sqrt(ab_safe)).clamp(-1, 1)
            loss, metrics = compute_total_loss(
                pred_noise, true_noise, alpha_bar, x0_hat, image, mask,
                phase=phase, color_fn=self.color_fn, context_fn=self.context_fn)

            if phase >= 2:
                # FIX: In Phase 2, remove .detach() so harmonizer reconstruction
                # loss backpropagates into the U-Net (teaches it to produce
                # outputs that are easier to harmonize).
                # In Phase 3, keep .detach() to avoid gradient conflicts with CLIP.
                harm_input = x0_hat.detach() if phase >= 3 else x0_hat
                harm   = self.core.harmonizer(harm_input, image, mask)
                l_harm = (((harm - image) ** 2) * mask).sum() / (mask.sum() + 1e-8)
                loss   = loss + 0.5 * l_harm
                metrics["harm"] = l_harm.item()

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        params  = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        finite  = all(
            p.grad is None or torch.isfinite(p.grad).all() for p in params)
        if finite:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
        else:
            self.optimizer.zero_grad()
            self.scaler.update()
        return metrics
