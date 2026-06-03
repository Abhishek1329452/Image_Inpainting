# === Cell 07 — Module colorad/config.py ===
"""Global configuration for COLORAD.

Resolution is set to 256 (not 512 as in the original spec): the model runs in
pixel space on an ADM-style U-Net, and 512px pixel-space diffusion is not
feasible on 2x T4 (15GB). 256px matches RAD's actual FFHQ setup.
"""
import torch

IMG_SIZE = 256                 # working resolution
COND_SIZE = IMG_SIZE // 8      # 32; conditioning grid (MAE/fusion) resolution
T_STEPS = 1000                 # diffusion timesteps
MAE_RES = 224                  # ViT input size

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"