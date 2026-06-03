# === Cell 18 — Module colorad/utils/color_utils.py ===
"""Image/mask pre- and post-processing helpers."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def preprocess(image_pil: Image.Image, size: int = 256) -> torch.Tensor:
    """PIL RGB -> (3,size,size) in [-1,1]."""
    img = image_pil.convert("RGB").resize((size, size), Image.LANCZOS)
    t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    return t * 2 - 1


def preprocess_mask(mask_pil: Image.Image, size: int = 256) -> torch.Tensor:
    """PIL grayscale (white=masked) -> (1,size,size) in {0,1}."""
    m = mask_pil.convert("L").resize((size, size), Image.NEAREST)
    t = torch.from_numpy(np.array(m)).float().unsqueeze(0) / 255.0
    return (t > 0.5).float()


def to_pil(tensor: torch.Tensor) -> Image.Image:
    """(3,H,W) in [-1,1] -> PIL RGB."""
    t = ((tensor.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(t, "RGB")