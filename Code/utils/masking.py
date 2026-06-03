# === Cell 17 — colorad/utils/masking.py  [multi-type masks] ===
"""
Mask types:
  perlin_training : varied scale & threshold → mixed blob sizes
  wide            : large flowing blob, high coverage
  perlin_fine     : high scale → many small scattered blobs
  box             : clean centred rectangle
  extreme         : very small centred rectangle
  brush           : free-form strokes
  mixed           : random from all types (use this for training)
"""
import random
import numpy as np
import torch
from PIL import Image, ImageDraw


def _perlin(h: int, w: int, scale: float) -> np.ndarray:
    gh   = max(2, int(h * scale))
    gw   = max(2, int(w * scale))
    grid = np.random.rand(gh, gw).astype(np.float32)
    img  = np.array(
        Image.fromarray((grid * 255).astype(np.uint8))
             .resize((w, h), Image.BICUBIC))
    img  = img.astype(np.float32) / 255.0
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def _brush(h: int, w: int) -> np.ndarray:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for _ in range(random.randint(2, 5)):
        x, y = random.randint(0, w), random.randint(0, h)
        for _ in range(random.randint(5, 15)):
            x2 = min(max(0, x + random.randint(-w // 5, w // 5)), w)
            y2 = min(max(0, y + random.randint(-h // 5, h // 5)), h)
            d.line([x, y, x2, y2], fill=255, width=random.randint(12, 35))
            x, y = x2, y2
    return np.array(m, dtype=np.float32) / 255.0


def generate_mask(height: int, width: int,
                  strategy: str = "mixed") -> torch.Tensor:

    # ── pick strategy randomly when mixed ────────────────────────────────────
    if strategy == "mixed":
        r = random.random()
        if   r < 0.40: strategy = "perlin_training"
        elif r < 0.58: strategy = "wide"
        elif r < 0.72: strategy = "perlin_fine"
        elif r < 0.83: strategy = "box"
        elif r < 0.92: strategy = "brush"
        else:          strategy = "extreme"

    # ── generate by type ─────────────────────────────────────────────────────
    if strategy == "perlin_training":
        # Wide threshold range → sometimes sparse, sometimes dense blobs
        scale = random.uniform(0.04, 0.12)
        thr   = random.uniform(0.22, 0.72)
        m = (_perlin(height, width, scale) > thr).astype(np.float32)

    elif strategy == "wide":
        # Large flowing coverage — low threshold, coarse grid
        scale = random.uniform(0.03, 0.06)
        thr   = random.uniform(0.12, 0.32)
        m = (_perlin(height, width, scale) > thr).astype(np.float32)

    elif strategy == "perlin_fine":
        # Many small scattered blobs — fine grid
        scale = random.uniform(0.12, 0.22)
        thr   = random.uniform(0.42, 0.65)
        m = (_perlin(height, width, scale) > thr).astype(np.float32)

    elif strategy == "box":
        m  = np.zeros((height, width), dtype=np.float32)
        bw = random.randint(width  // 4, width  // 2)
        bh = random.randint(height // 4, height // 2)
        x  = random.randint(width  // 8, width  - width  // 8 - bw)
        y  = random.randint(height // 8, height - height // 8 - bh)
        m[y:y + bh, x:x + bw] = 1.0

    elif strategy == "extreme":
        # Tiny centred box — very small mask
        m  = np.zeros((height, width), dtype=np.float32)
        bw = random.randint(width  // 8, width  // 5)
        bh = random.randint(height // 8, height // 5)
        x  = random.randint(width  // 3, width  * 2 // 3 - bw)
        y  = random.randint(height // 3, height * 2 // 3 - bh)
        m[y:y + bh, x:x + bw] = 1.0

    elif strategy == "brush":
        m = _brush(height, width)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    # ── safety clip ──────────────────────────────────────────────────────────
    ratio = m.mean()
    if ratio < 0.02:
        m[height * 3 // 8:height * 5 // 8,
          width  * 3 // 8:width  * 5 // 8] = 1.0
    elif ratio > 0.92:
        m[:, :] = 0.0
        m[height // 4:3 * height // 4,
          width  // 4:3 * width  // 4] = 1.0

    return torch.from_numpy(m).unsqueeze(0)