# === Cell 19 — Module colorad/utils/data.py ===
"""FFHQ (or any image folder) dataset for inpainting."""
import glob
import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
from .color_utils import preprocess
from .masking import generate_mask

_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def find_image_root(base: str) -> str:
    """Return the deepest folder under ``base`` that contains images."""
    if any(f.lower().endswith(_EXTS) for f in os.listdir(base)) if os.path.isdir(base) else False:
        return base
    for root, _, files in os.walk(base):
        if any(f.lower().endswith(_EXTS) for f in files):
            return root
    return base


class FFHQInpaintDataset(Dataset):
    """Loads images recursively; generates a fresh random mask per sample."""

    def __init__(self, root: str, img_size: int = 256, limit: int = None) -> None:
        self.img_size = img_size
        paths = []
        for ext in _EXTS:
            paths += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
        paths.sort()
        if limit:
            paths = paths[:limit]
        if not paths:
            raise FileNotFoundError(f"No images found under {root}")
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = preprocess(Image.open(self.paths[idx]), self.img_size)
        mask = generate_mask(self.img_size, self.img_size)
        return img, mask