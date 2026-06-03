# === Cell 22 — Module colorad/inference/inpainter.py ===
"""High-level inference wrapper."""
import torch
from PIL import Image
from ..models.colorad import COLORAD
from ..utils.color_utils import preprocess, preprocess_mask, to_pil


class COLORADInpainter:
    def __init__(self, checkpoint_path: str = None, device: str = "cuda",
                 img_size: int = 256, cond_size: int = 32) -> None:
        self.device = device
        self.img_size = img_size
        self.model = COLORAD(cond_size=cond_size, mae_pretrained=checkpoint_path is None)
        if checkpoint_path:
            sd = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(sd.get("model", sd), strict=False)
        self.model.eval().to(device)

    @torch.no_grad()
    def inpaint(self, image_pil: Image.Image, mask_pil: Image.Image,
                num_steps: int = 100) -> Image.Image:
        orig_size = image_pil.size
        image = preprocess(image_pil, self.img_size).unsqueeze(0).to(self.device)
        mask = preprocess_mask(mask_pil, self.img_size).unsqueeze(0).to(self.device)
        out = self.model.inpaint(image, mask, num_steps=num_steps)
        return to_pil(out.squeeze(0)).resize(orig_size, Image.LANCZOS)