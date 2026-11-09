"""The one frozen backbone: OpenAI CLIP ViT-L/14 via open_clip.

Nothing in this project fine-tunes it. Both heads (authenticity probe and
consistency zero-shot) read the same 768-d post-projection embedding, which is
exactly what UnivFD's released linear layer expects (see models/baselines.py).

Usage:
    backbone = ClipBackbone(device="cpu")
    feats = backbone.encode_images([pil, pil, ...])    # (N, 768) float32, L2-normalised
    tfe = backbone.encode_texts(["a photo of a red chair", ...])
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

from ..config import CLIP_MODEL_NAME, CLIP_PRETRAINED
from .device import is_cuda


class ClipBackbone:
    def __init__(self, device: Optional[str] = None, cache_dir: Optional[str] = None,
                 model_name: str = CLIP_MODEL_NAME, pretrained: str = CLIP_PRETRAINED,
                 fp16: Optional[bool] = None):
        import open_clip  # imported lazily so CPU-only tests do not need it

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = is_cuda(self.device) if fp16 is None else fp16   # autocast on any cuda:N
        cache_dir = cache_dir or os.environ.get("LV_CLIP_CACHE")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cache_dir, device=self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)       # frozen, explicitly
        self.model = model
        self.preprocess = preprocess      # resize 224 + centre crop + CLIP mean/std
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.logit_scale = float(model.logit_scale.exp().item())

    @torch.no_grad()
    def encode_images(self, images: Iterable[Image.Image], batch_size: int = 64,
                      normalise: bool = True) -> np.ndarray:
        out: List[np.ndarray] = []
        batch: List[torch.Tensor] = []

        def flush():
            if not batch:
                return
            x = torch.stack(batch).to(self.device)
            if self.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    f = self.model.encode_image(x)
            else:
                f = self.model.encode_image(x)
            f = f.float()
            if normalise:
                f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu().numpy())
            batch.clear()

        for img in images:
            batch.append(self.preprocess(img.convert("RGB")))
            if len(batch) >= batch_size:
                flush()
        flush()
        return np.concatenate(out, axis=0) if out else np.zeros((0, 768), dtype=np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], normalise: bool = True) -> np.ndarray:
        toks = self.tokenizer(texts).to(self.device)
        f = self.model.encode_text(toks).float()
        if normalise:
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()
