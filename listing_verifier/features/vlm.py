"""SmolVLM-500M as a multiple-choice oracle.

Design rule (risk 5 in the assessment): never ask open questions. Every query
is "choose one of these options", and we read the model's answer from the
next-token logits over the option letters, which gives a probability vector
instead of free text to parse. Generation is kept only as a debug fallback.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..config import VLM_MODEL_ID
from .device import pick_dtype

LETTERS = "ABCDEFGHIJKLMNOP"


def format_mc_question(question: str, options: List[str]) -> str:
    lines = [question.strip(), "Choose exactly one option and answer with just the letter."]
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {opt}")
    lines.append("Answer:")
    return "\n".join(lines)


class SmolVLM:
    def __init__(self, device: Optional[str] = None, model_id: str = VLM_MODEL_ID,
                 cache_dir: Optional[str] = None, image_splitting: bool = False):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = cache_dir or os.environ.get("LV_HF_CACHE")
        dtype = pick_dtype(self.device, "fp16")      # fp16 on T4 (no native bf16); fp32 on CPU
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        # One global view of a 256 px product shot is plenty; sub-image splitting
        # multiplies image tokens by ~5 and slows the CPU demo for no gain.
        if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "do_image_splitting"):
            self.processor.image_processor.do_image_splitting = image_splitting
        self.model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache_dir)
        self.model.to(self.device).eval()
        self._letter_ids = self._build_letter_ids()

    def _build_letter_ids(self):
        tok = self.processor.tokenizer
        ids = {}
        for L in LETTERS:
            variants = set()
            for s in (L, " " + L):
                enc = tok.encode(s, add_special_tokens=False)
                if enc:
                    variants.add(enc[-1])
            ids[L] = sorted(variants)
        missing = [L for L in LETTERS[:6] if not ids[L]]
        if missing:
            raise RuntimeError(f"tokenizer has no single-token id for option letters {missing}; multiple-choice scoring would be invalid")
        return ids

    def _prompt(self, text: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    @torch.no_grad()
    def option_probabilities(self, image: Image.Image, question: str, options: List[str]) -> np.ndarray:
        """P(option) from next-token logits restricted to the option letters."""
        assert 2 <= len(options) <= len(LETTERS)
        prompt = self._prompt(format_mc_question(question, options))
        inputs = self.processor(text=prompt, images=[image.convert("RGB")], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits[0, -1].float()
        scores = []
        for i in range(len(options)):
            ids = self._letter_ids[LETTERS[i]]
            scores.append(torch.logsumexp(logits[ids], dim=0).item() if ids else -1e9)
        scores = np.array(scores, dtype=np.float64)
        scores -= scores.max()
        p = np.exp(scores)
        p = (p / p.sum()).astype(np.float32)
        if not np.all(np.isfinite(p)):
            raise RuntimeError("non-finite option probabilities from the VLM (dtype/overflow problem?)")
        return p

    @torch.no_grad()
    def generate_answer(self, image: Image.Image, question: str, options: List[str],
                        max_new_tokens: int = 6) -> Tuple[Optional[int], str]:
        prompt = self._prompt(format_mc_question(question, options))
        inputs = self.processor(text=prompt, images=[image.convert("RGB")], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = re.search(r"\b([A-P])\b", text.strip().upper())
        idx = LETTERS.index(m.group(1)) if m else None
        if idx is not None and idx >= len(options):
            idx = None
        return idx, text
