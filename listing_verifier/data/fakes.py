"""Fake generation from ABO itself, in resumable chunks.

Modes
  t2i : prompt built from the item name / colour / material -> new image.
  i2i : the real ABO photo, upscaled to 512, is re-drawn at a given strength.
        Low strength (0.3) keeps framing and background identical to the real,
        which is the hardest, most confound-free kind of fake.

Every fake is saved at the canonical size (longest side 256) as JPEG q=92 so
reals and fakes have the same storage history before feature extraction.

Geometry (evaluation P1.4): generators output squares, ABO photos are not
square. The i2i source is letterboxed onto a white square (never stretched),
and every fake is centre-cropped back to its source listing's aspect ratio
before saving, so shape carries no label information.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image

from ..config import CANON_JPEG_QUALITY, CANON_MAX_SIDE, FAKE_IMAGE_SIZE, FAKE_KEEP_PNG_EVERY, GENERATORS
from ..features.device import describe, is_cuda, pick_dtype
from ..features.image_ops import crop_to_aspect, letterbox_to_square, load_rgb, resize_max_side

NEGATIVE_PROMPT = "blurry, text, watermark, logo, collage, multiple views, deformed"


def build_prompt(row: Dict) -> str:
    bits = ["product photo of"]
    colour = row.get("colour")
    material = row.get("material")
    name = str(row.get("item_name") or row.get("product_noun") or "item")
    name = name[:120]
    desc = name
    if isinstance(colour, str) and colour and colour.lower() not in name.lower():
        desc = f"{colour} {desc}"
    bits.append(desc)
    if isinstance(material, str) and material:
        bits.append(f", made of {material}")
    bits.append(", white background, studio lighting, centered, e-commerce listing")
    return " ".join(bits).replace(" ,", ",")


class FakeGenerator:
    def __init__(self, generator: str, device: str = "cuda", cache_dir: Optional[str] = None):
        import torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        if generator not in GENERATORS:
            raise ValueError(f"unknown generator {generator}; choose from {list(GENERATORS)}")
        self.generator = generator
        self.device = device
        model_id, self.licence = GENERATORS[generator]
        dtype = pick_dtype(device, "fp16")          # fp16 on any cuda:N, fp32 on CPU
        kwargs = {"torch_dtype": dtype, "cache_dir": cache_dir}
        if generator == "sdxl-turbo":
            kwargs["variant"] = "fp16"
        elif generator == "sd15":
            kwargs["safety_checker"] = None   # product photos only; avoids spurious black images
        self.t2i = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs).to(device)
        try:
            self.i2i = AutoPipelineForImage2Image.from_pipe(self.t2i).to(device)
        except Exception:  # Kandinsky's combined pipelines do not always share components via from_pipe
            self.i2i = AutoPipelineForImage2Image.from_pretrained(model_id, **kwargs).to(device)
        for p in (self.t2i, self.i2i):
            p.set_progress_bar_config(disable=True)
        self.torch = torch
        self.dtype = dtype
        print(f"[fakes] {generator}: {describe(device, self.t2i.unet if hasattr(self.t2i, 'unet') else None)} dtype={dtype}")

    def _gen(self, seed: int):
        return self.torch.Generator(device=self.device).manual_seed(int(seed))

    def _steps_and_guidance(self):
        # (steps, guidance) per family; Kandinsky's recommended guidance is ~4.
        return {"sdxl-turbo": (2, 0.0), "sd15": (25, 7.5), "kandinsky22": (25, 4.0)}[self.generator]

    def text_to_image(self, prompt: str, seed: int, size: int = FAKE_IMAGE_SIZE) -> Image.Image:
        steps, guidance = self._steps_and_guidance()
        if self.generator == "sdxl-turbo":
            return self.t2i(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
                            height=size, width=size, generator=self._gen(seed)).images[0]
        return self.t2i(prompt=prompt, negative_prompt=NEGATIVE_PROMPT, num_inference_steps=steps,
                        guidance_scale=guidance, height=size, width=size, generator=self._gen(seed)).images[0]

    def image_to_image(self, prompt: str, init: Image.Image, strength: float, seed: int,
                       size: int = FAKE_IMAGE_SIZE) -> Image.Image:
        init = letterbox_to_square(init.convert("RGB"), size)
        steps, guidance = self._steps_and_guidance()
        if self.generator == "sdxl-turbo":
            # diffusers runs int(steps * strength) denoising steps; ask for at least 2 effective ones
            steps = max(2, math.ceil(2.0 / max(strength, 1e-3)))
            return self.i2i(prompt=prompt, image=init, num_inference_steps=steps, strength=strength,
                            guidance_scale=guidance, generator=self._gen(seed)).images[0]
        return self.i2i(prompt=prompt, negative_prompt=NEGATIVE_PROMPT, image=init, num_inference_steps=steps,
                        strength=strength, guidance_scale=guidance, generator=self._gen(seed)).images[0]


def effective_steps(generator: str, mode: str, strength: float) -> int:
    """What diffusers will actually run (recorded per image in the chunk CSV)."""
    steps = {"sdxl-turbo": 2, "sd15": 25, "kandinsky22": 25}[generator]
    if mode == "t2i":
        return steps
    if generator == "sdxl-turbo":
        steps = max(2, math.ceil(2.0 / max(strength, 1e-3)))
    return int(steps * strength)


def save_fake(img: Image.Image, path: Path, keep_png: bool = False, aspect=None):
    """Crop to the source aspect, resize to canonical size, JPEG always; optional
    lossless PNG twin at the SAME size so the diagnostic compares format only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if aspect:
        img = crop_to_aspect(img, int(aspect[0]), int(aspect[1]))
    small = resize_max_side(img, CANON_MAX_SIDE)
    if keep_png:
        small.save(path.with_suffix(".png"), format="PNG")
    small.save(path, format="JPEG", quality=CANON_JPEG_QUALITY)


def generate_chunk(gen: FakeGenerator, rows: pd.DataFrame, mode: str, strength: float, out_dir: Path,
                   chunk_id: int, seed_base: int = 0, keep_png_every: int = FAKE_KEEP_PNG_EVERY,
                   key_prefix: str = "fake", log_every: int = 50) -> Path:
    """Generate one chunk; skip images that already exist; write chunk_XXXX.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"chunk_{chunk_id:04d}.csv"
    records: List[Dict] = []
    for i, (_, row) in enumerate(rows.iterrows()):
        seed = seed_base + chunk_id * 100000 + i
        tag = f"{mode}" + (f"_s{strength:.2f}" if mode == "i2i" else "")
        fname = f"{row['item_id']}_{tag}_{seed}.jpg"
        path = out_dir / fname
        prompt = build_prompt(row)
        aspect = (row.get("aspect_w"), row.get("aspect_h"))
        if not (aspect[0] and aspect[1]) and row.get("source_path"):
            aspect = load_rgb(row["source_path"]).size       # phone set: read it from the file
        if not path.exists():
            if mode == "t2i":
                img = gen.text_to_image(prompt, seed)
            else:
                img = gen.image_to_image(prompt, load_rgb(row["source_path"]), strength, seed)
            save_fake(img, path, keep_png=bool(keep_png_every) and i % keep_png_every == 0, aspect=aspect)
        records.append({"image_key": f"{key_prefix}:{gen.generator}:{mode}:{strength:.2f}:{row['item_id']}:{seed}",
                        "item_id": row["item_id"], "path": str(path.resolve()), "generator": gen.generator,
                        "mode": mode, "strength": strength if mode == "i2i" else 0.0, "seed": seed,
                        "prompt": prompt, "source_path": row.get("source_path", ""), "licence": gen.licence,
                        "png_twin": path.with_suffix(".png").exists(),
                        "aspect_w": aspect[0], "aspect_h": aspect[1],
                        "effective_steps": effective_steps(gen.generator, mode, strength)})
        if log_every and (i + 1) % log_every == 0:
            print(f"  chunk {chunk_id}: {i + 1}/{len(rows)}")
    tmp = csv_path.with_suffix(".csv.tmp")
    pd.DataFrame(records).to_csv(tmp, index=False)
    tmp.replace(csv_path)          # a chunk CSV is either complete or absent
    return csv_path


def run_persist_command(cmd: Optional[str]):
    """Optional hook run after each chunk, e.g. a Kaggle dataset version bump."""
    if cmd:
        print(f"[persist] {cmd}")
        subprocess.run(cmd, shell=True, check=False)
