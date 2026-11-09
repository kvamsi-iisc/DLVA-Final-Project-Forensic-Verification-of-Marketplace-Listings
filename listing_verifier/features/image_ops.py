"""Image canonicalisation and JPEG degradation.

One rule: every image is canonicalised (resize longest side to 256, JPEG q=92)
before anything looks at it. See config.py for why.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

from ..config import CANON_JPEG_QUALITY, CANON_MAX_SIDE, DCT_SIZE


def load_rgb(src) -> Image.Image:
    """ONE decoder for every entry point (file path, raw bytes, BytesIO or a PIL
    image from an upload): EXIF rotation applied, transparency composited on
    white. The demo server must use this too, or phone uploads would be
    preprocessed differently from training (evaluation P1.13)."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    img = src if isinstance(src, Image.Image) else Image.open(src)
    img = ImageOps.exif_transpose(img)  # phone photos carry rotation in EXIF
    if img.mode in ("RGBA", "LA", "P"):
        # Product PNGs with transparency: composite on white, like a catalogue page.
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        return bg
    return img.convert("RGB")


def resize_max_side(img: Image.Image, max_side: int = CANON_MAX_SIDE) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), subsampling="4:2:0", optimize=False)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def canonicalise(img: Image.Image, max_side: int = CANON_MAX_SIDE,
                 jpeg_quality: int = CANON_JPEG_QUALITY) -> Image.Image:
    return jpeg_roundtrip(resize_max_side(img, max_side), jpeg_quality)


def degrade(img: Image.Image, jpeg_quality: int = None) -> Image.Image:
    """Apply the robustness-curve degradation to an already canonical image."""
    if jpeg_quality is None:
        return img
    return jpeg_roundtrip(img, jpeg_quality)


def letterbox_to_square(img: Image.Image, size: int, fill=(255, 255, 255)) -> Image.Image:
    """Pad (never stretch) onto a square white canvas, object centred, like a catalogue page."""
    w, h = img.size
    scale = size / max(w, h)
    inner = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(inner, ((size - inner.width) // 2, (size - inner.height) // 2))
    return canvas


def crop_to_aspect(img: Image.Image, aspect_w: int, aspect_h: int) -> Image.Image:
    """Centre-crop to the aspect ratio aspect_w:aspect_h (the source listing's),
    so a generated square never betrays itself by its shape (evaluation P1.4)."""
    if not aspect_w or not aspect_h:
        return img
    w, h = img.size
    target = aspect_w / aspect_h
    if w / h > target:          # too wide
        nw = max(1, round(h * target))
        left = (w - nw) // 2
        return img.crop((left, 0, left + nw, h))
    nh = max(1, round(w / target))  # too tall
    top = (h - nh) // 2
    return img.crop((0, top, w, top + nh))


def to_square_gray(img: Image.Image, size: int = DCT_SIZE) -> np.ndarray:
    """Grayscale, resized (not cropped: product shots keep the object centred)."""
    g = img.convert("L").resize((size, size), Image.BICUBIC)
    return np.asarray(g, dtype=np.float32) / 255.0


def jpeg_bytes_size(img: Image.Image, quality: int) -> int:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    return buf.tell()
