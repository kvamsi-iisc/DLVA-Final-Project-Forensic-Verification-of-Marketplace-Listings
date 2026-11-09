"""Feature extraction over the index: CLIP embedding + DCT features per image,
at the canonical quality and at each JPEG robustness quality.

Output: data/features/features_q{Q}.npz with arrays
    image_key (N,) str, clip (N,768) f32, dct_map (N,64,64) f16, dct_radial (N,128) f32
Q = 0 means 'canonical, no extra degradation'.
Part files are written every `chunk_size` images so a disconnected session
resumes instead of restarting. Images already present in the final file are
skipped and the new ones are appended, so the train split can be added to a
degraded-quality file that already holds the test split (or a phone set can be
added later) without recomputing anything. With part_modulo=(i, n) a worker computes only
the parts with index % n == i and skips the merge; the launcher merges after
all workers finish (multi-GPU mode of scripts/06_extract_features.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import os

from ..config import (CANON_JPEG_QUALITY, CANON_MAX_SIDE, CLIP_MODEL_NAME, CLIP_PRETRAINED, DCT_MAP_SIZE,
                      DCT_RADIAL_BINS, DCT_SIZE, FEATURES_DIR)
from .dct import spectrum_features
from .image_ops import canonicalise, degrade, load_rgb, to_square_gray


def features_path(quality: int, features_dir: Path = FEATURES_DIR) -> Path:
    return Path(features_dir) / f"features_q{int(quality)}.npz"


def cache_fingerprint() -> str:
    """Everything that changes what a cached feature means. A cache written under
    a different fingerprint is refused instead of silently reused."""
    return (f"clip={CLIP_MODEL_NAME}/{CLIP_PRETRAINED};canon={CANON_MAX_SIDE}/{CANON_JPEG_QUALITY};"
            f"dct={DCT_SIZE}/{DCT_MAP_SIZE}/{DCT_RADIAL_BINS};fmt=1")


def atomic_savez(path: Path, **arrays):
    """Write to a temp file in the same directory, then rename: an interrupted
    run never leaves a half-written .npz that a resume would trust."""
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, fingerprint=np.array(cache_fingerprint()), **arrays)
    os.replace(tmp, path)


def check_fingerprint(z, path: Path):
    fp = str(z["fingerprint"]) if "fingerprint" in z.files else "<none>"
    if fp != cache_fingerprint():
        raise RuntimeError(f"{path} was written with fingerprint {fp!r}, current is {cache_fingerprint()!r}; "
                           f"delete the cache and re-extract")


def _extract_batch(backbone, paths: List[str], quality: int):
    imgs, maps, rads = [], [], []
    for p in paths:
        img = canonicalise(load_rgb(p))
        if quality:
            img = degrade(img, quality)
        imgs.append(img)
        f = spectrum_features(to_square_gray(img))
        maps.append(f["dct_map"])
        rads.append(f["dct_radial"])
    # stored UN-normalised: the probe normalises itself, UnivFD's released fc expects raw
    clip = backbone.encode_images(imgs, batch_size=len(imgs), normalise=False) if backbone is not None else np.zeros((len(imgs), 768), np.float32)
    return clip, np.stack(maps), np.stack(rads)


def part_path(quality: int, part: int, features_dir: Path) -> Path:
    return Path(features_dir) / f"features_q{quality}.part{part:03d}.npz"


def n_parts_for(n_images: int, chunk_size: int) -> int:
    return (n_images + chunk_size - 1) // chunk_size


def existing_keys(quality: int, features_dir: Path = FEATURES_DIR) -> set:
    final = features_path(quality, features_dir)
    if not final.exists():
        return set()
    z = np.load(final)
    check_fingerprint(z, final)
    return set(z["image_key"].astype(str).tolist())


def pending_index(index: pd.DataFrame, quality: int, features_dir: Path = FEATURES_DIR) -> pd.DataFrame:
    """Rows of `index` not yet in the cache for this quality (same function in
    the launcher and in the workers, so both agree on the part numbering)."""
    done = existing_keys(quality, features_dir)
    if not done:
        return index.reset_index(drop=True)
    return index[~index["image_key"].astype(str).isin(done)].reset_index(drop=True)


def merge_parts(quality: int, n_parts: int, features_dir: Path = FEATURES_DIR, verbose: bool = True) -> Path:
    """Concatenate part files (plus whatever the final file already holds) into the final file."""
    features_dir = Path(features_dir)
    final = features_path(quality, features_dir)
    part_paths = [part_path(quality, p, features_dir) for p in range(n_parts)]
    missing = [pp for pp in part_paths if not pp.exists()]
    if missing:
        raise FileNotFoundError(f"cannot merge quality {quality}: {len(missing)} part files missing, e.g. {missing[0]}")
    merged = {"image_key": [], "clip": [], "dct_map": [], "dct_radial": []}
    if final.exists():
        z = np.load(final)
        check_fingerprint(z, final)
        for k in merged:
            merged[k].append(z[k])
    for pp in part_paths:
        z = np.load(pp)
        check_fingerprint(z, pp)
        for k in merged:
            merged[k].append(z[k])
    atomic_savez(final, **{k: np.concatenate(v) for k, v in merged.items()})
    for pp in part_paths:
        pp.unlink()
    if verbose:
        print(f"[extract] merged {n_parts} parts -> {final}")
    return final


def extract(index: pd.DataFrame, backbone, quality: int = 0, features_dir: Path = FEATURES_DIR,
            chunk_size: int = 2000, batch_size: int = 64, verbose: bool = True,
            part_modulo=None, merge: bool = True) -> Optional[Path]:
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    index = pending_index(index, quality, features_dir)
    if len(index) == 0:
        if verbose:
            print(f"[extract q={quality}] nothing to do: all images already cached")
        return features_path(quality, features_dir) if merge else None
    keys = index["image_key"].astype(str).tolist()
    paths = index["path"].astype(str).tolist()
    n_parts = n_parts_for(len(keys), chunk_size)
    shard_i, shard_n = part_modulo or (0, 1)
    for part in range(n_parts):
        if part % shard_n != shard_i:
            continue
        pp = part_path(quality, part, features_dir)
        if pp.exists():
            continue
        lo, hi = part * chunk_size, min((part + 1) * chunk_size, len(keys))
        clips, maps, rads = [], [], []
        for i in range(lo, hi, batch_size):
            c, m, r = _extract_batch(backbone, paths[i:min(i + batch_size, hi)], quality)  # never cross the part boundary
            clips.append(c), maps.append(m), rads.append(r)
            if verbose and ((i - lo) // batch_size) % 10 == 0:
                print(f"[extract q={quality}] part {part + 1}/{n_parts} image {i}/{len(keys)}")
        clip_arr = np.concatenate(clips)
        assert len(clip_arr) == hi - lo, f"part {part}: {len(clip_arr)} features for {hi - lo} keys"
        atomic_savez(pp, image_key=np.array(keys[lo:hi]), clip=clip_arr,
                     dct_map=np.concatenate(maps), dct_radial=np.concatenate(rads))
    if not merge:
        return None
    return merge_parts(quality, n_parts, features_dir, verbose)


class FeatureStore:
    """Lazy access to cached features joined with the index."""

    def __init__(self, features_dir: Path = FEATURES_DIR):
        self.features_dir = Path(features_dir)
        self._cache = {}

    def load(self, quality: int = 0):
        if quality not in self._cache:
            path = features_path(quality, self.features_dir)
            z = np.load(path)
            check_fingerprint(z, path)
            keys = z["image_key"].astype(str)
            n = len(keys)
            if not (len(z["clip"]) == len(z["dct_map"]) == len(z["dct_radial"]) == n) or n == 0:
                raise RuntimeError(f"{path} is inconsistent or empty (keys {n}, clip {len(z['clip'])}); re-extract")
            for k in ("clip", "dct_map", "dct_radial"):
                if not np.all(np.isfinite(z[k])):
                    raise RuntimeError(f"{path}: non-finite values in {k} (fp16 overflow or a corrupt image); re-extract")
            self._cache[quality] = {"pos": {k: i for i, k in enumerate(keys)}, "clip": z["clip"],
                                    "dct_map": z["dct_map"], "dct_radial": z["dct_radial"]}
        return self._cache[quality]

    def get(self, image_keys, quality: int = 0):
        d = self.load(quality)
        idx = np.array([d["pos"][k] for k in image_keys if k in d["pos"]], dtype=np.int64)
        present = np.array([k in d["pos"] for k in image_keys], dtype=bool)
        return {"clip": d["clip"][idx], "dct_map": d["dct_map"][idx], "dct_radial": d["dct_radial"][idx],
                "present": present}

    def available_qualities(self) -> List[int]:
        return sorted(int(p.stem.split("_q")[1]) for p in self.features_dir.glob("features_q*.npz")
                      if ".part" not in p.name and ".tmp" not in p.name)
