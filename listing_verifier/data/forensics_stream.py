"""Stream a small, generator-diverse subset of CommunityForensics from HF.

Never download the full dataset (~260 GB). We iterate the streaming split,
keep at most `per_generator` images per generator name and stop at `max_total`.
Column names differ between dataset versions, so run with --inspect first and
pass the right names; the auto-detection below covers the common ones.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from PIL import Image

from ..config import COMMUNITY_FORENSICS_ID
from ..features.image_ops import canonicalise

IMAGE_COL_GUESSES = ["image_data", "image", "img", "jpg", "png"]
ID_COL_GUESSES = ["image_name", "id", "filename", "name"]
LABEL_COL_GUESSES = ["label", "is_fake", "fake", "target"]
GEN_COL_GUESSES = ["model_name", "generator", "model", "source", "generator_name"]


def _guess(features: Dict, guesses) -> Optional[str]:
    for g in guesses:
        if g in features:
            return g
    return None


def _to_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, dict) and "bytes" in x:
        return Image.open(io.BytesIO(x["bytes"]))
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x))
    raise TypeError(f"cannot decode image of type {type(x)}")


def stream_subset(out_dir: Path, dataset_id: str = COMMUNITY_FORENSICS_ID, config: Optional[str] = None,
                  split: str = "train", per_generator: int = 40, max_total: int = 3000,
                  image_col: Optional[str] = None, label_col: Optional[str] = None,
                  generator_col: Optional[str] = None, fake_label_value=1, inspect: bool = False,
                  shuffle_buffer: int = 2000, seed: int = 0, n_real: Optional[int] = None,
                  n_fake: Optional[int] = None, max_scan: int = 200000, id_col: Optional[str] = None) -> pd.DataFrame:
    """Explicit class quotas (default half/half of max_total), a per-generator cap
    on fakes, a bound on scanned rows, stable source identities (image_name) and
    a resumable manifest: re-running continues from forensics_subset.csv."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, name=config, split=split, streaming=True)
    features = dict(ds.features) if ds.features else {}
    print("[forensics] features:", {k: str(v) for k, v in features.items()})
    if inspect:
        for i, ex in enumerate(ds):
            print({k: (f"<{len(v)} bytes>" if isinstance(v, (bytes, bytearray)) else
                       f"<image {getattr(v, 'size', '?')}>" if hasattr(v, "size") else
                       (str(v)[:80] if isinstance(v, str) else v)) for k, v in ex.items()})
            if i >= 2:
                break
        return pd.DataFrame()
    image_col = image_col or _guess(features, IMAGE_COL_GUESSES)
    label_col = label_col or _guess(features, LABEL_COL_GUESSES)
    generator_col = generator_col or _guess(features, GEN_COL_GUESSES)
    id_col = id_col or _guess(features, ID_COL_GUESSES)
    n_real = n_real if n_real is not None else max_total // 2
    n_fake = n_fake if n_fake is not None else max_total - n_real
    if image_col is None or label_col is None:
        raise ValueError(f"could not identify image/label columns in {list(features)}; pass them explicitly")
    print(f"[forensics] using image={image_col} label={label_col} generator={generator_col}")

    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    out_dir = Path(out_dir)
    (out_dir / "real").mkdir(parents=True, exist_ok=True)
    (out_dir / "fake").mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "forensics_subset.csv"
    rows = pd.read_csv(manifest).to_dict("records") if manifest.exists() else []
    seen_ids = {str(r.get("source_id")) for r in rows}
    counts: Dict[str, int] = {}
    for r in rows:   # the SAME key as used below ('cf:<model>' for fakes, 'real'), or caps reset on resume
        k = str(r["generator"]) if r["label"] else "real"
        counts[k] = counts.get(k, 0) + 1
    n_real_have = sum(1 for r in rows if not r["label"])
    n_fake_have = sum(1 for r in rows if r["label"])
    scanned = 0
    for ex in ds:
        scanned += 1
        if scanned > max_scan or (n_real_have >= n_real and n_fake_have >= n_fake):
            break
        raw_label = ex[label_col]
        is_fake = int(str(raw_label) == str(fake_label_value) or raw_label is True)
        gen = str(ex[generator_col]) if generator_col else ("fake" if is_fake else "real")
        gen_key = f"cf:{gen}" if is_fake else "real"
        # image names repeat across generators, so the identity is generator + name
        source_id = f"{gen_key}/{ex[id_col]}" if id_col else f"{split}:{scanned}"
        if source_id in seen_ids:
            continue
        if is_fake and (n_fake_have >= n_fake or counts.get(gen_key, 0) >= per_generator):
            continue
        if not is_fake and n_real_have >= n_real:
            continue
        try:
            img = canonicalise(_to_pil(ex[image_col]).convert("RGB"))
        except Exception as e:  # corrupt sample: skip, never crash a 1-hour stream
            print("[forensics] skip:", e)
            continue
        n = len(rows)
        sub = "fake" if is_fake else "real"
        path = out_dir / sub / f"{n:06d}.jpg"
        img.save(path, format="JPEG", quality=92)
        counts[gen_key] = counts.get(gen_key, 0) + 1
        n_fake_have += is_fake
        n_real_have += 1 - is_fake
        seen_ids.add(source_id)
        rows.append({"image_key": f"cf:{n:06d}", "path": str(path.resolve()), "label": is_fake,
                     "generator": f"cf:{gen}" if is_fake else "real", "raw_label": raw_label,
                     "source_id": source_id, "architecture": ex.get("architecture", ""), "subset": ex.get("subset", "")})
        if len(rows) % 200 == 0:
            print(f"[forensics] {len(rows)} saved ({n_real_have} real / {n_fake_have} fake), "
                  f"{len([k for k in counts if k != 'real'])} generators, {scanned} rows scanned")
            pd.DataFrame(rows).to_csv(manifest.with_suffix(".csv.tmp"), index=False)
            manifest.with_suffix(".csv.tmp").replace(manifest)
    df = pd.DataFrame(rows)
    df.to_csv(manifest.with_suffix(".csv.tmp"), index=False)
    manifest.with_suffix(".csv.tmp").replace(manifest)
    print("[forensics] done:", df["label"].value_counts().to_dict(), "generators:", df["generator"].nunique(),
          "scanned:", scanned)
    if n_real_have < n_real or n_fake_have < n_fake:
        print(f"[forensics] WARNING: quota not met (real {n_real_have}/{n_real}, fake {n_fake_have}/{n_fake}); raise --max-scan or lower quotas")
    return df
