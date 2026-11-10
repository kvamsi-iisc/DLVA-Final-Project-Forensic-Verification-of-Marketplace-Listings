"""One table for every image the project touches: data/processed/index.csv.

Columns
  image_key   unique id, e.g. 'abo:B07XYZ' or 'fake:sdxl-turbo:i2i:0.5:B07XYZ:3'
  path        absolute path to the stored JPEG
  domain      catalogue | phone | forensics
  source      abo | fake | forensics | phone | phone_fake
  label       0 real, 1 fake
  generator   'real' or the generator family ('sdxl-turbo', 'sd15', 'cf:<name>')
  mode        none | t2i | i2i ;  strength (i2i only)
  split       train | val | test | phone | forensics   (family-inherited for fakes)
  item_id, family_key, product_noun, colour, material, count, count_verifiable
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config import FAKES_DIR, FORENSICS_DIR, PHONE_DIR, PROCESSED_DIR, RAW_DIR


def default_images_root() -> Path:
    """processed/images (the copied working subset) if it exists, else the raw ABO tree."""
    p = PROCESSED_DIR / "images"
    return p if p.exists() else RAW_DIR / "abo" / "images" / "small"


def _relocate(paths, base_dir: Path, sub: str = ""):
    """Absolute paths recorded on another machine (Kaggle) are re-pointed to
    base_dir/[sub]/<filename> when they no longer exist here."""
    out = []
    for p in paths:
        pp = Path(str(p))
        if pp.exists():
            out.append(str(pp))
        else:
            cand = Path(base_dir) / sub / pp.name
            out.append(str(cand) if cand.exists() else str(pp))
    return out

INDEX_COLUMNS = ["image_key", "path", "domain", "source", "label", "generator", "mode", "strength",
                 "split", "item_id", "family_key", "product_noun", "colour", "material", "count",
                 "count_verifiable", "item_name"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=INDEX_COLUMNS)


def one_row_per_item(abo_clean: pd.DataFrame) -> pd.DataFrame:
    """abo_clean.csv files written before 2026-11-10 can hold the same item_id twice
    (one row per ABO marketplace). Keep the first; the image key 'abo:<item_id>' must be unique."""
    dup = abo_clean["item_id"].duplicated(keep="first")
    if dup.any():
        print(f"[index] warning: {int(dup.sum())} repeated item_id rows in abo_clean (marketplace duplicates); keeping the first of each")
        return abo_clean[~dup]
    return abo_clean


def reals_from_abo(abo_clean: pd.DataFrame, images_root: Path) -> pd.DataFrame:
    abo_clean = one_row_per_item(abo_clean)
    df = abo_clean[abo_clean["keep"] & abo_clean["image_rel_path"].notna()].copy()
    out = pd.DataFrame({
        "image_key": "abo:" + df["item_id"].astype(str),
        "path": [str(Path(images_root) / p) for p in df["image_rel_path"]],
        "domain": "catalogue", "source": "abo", "label": 0, "generator": "real", "mode": "none",
        "strength": 0.0, "split": df["split"].values, "item_id": df["item_id"].values,
        "family_key": df["family_key"].values, "product_noun": df["product_noun"].values,
        "colour": df["colour"].values, "material": df["material"].values, "count": df["count"].values,
        "count_verifiable": df["count_verifiable"].values, "item_name": df["item_name"].values,
    })
    return out[INDEX_COLUMNS]


def fakes_from_chunks(fakes_dir: Path, abo_clean: Optional[pd.DataFrame], domain: str = "catalogue",
                      source: str = "fake") -> pd.DataFrame:
    csvs = sorted(Path(fakes_dir).rglob("chunk_*.csv"))
    if not csvs:
        return _empty()
    frames = []
    for c in csvs:
        f = pd.read_csv(c)
        f["path"] = _relocate(f["path"], c.parent)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df[df["path"].map(lambda p: Path(p).exists())]
    if len(df) < before:
        print(f"[index] warning: {before - len(df)} fake rows point at missing files (overwritten chunk CSV?)")
    dup = df["image_key"].duplicated(keep="first")
    if dup.any():
        print(f"[index] warning: {int(dup.sum())} duplicate fake keys across chunk CSVs; keeping the first")
        df = df[~dup]
    if abo_clean is not None:
        meta = one_row_per_item(abo_clean).set_index("item_id")[["split", "family_key", "product_noun", "colour", "material",
                                                                 "count", "count_verifiable", "item_name"]]
        df = df.join(meta, on="item_id", how="left")
    else:
        for c in ["family_key", "product_noun", "colour", "material", "count", "count_verifiable", "item_name"]:
            df[c] = None
    out = pd.DataFrame({
        "image_key": df["image_key"], "path": df["path"], "domain": domain, "source": source, "label": 1,
        "generator": df["generator"], "mode": df["mode"], "strength": df["strength"].fillna(0.0),
        "split": df["split"] if "split" in df else "phone", "item_id": df["item_id"],
        "family_key": df["family_key"], "product_noun": df["product_noun"], "colour": df["colour"],
        "material": df["material"], "count": df["count"], "count_verifiable": df["count_verifiable"],
        "item_name": df["item_name"],
    })
    return out[INDEX_COLUMNS]


def forensics_rows(forensics_dir: Path) -> pd.DataFrame:
    csv = Path(forensics_dir) / "forensics_subset.csv"
    if not csv.exists():
        return _empty()
    df = pd.read_csv(csv)
    df["path"] = [a if Path(a).exists() else str(Path(forensics_dir) / ("fake" if l else "real") / Path(a).name)
                  for a, l in zip(df["path"], df["label"])]
    df = df[df["path"].map(lambda p: Path(p).exists())]
    out = pd.DataFrame({
        "image_key": df["image_key"], "path": df["path"], "domain": "forensics", "source": "forensics",
        "label": df["label"].astype(int), "generator": df["generator"], "mode": "none", "strength": 0.0,
        "split": "forensics", "item_id": None, "family_key": None, "product_noun": None, "colour": None,
        "material": None, "count": None, "count_verifiable": False, "item_name": None,
    })
    return out[INDEX_COLUMNS]


def phone_rows(phone_dir: Path) -> pd.DataFrame:
    """Own phone photos listed in data/phone/labels.csv (see docs/PHONE_SET.md)."""
    csv = Path(phone_dir) / "labels.csv"
    if not csv.exists():
        return _empty()
    df = pd.read_csv(csv)
    df["path"] = [str(Path(phone_dir) / "real" / f) for f in df["filename"]]
    df = df[df["path"].map(lambda p: Path(p).exists())]
    df["item_id"] = df["filename"].map(lambda f: "phone_" + Path(f).stem)
    out = pd.DataFrame({
        "image_key": "phone:" + df["item_id"], "path": df["path"], "domain": "phone", "source": "phone",
        "label": 0, "generator": "real", "mode": "none", "strength": 0.0, "split": "phone",
        "item_id": df["item_id"], "family_key": "phone|" + df["item_id"],
        "product_noun": df.get("product_noun", pd.Series(["item"] * len(df))).fillna("item"),
        "colour": df.get("colour"), "material": df.get("material"), "count": df.get("count"),
        "count_verifiable": df.get("count", pd.Series([None] * len(df))).map(lambda c: pd.notna(c) and 1 <= int(c) <= 6),
        "item_name": df.get("item_name"),
    })
    return out[INDEX_COLUMNS]


def build_index(abo_clean: pd.DataFrame, images_root: Path, fakes_dir: Path = FAKES_DIR,
                forensics_dir: Path = FORENSICS_DIR, phone_dir: Path = PHONE_DIR) -> pd.DataFrame:
    parts: List[pd.DataFrame] = [reals_from_abo(abo_clean, images_root),
                                 fakes_from_chunks(fakes_dir, abo_clean)]
    parts.append(forensics_rows(forensics_dir))
    phone = phone_rows(phone_dir)
    parts.append(phone)
    phone_fakes_dir = Path(phone_dir) / "fake"
    if phone_fakes_dir.exists():
        pf = fakes_from_chunks(phone_fakes_dir, None, domain="phone", source="phone_fake")
        pf["split"] = "phone"
        if len(phone):
            meta = phone.set_index("item_id")[["family_key", "product_noun", "colour", "material", "count", "item_name"]]
            for c in meta.columns:
                pf[c] = pf["item_id"].map(meta[c])
        parts.append(pf)
    idx = pd.concat([p for p in parts if len(p)], ignore_index=True)
    assert idx["image_key"].is_unique, "duplicate image keys in index"
    return idx


def load_index(path: Path = PROCESSED_DIR / "index.csv") -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)
