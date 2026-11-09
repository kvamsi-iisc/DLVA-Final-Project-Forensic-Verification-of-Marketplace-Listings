"""Helpers for the own-phone-photo split (docs/PHONE_SET.md explains the protocol)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import PHONE_DIR

LABELS_TEMPLATE_COLUMNS = ["filename", "item_name", "product_noun", "colour", "material", "count", "notes"]


def write_labels_template(phone_dir: Path = PHONE_DIR) -> Path:
    phone_dir = Path(phone_dir)
    (phone_dir / "real").mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in (phone_dir / "real").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"})
    df = pd.DataFrame({"filename": files})
    for c in LABELS_TEMPLATE_COLUMNS[1:]:
        df[c] = ""
    out = phone_dir / "labels_template.csv"
    df.to_csv(out, index=False)
    return out


def phone_rows_for_generation(phone_dir: Path = PHONE_DIR) -> pd.DataFrame:
    """Rows in the shape generate_chunk expects (item_id, item_name, colour, material, source_path)."""
    df = pd.read_csv(Path(phone_dir) / "labels.csv")
    df["item_id"] = df["filename"].map(lambda f: "phone_" + Path(f).stem)
    df["source_path"] = [str(Path(phone_dir) / "real" / f) for f in df["filename"]]
    df["product_noun"] = df.get("product_noun", "item")
    return df
