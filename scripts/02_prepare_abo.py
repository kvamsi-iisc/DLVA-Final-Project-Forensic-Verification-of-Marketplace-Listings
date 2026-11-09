"""Clean ABO metadata, assign family-based splits, select the working subset.

Writes data/processed/
  abo_all_clean.parquet   every listing with labels + split (for reference)
  abo_clean.csv           the selected working subset used by everything else
  cleaning_report.json, split_report.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import ROOT, dump_json, set_seed
from listing_verifier.config import PROCESSED_DIR, RAW_DIR, SPLIT_SEED
from listing_verifier.data.abo_metadata import clean_listings, cleaning_report, load_image_index, load_listings
from listing_verifier.data.splits import assign_splits, leakage_report


def select_subset(df: pd.DataFrame, per_split: dict, cap_per_type: int, seed: int) -> pd.DataFrame:
    """Prefer listings with a colour label (needed by the consistency branch),
    cap each product type so sofas do not dominate, then sample per split."""
    df = df.copy()
    df["has_colour"] = df["colour"].notna()
    df = df.sample(frac=1.0, random_state=seed)                       # shuffle once
    df = df.sort_values(["has_colour"], ascending=False, kind="stable")  # labelled first
    df = df.groupby(["split", "product_type"], dropna=False).head(cap_per_type)
    parts = []
    for split, n in per_split.items():
        sub = df[df["split"] == split]
        parts.append(sub.head(n) if len(sub) > n else sub)
    return pd.concat(parts).sort_values("item_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW_DIR / "abo"))
    ap.add_argument("--out-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--train", type=int, default=7000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--cap-per-type", type=int, default=250, help="max listings per (split, product_type)")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--limit", type=int, default=None, help="debug: read only this many raw listings")
    ap.add_argument("--copy-images", action="store_true",
                    help="copy the selected listings' images to processed/images so later stages need only data/processed")
    args = ap.parse_args()
    set_seed(args.seed)
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[prepare] loading listings ...")
    df = load_listings(raw / "listings" / "metadata", limit=args.limit)
    print(f"[prepare] {len(df)} raw listings")
    df = clean_listings(df)
    images = load_image_index(raw / "images" / "metadata" / "images.csv.gz")
    df = df.merge(images, left_on="main_image_id", right_on="image_id", how="left")
    df.loc[df["image_rel_path"].isna() & df["keep"], "drop_reason"] = "image_missing"
    df["keep"] = df["drop_reason"] == ""
    df = assign_splits(df, "family_key", seed=args.seed)
    rep = cleaning_report(df)
    dump_json(rep, out / "cleaning_report.json")
    df.to_parquet(out / "abo_all_clean.parquet", index=False)

    kept = df[df["keep"]]
    subset = select_subset(kept, {"train": args.train, "val": args.val, "test": args.test}, args.cap_per_type, args.seed)
    leak = leakage_report(subset)
    dump_json(leak, out / "split_report.json")
    if args.copy_images:
        import shutil
        dst_root = out / "images"
        for rel in subset["image_rel_path"]:
            dst = dst_root / rel
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(raw / "images" / "small" / rel, dst)
        print(f"[prepare] copied {len(subset)} images to {dst_root}")
    subset.to_csv(out / "abo_clean.csv", index=False)
    print("[prepare] cleaning:", {k: rep[k] for k in ("n_listings", "n_kept", "n_with_colour", "n_with_material", "n_families")})
    print("[prepare] drop reasons:", rep["drop_reasons"])
    print("[prepare] subset:", leak)
    print("[prepare] wrote", out / "abo_clean.csv")


if __name__ == "__main__":
    main()
