"""Compute per-image option probabilities for the consistency branch and cache
them in data/features/consistency_probs.csv (resumable).

For each real listing (with at least one trusted attribute) and each attribute:
  clip_p : CLIP zero-shot softmax over the attribute's option list
  vlm_p  : SmolVLM multiple-choice probabilities over the same options
Both are computed ONCE per image, independent of any claim; the calibrator
later reads off the probability of whichever value the description claims.
Multi-GPU: --gpus auto. Worker i scores every n-th image into
consistency_probs.shard{i}.csv; the parent merges the shards into the main
file when all workers are done (resumable: shard files are read back too).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

from _common import ROOT, device_arg, parse_shard, resolve_gpus, spawn_per_gpu
from listing_verifier.config import FEATURES_DIR, PROCESSED_DIR
from listing_verifier.features.clip_backbone import ClipBackbone
from listing_verifier.features.extract import FeatureStore
from listing_verifier.features.image_ops import canonicalise, load_rgb
from listing_verifier.models.consistency import ClipZeroShot, VLM_QUESTIONS, vlm_options

PROBS_PATH = FEATURES_DIR / "consistency_probs.csv"


def fmt(p):
    return " ".join(f"{x:.5f}" for x in p)


def atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)          # a cache file is either complete or untouched


COLS = ["image_key", "attribute", "clip_p", "vlm_p"]


def merge_shards(out: Path) -> pd.DataFrame:
    """Main file + every shard file -> one de-duplicated table at `out`."""
    frames = [pd.read_csv(f) for f in [out] + sorted(out.parent.glob(out.stem + ".shard*.csv")) if f.exists()]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    df = df.drop_duplicates(["image_key", "attribute"], keep="last")
    atomic_csv(df, out)
    for f in out.parent.glob(out.stem + ".shard*.csv"):
        f.unlink()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(PROCESSED_DIR / "index.csv"))
    ap.add_argument("--features-dir", default=str(FEATURES_DIR))
    ap.add_argument("--out", default=str(PROBS_PATH))
    ap.add_argument("--splits", default="train,val,test,phone")
    ap.add_argument("--max-train", type=int, default=2500)
    ap.add_argument("--attributes", default="colour,material,count")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpus", default=None, help="'auto' or '0,1': fan out over several GPUs")
    ap.add_argument("--shard", default=None, help=argparse.SUPPRESS)  # set by the launcher: 'i,n'
    args = ap.parse_args()
    out = Path(args.out)
    gpus = resolve_gpus(args.gpus)
    if len(gpus) > 1 and args.shard is None:
        failures = spawn_per_gpu(__file__, sys.argv[1:], gpus, "shard", strip=[])
        df = merge_shards(out)
        print(f"[consistency] merged -> {out} ({len(df)} rows)")
        raise SystemExit(1 if failures else 0)
    shard_i, shard_n = parse_shard(args.shard)
    device = device_arg(args.device)
    attrs = args.attributes.split(",")
    idx = pd.read_csv(args.index, low_memory=False)
    rows = idx[(idx["label"] == 0) & idx["split"].isin(args.splits.split(","))].copy()
    has_any = rows["colour"].notna() | rows["material"].notna() | rows["count_verifiable"].astype(bool)
    rows = rows[has_any]
    parts = []
    for s, g in rows.groupby("split"):
        parts.append(g.sample(min(len(g), args.max_train), random_state=args.seed) if s == "train" else g)
    rows = pd.concat(parts).reset_index(drop=True)
    rows = rows.iloc[shard_i::shard_n].reset_index(drop=True)      # every n-th image for this worker

    # resume: rows already in the main file count as done; this worker appends to its own shard file
    main_done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=COLS)
    my_out = out if shard_n == 1 else out.with_name(f"{out.stem}.shard{shard_i}.csv")
    done = pd.read_csv(my_out) if (my_out != out and my_out.exists()) else (main_done if shard_n == 1 else pd.DataFrame(columns=COLS))
    done_set = set(zip(main_done["image_key"], main_done["attribute"])) | set(zip(done["image_key"], done["attribute"]))
    out = my_out
    print(f"[consistency] worker {shard_i + 1}/{shard_n}: {len(rows)} images x {attrs}; {len(done_set)} rows already cached")

    store = FeatureStore(args.features_dir)
    backbone = ClipBackbone(device=device)
    zs = ClipZeroShot(backbone)
    vlm = None
    if not args.no_vlm:
        from listing_verifier.features.vlm import SmolVLM
        vlm = SmolVLM(device=device)
    new = []
    for i, r in rows.iterrows():
        todo = [a for a in attrs if (r["image_key"], a) not in done_set
                and (a == "count" and bool(r["count_verifiable"]) or a != "count" and pd.notna(r[a]))]
        if not todo:
            continue
        feat = store.get([r["image_key"]])
        if not feat["present"][0]:
            continue
        clip_feat = feat["clip"][0] / np.linalg.norm(feat["clip"][0])
        img = canonicalise(load_rgb(r["path"])) if vlm is not None else None
        noun = str(r["product_noun"]) if pd.notna(r["product_noun"]) else "item"
        for a in todo:
            cp = zs.option_probabilities(clip_feat, a, noun)
            vp = vlm.option_probabilities(img, VLM_QUESTIONS[a], vlm_options(a)) if vlm is not None else None
            new.append({"image_key": r["image_key"], "attribute": a, "clip_p": fmt(cp), "vlm_p": fmt(vp) if vp is not None else ""})
        if len(new) and len(new) % args.save_every == 0:
            atomic_csv(pd.concat([done, pd.DataFrame(new)]), out)
            print(f"[consistency] {len(done) + len(new)} rows cached ({i + 1}/{len(rows)} images)")
    atomic_csv(pd.concat([done, pd.DataFrame(new)]), out)
    print(f"[consistency] done -> {out}")


if __name__ == "__main__":
    main()
