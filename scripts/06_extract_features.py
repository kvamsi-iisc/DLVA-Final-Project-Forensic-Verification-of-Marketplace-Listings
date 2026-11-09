"""Extract CLIP + DCT features for every image in the index.

  python scripts/06_extract_features.py                      # canonical (q=0) for all images
  python scripts/06_extract_features.py --qualities 90,70,50,30 --splits test,phone,forensics
  python scripts/06_extract_features.py --qualities 70,50 --splits train     # augmentation for training
Degraded qualities are needed on the evaluation splits (robustness curve) and,
for 70/50, on the train split (compression augmentation in 07). Re-running
appends only the images that are missing from each quality's cache.
Multi-GPU: --gpus auto (or 0,1). Each worker computes every n-th 2000-image
part on its own card; the parent merges the parts per quality afterwards.
"""
from __future__ import annotations

import argparse

import sys

import pandas as pd

from _common import ROOT, device_arg, parse_shard, resolve_gpus, spawn_per_gpu
from listing_verifier.config import FEATURES_DIR, PROCESSED_DIR
from listing_verifier.features.extract import extract, merge_parts, n_parts_for, pending_index


def select(idx, q, splits):
    if q != 0 and splits is None:
        splits = "test,phone,forensics"
    return idx[idx["split"].isin(splits.split(","))] if splits else idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(PROCESSED_DIR / "index.csv"))
    ap.add_argument("--features-dir", default=str(FEATURES_DIR))
    ap.add_argument("--qualities", default="0")
    ap.add_argument("--splits", default=None, help="comma list; default all for q=0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--clip-cache", default=None)
    ap.add_argument("--gpus", default=None, help="'auto' or '0,1': fan out over several GPUs")
    ap.add_argument("--part-modulo", default=None, help=argparse.SUPPRESS)  # set by the launcher: 'i,n'
    args = ap.parse_args()
    idx = pd.read_csv(args.index, low_memory=False)
    qualities = [int(x) for x in args.qualities.split(",")]

    gpus = resolve_gpus(args.gpus)
    if len(gpus) > 1 and args.part_modulo is None:
        pending = {q: len(pending_index(select(idx, q, args.splits), q, args.features_dir)) for q in qualities}
        print(f"[extract] pending images per quality: {pending}")
        failures = spawn_per_gpu(__file__, sys.argv[1:], gpus, "part-modulo", strip=[])
        if failures:
            raise SystemExit(f"{failures} worker(s) failed; re-run the same command to resume")
        for q, n in pending.items():
            if n:
                merge_parts(q, n_parts_for(n, args.chunk_size), args.features_dir)
        return

    from listing_verifier.features.clip_backbone import ClipBackbone
    backbone = ClipBackbone(device=device_arg(args.device), cache_dir=args.clip_cache)
    shard = parse_shard(args.part_modulo)
    for q in qualities:
        sub = select(idx, q, args.splits)
        print(f"[extract] quality {q}: {len(sub)} images (worker {shard[0] + 1}/{shard[1]})")
        extract(sub.reset_index(drop=True), backbone, quality=q, features_dir=args.features_dir,
                chunk_size=args.chunk_size, batch_size=args.batch_size,
                part_modulo=shard, merge=(shard[1] == 1))


if __name__ == "__main__":
    main()
