"""Generate fakes from ABO (or from the phone set) in resumable chunks.

Examples (one Kaggle run each, ~1-3 GPU hours):
  python scripts/03_generate_fakes.py --generator sdxl-turbo --mode t2i --per-split train=2100,val=300,test=600
  python scripts/03_generate_fakes.py --generator sdxl-turbo --mode i2i --strength 0.5 --per-split train=2100,val=300,test=600
  python scripts/03_generate_fakes.py --generator sd15 --mode i2i --strength 0.5 --per-split train=1400,val=200,test=400
  python scripts/03_generate_fakes.py --generator kandinsky22 --mode t2i --per-split train=700,val=100,test=200
  python scripts/03_generate_fakes.py --generator sdxl-turbo --mode i2i --strength 0.4 --source phone
Re-running the same command resumes: finished images are skipped, finished
chunks are re-listed. --start-chunk/--end-chunk let you split one job across
sessions. --persist-cmd runs a shell command after every chunk (e.g. a Kaggle
dataset version bump via scripts/kaggle_persist.py).

Multi-GPU (Kaggle "T4 x2"): add --gpus auto (or --gpus 0,1). The parent spawns
one worker per GPU; worker i generates the chunks with index % n == i, so both
cards work through the same job and each chunk still lands in its own CSV.
With --gpus the persist command runs once at the end, not after every chunk
(two workers bumping the same Kaggle dataset at once would collide).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

from _common import ROOT, device_arg, parse_shard, resolve_gpus, spawn_per_gpu
from listing_verifier.config import FAKE_CHUNK_SIZE, FAKE_KEEP_PNG_EVERY, FAKES_DIR, PHONE_DIR, PROCESSED_DIR, RAW_DIR
from listing_verifier.data.fakes import FakeGenerator, generate_chunk, run_persist_command
from listing_verifier.data.index import default_images_root
from listing_verifier.data.phone import phone_rows_for_generation


def parse_per_split(s: str) -> dict:
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True, choices=["sdxl-turbo", "sd15", "kandinsky22"])
    ap.add_argument("--mode", required=True, choices=["t2i", "i2i"])
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--source", default="abo", choices=["abo", "phone"])
    ap.add_argument("--per-split", default="train=2100,val=300,test=600")
    ap.add_argument("--abo-clean", default=str(PROCESSED_DIR / "abo_clean.csv"))
    ap.add_argument("--images-root", default=str(default_images_root()))
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--chunk-size", type=int, default=FAKE_CHUNK_SIZE)
    ap.add_argument("--start-chunk", type=int, default=0)
    ap.add_argument("--end-chunk", type=int, default=None, help="exclusive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-png-every", type=int, default=FAKE_KEEP_PNG_EVERY,
                    help="also store a lossless PNG twin for every N-th fake (0 = never); used by 14_png_diagnostic.py")
    ap.add_argument("--persist-cmd", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--hf-cache", default=None)
    ap.add_argument("--gpus", default=None, help="'auto' or '0,1': fan out over several GPUs")
    ap.add_argument("--chunk-modulo", default=None, help=argparse.SUPPRESS)  # set by the launcher: 'i,n'
    args = ap.parse_args()

    gpus = resolve_gpus(args.gpus)
    if len(gpus) > 1 and args.chunk_modulo is None:
        failures = spawn_per_gpu(__file__, sys.argv[1:], gpus, "chunk-modulo", strip=["--persist-cmd"])
        run_persist_command(args.persist_cmd)
        raise SystemExit(1 if failures else 0)
    shard_i, shard_n = parse_shard(args.chunk_modulo)

    tag = args.mode if args.mode == "t2i" else f"i2i_s{args.strength:.2f}"
    # The output folder names the run (per-split sizes + seed), so a pilot run and a
    # full run of the same generator/mode never share chunk filenames: merging two
    # Kaggle datasets with the same chunk_0000.csv would silently drop images.
    run_name = "n" + "-".join(f"{k}{v}" for k, v in parse_per_split(args.per_split).items()) + f"_seed{args.seed}"
    if args.source == "abo":
        abo = pd.read_csv(args.abo_clean)
        abo["source_path"] = [str(Path(args.images_root) / p) for p in abo["image_rel_path"]]
        abo["aspect_w"], abo["aspect_h"] = abo.get("width"), abo.get("height")   # from ABO images.csv
        per_split = parse_per_split(args.per_split)
        parts = []
        for split, n in per_split.items():
            sub = abo[abo["split"] == split].sort_values("item_id")
            parts.append(sub.sample(min(n, len(sub)), random_state=args.seed))
        rows = pd.concat(parts).reset_index(drop=True)
        out_dir = Path(args.out_root or FAKES_DIR) / args.generator / tag / run_name
        key_prefix = "fake"
    else:
        rows = phone_rows_for_generation(PHONE_DIR).reset_index(drop=True)
        out_dir = Path(args.out_root or (PHONE_DIR / "fake")) / args.generator / tag / f"phone_seed{args.seed}"
        key_prefix = "phonefake"

    n_chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size
    end = n_chunks if args.end_chunk is None else min(args.end_chunk, n_chunks)
    my_chunks = [c for c in range(args.start_chunk, end) if c % shard_n == shard_i]
    print(f"[fakes] {len(rows)} source rows -> {n_chunks} chunks of {args.chunk_size}; "
          f"this worker ({shard_i + 1}/{shard_n}) runs chunks {my_chunks}")
    print(f"[fakes] output: {out_dir}")
    gen = FakeGenerator(args.generator, device=device_arg(args.device), cache_dir=args.hf_cache)
    print(f"[fakes] licence: {gen.licence}")
    for c in my_chunks:
        chunk_rows = rows.iloc[c * args.chunk_size:(c + 1) * args.chunk_size]
        csv = generate_chunk(gen, chunk_rows, args.mode, args.strength, out_dir, c, seed_base=args.seed,
                             keep_png_every=args.keep_png_every, key_prefix=key_prefix)
        print(f"[fakes] chunk {c} done -> {csv}")
        if shard_n == 1:
            run_persist_command(args.persist_cmd)
    print("[fakes] finished")


if __name__ == "__main__":
    main()
