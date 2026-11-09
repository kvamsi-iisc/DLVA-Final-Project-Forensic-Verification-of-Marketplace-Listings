"""Stream a generator-diverse subset (default 3000 images) of CommunityForensics.
Run with --inspect first to see the column names of the dataset version you get."""
from __future__ import annotations

import argparse

from _common import ROOT
from listing_verifier.config import COMMUNITY_FORENSICS_ID, FORENSICS_DIR
from listing_verifier.data.forensics_stream import stream_subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=COMMUNITY_FORENSICS_ID)
    ap.add_argument("--config", default=None, help="HF config name, e.g. 'Small' if the dataset has one")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", default=str(FORENSICS_DIR))
    ap.add_argument("--per-generator", type=int, default=40)
    ap.add_argument("--max-total", type=int, default=3000)
    ap.add_argument("--image-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--generator-col", default=None)
    ap.add_argument("--fake-label-value", default="1")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-real", type=int, default=None, help="real quota (default max_total/2)")
    ap.add_argument("--n-fake", type=int, default=None, help="fake quota (default the rest)")
    ap.add_argument("--max-scan", type=int, default=200000, help="stop after scanning this many rows")
    ap.add_argument("--id-col", default=None)
    args = ap.parse_args()
    stream_subset(args.out_dir, args.dataset, args.config, args.split, args.per_generator, args.max_total,
                  args.image_col, args.label_col, args.generator_col, args.fake_label_value, args.inspect,
                  seed=args.seed, n_real=args.n_real, n_fake=args.n_fake, max_scan=args.max_scan, id_col=args.id_col)


if __name__ == "__main__":
    main()
