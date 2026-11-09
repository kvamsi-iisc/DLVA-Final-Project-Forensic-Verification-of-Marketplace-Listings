"""How much does JPEG storage cost the frequency cue?

Every N-th fake was also stored as a lossless PNG twin at the same 256 px size.
For those fakes we score the trained heads three ways and compare AUROC against
the canonical reals of the same split:
  jpeg_stored   the normal path: stored JPEG-92, then canonicalise() (a second JPEG-92)
  png_canonical PNG twin through canonicalise(): exactly ONE JPEG-92, like the reals
  png_lossless  PNG twin with no JPEG at all (upper bound, and a confounded one:
                reals are still JPEG, so part of any gain is 'JPEG vs PNG', not 'real vs fake')
A large png_lossless - jpeg_stored gap for the spectrum head, with little change
for the probe, means the storage step (not the model) is what blunts the
frequency cue. Report the number; do not switch the pipeline to lossless fakes,
because that would reintroduce the storage confound.

  python scripts/14_png_diagnostic.py --tag main --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT, device_arg, dump_json
from listing_verifier.config import CHECKPOINT_DIR, PROCESSED_DIR, RESULTS_DIR
from listing_verifier.eval.authenticity import load_models, score_features
from listing_verifier.eval.metrics import auroc
from listing_verifier.features.clip_backbone import ClipBackbone
from listing_verifier.features.dct import spectrum_features
from listing_verifier.features.image_ops import canonicalise, load_rgb, resize_max_side, to_square_gray


def score_images(models, backbone, images):
    clip = backbone.encode_images(images, normalise=False)
    feats = [spectrum_features(to_square_gray(im)) for im in images]
    return score_features(models, clip, np.stack([f["dct_map"] for f in feats]), np.stack([f["dct_radial"] for f in feats]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--index", default=str(PROCESSED_DIR / "index.csv"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-fakes", type=int, default=600)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = device_arg(args.device)
    idx = pd.read_csv(args.index, low_memory=False)
    sub = idx[(idx["split"] == args.split) & (idx["domain"] == "catalogue")]
    fakes = sub[(sub["label"] == 1) & sub["path"].map(lambda p: Path(p).with_suffix(".png").exists())]
    if len(fakes) == 0:
        print("no PNG twins found; generate fakes with --keep-png-every > 0")
        raise SystemExit(3)   # 'no data' code: notebooks skip only this
    fakes = fakes.head(args.max_fakes)
    reals = sub[sub["label"] == 0].sample(min(len(fakes) * 2, (sub["label"] == 0).sum()), random_state=0)
    print(f"[png] {len(fakes)} fakes with PNG twins vs {len(reals)} reals ({args.split})")
    models = load_models(CHECKPOINT_DIR / args.tag, device=device)
    backbone = ClipBackbone(device=device)

    real_imgs = [canonicalise(load_rgb(p)) for p in reals["path"]]
    variants = {
        "jpeg_stored": [canonicalise(load_rgb(p)) for p in fakes["path"]],
        "png_canonical": [canonicalise(load_rgb(Path(p).with_suffix(".png"))) for p in fakes["path"]],
        "png_lossless": [resize_max_side(load_rgb(Path(p).with_suffix(".png"))) for p in fakes["path"]],
    }
    s_real = score_images(models, backbone, real_imgs)
    y = np.r_[np.zeros(len(reals)), np.ones(len(fakes))]
    out = {}
    for name, imgs in variants.items():
        s_fake = score_images(models, backbone, imgs)
        out[name] = {m: auroc(y, np.r_[s_real[m], s_fake[m]]) for m in ("probe", "spectrum", "fused")}
        print(f"[png] {name:14s} " + "  ".join(f"{m}: {v:.3f}" for m, v in out[name].items()))
    out["n_fakes"], out["n_reals"] = int(len(fakes)), int(len(reals))
    out["note"] = "png_lossless is an upper bound and confounded (reals stay JPEG); report the gap, keep JPEG storage in the pipeline"
    dump_json(out, RESULTS_DIR / args.tag / "png_diagnostic.json")
    print("[png] wrote", RESULTS_DIR / args.tag / "png_diagnostic.json")


if __name__ == "__main__":
    main()
