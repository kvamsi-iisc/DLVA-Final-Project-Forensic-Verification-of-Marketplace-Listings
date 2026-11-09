"""Run the full verifier on one image from the command line (demo rehearsal).

  python scripts/13_verify_cli.py --image path.jpg --description "brown leather sofa, set of 2" --noun sofa --jpeg 50
"""
from __future__ import annotations

import argparse
import json

from _common import ROOT
from listing_verifier.config import DEMO_BUNDLE_DIR
from listing_verifier.demo.verifier import ListingVerifier
from listing_verifier.features.image_ops import load_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(DEMO_BUNDLE_DIR))
    ap.add_argument("--image", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--noun", default=None)
    ap.add_argument("--jpeg", type=int, default=None)
    ap.add_argument("--no-vlm", action="store_true")
    args = ap.parse_args()
    v = ListingVerifier(args.bundle, use_vlm=not args.no_vlm)
    res = v.verify(load_rgb(args.image), description=args.description, product_noun=args.noun, jpeg_quality=args.jpeg)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
