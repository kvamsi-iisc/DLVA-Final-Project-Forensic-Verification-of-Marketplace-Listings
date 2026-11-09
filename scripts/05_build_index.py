"""Build data/processed/index.csv from reals, fakes, forensics and phone sets."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import ROOT
from listing_verifier.config import FAKES_DIR, FORENSICS_DIR, PHONE_DIR, PROCESSED_DIR, RAW_DIR
from listing_verifier.data.index import build_index, default_images_root
from listing_verifier.eval.nuisance import nuisance_audit
from _common import dump_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abo-clean", default=str(PROCESSED_DIR / "abo_clean.csv"))
    ap.add_argument("--images-root", default=str(default_images_root()))
    ap.add_argument("--fakes-dir", default=str(FAKES_DIR))
    ap.add_argument("--forensics-dir", default=str(FORENSICS_DIR))
    ap.add_argument("--phone-dir", default=str(PHONE_DIR))
    ap.add_argument("--out", default=str(PROCESSED_DIR / "index.csv"))
    ap.add_argument("--no-audit", action="store_true", help="skip the nuisance-only shortcut audit")
    args = ap.parse_args()
    abo = pd.read_csv(args.abo_clean)
    idx = build_index(abo, Path(args.images_root), Path(args.fakes_dir), Path(args.forensics_dir), Path(args.phone_dir))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(args.out, index=False)
    print(idx.groupby(["domain", "split", "generator", "mode"]).size().to_string())
    print(f"[index] {len(idx)} images -> {args.out}")
    if idx["label"].nunique() < 2:
        print("[index] WARNING: no fakes in the index yet (reals only); training needs fakes, see 03_generate_fakes.py")
    for split, g in idx[idx["domain"] == "catalogue"].groupby("split"):
        print(f"[index] catalogue/{split}: {int((g['label'] == 0).sum())} real, {int((g['label'] == 1).sum())} fake, {g['family_key'].nunique()} families")
    # Coverage: what the index ACTUALLY contains, per split / generator / attribute,
    # so the report quotes achieved counts and missing combinations are visible now.
    cov = {"by_split_generator": {f"{s}/{g}": int(n) for (s, g), n in idx.groupby(["split", "generator"]).size().items()},
           "attributes_by_split": {}, "warnings": []}
    for s_, g in idx[idx["label"] == 0].groupby("split"):
        cov["attributes_by_split"][s_] = {"colour": int(g["colour"].notna().sum()), "material": int(g["material"].notna().sum()),
                                          "count_verifiable": int(g["count_verifiable"].astype(bool).sum()),
                                          "colour_values": g["colour"].value_counts().to_dict()}
    cat = idx[idx["domain"] == "catalogue"]
    train_gens = set(cat.loc[(cat["split"] == "train") & (cat["label"] == 1), "generator"])
    for s_ in ("val", "test"):
        have = set(cat.loc[(cat["split"] == s_) & (cat["label"] == 1), "generator"])
        for g in sorted(train_gens - have):
            cov["warnings"].append(f"generator {g} has train fakes but none in {s_}")
        if (cat["split"] == s_).sum() and not (cat.loc[cat["split"] == s_, "label"] == 1).any():
            cov["warnings"].append(f"{s_} split has no fakes at all")
    for w in cov["warnings"]:
        print("[index] WARNING:", w)
    dump_json(cov, Path(args.out).parent / "coverage.json")
    if not args.no_audit and (idx["domain"] == "catalogue").any() and (idx["label"] == 1).any():
        audit = nuisance_audit(idx)
        dump_json(audit, Path(args.out).parent / "nuisance_audit.json")
        print(f"[index] nuisance audit: AUROC {audit.get('nuisance_only_test_auroc', float('nan')):.3f} -> {audit['verdict']}")


if __name__ == "__main__":
    main()
