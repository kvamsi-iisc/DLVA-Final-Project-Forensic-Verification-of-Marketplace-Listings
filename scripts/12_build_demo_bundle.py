"""Assemble outputs/demo_bundle/ and PROVE its examples work.

  python scripts/12_build_demo_bundle.py --tag main            # scores every example (needs CLIP; CPU is fine)
  python scripts/12_build_demo_bundle.py --tag main --no-score # copy only (not a tested bundle)

Mandatory artifacts (missing -> hard failure): probe.npz, spectrum.pt, fusion.npz,
thresholds.json, calibrator.npz, calibrator_clip_only.npz, train_summary.json.
Examples are a structured demo set, not a random sample: reals with a colour
claim, one t2i and one i2i fake per generator, a count>1 real, phone items if
present. Every example is scored at JPEG none/70/50/30 in both inference modes;
outputs and latency go to examples/expected.json, artifact SHA-256s to
manifest.json. A `visible_ok` column in examples.csv is left for you to confirm
by eye that the metadata description matches what the photo shows (fakes may
have changed colour or count during generation: the description is inherited
from the SOURCE listing, so check it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from _common import ROOT, dump_json, log_invocation
from listing_verifier.config import (CHECKPOINT_DIR, CLIP_MODEL_NAME, CLIP_PRETRAINED, DEMO_BUNDLE_DIR, PROCESSED_DIR,
                                     VLM_MODEL_ID)

MANDATORY = ["probe.npz", "spectrum.pt", "fusion.npz", "thresholds.json", "calibrator.npz", "calibrator_clip_only.npz",
             "train_summary.json"]
OPTIONAL = ["probe.json", "fusion.json", "calibrator.json", "calibrator_clip_only.json", "train_keys.txt"]
JPEGS = [None, 70, 50, 30]


def describe(r) -> str:
    bits = [str(r[c]) for c in ("colour", "material") if pd.notna(r.get(c))]
    s = " ".join(bits + [str(r.get("product_noun") or "item")])
    if pd.notna(r.get("count")) and r.get("count_verifiable"):
        n = int(r["count"])
        s += f", set of {n}" if n > 1 else ", single item"
    return s[0].upper() + s[1:]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def pick_examples(idx: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    test = idx[idx["split"].isin(["test", "phone"]) & idx["colour"].notna()]
    parts = []
    reals = test[(test["label"] == 0) & (test["domain"] == "catalogue")]
    parts.append(reals[reals["count"].fillna(1).astype(int) > 1].head(1))            # a visible count > 1
    parts.append(reals.sample(min(3, len(reals)), random_state=seed))
    for gen, g in test[test["label"] == 1].groupby("generator"):
        for mode, gm in g.groupby("mode"):
            parts.append(gm.sample(1, random_state=seed))                            # one t2i + one i2i per generator
    phone = test[test["domain"] == "phone"]
    if len(phone):
        parts.append(phone.groupby("label", group_keys=False).apply(lambda g: g.head(1)))
    ex = pd.concat(parts).drop_duplicates("image_key")
    return ex.head(max(n, len(ex))) if len(ex) <= n else ex.head(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--index", default=str(PROCESSED_DIR / "index.csv"))
    ap.add_argument("--out", default=str(DEMO_BUNDLE_DIR))
    ap.add_argument("--n-examples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--no-vlm", action="store_true", help="score with the CLIP-only calibrator too (default scores both)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    log_invocation(out, {"config": vars(args)})
    ckpt = CHECKPOINT_DIR / args.tag
    missing = [f for f in MANDATORY if not (ckpt / f).exists()]
    if missing:
        raise SystemExit(f"[bundle] mandatory artifacts missing in {ckpt}: {missing} (run 07 and 10 first)")
    hashes = {}
    for f in MANDATORY + OPTIONAL:
        if (ckpt / f).exists():
            shutil.copy(ckpt / f, out / f)
            hashes[f] = sha256(out / f)

    idx = pd.read_csv(args.index, low_memory=False)
    ex = pick_examples(idx, args.n_examples, args.seed)
    if len(ex) == 0:
        raise SystemExit("[bundle] no eligible examples (need test-split listings with a colour label)")
    rows = []
    for _, r in ex.iterrows():
        fname = r["image_key"].replace(":", "_").replace(".", "_") + ".jpg"
        shutil.copy(r["path"], out / "examples" / fname)
        rows.append({"filename": fname, "description": describe(r), "product_noun": r["product_noun"],
                     "colour": r["colour"], "material": r["material"] if pd.notna(r["material"]) else "",
                     "count": int(r["count"]) if pd.notna(r["count"]) else "", "truth_label": int(r["label"]),
                     "generator": r["generator"], "mode": r["mode"], "domain": r["domain"], "image_key": r["image_key"],
                     "visible_ok": ""})
    pd.DataFrame(rows).to_csv(out / "examples" / "examples.csv", index=False)

    summary = json.loads((ckpt / "train_summary.json").read_text())
    manifest = {"tag": args.tag, "clip": f"{CLIP_MODEL_NAME}/{CLIP_PRETRAINED}", "vlm": VLM_MODEL_ID,
                "recommended_mode": summary.get("recommended_mode", "fused"), "artifact_sha256": hashes,
                "n_examples": len(rows), "scored": False,
                "launch": f"python -m listing_verifier.demo.server --bundle {out} --offline --mode {summary.get('recommended_mode', 'fused')}"}
    if not args.no_score:
        from listing_verifier.demo.verifier import ListingVerifier
        from listing_verifier.features.image_ops import load_rgb
        expected = {}
        for use_vlm in ([True, False] if not args.no_vlm else [False]):
            v = ListingVerifier(out, device="cpu", use_vlm=use_vlm, mode=manifest["recommended_mode"])
            key = "vlm" if use_vlm else "clip_only"
            expected[key] = {}
            for r in rows:
                img = load_rgb(out / "examples" / r["filename"])
                for q in JPEGS:
                    t0 = time.time()
                    res = v.verify(img, description=r["description"], product_noun=r["product_noun"], jpeg_quality=q)
                    res["latency_s"] = round(time.time() - t0, 2)
                    res["authenticity"].pop("clip_feature", None)
                    expected[key][f"{r['filename']}@{q or 'none'}"] = res
                    ok = (res["authenticity"]["verdict"] == "likely AI-generated") == bool(r["truth_label"])
                    print(f"[bundle] {key:9s} {r['filename'][:38]:38s} q={str(q or 'none'):4s} verdict={'OK ' if ok else 'MISS'} "
                          f"p={res['authenticity'][res['mode']]['p_fake']:.2f} claims={[c['verdict'] for c in res['consistency']]} {res['latency_s']}s")
        dump_json(expected, out / "examples" / "expected.json")
        lat = [r["latency_s"] for k in expected for r in expected[k].values()]
        manifest.update({"scored": True, "latency_median_s": sorted(lat)[len(lat) // 2], "latency_max_s": max(lat)})
        truth = {r["filename"]: bool(r["truth_label"]) for r in rows}
        first = expected[next(iter(expected))]
        hits = sum((res["authenticity"]["verdict"] == "likely AI-generated") == truth[k.split("@")[0]] for k, res in first.items())
        manifest["verdict_hits_at_all_jpegs"] = f"{hits}/{len(first)} (curated gallery, NOT a test-set result)"
    dump_json(manifest, out / "manifest.json")
    print(f"[bundle] {len(rows)} examples, scored={manifest['scored']} -> {out}")
    print("[bundle] now fill visible_ok in examples/examples.csv by looking at each image")


if __name__ == "__main__":
    main()
