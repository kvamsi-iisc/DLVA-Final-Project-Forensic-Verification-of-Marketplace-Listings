"""Train the authenticity branch on cached features.

  python scripts/07_train_authenticity.py --tag main
  python scripts/07_train_authenticity.py --tag turbo_only --train-generators sdxl-turbo
  python scripts/07_train_authenticity.py --tag control_mismatch --train-fake-domain forensics
Compression augmentation: with --augment-qualities 70,50 (default) the train
set is the union of the canonical features and the JPEG-70 / JPEG-50 features
of the same train images (if those caches exist). Both heads then see what
compression does to a real and to a fake, instead of meeting it for the first
time on the robustness curve. Validation stays canonical.

The control_mismatch tag is the Risk-1 control: fakes that do NOT look like catalogue
photos. If its test AUROC on ABO fakes collapses while its forensics AUROC is
high, the detector learned 'catalogue vs not'; that is the confound we wanted
to be able to see.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT, device_arg, dump_json, log_invocation, set_seed
from listing_verifier.config import CHECKPOINT_DIR, FEATURES_DIR, PROCESSED_DIR, SEEN_GENERATORS
from listing_verifier.data.abo_metadata import stable_hash
from listing_verifier.eval.authenticity import fit_thresholds, forensics_part, score_rows
from listing_verifier.features.extract import FeatureStore
from listing_verifier.models.fusion import fit_fusion
from listing_verifier.models.probe import train_probe
from listing_verifier.models.spectrum_head import save_spectrum_head, train_spectrum_head


def select_rows(idx: pd.DataFrame, split: str, generators, fake_domain: str, ctrl_part: str):
    reals = idx[(idx["split"] == split) & (idx["label"] == 0) & (idx["domain"] == "catalogue")]
    if fake_domain == "catalogue":
        fakes = idx[(idx["split"] == split) & (idx["label"] == 1) & (idx["domain"] == "catalogue")
                    & idx["generator"].isin(generators)]
    else:
        f = idx[(idx["domain"] == "forensics") & (idx["label"] == 1)].copy()
        f["ctrl"] = f["image_key"].map(forensics_part)
        fakes = f[f["ctrl"] == ctrl_part]
    return pd.concat([reals, fakes]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--index", default=str(PROCESSED_DIR / "index.csv"))
    ap.add_argument("--features-dir", default=str(FEATURES_DIR))
    ap.add_argument("--train-generators", default=",".join(SEEN_GENERATORS))
    ap.add_argument("--train-fake-domain", default="catalogue", choices=["catalogue", "forensics"])
    ap.add_argument("--augment-qualities", default="70,50",
                    help="comma list of JPEG qualities to add to train; '' disables. Each quality must cover EVERY "
                         "training image (exit 3 otherwise); there is deliberately no override flag")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    set_seed(args.seed)
    device = device_arg(args.device)
    gens = args.train_generators.split(",")
    out = Path(args.out_dir or CHECKPOINT_DIR / args.tag)
    out.mkdir(parents=True, exist_ok=True)
    log_invocation(out, {"config": vars(args)})

    idx = pd.read_csv(args.index, low_memory=False)
    store = FeatureStore(args.features_dir)
    tr = select_rows(idx, "train", gens, args.train_fake_domain, "ctrl_train")
    # Validation is split BY FAMILY into a selection half (C, early stopping) and a
    # calibration half (fusion weights, thresholds), so the thresholds are not fitted
    # on the same images that picked the model (evaluation P1.8).
    va_all = select_rows(idx, "val", gens, args.train_fake_domain, "ctrl_val")
    half = va_all["family_key"].fillna(va_all["image_key"]).map(lambda k: stable_hash("valsplit:" + str(k)) % 2)
    va, vc = va_all[half == 0], va_all[half == 1]
    ftr, fva, fvc = store.get(tr["image_key"].tolist()), store.get(va["image_key"].tolist()), store.get(vc["image_key"].tolist())
    tr, va, vc = tr[ftr["present"]].reset_index(drop=True), va[fva["present"]].reset_index(drop=True), vc[fvc["present"]].reset_index(drop=True)
    ytr, yva, yvc = tr["label"].to_numpy().astype(int), va["label"].to_numpy().astype(int), vc["label"].to_numpy().astype(int)
    EXIT_NO_DATA = 3   # notebooks skip an optional tag ONLY on this code; any other failure propagates
    if args.train_fake_domain == "catalogue":
        # Every requested generator must really be in the training data, or 'all3'
        # trained without Kandinsky would be indistinguishable from 'main'.
        have = tr.loc[tr["label"] == 1, "generator"].value_counts().to_dict()
        absent = [g for g in gens if have.get(g, 0) == 0]
        if absent:
            print(f"[train] requested generators with NO train fakes in the index: {absent} (have {have})")
            raise SystemExit(EXIT_NO_DATA)
    for name, y in (("train", ytr), ("val_select", yva), ("val_calib", yvc)):
        if len(y) < 20 or len(np.unique(y)) < 2:
            print(f"[train] {name} has {len(y)} rows / {len(np.unique(y))} classes: extract features first "
                  f"(06_extract_features.py) and check the index has reals AND fakes for these generators")
            raise SystemExit(EXIT_NO_DATA)
    print(f"[train] train {len(tr)} (fake {ytr.sum()}), val_select {len(va)} (fake {yva.sum()}), val_calib {len(vc)} (fake {yvc.sum()}); generators {gens}; fake domain {args.train_fake_domain}")

    aug_used = {}
    for q in [int(x) for x in args.augment_qualities.split(",") if x.strip()]:
        fa = store.get(tr["image_key"].tolist(), q) if q in store.available_qualities() else None
        n_have = int(fa["present"].sum()) if fa is not None else 0
        if n_have != len(tr):
            # Partial coverage is refused, not tolerated: an augmentation set that covers only
            # some training images changes the class/generator mix in ways nobody asked for.
            print(f"[train] augmentation q={q}: {n_have}/{len(tr)} training images cached; need all of them "
                  f"(run 06_extract_features.py --qualities {q} --splits train, which appends the missing ones)")
            raise SystemExit(EXIT_NO_DATA)
        ya = tr["label"].to_numpy().astype(int)[fa["present"]]
        for k in ("clip", "dct_map", "dct_radial"):
            ftr[k] = np.concatenate([ftr[k], fa[k]])
        ytr = np.concatenate([ytr, ya])
        aug_used[q] = n_have
        print(f"[train] augmentation q={q}: +{n_have} images (= n_train, full coverage) -> train size {len(ytr)}")

    print("[train] linear probe ...")
    probe = train_probe(ftr["clip"], ytr, fva["clip"], yva, seed=args.seed)
    print(f"[train] probe val AUROC {probe.meta['val_auroc']:.4f} (C={probe.meta['C']})")
    probe.save(out / "probe.npz")

    print("[train] spectrum head ...")
    res = train_spectrum_head(ftr["dct_map"], ftr["dct_radial"], ytr, fva["dct_map"], fva["dct_radial"], yva,
                              epochs=args.epochs, device=device, seed=args.seed)
    save_spectrum_head(res["model"], out / "spectrum.pt")
    print(f"[train] spectrum val AUROC {res['val_auroc']:.4f}")

    print("[train] fusion on the calibration half of validation ...")
    pl = probe.logits(fvc["clip"])
    sl = res["model"].predict_logits(fvc["dct_map"], fvc["dct_radial"])
    fusion = fit_fusion(pl, sl, yvc, seed=args.seed)
    fusion.save(out / "fusion.npz")
    print("[train] fusion weights:", fusion.meta)

    from listing_verifier.eval.authenticity import load_models
    models = load_models(out, device=device)
    sv = score_rows(models, store, vc)
    th = fit_thresholds(sv, 0.05)
    from listing_verifier.eval.authenticity import evaluate_scores
    calib_eval = evaluate_scores(sv, th, with_ci=False)
    fusion_helps = calib_eval["fused"]["auroc"] >= calib_eval["probe"]["auroc"]
    print(f"[train] on val_calib: probe {calib_eval['probe']['auroc']:.4f} fused {calib_eval['fused']['auroc']:.4f} -> recommended mode: {'fused' if fusion_helps else 'probe'}")
    dump_json(th, out / "thresholds.json")
    summary = {"tag": args.tag, "train_generators": gens, "train_fake_domain": args.train_fake_domain,
               "n_train": int(len(tr)), "n_train_with_augmentation": int(len(ytr)), "augmentation": aug_used,
               "n_val_select": int(len(va)), "n_val_calib": int(len(vc)), "probe": probe.meta,
               "val_calib_metrics": calib_eval, "recommended_mode": "fused" if fusion_helps else "probe",
               "train_image_keys_file": "train_keys.txt",
               "spectrum_val_auroc": res["val_auroc"], "spectrum_history": res["history"],
               "fusion": fusion.meta, "thresholds_at_5pct_fpr_on_val": th, "device": device}
    dump_json(summary, out / "train_summary.json")
    (out / "train_keys.txt").write_text("\n".join(tr["image_key"].astype(str)))   # provenance for the evaluator
    print("[train] saved to", out)


if __name__ == "__main__":
    main()
