"""Evaluate a trained authenticity tag on every test set we have.

Writes outputs/results/<tag>/: summary.csv, results.json, jpeg_curve.csv,
jpeg_curve.png, roc_*.png. Test sets:
  catalogue/<generator>        ABO test reals vs ABO test fakes of that generator (seen or in-domain unseen)
  catalogue/all_seen           pooled
  catalogue/<generator>/<mode>/<strength>   per-mode breakdown
  forensics/internal           CF reals vs CF fakes (unseen generators, unseen domain)
  forensics/fakes_vs_abo_reals cross-domain pairing (confounded; reported for completeness)
  phone/internal               own phone photos vs fakes made from them
  jpeg curve on catalogue/all_seen for every cached quality
  univfd baseline on the same sets, if outputs/checkpoints/univfd_fc_weights.pth exists
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT, device_arg, dump_json, log_invocation
from listing_verifier.config import CHECKPOINT_DIR, FEATURES_DIR, PROCESSED_DIR, RESULTS_DIR, SEEN_GENERATORS
from listing_verifier.eval.authenticity import MODEL_NAMES, evaluate_scores, forensics_part, load_models, score_rows
from listing_verifier.eval.metrics import auroc, summarise_detector
from listing_verifier.eval.plots import plot_jpeg_curve, plot_roc
from listing_verifier.features.extract import FeatureStore
from listing_verifier.models.baselines import load_univfd_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--index", default=str(PROCESSED_DIR / "index.csv"))
    ap.add_argument("--features-dir", default=str(FEATURES_DIR))
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--univfd-weights", default=str(CHECKPOINT_DIR / "univfd_fc_weights.pth"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = device_arg(args.device)
    ckpt = Path(args.ckpt_dir or CHECKPOINT_DIR / args.tag)
    out = RESULTS_DIR / args.tag
    out.mkdir(parents=True, exist_ok=True)
    log_invocation(out, {"config": vars(args)})
    models = load_models(ckpt, device=device)
    th = models["thresholds"]
    idx = pd.read_csv(args.index, low_memory=False)
    store = FeatureStore(args.features_dir)
    train_gens, trained_on_forensics, train_keys = None, False, set()
    ts = ckpt / "train_summary.json"
    if ts.exists():
        import json
        summary = json.loads(ts.read_text())
        train_gens = summary.get("train_generators")
        trained_on_forensics = summary.get("train_fake_domain") == "forensics"
    if (ckpt / "train_keys.txt").exists():
        train_keys = set((ckpt / "train_keys.txt").read_text().split())
    univfd = load_univfd_probe(args.univfd_weights) if Path(args.univfd_weights).exists() else None

    results, rows_out, roc_curves, prediction_frames = {}, [], {}, []

    def record(name: str, rows: pd.DataFrame, quality: int = 0, roc: bool = False):
        leaked = rows["image_key"].astype(str).isin(train_keys)
        assert not leaked.any(), f"{name}: {int(leaked.sum())} evaluation images were in the training set"
        if rows["label"].nunique() < 2 or len(rows) < 10:
            print(f"[eval] skip {name}: not enough data ({len(rows)} rows)")
            return None
        sc = score_rows(models, store, rows, quality)
        if univfd is not None:
            sc["univfd"] = univfd.probabilities(store.get(sc["image_key"].tolist(), quality)["clip"])
        res = evaluate_scores(sc, th)
        if univfd is not None:
            res["univfd"] = summarise_detector(sc["label"].to_numpy(), sc["univfd"].to_numpy())
        results[name] = res
        keep = [c for c in ("image_key", "label", "generator", "mode", "strength", "domain", "split") if c in sc]
        prediction_frames.append(sc[keep + [c for c in sc.columns if c in MODEL_NAMES or c.endswith("_logit") or c == "univfd"]]
                                 .assign(test_set=name, quality=quality))
        for m, r in res.items():
            rows_out.append({"test_set": name, "quality": quality, "model": m, **{k: v for k, v in r.items() if not isinstance(v, (list, tuple))}})
        if roc:
            roc_curves[name] = {m: (sc["label"].to_numpy(), sc[m].to_numpy()) for m in list(MODEL_NAMES) + (["univfd"] if univfd is not None else [])}
        print(f"[eval] {name:45s} " + "  ".join(f"{m}:{r['auroc']:.3f}" for m, r in res.items()))
        return sc

    cat = idx[(idx["domain"] == "catalogue") & (idx["split"] == "test")]
    reals = cat[cat["label"] == 0]
    for gen in sorted(cat.loc[cat["label"] == 1, "generator"].unique()):
        fakes = cat[(cat["label"] == 1) & (cat["generator"] == gen)]
        seen = "seen" if (train_gens is None or gen in train_gens) else "unseen_generator_same_domain"
        record(f"catalogue/{gen} [{seen}]", pd.concat([reals, fakes]), roc=True)
        for (mode, strength), fk in fakes.groupby(["mode", "strength"]):
            record(f"catalogue/{gen}/{mode}/s{strength:.2f}", pd.concat([reals, fk]))
    seen_fakes = cat[(cat["label"] == 1) & (cat["generator"].isin(train_gens or SEEN_GENERATORS))]
    all_seen = pd.concat([reals, seen_fakes])
    record("catalogue/all_seen", all_seen, roc=True)
    # Risk-1 control: ABO-derived fakes whose framing does not match the catalogue (t2i with
    # non-white backgrounds). Never trained on. A detector that is MUCH better here than on
    # catalogue/all_seen is reading style, not generator traces.
    mm = idx[(idx["domain"] == "mismatch") & (idx["split"] == "test") & (idx["label"] == 1)]
    for gen in sorted(mm["generator"].unique()):
        record(f"mismatch/{gen} [style control, not trained on]", pd.concat([reals, mm[mm["generator"] == gen]]), roc=True)

    fo = idx[idx["domain"] == "forensics"].copy()
    if len(fo):
        fo["ctrl"] = fo["image_key"].map(forensics_part)
        # The same reserved 40% (ctrl_test) for EVERY tag, so numbers are comparable
        # between the main models and the CF-trained control; the full subset only
        # for tags that never trained on CF (provenance from train_summary, not the tag name).
        fo_eval = fo[fo["ctrl"] == "ctrl_test"]
        record("forensics/internal [unseen generators + domain, reserved 40%]", fo_eval, roc=True)
        if not trained_on_forensics:
            record("forensics/all [full streamed subset]", fo)
        record("forensics/fakes_vs_abo_reals [cross-domain, confounded]", pd.concat([reals, fo_eval[fo_eval["label"] == 1]]))
        per_gen = []
        sc = score_rows(models, store, fo_eval)
        cf_reals = sc[sc["label"] == 0]
        for gen, g in sc[sc["label"] == 1].groupby("generator"):
            if len(g) >= 10:
                both = pd.concat([cf_reals, g])
                per_gen.append({"generator": gen, "n_fake": len(g), **{m: auroc(both["label"], both[m]) for m in MODEL_NAMES}})
        if per_gen:
            pd.DataFrame(per_gen).sort_values("fused").to_csv(out / "forensics_per_generator.csv", index=False)

    ph = idx[idx["domain"] == "phone"]
    if len(ph):
        record("phone/internal", ph, roc=True)

    # JPEG curve on IDENTICAL images at every quality (drop anything missing from any cache)
    keys_all = all_seen["image_key"].astype(str).tolist()
    coverage = {q: set(k for k, p in zip(keys_all, store.get(keys_all, q)["present"]) if p) for q in store.available_qualities()}
    # a quality cached only for training (augmentation) carries no test images: leave it out of the curve, say so
    qualities = [q for q, cov in coverage.items() if len(cov) >= 0.5 * len(keys_all)]
    for q, cov in coverage.items():
        if q not in qualities:
            print(f"[eval] quality {q}: only {len(cov)}/{len(keys_all)} test images cached; not part of the JPEG curve")
    common = set(keys_all)
    for q in qualities:
        common &= coverage[q]
    all_seen_common = all_seen[all_seen["image_key"].astype(str).isin(common)]
    print(f"[eval] JPEG curve on {len(all_seen_common)} images present at all qualities {qualities}")
    curve = {m: {} for m in MODEL_NAMES + (["univfd"] if univfd is not None else [])}
    for q in qualities:
        sc = record(f"catalogue/all_seen @jpeg{q if q else 'none'}", all_seen_common, quality=q)
        if sc is not None:
            for m in curve:
                if m in results[f"catalogue/all_seen @jpeg{q if q else 'none'}"]:
                    curve[m][q] = results[f"catalogue/all_seen @jpeg{q if q else 'none'}"][m]["auroc"]
        if len(ph):
            record(f"phone/internal @jpeg{q if q else 'none'}", ph, quality=q)
    curve = {m: v for m, v in curve.items() if v}
    if curve:
        pd.DataFrame(curve).to_csv(out / "jpeg_curve.csv")
        plot_jpeg_curve(curve, out / "jpeg_curve.png", f"AUROC vs JPEG quality ({args.tag}, catalogue test)")
    for name, cv in roc_curves.items():
        plot_roc(cv, out / ("roc_" + name.split(" ")[0].replace("/", "_") + ".png"), name)
    pd.DataFrame(rows_out).to_csv(out / "summary.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(out / "predictions.csv", index=False)   # every image, every set, every quality
    dump_json(results, out / "results.json")
    print("[eval] wrote", out)


if __name__ == "__main__":
    main()
