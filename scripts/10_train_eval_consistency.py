"""Train the claim calibrator on synthetic mismatch pairs and evaluate it.

Outputs (outputs/results/consistency/):
  results.json        per attribute: F1 (match / mismatch), accuracy, AUROC, ECE
                      for: calibrated (CLIP+VLM), calibrated CLIP-only, raw CLIP argmax, raw VLM argmax
  reliability_<attr>.png, f1_by_attribute.png
  hand_labels section if data/hand_labels/hand_labels_filled.csv exists
Calibrator saved to outputs/checkpoints/<tag>/calibrator.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT, dump_json, log_invocation
from listing_verifier.config import CHECKPOINT_DIR, FEATURES_DIR, HAND_LABELS_DIR, PROCESSED_DIR, RESULTS_DIR
from listing_verifier.data.mismatch import make_pairs
from listing_verifier.eval.metrics import auroc, expected_calibration_error, per_attribute_f1
from listing_verifier.eval.plots import plot_bars, plot_reliability
from listing_verifier.models.consistency import ClaimCalibrator, claim_features, option_index


def load_probs(path: Path):
    df = pd.read_csv(path)
    table = {}
    for k, a, c, v in zip(df["image_key"], df["attribute"], df["clip_p"], df["vlm_p"]):
        cp = np.array([float(x) for x in str(c).split()])
        vp = np.array([float(x) for x in str(v).split()]) if isinstance(v, str) and v.strip() else None
        table[(k, a)] = (cp, vp)
    return table


def featurise(pairs: pd.DataFrame, table, use_vlm: bool):
    X, keep = [], []
    if len(pairs) == 0:
        return np.zeros((0, 7), np.float32), np.zeros(0, bool)
    for k, a, claimed in zip(pairs["image_key"], pairs["attribute"], pairs["claimed"]):
        if (k, a) not in table:
            keep.append(False)
            continue
        cp, vp = table[(k, a)]
        X.append(claim_features(cp, vp if use_vlm else None, option_index(a, claimed)))
        keep.append(True)
    return np.array(X, dtype=np.float32), np.array(keep)


def raw_argmax_pred(pairs: pd.DataFrame, table, which: int):
    out = []
    for k, a, claimed in zip(pairs["image_key"], pairs["attribute"], pairs["claimed"]):
        probs = table[(k, a)][which]
        out.append(int(probs is not None and int(np.argmax(probs)) == option_index(a, claimed)))
    return np.array(out)


def evaluate(pairs: pd.DataFrame, p: np.ndarray, pred: np.ndarray) -> dict:
    df = pairs.copy()
    df["pred"] = pred
    res = per_attribute_f1(df, "attribute", "label", "pred")
    for a, g in df.groupby("attribute"):
        m = df["attribute"] == a
        res[a]["auroc"] = auroc(df.loc[m, "label"], p[m.to_numpy()])
        res[a]["ece"] = expected_calibration_error(df.loc[m, "label"], p[m.to_numpy()])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--index", default=str(PROCESSED_DIR / "index.csv"))
    ap.add_argument("--probs", default=str(FEATURES_DIR / "consistency_probs.csv"))
    ap.add_argument("--attributes", default="colour,material,count")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = RESULTS_DIR / "consistency"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / args.tag
    ckpt.mkdir(parents=True, exist_ok=True)
    log_invocation(out, {"config": vars(args)})
    idx = pd.read_csv(args.index, low_memory=False)
    table = load_probs(args.probs)
    has_vlm = any(v is not None for _, v in table.values())
    attrs = args.attributes.split(",")

    cal, cal_clip = ClaimCalibrator(), ClaimCalibrator()
    results = {"synthetic_test": {}, "n_pairs": {}}
    test_frames, test_preds = [], {"calibrated": [], "calibrated_clip_only": [], "raw_clip_argmax": [], "raw_vlm_argmax": []}
    test_probs = {"calibrated": [], "calibrated_clip_only": []}
    for a in attrs:
        tr = make_pairs(idx, a, "train", seed=args.seed)
        te = make_pairs(idx, a, "test", seed=args.seed + 1)
        if len(tr) == 0 or len(te) == 0:
            print(f"[consistency] {a}: no listings with a trusted {a} label in train/test; skip")
            continue
        Xtr, ktr = featurise(tr, table, True)
        Xte, kte = featurise(te, table, True)
        tr, te = tr[ktr].reset_index(drop=True), te[kte].reset_index(drop=True)
        if len(tr) < 20 or len(te) < 20:
            print(f"[consistency] {a}: not enough cached pairs (train {len(tr)}, test {len(te)}); skip")
            continue
        results["n_pairs"][a] = {"train": int(len(tr)), "test": int(len(te))}
        cal.fit(a, Xtr, tr["label"].to_numpy(), seed=args.seed)
        Xtr_c, _ = featurise(tr, table, False)
        Xte_c, _ = featurise(te, table, False)
        cal_clip.fit(a, Xtr_c, tr["label"].to_numpy(), seed=args.seed)
        p = cal.predict_proba(a, Xte)
        pc = cal_clip.predict_proba(a, Xte_c)
        test_frames.append(te)
        test_probs["calibrated"].append(p), test_probs["calibrated_clip_only"].append(pc)
        test_preds["calibrated"].append((p >= 0.5).astype(int))
        test_preds["calibrated_clip_only"].append((pc >= 0.5).astype(int))
        test_preds["raw_clip_argmax"].append(raw_argmax_pred(te, table, 0))
        test_preds["raw_vlm_argmax"].append(raw_argmax_pred(te, table, 1))
        plot_reliability(te["label"].to_numpy(), p, out / f"reliability_{a}.png", f"Calibration: {a} (synthetic test pairs)")
    if not test_frames:
        raise SystemExit("no attribute had enough pairs; run 09_run_consistency.py first")
    te_all = pd.concat(test_frames).reset_index(drop=True)
    export = te_all.copy()
    for name, preds in test_preds.items():
        pred = np.concatenate(preds)
        prob = np.concatenate(test_probs[name]) if name in test_probs else pred.astype(float)
        results["synthetic_test"][name] = evaluate(te_all, prob, pred)
        export[f"p_{name}"] = prob
        export[f"pred_{name}"] = pred
    export.to_csv(out / "predictions_synthetic_test.csv", index=False)   # every pair, every method
    results["vlm_available"] = has_vlm

    # Phone split (own photos with human-entered labels): synthetic pairs again, but on
    # out-of-catalogue images. Small and exploratory; reported with its n.
    phone_frames, phone_p = [], []
    for a in cal.models:
        ph = make_pairs(idx, a, "phone", seed=args.seed + 2)
        if len(ph) == 0:
            continue
        Xp, kp = featurise(ph, table, True)
        ph = ph[kp].reset_index(drop=True)
        if len(ph):
            phone_frames.append(ph)
            phone_p.append(cal.predict_proba(a, Xp))
    if phone_frames:
        ph_all = pd.concat(phone_frames).reset_index(drop=True)
        pp = np.concatenate(phone_p)
        results["phone_synthetic"] = evaluate(ph_all, pp, (pp >= 0.5).astype(int))
        ph_all.assign(p_calibrated=pp).to_csv(out / "predictions_phone.csv", index=False)
        print("[consistency] phone split:", {a: f"n={r['n']} acc={r['accuracy']:.2f}" for a, r in results["phone_synthetic"].items()})
    cal.save(ckpt / "calibrator.npz")
    cal_clip.save(ckpt / "calibrator_clip_only.npz")
    plot_bars({a: r["f1_mismatch"] for a, r in results["synthetic_test"]["calibrated"].items()},
              out / "f1_by_attribute.png", "Mismatch-class F1 per attribute (calibrated)")

    hand = HAND_LABELS_DIR / "hand_labels_filled.csv"
    if hand.exists():
        h = pd.read_csv(hand)
        h = h[h["human_label"].isin([0, 1, "0", "1"])].copy()
        h["label"] = h["human_label"].astype(int)
        h = h[h["attribute"].isin(cal.models)]
        X, k = featurise(h, table, True)
        h = h[k].reset_index(drop=True)
        if len(h):
            p = np.array([cal.predict_proba(a, x[None])[0] for a, x in zip(h["attribute"], X)])
            results["hand_labelled"] = evaluate(h, p, (p >= 0.5).astype(int))
            h.assign(p_calibrated=p).to_csv(out / "predictions_hand_labelled.csv", index=False)
            results["hand_labelled"]["metadata_noise_rate"] = float(1 - h["label"].mean())
            print("[consistency] hand-labelled:", {a: round(r["accuracy"], 3) for a, r in results["hand_labelled"].items() if isinstance(r, dict)})
    dump_json(results, out / "results.json")
    for name, res in results["synthetic_test"].items():
        print(f"[consistency] {name:22s} " + "  ".join(f"{a}: F1mis {r['f1_mismatch']:.3f} acc {r['accuracy']:.3f}" for a, r in res.items()))
    print("[consistency] wrote", out, "and", ckpt / "calibrator.npz")


if __name__ == "__main__":
    main()
