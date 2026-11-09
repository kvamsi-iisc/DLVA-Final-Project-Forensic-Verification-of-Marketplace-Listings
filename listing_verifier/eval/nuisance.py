"""Shortcut audit: can a classifier that sees ONLY nuisance variables (image
width, height, aspect ratio, file size, JPEG quality estimate) tell real from
fake? If yes, the dataset leaks the label through geometry or storage and the
detector's score is not evidence of anything (evaluation P1.4).

Run by 05_build_index.py; written to data/processed/nuisance_audit.json.
Target: AUROC close to 0.5 on the test split. Anything above ~0.6 needs fixing
before training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

from .metrics import auroc


def image_nuisance_row(path: str) -> Dict:
    p = Path(path)
    try:
        with Image.open(p) as im:
            w, h = im.size
            fmt = im.format or ""
            q = None
            if fmt == "JPEG" and getattr(im, "quantization", None):
                # mean of the luma quantisation table: lower = higher JPEG quality
                q = float(np.mean(list(im.quantization.get(0, [0]))))
        return {"width": w, "height": h, "aspect": w / max(h, 1), "bytes": p.stat().st_size,
                "is_square": int(w == h), "jpeg_qtable_mean": q if q is not None else -1.0, "format": fmt}
    except Exception:
        return {"width": -1, "height": -1, "aspect": -1.0, "bytes": -1, "is_square": -1, "jpeg_qtable_mean": -1.0, "format": "ERR"}


FEATURES = ["width", "height", "aspect", "bytes", "is_square", "jpeg_qtable_mean"]


def nuisance_audit(index: pd.DataFrame, domain: str = "catalogue", max_per_split: int = 4000, seed: int = 0) -> Dict:
    sub = index[index["domain"] == domain]
    parts = [g.sample(min(len(g), max_per_split), random_state=seed) for _, g in sub.groupby("split")]
    sub = pd.concat(parts) if parts else sub
    rows = pd.DataFrame([image_nuisance_row(p) for p in sub["path"]], index=sub.index)
    df = pd.concat([sub[["split", "label", "generator"]], rows], axis=1)
    df = df[df["width"] > 0]
    out: Dict = {"n": int(len(df)), "by_label": {}}
    for lab, g in df.groupby("label"):
        out["by_label"][int(lab)] = {"n": int(len(g)), "frac_square": float(g["is_square"].mean()),
                                     "aspect_mean": float(g["aspect"].mean()), "bytes_mean": float(g["bytes"].mean()),
                                     "width_mean": float(g["width"].mean()), "height_mean": float(g["height"].mean()),
                                     "jpeg_qtable_mean": float(g["jpeg_qtable_mean"].mean()),
                                     "formats": g["format"].value_counts().to_dict()}
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    if tr["label"].nunique() == 2 and te["label"].nunique() == 2:
        X = tr[FEATURES].to_numpy(dtype=float)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit((X - mu) / sd, tr["label"])
        s = clf.decision_function((te[FEATURES].to_numpy(dtype=float) - mu) / sd)
        out["nuisance_only_test_auroc"] = auroc(te["label"].to_numpy(), s)
        out["coefficients"] = dict(zip(FEATURES, map(float, clf.coef_[0])))
        out["verdict"] = ("OK: nuisance variables do not predict the label" if out["nuisance_only_test_auroc"] < 0.6
                          else "WARNING: geometry/storage leaks the label; fix before training")
    else:
        out["verdict"] = "not enough data for the nuisance classifier"
    return out
