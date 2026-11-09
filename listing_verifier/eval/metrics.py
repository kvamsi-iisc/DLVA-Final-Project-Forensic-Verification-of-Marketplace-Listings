"""Evaluation metrics. Written before the models (the lesson from the Stanford
reports): a metric file that is tested on toy data cannot silently flatter a
model later.

Conventions: label 1 = fake (authenticity) or 1 = claim matches (consistency);
scores are higher-is-more-positive.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float = 0.05) -> float:
    """TPR at a fixed false-positive rate: 'if we only allow 5% of real photos
    to be flagged, what fraction of fakes do we still catch?' Buyers care about
    this number more than about AUROC."""
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(target_fpr, fpr, tpr))


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float = 0.05) -> float:
    """Score threshold on *real* images such that only target_fpr of them exceed it."""
    reals = np.asarray(s)[np.asarray(y) == 0]
    return float(np.quantile(reals, 1.0 - target_fpr))


def accuracy(y: np.ndarray, s: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((np.asarray(s) >= threshold) == (np.asarray(y) == 1)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE: average |confidence - accuracy| over equal-width probability bins."""
    y, p = np.asarray(y).astype(float), np.clip(np.asarray(p, dtype=float), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def operating_point(y: np.ndarray, s: np.ndarray, threshold: float) -> Dict[str, float]:
    """What actually happens at the FROZEN threshold: TPR, FPR and the confusion
    counts. Different from the ROC-interpolated TPR@5%FPR, which assumes the
    threshold could be re-tuned on this very test set."""
    y, pred = np.asarray(y) == 1, np.asarray(s) >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    return {"tpr_at_threshold": tp / max(tp + fn, 1), "fpr_at_threshold": fp / max(fp + tn, 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def summarise_detector(y: np.ndarray, s: np.ndarray, threshold: Optional[float] = None) -> Dict[str, float]:
    out = {
        "auroc": auroc(y, s),
        "tpr_at_5fpr": tpr_at_fpr(y, s, 0.05),
        "tpr_at_1fpr": tpr_at_fpr(y, s, 0.01),
        "ap": float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan"),
        "n_real": int(np.sum(np.asarray(y) == 0)),
        "n_fake": int(np.sum(np.asarray(y) == 1)),
    }
    if threshold is not None:
        out["acc_at_threshold"] = accuracy(y, s, threshold)
        out["threshold"] = float(threshold)
        out.update(operating_point(y, s, threshold))
    return out


def per_attribute_f1(df, attr_col: str = "attribute", y_col: str = "label", pred_col: str = "pred") -> Dict[str, dict]:
    """Per attribute: F1 for the 'mismatch' class (that is what a buyer wants
    caught), accuracy, and support."""
    out = {}
    for attr, g in df.groupby(attr_col):
        y = g[y_col].astype(int).to_numpy()
        pred = g[pred_col].astype(int).to_numpy()
        out[attr] = {
            "f1_match": float(f1_score(y, pred, pos_label=1, zero_division=0)),
            "f1_mismatch": float(f1_score(y, pred, pos_label=0, zero_division=0)),
            "accuracy": float(np.mean(y == pred)),
            "n": int(len(g)),
        }
    return out


def bootstrap_ci(metric: Callable[[np.ndarray, np.ndarray], float], y: np.ndarray, s: np.ndarray,
                 n_boot: int = 500, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        vals.append(metric(y[idx], s[idx]))
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
