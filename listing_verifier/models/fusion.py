"""Late fusion of the semantic probe and the frequency head.

Two logits in, one probability out, through a logistic regression fitted on the
VALIDATION split (never on test). We also report the naive mean of the two
branch probabilities so the reader can see whether the learned weights matter.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression


class LateFusion:
    def __init__(self, w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray, meta: Dict = None):
        self.w, self.b, self.mean, self.std, self.meta = w, b, mean, std, meta or {}

    def logits(self, probe_logit: np.ndarray, spec_logit: np.ndarray) -> np.ndarray:
        x = (np.stack([probe_logit, spec_logit], axis=1) - self.mean) / self.std
        return x @ self.w + self.b

    def probabilities(self, probe_logit, spec_logit):
        return 1.0 / (1.0 + np.exp(-self.logits(probe_logit, spec_logit)))

    @staticmethod
    def mean_probability(probe_logit, spec_logit):
        p1 = 1 / (1 + np.exp(-np.asarray(probe_logit)))
        p2 = 1 / (1 + np.exp(-np.asarray(spec_logit)))
        return 0.5 * (p1 + p2)

    def save(self, path: Path):
        path = Path(path)
        np.savez(path, w=self.w.astype(np.float32), b=np.float32(self.b),
                 mean=self.mean.astype(np.float32), std=self.std.astype(np.float32))
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LateFusion":
        path = Path(path)
        z = np.load(path)
        meta = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
        return cls(z["w"], float(z["b"]), z["mean"], z["std"], meta)


def fit_fusion(probe_logit_val, spec_logit_val, y_val, seed: int = 0) -> LateFusion:
    x = np.stack([probe_logit_val, spec_logit_val], axis=1).astype(np.float64)
    mean, std = x.mean(0), x.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit((x - mean) / std, y_val)
    w = clf.coef_[0]
    return LateFusion(w, float(clf.intercept_[0]), mean, std,
                      meta={"weight_probe": float(w[0]), "weight_spectrum": float(w[1]),
                            "relative_weight_spectrum": float(abs(w[1]) / (abs(w[0]) + abs(w[1]) + 1e-9))})
