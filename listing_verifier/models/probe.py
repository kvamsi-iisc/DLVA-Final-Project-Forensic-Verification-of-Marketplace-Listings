"""Linear probe on frozen CLIP features (the UnivFD recipe).

One weight vector over 768 numbers. That it works at all is the point of
Ojha et al.: a feature space that was never trained to spot fakes does not
pick up the generator-specific shortcut that end-to-end CNNs learn.
Saved as plain numpy so the CPU demo needs no sklearn pickle.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..eval.metrics import auroc


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


class LinearProbe:
    def __init__(self, w: Optional[np.ndarray] = None, b: float = 0.0, normalise: bool = True,
                 meta: Optional[Dict] = None):
        self.w = w
        self.b = b
        self.normalise = normalise
        self.meta = meta or {}

    def logits(self, feats: np.ndarray) -> np.ndarray:
        x = _l2n(feats) if self.normalise else feats
        return x @ self.w + self.b

    def probabilities(self, feats: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logits(feats)))

    def save(self, path: Path):
        path = Path(path)
        np.savez(path, w=self.w.astype(np.float32), b=np.float32(self.b), normalise=self.normalise)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LinearProbe":
        path = Path(path)
        z = np.load(path)
        meta = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
        return cls(z["w"], float(z["b"]), bool(z["normalise"]), meta)


def train_probe(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                cs=(0.1, 0.3, 1.0, 3.0, 10.0), normalise: bool = True, seed: int = 0) -> LinearProbe:
    """Pick the inverse-regularisation C by validation AUROC. Takes seconds."""
    xt = _l2n(x_train) if normalise else x_train
    xv = _l2n(x_val) if normalise else x_val
    best = None
    history = {}
    for c in cs:
        clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=seed)
        clf.fit(xt, y_train)
        score = auroc(y_val, clf.decision_function(xv))
        history[str(c)] = score
        if best is None or score > best[0]:
            best = (score, c, clf)
    score, c, clf = best
    return LinearProbe(clf.coef_[0].astype(np.float32), float(clf.intercept_[0]), normalise,
                       meta={"C": c, "val_auroc": score, "val_auroc_by_C": history,
                             "n_train": int(len(y_train)), "n_val": int(len(y_val))})
