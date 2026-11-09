"""Scoring and evaluation helpers shared by the training, evaluation and demo code."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..features.extract import FeatureStore
from ..models.fusion import LateFusion
from ..models.probe import LinearProbe
from ..models.spectrum_head import load_spectrum_head
from .metrics import auroc, bootstrap_ci, summarise_detector, threshold_at_fpr

MODEL_NAMES = ["probe", "spectrum", "fused", "mean"]


def forensics_part(image_key: str) -> str:
    """Deterministic 50/10/40 split of the forensics subset, used only by the
    style-mismatch control experiment (train on non-product fakes)."""
    from ..data.abo_metadata import stable_hash
    u = (stable_hash("ctrl:" + image_key) % 1000) / 1000
    return "ctrl_train" if u < 0.5 else ("ctrl_val" if u < 0.6 else "ctrl_test")


def load_models(ckpt_dir: Path, device: str = "cpu", require_fusion: bool = True) -> Dict:
    ckpt_dir = Path(ckpt_dir)
    if not (ckpt_dir / "probe.npz").exists():
        raise FileNotFoundError(f"{ckpt_dir / 'probe.npz'} missing: train with scripts/07_train_authenticity.py")
    out = {"probe": LinearProbe.load(ckpt_dir / "probe.npz")}
    have = (ckpt_dir / "spectrum.pt").exists() and (ckpt_dir / "fusion.npz").exists()
    if have:
        out["spectrum"] = load_spectrum_head(ckpt_dir / "spectrum.pt", device=device)
        out["fusion"] = LateFusion.load(ckpt_dir / "fusion.npz")
    elif require_fusion:
        raise FileNotFoundError(f"spectrum.pt / fusion.npz missing in {ckpt_dir}; use mode='probe' or train them")
    th = ckpt_dir / "thresholds.json"
    out["thresholds"] = json.loads(th.read_text()) if th.exists() else {}
    return out


def score_features(models: Dict, clip: np.ndarray, dct_map: np.ndarray, dct_radial: np.ndarray) -> Dict[str, np.ndarray]:
    probe_logit = models["probe"].logits(clip)
    if "spectrum" not in models:
        return {"probe_logit": probe_logit, "probe": 1 / (1 + np.exp(-probe_logit))}
    spec_logit = models["spectrum"].predict_logits(dct_map, dct_radial)
    return {
        "probe_logit": probe_logit, "spectrum_logit": spec_logit,
        "probe": 1 / (1 + np.exp(-probe_logit)), "spectrum": 1 / (1 + np.exp(-spec_logit)),
        "fused": models["fusion"].probabilities(probe_logit, spec_logit),
        "mean": LateFusion.mean_probability(probe_logit, spec_logit),
    }


def score_rows(models: Dict, store: FeatureStore, rows: pd.DataFrame, quality: int = 0) -> pd.DataFrame:
    keys = rows["image_key"].astype(str).tolist()
    f = store.get(keys, quality)
    sub = rows[f["present"]].reset_index(drop=True).copy()
    s = score_features(models, f["clip"], f["dct_map"], f["dct_radial"])
    for k, v in s.items():
        sub[k] = v
    return sub


def evaluate_scores(scores: pd.DataFrame, thresholds: Optional[Dict] = None, with_ci: bool = True) -> Dict[str, Dict]:
    out = {}
    y = scores["label"].to_numpy().astype(int)
    for m in MODEL_NAMES:
        if m not in scores:
            continue
        s = scores[m].to_numpy()
        th = (thresholds or {}).get(m)
        res = summarise_detector(y, s, th)
        if with_ci and len(np.unique(y)) > 1:
            res["auroc_ci95"] = bootstrap_ci(auroc, y, s, n_boot=300)
        out[m] = res
    return out


def fit_thresholds(scores_val: pd.DataFrame, target_fpr: float = 0.05) -> Dict[str, float]:
    y = scores_val["label"].to_numpy().astype(int)
    return {m: threshold_at_fpr(y, scores_val[m].to_numpy(), target_fpr) for m in MODEL_NAMES if m in scores_val}
