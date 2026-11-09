"""Consistency branch: does a claimed attribute match the image?

Three ingredients, all on the frozen backbone + the small VLM:
  1. CLIP zero-shot contrast per claim: "a photo of a red chair" vs the same
     prompt with every other vocab colour. Gives P(claim | image) over options.
  2. SmolVLM multiple-choice answer over the same option list.
  3. A tiny calibration layer (logistic regression on ~6 numbers) trained on
     synthetic mismatch pairs, turning the two opinions into one calibrated
     P(claim matches).

The calibrator is what makes the output honest: raw CLIP softmax over 12
colours is confidently wrong on pastel shades; the calibrator learns how far
to trust it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..config import COLOUR_VOCAB, COUNT_MAX, COUNT_MIN, MATERIAL_VOCAB

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
COUNT_OPTIONS = list(range(COUNT_MIN, COUNT_MAX + 1))

ATTRIBUTE_OPTIONS = {
    "colour": COLOUR_VOCAB,
    "material": MATERIAL_VOCAB,
    "count": COUNT_OPTIONS,
}

# Prompt ensembles: averaging a few phrasings makes CLIP zero-shot noticeably
# steadier (the standard trick from the CLIP paper).
COLOUR_TEMPLATES = [
    "a photo of a {v} {noun}.",
    "a product photo of a {v} {noun} on a white background.",
    "a {v} {noun}.",
]
MATERIAL_TEMPLATES = [
    "a photo of a {noun} made of {v}.",
    "a product photo of a {v} {noun}.",
    "a {noun} made of {v}.",
]
COUNT_TEMPLATES = [
    "a photo of {v}.",
    "a product photo of {v} on a white background.",
]

VLM_QUESTIONS = {
    "colour": "What is the main colour of the product in this photo?",
    "material": "What material is the product in this photo mainly made of?",
    "count": "How many separate units of the product are shown in this photo?",
}


def _plural(noun: str) -> str:
    return noun if noun.endswith("s") else noun + "s"


def count_phrase(n: int, noun: str) -> str:
    if n == 1:
        return f"a single {noun}"
    return f"a set of {NUMBER_WORDS[n]} {_plural(noun)}"


def prompts_for(attribute: str, noun: str) -> List[List[str]]:
    """List (per option) of prompt variants."""
    if attribute == "colour":
        return [[t.format(v=v, noun=noun) for t in COLOUR_TEMPLATES] for v in COLOUR_VOCAB]
    if attribute == "material":
        return [[t.format(v=v, noun=noun) for t in MATERIAL_TEMPLATES] for v in MATERIAL_VOCAB]
    if attribute == "count":
        return [[t.format(v=count_phrase(n, noun)) for t in COUNT_TEMPLATES] for n in COUNT_OPTIONS]
    raise ValueError(attribute)


def vlm_options(attribute: str) -> List[str]:
    if attribute == "count":
        return [str(n) for n in COUNT_OPTIONS]
    return [str(v) for v in ATTRIBUTE_OPTIONS[attribute]]


class ClipZeroShot:
    """Caches the ensembled text embeddings per (attribute, product noun)."""

    def __init__(self, backbone):
        self.backbone = backbone
        self._cache: Dict[tuple, np.ndarray] = {}

    def option_embeddings(self, attribute: str, noun: str) -> np.ndarray:
        key = (attribute, noun)
        if key not in self._cache:
            groups = prompts_for(attribute, noun)
            flat = [p for g in groups for p in g]
            emb = self.backbone.encode_texts(flat)
            k = len(groups[0])
            emb = emb.reshape(len(groups), k, -1).mean(axis=1)
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            self._cache[key] = emb.astype(np.float32)
        return self._cache[key]

    def option_probabilities(self, image_feat: np.ndarray, attribute: str, noun: str) -> np.ndarray:
        """image_feat: (768,) L2-normalised. Returns softmax over options."""
        emb = self.option_embeddings(attribute, noun)
        logits = self.backbone.logit_scale * (emb @ image_feat)
        logits = logits - logits.max()
        p = np.exp(logits)
        return (p / p.sum()).astype(np.float32)


def option_index(attribute: str, value) -> int:
    opts = ATTRIBUTE_OPTIONS[attribute]
    return opts.index(int(value) if attribute == "count" else str(value))


FEATURE_NAMES = ["clip_p_claim", "clip_logodds_claim", "clip_margin", "clip_rank",
                 "vlm_p_claim", "vlm_top_is_claim", "vlm_entropy"]


def claim_features(clip_p: np.ndarray, vlm_p: Optional[np.ndarray], claim_idx: int) -> np.ndarray:
    """The ~7 numbers the calibrator sees for one (image, claim)."""
    clip_p = np.asarray(clip_p, dtype=np.float64)
    pc = float(clip_p[claim_idx])
    others = np.delete(clip_p, claim_idx)
    rank = float((clip_p > pc).sum()) / max(len(clip_p) - 1, 1)
    f = [pc, np.log(pc + 1e-6) - np.log(1 - pc + 1e-6), np.log(pc + 1e-6) - np.log(others.max() + 1e-6), rank]
    if vlm_p is None:
        f += [0.5, 0.0, 0.0]
    else:
        vlm_p = np.asarray(vlm_p, dtype=np.float64)
        ent = float(-(vlm_p * np.log(vlm_p + 1e-9)).sum())
        f += [float(vlm_p[claim_idx]), float(vlm_p.argmax() == claim_idx), ent]
    return np.array(f, dtype=np.float32)


class ClaimCalibrator:
    """Per-attribute logistic regression over claim_features -> P(match)."""

    def __init__(self):
        self.models: Dict[str, Dict] = {}

    def fit(self, attribute: str, X: np.ndarray, y: np.ndarray, seed: int = 0):
        mean, std = X.mean(0), X.std(0) + 1e-6
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        clf.fit((X - mean) / std, y)
        self.models[attribute] = {"w": clf.coef_[0], "b": float(clf.intercept_[0]), "mean": mean, "std": std,
                                  "n": int(len(y))}

    def predict_proba(self, attribute: str, X: np.ndarray) -> np.ndarray:
        m = self.models[attribute]
        z = ((np.atleast_2d(X) - m["mean"]) / m["std"]) @ m["w"] + m["b"]
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path: Path):
        path = Path(path)
        arrays = {}
        for a, m in self.models.items():
            for k in ("w", "mean", "std"):
                arrays[f"{a}__{k}"] = np.asarray(m[k], dtype=np.float32)
            arrays[f"{a}__b"] = np.float32(m["b"])
        np.savez(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps(
            {a: {"n": m["n"], "weights": dict(zip(FEATURE_NAMES, map(float, m["w"])))} for a, m in self.models.items()},
            indent=2))

    @classmethod
    def load(cls, path: Path) -> "ClaimCalibrator":
        z = np.load(Path(path))
        cal = cls()
        attrs = sorted({k.split("__")[0] for k in z.files})
        for a in attrs:
            cal.models[a] = {"w": z[f"{a}__w"], "b": float(z[f"{a}__b"]), "mean": z[f"{a}__mean"],
                             "std": z[f"{a}__std"], "n": -1}
        return cal

    def has(self, attribute: str) -> bool:
        return attribute in self.models
