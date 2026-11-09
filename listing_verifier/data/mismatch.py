"""Synthetic mismatch pairs for the consistency branch.

For every listing with a trusted attribute value we create
    (image, attribute, claimed = true value,  label 1)
    (image, attribute, claimed = other value, label 0)
The "other" value is drawn from the empirical distribution of that attribute
in the same split, so negatives are not dominated by rare colours (a model
that only learns 'purple is rare' would look good otherwise).

Student honesty note: this makes the test *symmetric and easy* compared with
real fraud (a seller usually lies in a plausible direction, e.g. 'leather' for
PU). The 100 hand-labelled listings are the reality check.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import COUNT_MAX, COUNT_MIN

PAIR_COLUMNS = ["image_key", "product_noun", "attribute", "claimed", "truth", "label"]


def make_pairs(index: pd.DataFrame, attribute: str, split: str, seed: int = 0,
               max_items: int = None) -> pd.DataFrame:
    df = index[(index["split"] == split) & (index["label"] == 0)].copy()
    if attribute == "count":
        df = df[df["count_verifiable"].astype(bool) & df["count"].notna()]
        values = df["count"].astype(int)
        universe = list(range(COUNT_MIN, COUNT_MAX + 1))
    else:
        df = df[df[attribute].notna()]
        values = df[attribute].astype(str)
        universe = sorted(values.unique().tolist())
    if max_items and len(df) > max_items:
        df = df.sample(max_items, random_state=seed)
        values = values.loc[df.index]
    rng = np.random.default_rng(seed)
    freq = values.value_counts(normalize=True)
    rows: List[Dict] = []
    for key, noun, v in zip(df["image_key"], df["product_noun"], values):
        others = [u for u in universe if u != v]
        if not others:
            continue
        w = np.array([freq.get(u, 0.0) + 1e-3 for u in others])
        neg = others[rng.choice(len(others), p=w / w.sum())]
        rows.append({"image_key": key, "product_noun": noun, "attribute": attribute, "claimed": v, "truth": v, "label": 1})
        rows.append({"image_key": key, "product_noun": noun, "attribute": attribute, "claimed": neg, "truth": v, "label": 0})
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)   # schema survives an empty result
