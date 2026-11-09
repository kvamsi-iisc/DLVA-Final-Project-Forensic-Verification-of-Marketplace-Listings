"""Leakage-free train/val/test splitting by product family.

Why not split by listing: ABO contains many near-duplicate listings (the same
sofa in four colours, photographed on the same template). If one colour lands
in train and another in test, the probe memorises the template and the test
score is inflated. The Stanford CS231N ABO report hit exactly this (CIDEr 473).

Mechanism: the split is a deterministic function of the family key:
    bucket = md5(f"{seed}:{family_key}") / 2**32  ->  train | val | test
Every listing with the same family key lands in the same split, on every
machine, on every run. A fake generated FROM a listing inherits its source's
family key, so it lands in the same split as its source as well.
"""
from __future__ import annotations

import pandas as pd

from ..config import SPLIT_FRACTIONS, SPLIT_SEED
from .abo_metadata import stable_hash


def split_for_key(key: str, seed: int = SPLIT_SEED, fractions: dict = None) -> str:
    fractions = fractions or SPLIT_FRACTIONS
    u = (stable_hash(f"{seed}:{key}") % 2**32) / 2**32
    acc = 0.0
    names = list(fractions)
    for name in names:
        acc += fractions[name]
        if u < acc:
            return name
    return names[-1]


def assign_splits(df: pd.DataFrame, group_col: str = "family_key", seed: int = SPLIT_SEED,
                  fractions: dict = None) -> pd.DataFrame:
    out = df.copy()
    out["split"] = out[group_col].map(lambda k: split_for_key(str(k), seed, fractions))
    return out


def leakage_report(df: pd.DataFrame, group_col: str = "family_key", brand_col: str = "brand") -> dict:
    """Assert no group spans two splits; report how much *brand* overlap remains.

    Brand overlap is reported, not forbidden: splitting purely by brand would
    put all of AmazonBasics in one split and make the splits unusable.
    """
    per_group = df.groupby(group_col)["split"].nunique()
    leaking = per_group[per_group > 1]
    if len(leaking):
        raise AssertionError(f"{len(leaking)} groups span more than one split: {list(leaking.index[:5])}")
    brands_by_split = {s: set(df.loc[df["split"] == s, brand_col].dropna()) for s in df["split"].unique()}
    test_brands = brands_by_split.get("test", set())
    train_brands = brands_by_split.get("train", set())
    return {
        "split_sizes": df["split"].value_counts().to_dict(),
        "groups_per_split": df.groupby("split")[group_col].nunique().to_dict(),
        "n_brands_in_both_train_and_test": len(test_brands & train_brands),
        "frac_test_listings_with_brand_seen_in_train": float(
            df.loc[df["split"] == "test", brand_col].isin(train_brands).mean()) if len(test_brands) else 0.0,
    }
