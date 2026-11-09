"""Verify that promised artifacts exist AND have content. Exit 1 listing every problem.

  python scripts/verify_outputs.py data/features/features_q0.npz outputs/checkpoints/main/probe.npz ...
Rules: directory -> non-empty; .csv -> at least one row; .npz -> loads, every
array non-empty and finite, and the file matches the schema its name implies
(probe / fusion / calibrator / features_q*); .json -> parses (thresholds.json
needs finite model thresholds); .zip -> has members; .pt -> loads with
weights_only and, for spectrum.pt, has the expected keys; anything else -> size > 0.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Optional

from _common import ROOT  # noqa: F401  (sys.path bootstrap)


NPZ_SCHEMAS = {
    # filename prefix -> {array: expected shape (None = any length on that axis)}
    "probe": {"w": (768,), "b": ()},
    "fusion": {"w": (2,), "b": (), "mean": (2,), "std": (2,)},
    "features_q": {"image_key": (None,), "clip": (None, 768), "dct_map": (None, 64, 64), "dct_radial": (None, 128), "fingerprint": ()},
}


def _shape_ok(arr, expected) -> bool:
    if len(arr.shape) != len(expected):
        return False
    return all(e is None or a == e for a, e in zip(arr.shape, expected))


def _check_npz(path: Path, z) -> Optional[str]:
    import numpy as np
    if not z.files:
        return f"npz has no arrays: {path}"
    for k in z.files:
        a = np.asarray(z[k])
        if a.size == 0:
            return f"empty array {k!r}: {path}"
        if np.issubdtype(a.dtype, np.floating) and not np.all(np.isfinite(a)):
            return f"non-finite values in {k!r}: {path}"
    schema = next((v for prefix, v in NPZ_SCHEMAS.items() if path.name.startswith(prefix)), None)
    if schema:
        for k, shape in schema.items():
            if k not in z.files:
                return f"{path.name} lacks array {k!r}"
            if not _shape_ok(np.asarray(z[k]), shape):
                return f"{path.name}: {k!r} has shape {np.asarray(z[k]).shape}, expected {shape}"
        if path.name.startswith("features_q"):
            n = len(z["image_key"])
            if not all(len(z[k]) == n for k in ("clip", "dct_map", "dct_radial")):
                return f"{path.name}: row counts disagree with image_key ({n})"
    if path.name.startswith("calibrator"):
        attrs = {k.split("__")[0] for k in z.files if "__" in k}
        if not attrs:
            return f"{path.name}: no calibrated attribute"
        for a in attrs:
            for part, shape in (("w", (7,)), ("mean", (7,)), ("std", (7,)), ("b", ())):
                key = f"{a}__{part}"
                if key not in z.files or not _shape_ok(np.asarray(z[key]), shape):
                    return f"{path.name}: {key!r} missing or wrong shape"
    return None


def verify_output(path) -> Optional[str]:
    path = Path(path)
    if not path.exists():
        return f"missing: {path}"
    try:
        if path.is_dir():
            return None if any(path.iterdir()) else f"empty dir: {path}"
        suf = path.suffix.lower()
        if suf == ".csv":
            import pandas as pd
            return None if len(pd.read_csv(path)) > 0 else f"empty csv: {path}"
        if suf == ".npz":
            import numpy as np
            return _check_npz(path, np.load(path))
        if suf == ".json":
            obj = json.load(open(path))
            if path.name == "thresholds.json":
                import math
                if not isinstance(obj, dict) or "probe" not in obj or not all(
                        isinstance(v, (int, float)) and math.isfinite(v) for v in obj.values()):
                    return f"thresholds.json malformed: {obj}"
            return None
        if suf == ".zip":
            return None if zipfile.ZipFile(path).namelist() else f"empty zip: {path}"
        if suf == ".pt":
            import torch
            obj = torch.load(path, map_location="cpu", weights_only=True)
            if path.name == "spectrum.pt":
                if not isinstance(obj, dict) or not {"state_dict", "map_size", "radial_bins"} <= set(obj):
                    return f"spectrum.pt lacks state_dict/map_size/radial_bins"
                bad = [k for k, v in obj["state_dict"].items() if v.is_floating_point() and not torch.isfinite(v).all()]
                if bad:
                    return f"spectrum.pt has non-finite tensors: {bad[:3]}"
            return None
        return None if path.stat().st_size > 0 else f"empty file: {path}"
    except Exception as e:
        return f"unreadable {path}: {type(e).__name__}: {e}"


def main(argv):
    problems = [p for p in (verify_output(a) for a in argv) if p]
    for p in problems:
        print("[verify] PROBLEM:", p)
    if problems:
        sys.exit(1)
    print(f"[verify] ok: {len(argv)} artifacts verified")


if __name__ == "__main__":
    main(sys.argv[1:])
