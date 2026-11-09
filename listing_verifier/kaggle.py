"""Helpers the Kaggle notebooks import from the repo, so fixes land with a
`git pull` instead of a notebook re-import.

    from listing_verifier.kaggle import init, run, attach, merge_fakes
    WORK, REPO, DATA, OUT, LOGS = init("/kaggle/working")

Attached inputs are found in both Kaggle layouts: classic `/kaggle/input/<slug>` and
the newer `/kaggle/input/datasets/<owner>/<slug>` (`dataset_root`, `input_file`, `is_attached`).
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

INPUT = Path("/kaggle/input")
_STATE = {}


def init(work="/kaggle/working"):
    # never let transformers pull in the image's TensorFlow (protobuf gencode/runtime clash); workers inherit this
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    work = Path(work)
    paths = {"WORK": work, "REPO": work / "repo", "DATA": work / "data", "OUT": work / "outputs", "LOGS": work / "outputs" / "logs"}
    for p in ("DATA", "OUT", "LOGS"):
        paths[p].mkdir(parents=True, exist_ok=True)
    _STATE.update(paths)
    return paths["WORK"], paths["REPO"], paths["DATA"], paths["OUT"], paths["LOGS"]


def run(*args):
    """Run a pipeline script; stream output to the cell and to LOGS/<script>_<time>.log,
    append a completion record to LOGS/completed.jsonl, raise on a non-zero exit."""
    logs = _STATE["LOGS"]
    cmd = [sys.executable, *[str(a) for a in args]]
    name = Path(str(args[0])).stem
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")   # microseconds: no two logs collide
    log_path = logs / f"{name}_{stamp}.log"
    print(">>>", " ".join(cmd[1:]), "->", log_path.name, flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = proc.wait()
    record = {"script": name, "args": cmd[2:], "returncode": rc, "seconds": round(time.time() - t0, 1),
              "log": str(log_path), "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with open(logs / "completed.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _describe(base: Path, depth: int = 2) -> str:
    lines = []
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(base)
        if len(rel.parts) <= depth:
            lines.append(("  " * (len(rel.parts) - 1)) + rel.parts[-1] + ("/" if p.is_dir() else ""))
        if len(lines) >= 40:
            lines.append("  ...")
            break
    return "\n".join(lines)


def dataset_root(slug: str, input_root: Path = INPUT) -> Optional[Path]:
    """Folder of an attached dataset, whichever layout Kaggle uses:
    classic `/kaggle/input/<slug>` or the newer `/kaggle/input/datasets/<owner>/<slug>`
    (models and competitions get their own prefixes the same way). Falls back to any
    directory named `slug` within three levels, so a renamed prefix does not break us."""
    input_root = Path(input_root)
    if not input_root.exists():
        return None
    direct = input_root / slug
    if direct.exists():
        return direct
    for kind in ("datasets", "models", "competitions"):
        hits = sorted((input_root / kind).glob(f"*/{slug}")) + sorted((input_root / kind).glob(slug))
        if hits:
            return hits[0]
    for pattern in (f"*/{slug}", f"*/*/{slug}", f"*/*/*/{slug}"):
        hits = [h for h in sorted(input_root.glob(pattern)) if h.is_dir()]
        if hits:
            return hits[0]
    return None


def is_attached(slug: str, input_root: Path = INPUT) -> bool:
    return dataset_root(slug, input_root) is not None


def input_file(slug: str, rel: str, input_root: Path = INPUT) -> Optional[Path]:
    """`rel` inside the attached dataset `slug` (any layout), or None when either is absent.
    Searches for the file name when the dataset was uploaded with extra nesting."""
    base = dataset_root(slug, input_root)
    if base is None:
        return None
    direct = base / rel
    if direct.exists():
        return direct
    hits = sorted(base.rglob(Path(rel).name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def list_attached(input_root: Path = INPUT) -> List[str]:
    """Slugs of every attached input as `owner/slug` (new layout) or `slug` (classic)."""
    input_root = Path(input_root)
    if not input_root.exists():
        return []
    out = []
    for p in sorted(input_root.iterdir()):
        if p.name in ("datasets", "models", "competitions") and p.is_dir():
            for owner in sorted(p.iterdir()):
                if owner.is_dir():
                    out.extend(f"{p.name}/{owner.name}/{d.name}" for d in sorted(owner.iterdir()) if d.is_dir())
        else:
            out.append(p.name)
    return out


def find_source(slug: str, marker: str, input_root: Path = INPUT) -> Path:
    """Locate the folder inside the attached dataset that contains `marker`
    (a file or directory name), whatever nesting Kaggle gave the upload.
    Picks the shallowest match; raises with a listing of the dataset if none."""
    base = dataset_root(slug, input_root)
    if base is None:
        raise FileNotFoundError(f"dataset '{slug}' is not attached. Attached inputs: {list_attached(input_root)}")
    hits = sorted(base.rglob(marker), key=lambda p: len(p.parts))
    if not hits:
        raise FileNotFoundError(f"'{marker}' not found anywhere inside {base}. Dataset layout:\n{_describe(base)}")
    return hits[0].parent


def attach(slug: str, sub: str, dest, must_contain: str, input_root: Path = INPUT) -> Path:
    """Copy the dataset folder holding `must_contain` to `dest` (copy, not symlink:
    scripts write next to inputs). An existing complete dest is kept; a stale or
    incomplete one is replaced. `sub` is only a hint for the message."""
    dest = Path(dest)
    if dest.exists() and (dest / must_contain).exists():
        print(f"exists: {dest}")
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    src = find_source(slug, must_contain, input_root)
    print(f"copy {src} -> {dest}  (expected {sub}, marker {must_contain})")
    shutil.copytree(src, dest)
    if not (dest / must_contain).exists():
        raise FileNotFoundError(f"copied {src} but {dest / must_contain} is missing")
    return dest


def merge_fakes(slugs: Iterable[str], dest, input_root: Path = INPUT) -> int:
    """Merge every attached fake dataset into DATA/fakes (each run has its own folder)."""
    dest = Path(dest)
    for slug in slugs:
        base = dataset_root(slug, input_root)
        if base is None:
            print("not attached:", slug)
            continue
        chunks = list(base.rglob("chunk_*.csv"))
        if not chunks:
            print(f"warning: {slug} has no chunk_*.csv; layout:\n{_describe(base)}")
            continue
        # the fakes root is the folder named 'fakes' above the chunks, else the common parent
        root = next((p for p in chunks[0].parents if p.name == "fakes"), chunks[0].parent.parent.parent)
        collisions = [c for c in chunks if (dest / c.relative_to(root)).exists()]
        if collisions:
            print(f"WARNING: {slug}: {len(collisions)} chunk files already exist in {dest} and will be overwritten "
                  f"(same run folder in two datasets?), e.g. {collisions[0].relative_to(root)}")
        shutil.copytree(root, dest, dirs_exist_ok=True)
        print(f"merged {slug}: {len(chunks)} chunk files from {root}")
    n = len(list(dest.rglob("chunk_*.csv"))) if dest.exists() else 0
    print("fake chunks total:", n)
    return n


def optional_attach(slug: str, sub: str, dest, must_contain: str, input_root: Path = INPUT) -> Optional[Path]:
    if is_attached(slug, input_root):
        return attach(slug, sub, dest, must_contain, input_root)
    print(f"optional dataset {slug} not attached; skipping")
    return None
