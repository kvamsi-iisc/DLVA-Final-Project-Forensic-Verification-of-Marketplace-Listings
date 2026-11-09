"""Helpers the Kaggle notebooks import from the repo, so fixes land with a
`git pull` instead of a notebook re-import.

    from listing_verifier.kaggle import init, run, attach, merge_fakes
    WORK, REPO, DATA, OUT, LOGS = init("/kaggle/working")
"""
from __future__ import annotations

import datetime
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

INPUT = Path("/kaggle/input")
_STATE = {}


def init(work="/kaggle/working"):
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


def find_source(slug: str, marker: str, input_root: Path = INPUT) -> Path:
    """Locate the folder inside the attached dataset that contains `marker`
    (a file or directory name), whatever nesting Kaggle gave the upload.
    Picks the shallowest match; raises with a listing of the dataset if none."""
    base = input_root / slug
    if not base.exists():
        attached = sorted(p.name for p in input_root.iterdir()) if input_root.exists() else []
        raise FileNotFoundError(f"dataset '{slug}' is not attached. Attached inputs: {attached}")
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
        base = input_root / slug
        if not base.exists():
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
    if (input_root / slug).exists():
        return attach(slug, sub, dest, must_contain, input_root)
    print(f"optional dataset {slug} not attached; skipping")
    return None
