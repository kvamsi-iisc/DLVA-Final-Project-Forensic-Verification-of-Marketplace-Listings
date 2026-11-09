"""Shared bootstrap for scripts: makes `listing_verifier` importable when a
script is run from anywhere (Kaggle runs `python scripts/x.py` from the repo)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def dump_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def device_arg(s: str = None) -> str:
    if s:
        return s
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Multi-GPU fan-out. Kaggle's "GPU T4 x2" gives two 15 GB cards; the weekly
# quota is charged per session hour, not per card, so sharding a job across
# both is free throughput. Pattern: the parent process loads no model, spawns
# one copy of the same script per GPU with `--device cuda:i` and a shard
# argument, streams their logs, waits, then merges whatever needs merging.
# ---------------------------------------------------------------------------
import os
import subprocess
from typing import List, Optional


def resolve_gpus(spec: Optional[str]) -> List[int]:
    """'auto' -> every visible GPU; '0,1' -> [0, 1]; None/'' -> [] (single-device mode)."""
    if not spec:
        return []
    if spec == "auto":
        try:
            import torch
            return list(range(torch.cuda.device_count()))
        except ImportError:
            return []
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def strip_args(argv: List[str], names: List[str]) -> List[str]:
    """Remove `--name value` and `--name=value` occurrences from an argv list."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in names:
            skip = True
            continue
        if any(a.startswith(n + "=") for n in names):
            continue
        out.append(a)
    return out


def spawn_per_gpu(script: str, argv: List[str], gpus: List[int], shard_flag: str,
                  strip: List[str], extra: Optional[List[str]] = None) -> int:
    """Run `script` once per GPU with `--device cuda:i --<shard_flag> i,n`.
    Returns the number of failed workers. Logs are prefixed with the GPU id."""
    base = strip_args(argv, strip + ["--device", "--gpus"])
    procs = []
    for i, g in enumerate(gpus):
        cmd = [sys.executable, script] + base + ["--device", f"cuda:{g}", f"--{shard_flag}", f"{i},{len(gpus)}"] + (extra or [])
        env = {**os.environ, "LV_WORKER": str(i)}
        print(f"[gpu{g}] launching: {' '.join(cmd[2:])}", flush=True)
        procs.append((g, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)))
    failures = 0
    # Stream logs: round-robin readline keeps both workers visible in the notebook.
    import selectors
    sel = selectors.DefaultSelector()
    for g, p in procs:
        sel.register(p.stdout, selectors.EVENT_READ, data=g)
    open_streams = len(procs)
    while open_streams:
        for key, _ in sel.select(timeout=1.0):
            line = key.fileobj.readline()
            if line == "":
                sel.unregister(key.fileobj)
                open_streams -= 1
            else:
                print(f"[gpu{key.data}] {line.rstrip()}", flush=True)
    for g, p in procs:
        rc = p.wait()
        if rc != 0:
            failures += 1
            print(f"[gpu{g}] FAILED with exit code {rc}", flush=True)
    return failures


def parse_shard(spec: Optional[str]):
    """'i,n' -> (i, n); None -> (0, 1)."""
    if not spec:
        return 0, 1
    i, n = (int(x) for x in spec.split(","))
    assert 0 <= i < n
    return i, n


def log_invocation(out_dir, extra: Optional[dict] = None):
    """Record how a result was produced: argv, time, code version, environment.
    Written as <out_dir>/invocation.json next to the artifacts it explains."""
    import datetime
    import platform
    from pathlib import Path as _P
    info = {"argv": sys.argv, "cwd": os.getcwd(), "time_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "python": sys.version.split()[0], "platform": platform.platform(),
            "env": {k: v for k, v in os.environ.items() if k.startswith("LV_") or k in ("HF_HOME", "LV_WORKER")}}
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT,
                                            timeout=5).stdout.strip() or None
    except Exception:
        info["git_commit"] = None
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    if extra:
        info.update(extra)
    out_dir = _P(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(info, out_dir / "invocation.json")
    return info
