"""Push a folder to a Kaggle Dataset (create on first call, new version after).

Needs KAGGLE_USERNAME / KAGGLE_KEY in the environment (on Kaggle: add them as
notebook Secrets and export them in a cell). Used as the --persist-cmd hook of
03_generate_fakes.py so each 500-image chunk survives a session disconnect:
  --persist-cmd "python scripts/kaggle_persist.py --folder data/fakes --slug abo-fakes"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--slug", required=True, help="dataset slug, e.g. abo-fakes")
    ap.add_argument("--title", default=None)
    ap.add_argument("--message", default="chunk update")
    args = ap.parse_args()
    user = os.environ.get("KAGGLE_USERNAME")
    if not user:
        raise SystemExit("KAGGLE_USERNAME / KAGGLE_KEY not set")
    folder = Path(args.folder)
    meta = folder / "dataset-metadata.json"
    first = not meta.exists()
    meta.write_text(json.dumps({"title": args.title or args.slug, "id": f"{user}/{args.slug}",
                                "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}, indent=2))
    if first:
        cmd = ["kaggle", "datasets", "create", "-p", str(folder), "--dir-mode", "zip"]
    else:
        cmd = ["kaggle", "datasets", "version", "-p", str(folder), "-m", args.message, "--dir-mode", "zip"]
    print("[persist]", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    if first and r.returncode != 0 and "already" in (r.stdout + r.stderr).lower():
        cmd[2] = "version"
        cmd += ["-m", args.message]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit(f"[persist] FAILED (exit {r.returncode}); the local files are intact, retry later")
    print("[persist] ok")


if __name__ == "__main__":
    main()
