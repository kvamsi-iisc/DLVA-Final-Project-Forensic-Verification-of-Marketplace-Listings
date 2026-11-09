"""Print and save the exact environment a run used (for the report's appendix)."""
from __future__ import annotations

import argparse
import importlib
import platform
import sys

from _common import ROOT, dump_json

PKGS = ["torch", "torchvision", "open_clip", "transformers", "diffusers", "datasets", "accelerate",
        "sklearn", "numpy", "PIL", "pandas", "fastapi", "huggingface_hub"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "results" / "env_report.json"))
    args = ap.parse_args()
    info = {"python": sys.version, "platform": platform.platform()}
    for p in PKGS:
        try:
            m = importlib.import_module(p)
            info[p] = getattr(m, "__version__", "?")
        except Exception as e:
            info[p] = f"not installed ({type(e).__name__})"
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:
        pass
    for k, v in info.items():
        print(f"{k:18s} {v}")
    dump_json(info, args.out)


if __name__ == "__main__":
    main()
