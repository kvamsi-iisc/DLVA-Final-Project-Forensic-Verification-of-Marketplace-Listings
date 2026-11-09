"""Published detectors as baselines (never the deliverable).

UnivFD (Ojha et al. 2023) released `fc_weights.pth`: a single nn.Linear(768, 1)
over the CLIP ViT-L/14 image embedding, the same embedding our backbone yields.
So the baseline is literally our LinearProbe class with their weights plugged
in, scored on OUR product-photo test set. Their number on generic scenes is
not assumed to carry over; we measure it.

Download: https://github.com/WisconsinAIVision/UniversalFakeDetect (pretrained
weights link in the README) -> place at outputs/checkpoints/univfd_fc_weights.pth

AIDE (Yan et al. 2025) is a larger model (ConvNeXt on DCT patches + CLIP). We
do not re-implement it; `export_file_list` writes our test images in the
format its evaluation script reads, so you can run their repo on our data and
paste the resulting AUROC into the results table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .probe import LinearProbe


def load_univfd_probe(weights_path: Path) -> LinearProbe:
    import torch
    sd = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
    if "weight" in sd:
        w, b = sd["weight"], sd["bias"]
    else:  # some mirrors wrap it as {'fc.weight': ..., 'fc.bias': ...}
        w = next(v for k, v in sd.items() if k.endswith("weight"))
        b = next(v for k, v in sd.items() if k.endswith("bias"))
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    b = float(np.asarray(b).reshape(-1)[0])
    # UnivFD applies the linear layer to the raw (un-normalised) CLIP embedding.
    return LinearProbe(w, b, normalise=False, meta={"source": "UnivFD released fc_weights.pth"})


def export_file_list(index: pd.DataFrame, out_path: Path, split: str = "test"):
    sub = index[index["split"] == split]
    rows = [f"{p}\t{int(l)}" for p, l in zip(sub["path"], sub["label"])]
    Path(out_path).write_text("\n".join(rows) + "\n")
    return len(rows)
