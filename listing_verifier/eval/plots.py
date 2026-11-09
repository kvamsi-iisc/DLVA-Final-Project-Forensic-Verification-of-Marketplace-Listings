"""Report figures. matplotlib with the Agg backend so it works on Kaggle and CI."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

MODEL_STYLES = {"probe": ("CLIP linear probe (semantic)", "tab:blue", "-"),
                "spectrum": ("log-DCT head (frequency)", "tab:red", "--"),
                "fused": ("late fusion", "tab:green", "-."),
                "mean": ("mean of probabilities", "tab:gray", ":"),
                "univfd": ("UnivFD released weights", "tab:purple", "-")}


def plot_jpeg_curve(curve: Dict[str, Dict[int, float]], out: Path, title: str = "AUROC vs JPEG quality"):
    """curve[model][quality] = AUROC; quality 0 is drawn as 'none' at the left."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, qs in curve.items():
        label, colour, ls = MODEL_STYLES.get(model, (model, None, "-"))
        xs = sorted(qs, key=lambda q: (q == 0, -q))   # 0 (none) first, then 90..30
        x_pos = list(range(len(xs)))
        ax.plot(x_pos, [qs[q] for q in xs], marker="o", label=label, color=colour, linestyle=ls)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["none" if q == 0 else str(q) for q in xs])
    ax.set_xlabel("JPEG quality applied to test images")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc(curves: Dict[str, tuple], out: Path, title: str = "ROC"):
    fig, ax = plt.subplots(figsize=(5, 5))
    for model, (y, s) in curves.items():
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        label, colour, ls = MODEL_STYLES.get(model, (model, None, "-"))
        ax.plot(fpr, tpr, label=label, color=colour, linestyle=ls)
    ax.plot([0, 1], [0, 1], "k:", lw=0.5)
    ax.axvline(0.05, color="orange", lw=0.7, label="5% FPR")
    ax.set_xlabel("false positive rate (real flagged as fake)")
    ax.set_ylabel("true positive rate (fakes caught)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_reliability(y: np.ndarray, p: np.ndarray, out: Path, title: str = "Reliability", n_bins: int = 10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            xs.append(p[m].mean()), ys.append(y[m].mean()), ns.append(m.sum())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k:", lw=0.7)
    ax.scatter(xs, ys, s=[20 + 2 * np.sqrt(n) for n in ns])
    ax.plot(xs, ys, lw=0.8)
    ax.set_xlabel("predicted P(claim matches)")
    ax.set_ylabel("observed match rate")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_bars(values: Dict[str, float], out: Path, title: str, ylabel: str = "F1 (mismatch class)"):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(list(values), list(values.values()))
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
