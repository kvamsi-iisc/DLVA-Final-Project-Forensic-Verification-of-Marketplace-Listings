"""Small head on log-DCT spectra (the frequency cue from AIDE).

Input per image: a 64x64 pooled log-DCT map and a 128-bin radial profile.
Output: one logit. ~60k parameters; trains on a laptop CPU in a few minutes.
Standardisation statistics are computed on train and stored in the checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..eval.metrics import auroc


class SpectrumHead(nn.Module):
    def __init__(self, map_size: int = 64, radial_bins: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.radial = nn.Sequential(nn.Linear(radial_bins, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64 + 64, 1))
        self.register_buffer("map_mean", torch.zeros(1, 1, map_size, map_size))
        self.register_buffer("map_std", torch.ones(1, 1, map_size, map_size))
        self.register_buffer("rad_mean", torch.zeros(1, radial_bins))
        self.register_buffer("rad_std", torch.ones(1, radial_bins))

    def set_stats(self, maps: np.ndarray, radials: np.ndarray):
        self.map_mean.copy_(torch.from_numpy(maps.mean(0, keepdims=True)[:, None].astype(np.float32)))
        self.map_std.copy_(torch.from_numpy((maps.std(0, keepdims=True) + 1e-4)[:, None].astype(np.float32)))
        self.rad_mean.copy_(torch.from_numpy(radials.mean(0, keepdims=True).astype(np.float32)))
        self.rad_std.copy_(torch.from_numpy((radials.std(0, keepdims=True) + 1e-4).astype(np.float32)))

    def forward(self, dct_map: torch.Tensor, dct_radial: torch.Tensor) -> torch.Tensor:
        m = (dct_map.unsqueeze(1) - self.map_mean) / self.map_std
        r = (dct_radial - self.rad_mean) / self.rad_std
        return self.head(torch.cat([self.conv(m), self.radial(r)], dim=1)).squeeze(1)

    @torch.no_grad()
    def predict_logits(self, maps: np.ndarray, radials: np.ndarray, batch_size: int = 512) -> np.ndarray:
        self.eval()
        dev = self.map_mean.device
        out = []
        for i in range(0, len(maps), batch_size):
            m = torch.from_numpy(maps[i:i + batch_size].astype(np.float32)).to(dev)
            r = torch.from_numpy(radials[i:i + batch_size].astype(np.float32)).to(dev)
            out.append(self(m, r).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def train_spectrum_head(maps_tr, rad_tr, y_tr, maps_va, rad_va, y_va, epochs: int = 30, lr: float = 1e-3,
                        batch_size: int = 256, device: Optional[str] = None, seed: int = 0,
                        patience: int = 6, verbose: bool = True) -> Dict:
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrumHead(map_size=maps_tr.shape[-1], radial_bins=rad_tr.shape[-1]).to(device)
    model.set_stats(maps_tr.astype(np.float32), rad_tr.astype(np.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = float(y_tr.sum())
    pos_weight = torch.tensor([(len(y_tr) - pos) / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    X_m = torch.from_numpy(maps_tr.astype(np.float32))
    X_r = torch.from_numpy(rad_tr.astype(np.float32))
    Y = torch.from_numpy(y_tr.astype(np.float32))
    best_auc, best_state, bad, history = -1.0, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Y))
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_m[idx].to(device), X_r[idx].to(device))
            loss = loss_fn(logits, Y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_auc = auroc(y_va, model.predict_logits(maps_va, rad_va))
        history.append({"epoch": ep, "train_loss": total / len(Y), "val_auroc": val_auc})
        if verbose:
            print(f"[spectrum] epoch {ep:02d} loss {total / len(Y):.4f} val AUROC {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc, bad = val_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.to(device).eval()
    return {"model": model, "val_auroc": best_auc, "history": history}


def save_spectrum_head(model: SpectrumHead, path: Path):
    torch.save({"state_dict": model.state_dict(), "map_size": model.map_mean.shape[-1],
                "radial_bins": model.rad_mean.shape[-1]}, path)


def load_spectrum_head(path: Path, device: str = "cpu") -> SpectrumHead:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = SpectrumHead(ckpt["map_size"], ckpt["radial_bins"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
