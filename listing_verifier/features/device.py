"""Device and precision selection that works for 'cuda', 'cuda:1' and 'cpu'.

Bug this replaces: three loaders tested `device == "cuda"`, so the multi-GPU
workers (which get 'cuda:0' / 'cuda:1') silently loaded float32 models.
"""
from __future__ import annotations

from typing import Optional

import torch


def device_type(device) -> str:
    return torch.device(device).type


def is_cuda(device) -> bool:
    return device_type(device) == "cuda"


def bf16_supported(device) -> bool:
    if not is_cuda(device):
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def pick_dtype(device, prefer: str = "fp16") -> torch.dtype:
    """float32 on CPU. On CUDA: fp16 by default (T4 has no native bf16);
    bf16 only when asked for AND supported by the card."""
    if not is_cuda(device):
        return torch.float32
    if prefer == "bf16" and bf16_supported(device):
        return torch.bfloat16
    return torch.float16


def describe(device, model: Optional[torch.nn.Module] = None) -> str:
    bits = [f"device={device}"]
    if model is not None:
        try:
            p = next(model.parameters())
            bits.append(f"param_dtype={p.dtype}")
        except StopIteration:
            pass
    if is_cuda(device):
        idx = torch.device(device).index or 0
        bits.append(f"gpu={torch.cuda.get_device_name(idx)}")
        bits.append(f"allocated_gb={torch.cuda.memory_allocated(idx) / 2**30:.2f}")
    return " ".join(bits)
