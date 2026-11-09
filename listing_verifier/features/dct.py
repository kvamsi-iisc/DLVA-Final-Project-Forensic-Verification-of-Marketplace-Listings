"""Frequency features: log-magnitude 2D DCT of the grayscale image.

Why a frequency cue at all (AIDE, Yan et al. ICLR 2025): diffusion decoders
leave periodic, high-frequency fingerprints (upsampling grids, smooth noise
statistics) that the semantic CLIP feature barely sees. Why we expect it to
die under JPEG: JPEG quantisation zeroes exactly those high-frequency DCT
coefficients, so the fingerprint is erased along with them. That collapse is
the planned robustness finding, not a bug.

Implemented with plain numpy matrix multiplies (orthonormal DCT-II basis) so
there is no scipy dependency at demo time.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import DCT_MAP_SIZE, DCT_RADIAL_BINS, DCT_SIZE


@lru_cache(maxsize=4)
def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis as an (n, n) matrix: X_dct = D @ x."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    D = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    D[0, :] /= np.sqrt(2.0)
    return D.astype(np.float32)


def log_dct_2d(gray: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    assert gray.ndim == 2 and gray.shape[0] == gray.shape[1]
    D = dct_matrix(gray.shape[0])
    coeff = D @ gray.astype(np.float32) @ D.T
    return np.log(np.abs(coeff) + eps)


@lru_cache(maxsize=4)
def _radial_bins(n: int, nbins: int) -> np.ndarray:
    u, v = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r = np.sqrt(u.astype(np.float32) ** 2 + v.astype(np.float32) ** 2)
    r = r / r.max()
    return np.minimum((r * nbins).astype(np.int32), nbins - 1)


def radial_profile(spec: np.ndarray, nbins: int = DCT_RADIAL_BINS) -> np.ndarray:
    """Azimuthal average of the log-spectrum: 1-D 'how much energy at each
    frequency radius'. DC is at the (0,0) corner for a DCT, so radius is
    measured from the corner."""
    bins = _radial_bins(spec.shape[0], nbins)
    sums = np.bincount(bins.ravel(), weights=spec.ravel(), minlength=nbins)
    counts = np.bincount(bins.ravel(), minlength=nbins)
    return (sums / np.maximum(counts, 1)).astype(np.float32)


def pool_map(spec: np.ndarray, out_size: int = DCT_MAP_SIZE) -> np.ndarray:
    n = spec.shape[0]
    f = n // out_size
    return spec[: f * out_size, : f * out_size].reshape(out_size, f, out_size, f).mean(axis=(1, 3))


def spectrum_features(gray: np.ndarray) -> dict:
    """Return {'dct_map': (64,64) float16, 'dct_radial': (128,) float32}."""
    if gray.shape[0] != DCT_SIZE:
        raise ValueError(f"expected {DCT_SIZE}x{DCT_SIZE} grayscale, got {gray.shape}")
    spec = log_dct_2d(gray)
    return {
        "dct_map": pool_map(spec).astype(np.float16),
        "dct_radial": radial_profile(spec),
    }
