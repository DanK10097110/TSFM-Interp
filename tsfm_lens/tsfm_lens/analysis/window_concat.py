"""Lossless window representation (CONCAT) and its CKA through Gram sums (ROADMAP.md sec 44.4).

Why this module exists. The pipeline's window vector is the overlap-weighted MEAN of a window's
tokens (`extraction/alignment.py::pooling_matrix`). For a model whose tokens are shorter than the
window that average discards within-window order (sec 44.2), while a model with one token per
window keeps it, so window-level CKA is not symmetric across architectures. CONCAT represents a
window by its tokens concatenated in time order: 16-step models give 2 tokens, 1-step models 32,
TimesFM 1, and a token wider than the window is copied exactly as MEAN copies it.

Linear CKA only needs the row Gram matrix, and the Gram of a concatenation is the sum of the
per-position Grams, K = sum_p X_p X_p^T, so the (window x D) concatenation (32 x 768 wide for a
1-step model) is never materialized. Centering K as H K H equals the Gram of the column-centered
concatenation, so `cka_from_grams` is exactly feature-space linear CKA of the concatenations.

It inherits the token-time spans of the adapter and fails loudly (never guesses) when tokens
straddle a window boundary or windows would hold different numbers of tokens.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from ..extraction.alignment import pooling_matrix


def concat_grid(spans: np.ndarray, context_len: int, window: int) -> int:
    """Width g of the positions that make up a window: the token width, or `window` for wider tokens.

    Raises if tokens are not all the same width or straddle a g-step boundary, since CONCAT would
    then give windows different shapes.
    """
    spans = np.asarray(spans, dtype=np.float64)
    width = spans[:, 1] - spans[:, 0]
    wmax = float(width.max())
    if wmax >= window:
        return int(window)
    g = int(round(wmax))
    if abs(wmax - g) > 1e-9 or window % g or context_len % g:
        raise ValueError(f"concat: token width {wmax} does not tile a {window}-step window")
    if np.any(np.abs(width - wmax) > 1e-9) or np.any(
            np.floor(spans[:, 0] / g + 1e-9) != np.floor((spans[:, 1] - 1e-9) / g)):
        raise ValueError("concat: tokens are ragged or straddle a position boundary")
    pooling_matrix(spans, context_len, g)
    return g


def position_features(h: torch.Tensor, spans: np.ndarray, context_len: int, window: int,
                      grid: int | None = None) -> torch.Tensor:
    """Token states [B, T, D] -> per-window position vectors [B, n_windows, P, D], P = window // grid."""
    g = concat_grid(spans, context_len, window) if grid is None else int(grid)
    pool = pooling_matrix(spans, context_len, g).to(h.device, h.dtype)
    p = torch.einsum("wt,btd->bwd", pool, h)
    b, n, d = p.shape
    k = window // g
    return p.reshape(b, n // k, k, d)


def centered_gram(blocks: Iterable[torch.Tensor], device: str | torch.device = "cpu") -> torch.Tensor:
    """H (sum_p X_p X_p^T) H over row-aligned blocks X_p [N, D_p], in float32 on `device`."""
    k = None
    for x in blocks:
        x = x.to(device).float()
        g = x @ x.T
        k = g if k is None else k + g
    k = k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()
    return k


def cka_from_grams(ka: torch.Tensor, kb: torch.Tensor) -> float:
    """Linear CKA of two CENTERED Gram matrices (float64 reduction)."""
    ka, kb = ka.double(), kb.double()
    return float(((ka * kb).sum() / torch.sqrt((ka * ka).sum() * (kb * kb).sum()).clamp_min(1e-300)).item())
