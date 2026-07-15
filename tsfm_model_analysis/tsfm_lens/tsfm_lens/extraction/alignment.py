"""Cross-architecture time alignment.

Both models' token representations are pooled onto a shared grid of
fixed-width time windows using an overlap-weighted pooling matrix built from
each adapter's declared `token_time_spans`. Because declared spans can drift
from library internals across versions, `impulse_alignment_check` verifies
the mapping empirically: an input impulse in window w must perturb aligned
window w more than any other.
"""

from __future__ import annotations

import numpy as np
import torch

from ..models.base import ModelAdapter
from ..utils import log
from .hooks import ActivationCatcher


def pooling_matrix(spans: np.ndarray, context_len: int, window: int) -> torch.Tensor:
    """Overlap-weighted, row-normalized pooling matrix of shape [n_windows, n_tokens]."""
    n_windows = context_len // window
    starts = np.arange(n_windows) * window
    ends = starts + window
    overlap = np.maximum(
        0.0,
        np.minimum(ends[:, None], spans[None, :, 1]) - np.maximum(starts[:, None], spans[None, :, 0]),
    )
    row_sum = overlap.sum(axis=1, keepdims=True)
    if np.any(row_sum == 0):
        raise ValueError("some windows receive no tokens; check token_time_spans")
    return torch.from_numpy((overlap / row_sum).astype(np.float32))


def align(hidden: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    """Pool token states [B, T, D] into window states [B, W, D] on the tensor's device."""
    return torch.einsum("wt,btd->bwd", pool.to(hidden.device, hidden.dtype), hidden)


def impulse_alignment_check(adapter: ModelAdapter, window: int,
                            layers: list | None = None) -> dict:
    """Empirically verify token-to-time alignment with an impulse-response probe.

    Returns per-layer diagonal-dominance fractions; values near 1.0 mean the
    declared spans match the model's actual receptive structure. Attention
    mixes information across positions, so mid-depth values below 1.0 are
    expected; near-zero everywhere signals a broken mapping.
    """
    adapter.ensure_loaded()
    context_len = adapter.data_cfg.context_len
    n_windows = context_len // window
    layer_names = layers or adapter.layer_names()
    pool = pooling_matrix(adapter.token_time_spans(), context_len, window)

    t = np.arange(context_len, dtype=np.float32)
    base = np.sin(2 * np.pi * t / (context_len / 4)).astype(np.float32)
    batch = np.tile(base, (n_windows + 1, 1))
    for w in range(n_windows):
        batch[w + 1, w * window + window // 2] += 8.0

    with torch.no_grad(), ActivationCatcher(adapter.module, layer_names) as catcher:
        adapter.forward(adapter.prepare(batch))
        acts = catcher.collect()

    results = {}
    for name, hidden in acts.items():
        aligned = align(adapter.postprocess_tokens(name, hidden).float(), pool)
        delta = (aligned[1:] - aligned[0:1]).norm(dim=-1).cpu().numpy()
        hits = (delta.argmax(axis=1) == np.arange(n_windows)).mean()
        results[name] = float(hits)
    worst = min(results.values())
    log.info("alignment check '%s': diagonal-hit fraction min=%.2f mean=%.2f",
             adapter.name, worst, float(np.mean(list(results.values()))))
    if worst == 0.0:
        log.warning("alignment check '%s': a layer shows zero diagonal hits; "
                    "token_time_spans is likely wrong for this checkpoint", adapter.name)
    return results
