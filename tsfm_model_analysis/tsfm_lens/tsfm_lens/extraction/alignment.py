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
    # Impulse size relative to the base signal's own amplitude, not an
    # absolute constant: verified live against chronos-t5-small that a large
    # absolute impulse (previously a hardcoded 8.0, ~8x the base's unit
    # amplitude) rescales Chronos's context-adaptive quantization tokenizer's
    # global bin edges, shifting hundreds of unrelated tokens across the
    # whole sequence and burying the local diagonal-dominance signal in noise
    # unrelated to alignment (measured min diagonal-hit fraction 0.06 at
    # amp=8.0 vs. a perfect 1.00 at amp<=0.3x base amplitude, for every
    # layer). TimesFM's patch embeddings are insensitive to this (near-1.0
    # across the same sweep) since they don't do sequence-global rescaling,
    # so this smaller, relative impulse is safe for both tokenization styles.
    impulse = 0.25 * float(np.max(np.abs(base)))
    batch = np.tile(base, (n_windows + 1, 1))
    for w in range(n_windows):
        batch[w + 1, w * window + window // 2] += impulse

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


def run_alignment_gate(adapter: ModelAdapter, window: int, layers: list,
                       min_diagonal_frac: float = 0.5, on_failure: str = "fail") -> dict:
    """Run the impulse alignment check, persist a full record, and enforce a gate.

    `CLAUDE.md` sec 6.3/sec 7 invariant 7 say a near-zero diagonal-hit fraction
    means the declared `token_time_spans` are wrong for the installed library
    version and no cross-model number should be trusted until fixed -- but
    until this function existed (ROADMAP.md sec 15 A2), the check ran, printed
    one INFO line, and its result was discarded: nothing enforced the "before
    trusting any cross-model number" half of that sentence. This probes every
    captured layer (not a strided subset -- the check is one extra forward
    pass, effectively free) and fails loudly on a broken mapping by default,
    exactly like any other broken assumption in this codebase (`CLAUDE.md`
    sec 2.5) rather than degrading to a log line.
    """
    hits = impulse_alignment_check(adapter, window, layers)
    names = list(hits.keys())
    values = [hits[n] for n in names]
    shallowest = names[0]
    shallowest_frac = hits[shallowest]
    passed = shallowest_frac >= min_diagonal_frac
    record = {
        "per_layer": hits,
        "layers_probed": names,
        "n_layers_probed": len(names),
        "min": float(min(values)),
        "mean": float(sum(values) / len(values)),
        "window": window,
        "shallowest_layer": shallowest,
        "shallowest_frac": float(shallowest_frac),
        "min_diagonal_frac_threshold": min_diagonal_frac,
        "on_failure": on_failure,
        "passed": passed,
    }
    if not passed:
        msg = (f"alignment check FAILED for model '{adapter.name}': shallowest probed "
              f"layer '{shallowest}' has diagonal-hit fraction {shallowest_frac:.2f}, "
              f"below alignment.min_diagonal_frac={min_diagonal_frac}. This means "
              f"token_time_spans is likely wrong for this checkpoint/library version -- "
              f"CLAUDE.md sec 6.3, sec 7 invariant 7. Inspect every layer with "
              f"`python run.py --config <this config> --check-alignment {adapter.name}`, "
              f"fix the adapter's declared spans, or set alignment.on_failure: warn to "
              f"proceed anyway at your own risk (not recommended -- see ROADMAP.md sec 15 A2).")
        if on_failure == "fail":
            raise RuntimeError(msg)
        log.warning(msg)
    return record
