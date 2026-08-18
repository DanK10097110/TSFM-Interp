"""Cross-architecture time alignment.

Both models' token representations are pooled onto a shared grid of
fixed-width time windows using an overlap-weighted pooling matrix built from
each adapter's declared `token_time_spans`. Because declared spans can drift
from library internals across versions, `impulse_alignment_check` verifies
the mapping empirically: an input impulse in window w must perturb aligned
window w more than any other.

`calibrate_impulse_amplitude` (ROADMAP.md sec 15 A20) picks that probe's
impulse size per run instead of trusting one fixed constant: sec 11.16 found
the historical 0.25x-base-amplitude default confounds Chronos-T5's
context-adaptive tokenizer at some (checkpoint, context_len) combinations it
was never measured against, silently burying the alignment signal in
tokenizer-rescaling noise rather than a broken span. `run_alignment_gate`
calls it by default; `impulse_alignment_check` itself keeps a plain
`amplitude` parameter (default 0.25, unchanged) so every other caller's
behavior is untouched unless it opts in.
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
                            layers: list | None = None,
                            amplitude: float = 0.25) -> dict:
    """Empirically verify token-to-time alignment with an impulse-response probe.

    Returns per-layer diagonal-dominance fractions; values near 1.0 mean the
    declared spans match the model's actual receptive structure. Attention
    mixes information across positions, so mid-depth values below 1.0 are
    expected; near-zero everywhere signals a broken mapping.

    `amplitude` (default 0.25, unchanged from sec 11.16's original fix) sizes
    the impulse relative to the probe signal's own amplitude, not as an
    absolute constant -- see the comment at its use below. This default is
    calibrated for chronos-t5-small at context_len 512 specifically and is
    **not** safe at every (checkpoint, context_len) pair (sec 15 A20) --
    callers that need a value calibrated for the adapter actually in hand
    should get one from `calibrate_impulse_amplitude` instead of relying on
    this default. `run_alignment_gate` does this automatically.
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
    # so this smaller, relative impulse is safe for both tokenization styles
    # -- AT THIS ONE (checkpoint, context_len) PAIR. sec 15 A20 found the same
    # confound recurs at other context lengths of the same checkpoint family
    # (context_len 448 needs amplitude <=0.05, not 0.25); `amplitude` is now a
    # parameter specifically so a caller can correct for that instead of the
    # value silently drifting out of calibration.
    impulse = amplitude * float(np.max(np.abs(base)))
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


def calibrate_impulse_amplitude(adapter: ModelAdapter, window: int,
                                candidates: tuple = (0.25, 0.15, 0.10, 0.05, 0.02),
                                max_unrelated_frac: float = 0.02) -> dict:
    """Pick the largest impulse amplitude that does not confound tokenization.

    ROADMAP.md sec 15 A20: `impulse_alignment_check`'s fixed 0.25 default was
    calibrated once, against chronos-t5-small at context_len 512, by directly
    diffing tokenized ids before/after the perturbation (`CLAUDE.md` sec
    11.16). That diagnosis generalizes into an automatic sweep instead of a
    one-off measurement: for adapters with a re-quantizing tokenizer
    (`token_ids()` returns non-None -- currently Chronos-T5 only, via its
    context-adaptive `MeanScaleUniformBins` bin edges), this perturbs one
    probe window at each candidate amplitude (largest first) and counts how
    many tokens *outside* that window's own span change id -- the direct
    measurement of the global-rescaling confound sec 11.16 diagnosed, not a
    proxy for it. The first amplitude whose unrelated-token-change fraction
    stays at or below `max_unrelated_frac` is chosen; the sweep always tries
    0.25 first, so any (checkpoint, context_len) pair the historical constant
    was already safe for -- e.g. chronos-t5-small at 512 -- reproduces 0.25
    exactly and every number recorded at that setting is unchanged.

    Adapters with no re-quantizing tokenizer (`token_ids()` returns None --
    TimesFM/Sundial/Chronos-Bolt/Chronos-2's continuous patch-MLP embeddings)
    have nothing for an impulse to confound this way: calibration is a no-op
    that returns `candidates[0]` (0.25) unchanged, matching sec 11.16's
    original finding that TimesFM stayed at a perfect 1.00 diagonal-hit
    fraction across the entire amplitude sweep tested.
    """
    adapter.ensure_loaded()
    context_len = adapter.data_cfg.context_len
    n_windows = context_len // window
    spans = adapter.token_time_spans()

    t = np.arange(context_len, dtype=np.float32)
    base = np.sin(2 * np.pi * t / (context_len / 4)).astype(np.float32)
    probe_window = n_windows // 2
    start, end = probe_window * window, probe_window * window + window

    base_ids = adapter.token_ids(adapter.prepare(base[None, :]))
    if base_ids is None:
        return {"amplitude": candidates[0], "calibrated": False, "sweep": [],
                "reason": "adapter.token_ids() is None -- no re-quantizing tokenizer "
                          "for an impulse to confound; calibration is a no-op "
                          "(ROADMAP.md sec 15 A20(d))"}
    base_ids = np.asarray(base_ids).reshape(-1)
    n_tok = min(len(base_ids), len(spans))
    related = (spans[:n_tok, 1] > start) & (spans[:n_tok, 0] < end)
    n_unrelated = int((~related).sum()) or 1

    sweep = []
    chosen = candidates[-1]
    for amp in candidates:
        impulse = amp * float(np.max(np.abs(base)))
        probe = base.copy()
        probe[start + window // 2] += impulse
        probe_ids = adapter.token_ids(adapter.prepare(probe[None, :]))
        probe_ids = np.asarray(probe_ids).reshape(-1)
        m = min(n_tok, len(probe_ids))
        changed = base_ids[:m] != probe_ids[:m]
        unrelated_changed = int((changed & ~related[:m]).sum())
        unrelated_frac = unrelated_changed / n_unrelated
        sweep.append({"amplitude": amp, "unrelated_tokens_changed": unrelated_changed,
                      "unrelated_frac": unrelated_frac})
        if unrelated_frac <= max_unrelated_frac:
            chosen = amp
            break
    log.info("alignment calibration '%s': chose amplitude=%.2f (sweep: %s)",
             adapter.name, chosen, sweep)
    return {"amplitude": chosen, "calibrated": True, "sweep": sweep}


def run_alignment_gate(adapter: ModelAdapter, window: int, layers: list,
                       min_diagonal_frac: float = 0.5, on_failure: str = "fail",
                       calibrate: bool = True) -> dict:
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

    `calibrate=True` (default) self-calibrates the probe amplitude via
    `calibrate_impulse_amplitude` before probing (ROADMAP.md sec 15 A20) --
    the chosen amplitude and the sweep that produced it are recorded below so
    a reader can tell a self-calibrated run from a fixed-amplitude one.
    """
    calibration = calibrate_impulse_amplitude(adapter, window) if calibrate else None
    amplitude = calibration["amplitude"] if calibration else 0.25
    hits = impulse_alignment_check(adapter, window, layers, amplitude=amplitude)
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
        "amplitude": amplitude,
        "calibration": calibration,
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
