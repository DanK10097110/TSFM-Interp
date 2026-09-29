"""Quantization-churn probe: is a noise corruption's effect on a token-based
model driven by discrete re-tokenization (a quantization instability), not
just "more input perturbation"?

`run_noise_snr_sweep.py` (ROADMAP.md §7) found the TimesFM/Chronos noise-SNR
depth-profile rank agreement flips sign between mild and heavy noise, then
plateaus at strong anti-correlation, but left *why* open, naming a direct
tokenizer inspection as the concrete next step -- the same diagnostic method
`CLAUDE.md` §11.16 used to find that `impulse_alignment_check`'s hardcoded
impulse amplitude was silently miscalibrated for Chronos's context-adaptive
quantization tokenizer (a large perturbation shifts the *whole sequence's*
quantization bin edges, not just the tokens near the perturbation).

This module holds the pure, testable statistic -- given clean vs. corrupted
token-ID arrays for the same series, how much of the tokenization actually
changed, and how large the jumps are when it does. The I/O (tokenizing real
contexts through a live model's tokenizer) lives in the CLI script
(`run_quantization_churn_sweep.py`), mirroring the existing split between
`sae/ground_truth.py`'s pure matching function and its I/O wrapper.
"""

from __future__ import annotations

import numpy as np

from .stats import mean_ci


def token_churn_stats(ids_clean: np.ndarray, ids_corrupt: np.ndarray, mask: np.ndarray,
                      n_boot: int = 500, seed: int = 0, ci: float = 0.95) -> dict:
    """Per-series fraction of tokens whose quantization bin changed, plus jump size.

    `ids_clean`/`ids_corrupt` are `[n_series, n_tokens]` integer token-ID
    arrays for the same series under two different inputs (e.g. clean vs.
    noise-corrupted); `mask` is a same-shaped boolean/int array, 1 where a
    position is a real context token (not padding/EOS bookkeeping the caller
    already resolved). The resampling unit is the series (`CLAUDE.md` §6.6),
    so the churn-fraction point estimate carries a series-level bootstrap CI.

    A high churn fraction with large jumps is the quantization-instability
    signature this probe exists to detect; a near-zero, slowly-growing churn
    fraction (or one that ramps smoothly with no relation to a downstream
    effect's onset) argues against quantization being the driver of that
    effect.
    """
    ids_clean = np.asarray(ids_clean)
    ids_corrupt = np.asarray(ids_corrupt)
    mask = np.asarray(mask).astype(bool)
    if ids_clean.shape != ids_corrupt.shape or ids_clean.shape != mask.shape:
        raise ValueError(f"shape mismatch: clean={ids_clean.shape} corrupt={ids_corrupt.shape} "
                         f"mask={mask.shape}")
    valid_per_series = mask.sum(axis=1)
    if np.any(valid_per_series == 0):
        raise ValueError("at least one series has zero valid (unmasked) token positions")

    changed = (ids_clean != ids_corrupt) & mask
    churn_frac = changed.sum(axis=1) / valid_per_series
    delta = np.abs(ids_clean.astype(np.int64) - ids_corrupt.astype(np.int64))
    changed_deltas = delta[changed]

    return {
        "churn_frac": mean_ci(churn_frac, n_boot=n_boot, seed=seed, ci=ci),
        "per_series_churn_frac": churn_frac.tolist(),
        "frac_series_untouched": float((churn_frac == 0).mean()),
        "mean_abs_id_jump_when_changed": float(changed_deltas.mean()) if changed_deltas.size else 0.0,
        "median_abs_id_jump_when_changed": float(np.median(changed_deltas)) if changed_deltas.size else 0.0,
        "n_tokens_valid": int(valid_per_series.sum()),
        "n_tokens_changed": int(changed.sum()),
    }
