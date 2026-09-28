"""Context-truncation-from-the-back probe (`ROADMAP.md` sec 16 E17).

Distinct from two already-built, superficially similar diagnostics:

- `analysis/phase_sensitivity.py` trims 0..patch_width-1 points off the
  context's FRONT (the oldest end) to test patch-boundary aliasing -- a
  small, sub-patch-width shift, holding the true context endpoint fixed.
- `analysis/context_scaling.py` (ROADMAP.md sec 16 E20) also trims off the
  FRONT: as context length shrinks, it keeps the trailing (most recent)
  history intact right up to the fixed target window, and drops OLDER
  history. That answers "does more history help."

This module asks the opposite-direction question: what happens when the
MOST RECENT history -- the data closest to the forecast origin -- is what's
missing (a reporting-lag / data-staleness scenario), while older history
further back is still available? The available context is the FRONT
(oldest) `avail_len` points of an otherwise-fixed context window; the
`context_len - avail_len` most recent points are treated as unavailable.
To still evaluate against the SAME fixed target window every truncation
level shares (the same design principle `context_scaling.py` uses, for the
same reason: a dose-response curve needs one target, not a different series
per point), the model is asked to forecast `horizon + lag` steps starting
right after the available context, and only the LAST `horizon` of that
forecast -- the part that actually reaches the real target -- is scored.
This is the direct behavioral consequence of losing recent context: the
model must forecast further into the future to reach the same calendar
target.

Pure numpy reduction over an already-computed `{avail_len: per-series MASE
array}` mapping, mirroring `phase_sensitivity.py`'s own split between pure
stats (here) and the I/O that calls `adapter.predict()`
(`analysis/frontend.py`).
"""

from __future__ import annotations

import numpy as np

from .stats import mean_ci


def context_truncation_stats(mase_by_avail_len: dict, n_boot: int = 500, seed: int = 0,
                             ci: float = 0.95, cliff_frac_threshold: float = 0.6) -> dict:
    """Characterize forecast degradation as recent context is lost.

    `mase_by_avail_len` maps the AVAILABLE (non-stale) context length -> a
    per-series MASE array against the same fixed target, all series aligned
    in the same order across lengths (the caller's responsibility, same
    contract as `phase_sensitivity_stats`'s `mase_by_shift`). Needs at least
    3 lengths to characterize a *shape* (2 points can only be "up" or
    "down," never distinguish a cliff from a ramp).

    Reports the sequence ordered from the fullest available context (the
    baseline, no missing recent data) to the most truncated, and classifies
    the degradation shape by how much of the total baseline-to-most-
    truncated increase in MASE is concentrated in a single step: `"cliff"`
    when one step accounts for at least `cliff_frac_threshold` of the total
    (a hard floor below which the model breaks), `"graceful"` when
    degradation is spread across steps, and `"no_degradation"` when the
    most-truncated point is not worse than the baseline at all.

    Shape classification and `baseline_mase`/`most_truncated_mase`/
    `total_degradation`/`worst_step_*` all use the per-length MEDIAN, not the
    mean, of the per-series MASE array -- found necessary, not stylistic,
    against a real checkpoint: MASE's denominator is each series' own naive
    (mean-abs-diff) scale computed over the *available* (truncated) context,
    and at the shortest truncation levels a handful of series are locally
    near-flat over that short a window, sending the scale toward `mase()`'s
    own floor and the per-series MASE toward astronomical values regardless
    of how good or bad the actual forecast is. A handful of such series
    (measured: ~6-8% of a 64-series real sample at the two shortest lengths)
    dominates a raw mean by many orders of magnitude and manufactures a
    "cliff" that is really a metric artifact, not a finding about the model
    -- the same "isolated extreme values swamp an aggregate" failure shape
    already diagnosed for `CO_trev_1_num` (`CLAUDE.md` sec 11.3) and fixed
    by winsorization elsewhere in this repo (sec 5). The raw mean is still
    reported (`mean_mase_by_length_desc`) precisely so the gap between it
    and the median is itself visible -- a large gap is the diagnostic's own
    tell that a length is running into this floor, not a fact hidden from
    the reader.
    """
    lens_ = sorted(mase_by_avail_len.keys())
    if len(lens_) < 3:
        raise ValueError("need at least 3 available-context lengths to characterize a degradation shape")
    mat = np.stack([np.asarray(mase_by_avail_len[l], dtype=np.float64) for l in lens_], axis=1)
    mean_by_len = mat.mean(axis=0)
    median_by_len = np.median(mat, axis=0)

    # `lens_` is ascending (least truncated last); read the sequence from
    # fullest context (baseline) down to most truncated.
    order = list(range(len(lens_)))[::-1]
    seq = median_by_len[order]
    mean_seq = mean_by_len[order]
    seq_lens = [lens_[i] for i in order]
    baseline, most_truncated = float(seq[0]), float(seq[-1])
    total_degradation = most_truncated - baseline

    diffs = np.diff(seq)
    if total_degradation > 1e-8:
        worst_step = int(np.argmax(diffs))
        worst_step_frac = (float(diffs[worst_step]) / total_degradation
                           if diffs[worst_step] > 0 else 0.0)
        shape = "cliff" if worst_step_frac >= cliff_frac_threshold else "graceful"
    else:
        worst_step, worst_step_frac, shape = None, None, "no_degradation"

    return {
        "available_context_lengths_desc": [int(l) for l in seq_lens],
        "n_series": int(mat.shape[0]),
        "median_mase_by_length_desc": [float(v) for v in seq],
        "mean_mase_by_length_desc": [float(v) for v in mean_seq],
        "mase_ci_by_length_desc": [mean_ci(mat[:, order[i]], n_boot=n_boot, seed=seed + i, ci=ci)
                                   for i in range(len(order))],
        "baseline_mase": baseline,
        "most_truncated_mase": most_truncated,
        "total_degradation": total_degradation,
        "worst_step_index": worst_step,
        "worst_step_frac_of_total": worst_step_frac,
        "shape": shape,
    }
