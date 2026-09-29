"""Phase-sensitivity probe unit tests (ROADMAP.md sec 16 E17).

Synthetic, planted-answer only -- no real checkpoint or `adapter.predict()`
I/O (that lives in a live-run check reported separately, per CLAUDE.md
sec 2.4). Builds one `{shift: per-series MASE array}` mapping for a
"phase-sensitive" model (MASE genuinely depends on the shift, on every
series, by construction) and one for a "phase-invariant" model (MASE is
flat across shifts plus only tiny per-series noise), and asserts the
sensitive one's coefficient of variation and worst-minus-best gap are
clearly larger.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.phase_sensitivity import phase_sensitivity_stats


def test_phase_sensitivity_requires_at_least_two_shifts():
    try:
        phase_sensitivity_stats({0: np.array([1.0, 2.0])})
        assert False, "expected ValueError for a single shift"
    except ValueError:
        pass
    print("single-shift rejection test passed")


def test_sensitive_model_shows_higher_cv_and_wider_gap_than_invariant_model():
    rng = np.random.default_rng(0)
    n_series = 30
    n_shifts = 8

    # Phase-sensitive: each series' MASE swings by a real, shift-dependent
    # factor (planted signal) plus small noise -- by construction, the
    # per-series mean-across-shifts is never near zero, so this isn't
    # relying on the near-zero-mean exclusion path at all.
    base = rng.uniform(0.5, 1.5, size=n_series)
    shift_factor = 1.0 + 0.6 * np.sin(np.arange(n_shifts) / n_shifts * 2 * np.pi)
    sensitive = {}
    for s in range(n_shifts):
        noise = rng.normal(0, 0.01, size=n_series)
        sensitive[s] = np.clip(base * shift_factor[s] + noise, 1e-3, None)

    # Phase-invariant: same base MASE level, but shift has no systematic
    # effect -- only small per-series noise varies it across shifts.
    invariant = {}
    for s in range(n_shifts):
        noise = rng.normal(0, 0.01, size=n_series)
        invariant[s] = np.clip(base + noise, 1e-3, None)

    stats_sensitive = phase_sensitivity_stats(sensitive, seed=0)
    stats_invariant = phase_sensitivity_stats(invariant, seed=0)

    assert stats_sensitive["cv"]["value"] > 3 * stats_invariant["cv"]["value"], (
        "the planted phase-sensitive model should show a clearly higher "
        "MASE coefficient of variation than the planted phase-invariant one")
    assert stats_sensitive["worst_minus_best_mase"] > 3 * stats_invariant["worst_minus_best_mase"]
    assert stats_sensitive["n_series_used_for_cv"] == n_series
    assert stats_invariant["n_series_used_for_cv"] == n_series
    print("sensitive-vs-invariant planted-signal test passed")


def test_near_zero_mase_series_excluded_from_cv_but_still_counted():
    # One series is a near-perfect forecast at every shift (mean MASE ~0) --
    # it should be excluded from the CV average (a near-infinite ratio would
    # be meaningless) but still reflected in n_series and mean_mase_by_shift.
    mase_by_shift = {
        0: np.array([1e-9, 1.0, 0.8]),
        1: np.array([1e-9, 1.1, 0.9]),
        2: np.array([1e-9, 0.9, 1.0]),
    }
    stats = phase_sensitivity_stats(mase_by_shift, seed=0)
    assert stats["n_series"] == 3
    assert stats["n_series_used_for_cv"] == 2
    assert stats["per_series_cv"][0] is None
    assert stats["per_series_cv"][1] is not None
    print("near-zero-MASE exclusion test passed")


if __name__ == "__main__":
    test_phase_sensitivity_requires_at_least_two_shifts()
    test_sensitive_model_shows_higher_cv_and_wider_gap_than_invariant_model()
    test_near_zero_mase_series_excluded_from_cv_but_still_counted()
    print("All phase sensitivity tests passed")
