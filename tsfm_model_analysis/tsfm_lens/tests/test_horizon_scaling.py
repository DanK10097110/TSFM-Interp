"""Horizon scaling sweep tests (ROADMAP.md sec 16 E20a).

Runnable directly (`python tests/test_horizon_scaling.py`) or via pytest.
`horizon_values`/`generate_horizon_sweep_series`/`score_horizon_sweep` are
pure/deterministic (no model, no GPU -- `parametric()` is plain numpy), so
these are exercised for real rather than mocked; the live-model scoring
loop lives in `run_horizon_scaling_sweep.py` itself and is exercised there
against real checkpoints, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.horizon_scaling import (
    generate_horizon_sweep_series,
    horizon_values,
    score_horizon_sweep,
    summarize_horizon_sweep,
)


def test_horizon_values_ordering_and_end_at_max():
    hs = horizon_values(max_horizon=64, min_horizon=8, n_points=5)
    assert hs[-1] == 64
    assert hs == sorted(hs)
    assert len(set(hs)) == len(hs)
    print("horizon_values ordering/end-at-max test passed")


def test_horizon_values_rejects_max_below_min():
    with pytest.raises(ValueError):
        horizon_values(max_horizon=4, min_horizon=8, n_points=3)
    print("max-below-min guard test passed")


def test_generate_horizon_sweep_series_shape_and_determinism():
    a = generate_horizon_sweep_series(context_len=128, max_horizon=32, n_series=5, seed=0)
    b = generate_horizon_sweep_series(context_len=128, max_horizon=32, n_series=5, seed=0)
    assert a.shape == (5, 160)
    assert np.array_equal(a, b)
    print("horizon-sweep series shape/determinism test passed")


def test_generate_horizon_sweep_series_validates_positive_args():
    with pytest.raises(ValueError):
        generate_horizon_sweep_series(context_len=0, max_horizon=32, n_series=5, seed=0)
    with pytest.raises(ValueError):
        generate_horizon_sweep_series(context_len=128, max_horizon=32, n_series=0, seed=0)
    print("positive-args guard test passed")


def test_score_horizon_sweep_shares_the_same_context_across_all_horizons():
    context_len, max_horizon, n = 128, 32, 4
    full = generate_horizon_sweep_series(context_len, max_horizon, n, seed=1)
    horizons = horizon_values(max_horizon, min_horizon=8, n_points=3)

    seen_contexts = []

    def predict_fn(contexts: np.ndarray, h: int) -> np.ndarray:
        # naive last-value forecast; records its own input context to
        # confirm the context slice truly never moves across sweep points.
        seen_contexts.append(contexts.copy())
        return np.repeat(contexts[:, -1:], h, axis=1)

    metrics = score_horizon_sweep(predict_fn, full, horizons, context_len)
    assert set(metrics["horizon"].unique().tolist()) == set(horizons)
    assert len(metrics) == sum(h for h in horizons) * 0 + len(horizons) * n
    for c in seen_contexts[1:]:
        assert np.array_equal(c, seen_contexts[0])
    print("score_horizon_sweep fixed-context invariant test passed")


def test_score_horizon_sweep_recovers_planted_horizon_degradation():
    # a "model" that forecasts perfectly for the first 8 steps and then
    # zeros for anything beyond -- summarize_horizon_sweep must show a
    # clean, unambiguous MASE jump once horizon exceeds that capacity.
    context_len, max_horizon, n = 128, 32, 20
    full = generate_horizon_sweep_series(context_len, max_horizon, n, seed=2)
    horizons = [8, 16, 32]

    def predict_fn(contexts: np.ndarray, h: int) -> np.ndarray:
        target = full[:, context_len:context_len + h]
        if h <= 8:
            return target.copy()
        out = np.zeros_like(target)
        out[:, :8] = target[:, :8]
        return out

    metrics = score_horizon_sweep(predict_fn, full, horizons, context_len)
    summary = {row["horizon"]: row for row in summarize_horizon_sweep(metrics, n_boot=200, seed=0)}
    # h=8 is perfectly forecast (zero error); h=16 and h=32 both include the
    # zeroed tail and must clearly exceed h=8. MASE is scale-normalized per
    # series rather than horizon-length-normalized, so 16 vs 32 need not be
    # strictly ordered themselves -- only "beyond capacity is clearly worse
    # than within capacity" is the actual claim this test makes.
    assert summary[8]["value"] == pytest.approx(0.0, abs=1e-6)
    assert summary[16]["value"] > summary[8]["value"]
    assert summary[32]["value"] > summary[8]["value"]
    print("planted horizon-degradation recovery test passed")


if __name__ == "__main__":
    test_horizon_values_ordering_and_end_at_max()
    test_horizon_values_rejects_max_below_min()
    test_generate_horizon_sweep_series_shape_and_determinism()
    test_generate_horizon_sweep_series_validates_positive_args()
    test_score_horizon_sweep_shares_the_same_context_across_all_horizons()
    test_score_horizon_sweep_recovers_planted_horizon_degradation()
    print("All horizon scaling sweep tests passed")
