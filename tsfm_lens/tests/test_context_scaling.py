"""Context-length scaling sweep tests (ROADMAP.md sec 16 E20).

Runnable directly (`python tests/test_context_scaling.py`) or via pytest.
`context_length_values`/`generate_context_sweep_series`/
`score_context_length_sweep` are pure/deterministic (no model, no GPU --
`parametric()` is plain numpy), so these are exercised for real rather than
mocked; the live-model scoring loop lives in `run_context_scaling_sweep.py`
(study driver, dev branch) itself and is exercised there against real
checkpoints, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.context_scaling import (
    context_length_values,
    generate_context_sweep_series,
    score_context_length_sweep,
    summarize_context_sweep,
)


def test_context_length_values_are_multiples_of_window_and_end_at_max():
    lens = context_length_values(max_context_len=512, min_context_len=128, window=32, n_points=5)
    assert lens[-1] == 512
    assert all(v % 32 == 0 for v in lens)
    assert lens == sorted(lens)
    assert len(set(lens)) == len(lens)
    print("context_length_values multiple/ordering/end-at-max test passed")


def test_context_length_values_rejects_non_multiple_max_context():
    with pytest.raises(ValueError):
        context_length_values(max_context_len=500, min_context_len=128, window=32, n_points=3)
    print("non-multiple max_context guard test passed")


def test_context_length_values_rejects_max_below_min():
    with pytest.raises(ValueError):
        context_length_values(max_context_len=64, min_context_len=128, window=32, n_points=3)
    print("max-below-min guard test passed")


def test_generate_context_sweep_series_shape_and_determinism():
    a = generate_context_sweep_series(max_context_len=256, horizon=16, n_series=5, seed=0)
    b = generate_context_sweep_series(max_context_len=256, horizon=16, n_series=5, seed=0)
    assert a.shape == (5, 272)
    assert np.array_equal(a, b)
    print("context-sweep series shape/determinism test passed")


def test_generate_context_sweep_series_validates_positive_args():
    with pytest.raises(ValueError):
        generate_context_sweep_series(max_context_len=0, horizon=16, n_series=5, seed=0)
    with pytest.raises(ValueError):
        generate_context_sweep_series(max_context_len=256, horizon=16, n_series=0, seed=0)
    print("positive-args guard test passed")


def test_score_context_length_sweep_shares_the_same_target_across_all_lengths():
    max_context, horizon, n = 256, 16, 4
    full = generate_context_sweep_series(max_context, horizon, n, seed=1)
    lens = context_length_values(max_context, min_context_len=64, window=32, n_points=3)

    seen_targets = []

    def predict_fn(contexts: np.ndarray, h: int) -> np.ndarray:
        # planted "model": naive last-value forecast, so MASE is finite and
        # comparable, and its inputs are recorded to confirm the target
        # window truly never moves across sweep points.
        seen_targets.append(full[:, max_context:max_context + h].copy())
        return np.repeat(contexts[:, -1:], h, axis=1)

    metrics = score_context_length_sweep(predict_fn, full, lens, max_context, horizon)
    assert set(metrics["context_len"].unique().tolist()) == set(lens)
    assert len(metrics) == len(lens) * n
    for t in seen_targets[1:]:
        assert np.array_equal(t, seen_targets[0])
    print("score_context_length_sweep fixed-target invariant test passed")


def test_score_context_length_sweep_recovers_planted_context_length_advantage():
    # a "model" that forecasts perfectly once it sees at least 128 timesteps
    # of context (simulating a real crystallization-style capacity floor)
    # and forecasts zeros otherwise -- summarize_context_sweep must show a
    # large, unambiguous MASE gap at exactly that threshold.
    max_context, horizon, n = 256, 8, 20
    full = generate_context_sweep_series(max_context, horizon, n, seed=2)
    lens = [64, 128, 256]

    def predict_fn(contexts: np.ndarray, h: int) -> np.ndarray:
        target = full[:, max_context:max_context + h]
        if contexts.shape[1] >= 128:
            return target.copy()
        return np.zeros_like(target)

    metrics = score_context_length_sweep(predict_fn, full, lens, max_context, horizon)
    summary = {row["context_len"]: row for row in summarize_context_sweep(metrics, n_boot=200, seed=0)}
    assert summary[128]["value"] == pytest.approx(0.0, abs=1e-6)
    assert summary[256]["value"] == pytest.approx(0.0, abs=1e-6)
    assert summary[64]["value"] > summary[128]["value"]
    print("planted context-length-advantage recovery test passed")


if __name__ == "__main__":
    test_context_length_values_are_multiples_of_window_and_end_at_max()
    test_context_length_values_rejects_non_multiple_max_context()
    test_context_length_values_rejects_max_below_min()
    test_generate_context_sweep_series_shape_and_determinism()
    test_generate_context_sweep_series_validates_positive_args()
    test_score_context_length_sweep_shares_the_same_target_across_all_lengths()
    test_score_context_length_sweep_recovers_planted_context_length_advantage()
    print("All context-length scaling sweep tests passed")
