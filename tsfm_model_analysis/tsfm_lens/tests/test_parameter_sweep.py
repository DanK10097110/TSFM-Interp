"""Controlled parameter-sweep tests (ROADMAP.md §7 Phase 3, bullet 1).

Runnable directly (`python tests/test_parameter_sweep.py`) or via pytest.
`build_recipe`/`generate_sweep_data` are pure/deterministic (no model, no
GPU -- `parametric()` is plain numpy), so these are exercised for real
rather than mocked; the live-model scoring loop lives in
`run_parameter_sweep.py` itself and is exercised there against real
checkpoints, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.parameter_sweep import (
    build_recipe,
    generate_sweep_data,
    score_sweep,
    summarize_sweep,
)


def test_build_recipe_seasonal_period_varies_only_period():
    r8 = build_recipe("seasonal_period", 8.0)
    r32 = build_recipe("seasonal_period", 32.0)
    assert r8["seasonalities"][0]["period"] == 8.0
    assert r32["seasonalities"][0]["period"] == 32.0
    # everything else held fixed
    assert r8["seasonalities"][0]["amplitude"] == r32["seasonalities"][0]["amplitude"]
    assert r8["noise_scale"] == r32["noise_scale"]
    print("seasonal_period recipe test passed")


def test_build_recipe_noise_scale_varies_only_noise():
    lo = build_recipe("noise_scale", 0.1)
    hi = build_recipe("noise_scale", 1.0)
    assert lo["noise_scale"] == 0.1
    assert hi["noise_scale"] == 1.0
    assert lo["seasonalities"] == hi["seasonalities"]
    print("noise_scale recipe test passed")


def test_build_recipe_intermittency_varies_only_rate():
    lo = build_recipe("intermittency_rate", 0.0)
    hi = build_recipe("intermittency_rate", 0.8)
    assert lo["intermittency"]["rate"] == 0.0
    assert hi["intermittency"]["rate"] == 0.8
    assert lo["seasonalities"] == hi["seasonalities"]
    assert lo["noise_scale"] == hi["noise_scale"]
    print("intermittency_rate recipe test passed")


def test_build_recipe_unknown_param_raises():
    with pytest.raises(ValueError):
        build_recipe("not_a_real_param", 1.0)
    print("unknown-param guard test passed")


def test_generate_sweep_data_shapes_and_family_labels():
    data = generate_sweep_data("seasonal_period", [8.0, 32.0], n_per_point=3,
                               context_len=64, horizon=16, seed=0)
    assert data.values.shape == (6, 80)
    assert data.context_len == 64 and data.horizon == 16
    assert sorted(data.meta["family"].unique().tolist()) == ["seasonal_period=32", "seasonal_period=8"]
    assert (data.meta["family"] == "seasonal_period=8").sum() == 3
    assert (data.meta["family"] == "seasonal_period=32").sum() == 3
    assert data.meta["series_id"].nunique() == 6  # no accidental seed collisions
    print("sweep-data shape/label test passed")


def test_generate_sweep_data_is_deterministic():
    a = generate_sweep_data("noise_scale", [0.2, 0.6], n_per_point=2,
                            context_len=48, horizon=8, seed=5)
    b = generate_sweep_data("noise_scale", [0.2, 0.6], n_per_point=2,
                            context_len=48, horizon=8, seed=5)
    assert np.array_equal(a.values, b.values)
    print("sweep-data determinism test passed")


def test_generate_sweep_data_validates_positive_args():
    with pytest.raises(ValueError):
        generate_sweep_data("seasonal_period", [8.0], n_per_point=1,
                            context_len=0, horizon=8, seed=0)
    with pytest.raises(ValueError):
        generate_sweep_data("seasonal_period", [8.0], n_per_point=0,
                            context_len=32, horizon=8, seed=0)
    print("positive-args guard test passed")


def test_score_and_summarize_sweep_recovers_planted_difference():
    data = generate_sweep_data("seasonal_period", [8.0, 64.0], n_per_point=20,
                               context_len=64, horizon=16, seed=1)
    contexts, targets = data.contexts(), data.targets()
    # a planted "model" that forecasts perfectly for the short-period family
    # and badly (zeros) for the long-period one -- summarize_sweep must show
    # a large, unambiguous MASE gap in the expected direction.
    point = np.zeros_like(targets)
    short_mask = (data.meta["family"] == "seasonal_period=8").to_numpy()
    point[short_mask] = targets[short_mask]

    metrics = score_sweep(point, targets, contexts, data.meta)
    summary = {row["family"]: row for row in summarize_sweep(metrics, n_boot=200, seed=0)}
    assert summary["seasonal_period=8"]["value"] == pytest.approx(0.0, abs=1e-6)
    assert summary["seasonal_period=64"]["value"] > summary["seasonal_period=8"]["value"]
    print("score/summarize planted-difference test passed")


if __name__ == "__main__":
    test_build_recipe_seasonal_period_varies_only_period()
    test_build_recipe_noise_scale_varies_only_noise()
    test_build_recipe_intermittency_varies_only_rate()
    test_build_recipe_unknown_param_raises()
    test_generate_sweep_data_shapes_and_family_labels()
    test_generate_sweep_data_is_deterministic()
    test_generate_sweep_data_validates_positive_args()
    test_score_and_summarize_sweep_recovers_planted_difference()
