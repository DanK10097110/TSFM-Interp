"""Tests for the real-vs-untrained-null bootstrap significance test
(ROADMAP.md sec 16 E9 / sec 13's open question), `analysis/null_baseline.py`.

Builds two small mock-pipeline run directories on CPU (no GPU, no network):
a "real-like" cross-architecture pair (mock_patch vs mock_wave) and a
"null-like" pair (mock_patch vs its own random_init twin), both over the
identical smoke corpus config so their per-series metadata lines up
row-for-row -- exactly the condition `compare_l1_peak_cka`/
`compare_l2_best_gain` need to use the (tighter) paired bootstrap.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.null_baseline import (
    compare_l1_depth_curve,
    compare_l1_peak_cka,
    compare_l2_best_gain,
    compare_l2_depth_curve,
    run_null_baseline_comparison,
)
from tsfm_lens.config import config_from_dict
from tsfm_lens.pipeline import run_pipeline
from tsfm_lens.utils import load_json


def _base_data_cfg(n_per_family: int = 20) -> dict:
    return {"source": "smoke", "context_len": 128, "horizon": 32,
            "smoke_series_per_family": n_per_family}


def _build_run(tmp_path: Path, name: str, models: list, n_per_family: int = 20, seed: int = 0) -> Path:
    cfg = config_from_dict({
        "run": {"name": name, "out_dir": str(tmp_path), "device": "cpu",
                "dtype": "float32", "seed": seed},
        "data": _base_data_cfg(n_per_family),
        "alignment": {"window": 32, "sanity_check": False},
        "models": models,
        "l1": {"layer_stride": 1, "max_rows": 20000, "min_family_series": 10, "rsa": False},
        "l2": {"layer_stride": 1, "max_rows": 2000, "val_frac": 0.3, "input_baseline": True},
        "l3": {"enabled": False}, "lens": {"enabled": False}, "attention": {"enabled": False},
        "exemplars": {"enabled": False}, "clustering": {"enabled": False},
        "internals": {"enabled": False}, "confirm": {"enabled": False}, "report": {"enabled": False},
        "stats": {"enabled": True, "n_boot": 20, "n_boot_heavy": 20, "min_series": 8},
    })
    run_pipeline(cfg, stages=["extract", "l1", "l2"])
    return cfg.run_dir()


@pytest.fixture(scope="module")
def runs():
    tmp = Path(tempfile.mkdtemp())
    real = _build_run(tmp, "real_pair", [
        {"name": "A", "adapter": "mock_patch", "batch_size": 64},
        {"name": "B", "adapter": "mock_wave", "batch_size": 64},
    ])
    null = _build_run(tmp, "null_pair", [
        {"name": "A", "adapter": "mock_patch", "batch_size": 64},
        {"name": "A-random", "adapter": "mock_patch", "batch_size": 64, "random_init": True},
    ])
    mismatched = _build_run(tmp, "mismatched_pair", [
        {"name": "A", "adapter": "mock_patch", "batch_size": 64},
        {"name": "A-random", "adapter": "mock_patch", "batch_size": 64, "random_init": True},
    ], n_per_family=35)  # different corpus size -> deliberately misaligned
    null_b = _build_run(tmp, "null_pair_b", [
        {"name": "B", "adapter": "mock_wave", "batch_size": 64},
        {"name": "B-random", "adapter": "mock_wave", "batch_size": 64, "random_init": True},
    ])
    return {"real": real, "null": null, "null_b": null_b, "mismatched": mismatched}


def test_l1_peak_cka_paired_comparison_runs_and_is_finite(runs):
    result = compare_l1_peak_cka(runs["real"], runs["null"], n_boot=30, seed=0)
    assert result["paired"] is True
    for key in ("a", "b", "diff", "diff_lo", "diff_hi", "p"):
        assert np.isfinite(result[key])
    assert -1.0 - 1e-6 <= result["a"] <= 1.0 + 1e-6
    assert -1.0 - 1e-6 <= result["b"] <= 1.0 + 1e-6
    assert result["diff_lo"] <= result["diff"] <= result["diff_hi"]
    assert 0.0 <= result["p"] <= 1.0
    assert result["real_pair"]["model_a"] == "A"
    assert result["null_pair"]["model_a"] == "A"


def test_l2_best_gain_paired_comparison_runs_and_is_finite(runs):
    result = compare_l2_best_gain(runs["real"], runs["null"], n_boot=30, seed=0)
    assert result["paired"] is True
    for key in ("a", "b", "diff", "diff_lo", "diff_hi", "p"):
        assert np.isfinite(result[key])
    assert result["diff_lo"] <= result["diff"] <= result["diff_hi"]
    assert "->" in result["real_direction"]
    assert "->" in result["null_direction"]


def test_l1_depth_curve_returns_one_row_per_model_a_layer_with_both_null_sides(runs):
    rows = compare_l1_depth_curve(runs["real"], runs["null"], runs["null_b"], n_boot=15, seed=0)
    from tsfm_lens.extraction.store import ActivationStore
    store = ActivationStore(runs["real"] / "activations.zarr", mode="r")
    assert len(rows) == len(store.layers("A"))
    for row in rows:
        assert row["layer_a"] in store.layers("A")
        assert row["layer_b"] in store.layers("B")
        assert -1.0 - 1e-6 <= row["real_cka"] <= 1.0 + 1e-6
        for side in ("vs_null_a", "vs_null_b"):
            assert np.isfinite(row[side]["diff"])
            assert row[side]["diff_lo"] <= row[side]["diff"] <= row[side]["diff_hi"]
            assert row[side]["paired"] is True


def test_l1_depth_curve_raises_on_mismatched_null_run_order(runs):
    with pytest.raises(ValueError, match="comparison_pair"):
        compare_l1_depth_curve(runs["real"], runs["null_b"], runs["null"], n_boot=10, seed=0)


def _real_best_direction(runs) -> tuple:
    stitch = load_json(runs["real"] / "l2" / "stitching.json")
    best_dir = max(stitch["directions"], key=lambda d: stitch["directions"][d]["best_gain"])
    src, dst = best_dir.split("->")
    return stitch, best_dir, src, dst


def test_l2_depth_curve_returns_one_row_per_src_layer_with_both_null_sides(runs):
    stitch, best_dir, src, dst = _real_best_direction(runs)
    null_by_model = {"A": runs["null"], "B": runs["null_b"]}
    rows = compare_l2_depth_curve(runs["real"], null_by_model[src], null_by_model[dst], n_boot=15, seed=0)
    assert len(rows) == len(stitch["layers"][src])
    for row in rows:
        assert row["direction"] == best_dir
        assert np.isfinite(row["real_gain"])
        for side in ("vs_null_a", "vs_null_b"):
            assert np.isfinite(row[side]["diff"])
            assert row[side]["diff_lo"] <= row[side]["diff"] <= row[side]["diff_hi"]
            assert row[side]["paired"] is True


def test_l2_depth_curve_raises_on_mismatched_null_run_order(runs):
    _, _, src, dst = _real_best_direction(runs)
    null_by_model = {"A": runs["null"], "B": runs["null_b"]}
    with pytest.raises(ValueError, match="best direction"):
        compare_l2_depth_curve(runs["real"], null_by_model[dst], null_by_model[src], n_boot=10, seed=0)


def test_l1_falls_back_to_unpaired_on_series_mismatch_without_crashing(runs):
    result = compare_l1_peak_cka(runs["real"], runs["mismatched"], n_boot=20, seed=0)
    assert result["paired"] is False
    assert np.isfinite(result["diff"])


def test_l2_falls_back_to_unpaired_on_series_mismatch_without_crashing(runs):
    result = compare_l2_best_gain(runs["real"], runs["mismatched"], n_boot=20, seed=0)
    assert result["paired"] is False
    assert np.isfinite(result["diff"])


def test_top_level_comparison_wraps_both_sections(runs):
    result = run_null_baseline_comparison(runs["real"], runs["null"], n_boot=20, seed=0)
    assert "l1_peak_cka" in result and "l2_best_gain" in result
    assert "error" not in result["l1_peak_cka"]
    assert "error" not in result["l2_best_gain"]


def test_top_level_comparison_degrades_per_section_on_missing_artifact(tmp_path, runs):
    """A run directory with no l2/ artifact must not take down the L1 section too."""
    import shutil
    broken = tmp_path / "broken_null"
    shutil.copytree(runs["null"], broken)
    shutil.rmtree(broken / "l2")
    result = run_null_baseline_comparison(runs["real"], broken, n_boot=10, seed=0)
    assert "error" not in result["l1_peak_cka"]
    assert "error" in result["l2_best_gain"]
