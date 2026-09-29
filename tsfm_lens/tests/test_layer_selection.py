"""Layer-selection correlation study tests (ROADMAP.md Phase 2a, §6.1).

Runnable directly (`python tests/test_layer_selection.py`) or via pytest.
Regression-tests the cluster-bootstrap Spearman primitive against synthetic
data with a known correlation (must find it) and known independence (CI
must straddle zero) before it's trusted on real run artifacts, plus the
record-collection function's graceful degrade when a run is missing stages.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.layer_selection import (
    _entropy,
    cluster_bootstrap_spearman,
    collect_layer_records,
    recommend_layers,
)
from tsfm_lens.utils import save_json


def _synthetic_records(rng, correlated: bool) -> list[dict]:
    records = []
    for run in ["r1", "r2", "r3", "r4", "r5"]:
        for model in ["A", "B"]:
            x = np.sort(rng.normal(size=8)) if correlated else rng.normal(size=8)
            y = (x + rng.normal(scale=0.3, size=8)) if correlated else rng.normal(size=8)
            for xi, yi in zip(x, y):
                records.append({"run": run, "model": model, "x": float(xi), "y": float(yi)})
    return records


def test_cluster_bootstrap_finds_known_correlation_and_rejects_null():
    rng = np.random.default_rng(0)
    correlated = cluster_bootstrap_spearman(_synthetic_records(rng, correlated=True), "x", "y")
    assert correlated is not None and correlated["significant"], correlated
    assert correlated["value"] > 0.7, correlated

    independent = cluster_bootstrap_spearman(_synthetic_records(rng, correlated=False), "x", "y")
    assert independent is not None and not independent["significant"], independent
    print("cluster-bootstrap correlated/null test passed")


def test_cluster_bootstrap_needs_at_least_three_groups():
    records = [{"run": "r1", "model": "A", "x": 1.0, "y": 1.0},
               {"run": "r1", "model": "A", "x": 2.0, "y": 2.0},
               {"run": "r2", "model": "A", "x": 3.0, "y": 3.0}]
    assert cluster_bootstrap_spearman(records, "x", "y") is None
    print("too-few-groups guard test passed")


def test_entropy_bounds():
    assert _entropy(np.array([1.0, 0.0, 0.0])) == 0.0
    assert abs(_entropy(np.array([1.0, 1.0, 1.0, 1.0])) - 1.0) < 1e-9
    print("entropy bounds test passed")


def test_collect_layer_records_degrades_when_stages_missing():
    """A run with only `internals/profile.json` must contribute effective_dim/
    input_cka/probe_decodability records but no tuned_r2_model/l3_* fields."""
    out = Path(tempfile.mkdtemp())
    run_dir = out / "internals_only"
    save_json(run_dir / "internals" / "profile.json", {
        "M": {"layers": ["l0", "l1"], "rel_depth": [0.0, 1.0],
              "effective_dim": [3.0, 5.0], "input_cka": [0.1, 0.2],
              "probe": [{"value": 0.5}, {"value": 0.9}]},
    })
    records = collect_layer_records([run_dir])
    assert len(records) == 2
    assert all("tuned_r2_model" not in r and "l3_mean_sensitivity" not in r for r in records)
    assert records[1]["effective_dim"] == 5.0
    print("missing-stage degrade test passed")


def test_collect_layer_records_picks_up_sae_layer_sweep_when_present():
    """A run with both internals/profile.json and sae_layer_sweep.json must
    contribute `sae_ground_truth_rho` (ROADMAP.md §6.1's "feed SAE eval back
    into layer-selection" item), one value per (model, layer) the sweep
    covers, keyed off ground_truth_alignment's mean_abs_rho_matched."""
    out = Path(tempfile.mkdtemp())
    run_dir = out / "with_sae_sweep"
    save_json(run_dir / "internals" / "profile.json", {
        "M": {"layers": ["l0", "l1"], "rel_depth": [0.0, 1.0],
              "effective_dim": [3.0, 5.0], "input_cka": [0.1, 0.2],
              "probe": [{"value": 0.5}, {"value": 0.9}]},
    })
    save_json(run_dir / "sae_layer_sweep.json", {
        "run": str(run_dir), "models": ["M"],
        "results": {
            "M/l0": {"ground_truth_alignment": {"mean_abs_rho_matched": 0.12, "n_features_matched": 3}},
            "M/l1": {"ground_truth_alignment": {"mean_abs_rho_matched": 0.31, "n_features_matched": 7}},
        },
    })
    records = collect_layer_records([run_dir])
    assert len(records) == 2
    by_layer = {r["layer"]: r for r in records}
    assert by_layer["l0"]["sae_ground_truth_rho"] == 0.12
    assert by_layer["l1"]["sae_ground_truth_rho"] == 0.31
    print("sae_layer_sweep ingestion test passed")


def test_collect_layer_records_omits_sae_proxy_when_sweep_missing_or_errored():
    """No sae_layer_sweep.json at all -> no `sae_ground_truth_rho` field (not
    a zero); a sweep entry with an `error` or zero matched features must be
    skipped the same way, not recorded as a spurious 0.0."""
    out = Path(tempfile.mkdtemp())
    no_sweep = out / "no_sweep"
    save_json(no_sweep / "internals" / "profile.json", {
        "M": {"layers": ["l0"], "rel_depth": [0.0], "effective_dim": [3.0],
              "input_cka": [0.1], "probe": [{"value": 0.5}]},
    })
    records = collect_layer_records([no_sweep])
    assert "sae_ground_truth_rho" not in records[0]

    errored = out / "errored_sweep"
    save_json(errored / "internals" / "profile.json", {
        "M": {"layers": ["l0"], "rel_depth": [0.0], "effective_dim": [3.0],
              "input_cka": [0.1], "probe": [{"value": 0.5}]},
    })
    save_json(errored / "sae_layer_sweep.json", {
        "results": {"M/l0": {"ground_truth_alignment": {"error": "too few series with ground truth"}}},
    })
    records = collect_layer_records([errored])
    assert "sae_ground_truth_rho" not in records[0]

    zero_matched = out / "zero_matched_sweep"
    save_json(zero_matched / "internals" / "profile.json", {
        "M": {"layers": ["l0"], "rel_depth": [0.0], "effective_dim": [3.0],
              "input_cka": [0.1], "probe": [{"value": 0.5}]},
    })
    save_json(zero_matched / "sae_layer_sweep.json", {
        "results": {"M/l0": {"ground_truth_alignment": {"mean_abs_rho_matched": 0.0, "n_features_matched": 0}}},
    })
    records = collect_layer_records([zero_matched])
    assert "sae_ground_truth_rho" not in records[0]
    print("missing/errored/zero-matched sae sweep degrade test passed")


def test_recommend_layers_directions_and_goal_validation():
    records = [
        {"run": "r1", "model": "M", "layer": "l0", "effective_dim": 10.0, "l3_entropy": 0.2},
        {"run": "r1", "model": "M", "layer": "l1", "effective_dim": 3.0, "l3_entropy": 0.9},
        {"run": "r1", "model": "M", "layer": "l2", "effective_dim": 6.0, "l3_entropy": 0.5},
    ]
    readable = recommend_layers(records, "r1", "M", "forecast_readability", top_k=1)
    assert readable[0]["layer"] == "l1", readable  # lowest effective_dim

    decodable = recommend_layers(records, "r1", "M", "family_decodability", top_k=1)
    assert decodable[0]["layer"] == "l1", decodable  # highest l3_entropy

    try:
        recommend_layers(records, "r1", "M", "nonsense_goal")
        assert False, "expected ValueError for an unknown goal"
    except ValueError:
        pass
    print("recommend_layers directions/goal-validation test passed")


if __name__ == "__main__":
    test_cluster_bootstrap_finds_known_correlation_and_rejects_null()
    test_cluster_bootstrap_needs_at_least_three_groups()
    test_entropy_bounds()
    test_collect_layer_records_degrades_when_stages_missing()
    test_collect_layer_records_picks_up_sae_layer_sweep_when_present()
    test_collect_layer_records_omits_sae_proxy_when_sweep_missing_or_errored()
    test_recommend_layers_directions_and_goal_validation()
