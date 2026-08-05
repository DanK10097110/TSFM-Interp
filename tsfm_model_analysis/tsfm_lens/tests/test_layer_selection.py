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
    test_recommend_layers_directions_and_goal_validation()
