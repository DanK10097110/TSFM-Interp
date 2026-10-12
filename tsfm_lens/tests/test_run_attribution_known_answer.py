"""The pre-fixed gate arithmetic of `run_attribution_known_answer.py` (ROADMAP.md sec 42, 5a).

Pure reductions over per-seed rows: seed 0 is the tuning seed and must never
enter the verdict, the bar is the MEAN over held-out seeds 1-4 of the per-seed
pooled recall, and a missing held-out seed is `not scorable`, not a pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_attribution_known_answer import _kind, gate_verdict, pooled  # noqa: E402


def _rows(recalls: dict) -> dict:
    return {s: {"pooled": {"attribution": {"recall": r}}} for s, r in recalls.items()}


def test_gate_passes_on_the_mean_of_held_out_seeds_only():
    rows = _rows({0: 0.0, 1: 0.9, 2: 0.8, 3: 0.8, 4: 0.7})
    v = gate_verdict(rows)
    assert v["verdict"] == "pass" and v["mean_recall"] == pytest.approx(0.8)


def test_tuning_seed_cannot_rescue_a_failing_held_out_mean():
    v = gate_verdict(_rows({0: 1.0, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}))
    assert v["verdict"] == "fail" and v["mean_recall"] == pytest.approx(0.5)


def test_a_missing_held_out_seed_is_not_scorable():
    assert gate_verdict(_rows({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}))["verdict"] == "not scorable"
    assert gate_verdict(_rows({1: 1.0, 2: 1.0, 3: 1.0, 4: None}))["verdict"] == "not scorable"


def test_pooled_recall_weights_models_by_their_recovered_concepts():
    a = {"per_rule": {"attribution": {"n_hit": 1, "n_real_recovered": 2, "hit": ["shared_trend_0"],
                                      "missed": ["unique_seasonal_0"]}}}
    b = {"per_rule": {"attribution": {"n_hit": 3, "n_real_recovered": 3,
                                      "hit": ["opposite_level_0"] * 3, "missed": []}}}
    p = pooled([a, b], "attribution")
    assert (p["n_hit"], p["n"], p["recall"]) == (4, 5, 0.8)
    assert p["by_kind"]["seasonal"] == {"n_hit": 0, "n": 1}


def _two_rule_rows(recalls: dict) -> dict:
    return {s: {"pooled": {"a": {"recall": ra}, "b": {"recall": rb}}} for s, (ra, rb) in recalls.items()}


def test_regate_reads_each_rule_on_its_own_unseen_seeds_only():
    rows = _two_rule_rows({1: (0.0, 0.0), 5: (0.9, 0.5), 6: (0.8, 0.5), 7: (0.8, 0.5), 8: (0.8, 0.5)})
    a = gate_verdict(rows, "a", (5, 6, 7, 8))
    b = gate_verdict(rows, "b", (5, 6, 7, 8))
    assert (a["verdict"], a["rule"], a["mean_recall"]) == ("pass", "a", pytest.approx(0.825))
    assert (b["verdict"], b["mean_recall"]) == ("fail", pytest.approx(0.5))


def test_a_rule_without_a_recall_on_a_held_out_seed_is_not_scorable():
    rows = _two_rule_rows({5: (0.9, 0.5), 6: (0.8, 0.5), 7: (0.8, 0.5), 8: (0.8, 0.5)})
    assert gate_verdict(rows, "missing", (5, 6, 7, 8))["verdict"] == "not scorable"


def test_kind_is_read_from_the_planted_id():
    assert _kind("convergent_dispersion_1") == "dispersion"
    assert _kind("shared_trend_0") == "trend"
