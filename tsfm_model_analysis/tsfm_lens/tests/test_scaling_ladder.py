"""Scaling-ladder reducer (ROADMAP.md sec 20 H1) -- synthetic, planted answers.

The five real rungs are GPU work; everything statistical about the ladder is
not, so it is pinned here against data whose right answer is known by
construction. Three of these are load-bearing negatives: a ladder cannot be
placed on a checkpoint name, a five-rung p-value has a floor that must be
reported next to it, and a degenerate ladder must abstain rather than answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.scaling_ladder import (  # noqa: E402
    classify, ladder_size, run_scaling_ladder, spearman_exact,
)

SIZES = [8e6, 20e6, 46e6, 200e6, 710e6]


def test_a_perfect_ramp_is_monotone_at_the_exact_p_floor():
    out = spearman_exact(SIZES, [0.1, 0.2, 0.3, 0.4, 0.5])
    assert out["rho"] == pytest.approx(1.0)
    assert out["exact"] is True
    # 5! = 120 orderings; two of them (the ramp and its reverse) are at |rho|=1.
    assert out["p"] == pytest.approx(2 / 120)
    assert out["p"] == pytest.approx(out["p_floor"])


def test_the_p_floor_is_reported_because_no_five_rung_ladder_can_beat_it():
    """CLAUDE.md sec 11.35: a bounded statistic's own limit must travel with it."""
    strong = spearman_exact(SIZES, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert strong["p"] >= strong["p_floor"]
    assert strong["p_floor"] > 0.01     # nothing here can ever reach p < 0.01


def test_a_non_monotone_metric_is_flagged_not_smoothed():
    out = classify(SIZES, [0.1, 0.9, 0.2, 0.8, 0.3], ci_widths=[0.01] * 5)
    assert out["shape"] == "non_monotone"
    assert out["flat"] is False


def test_flatness_needs_a_within_run_ci_and_is_withheld_without_one():
    values = [0.500, 0.502, 0.499, 0.501, 0.500]
    wide = classify(SIZES, values, ci_widths=[0.05] * 5)
    assert wide["flat"] is True         # moved less than one run's own noise
    tight = classify(SIZES, values, ci_widths=[0.0001] * 5)
    assert tight["flat"] is False       # the same wobble is real at this precision
    unbacked = classify(SIZES, values, ci_widths=None)
    assert unbacked["flat"] is None and "no within-run CI" in unbacked["flat_reason"]


def test_a_degenerate_ladder_abstains_rather_than_answering_confidently():
    """One point makes `all(diff > 0)` vacuously true -- sec 11.37's failure shape."""
    one = classify([1e6], [0.4], ci_widths=[0.01])
    assert one["shape"] == "too_few_rungs" and one["flat"] is None
    assert spearman_exact([1e6, 2e6], [0.4, 0.5])["rho"] is None


def _write_rung(tmp: Path, name: str, params: int | None, mase: float) -> Path:
    rd = tmp / name
    (rd / "budget").mkdir(parents=True)
    (rd / "l0").mkdir(parents=True)
    (rd / "config_resolved.yaml").write_text(
        f"run:\n  name: {name}\nmodels:\n  - name: M\n    checkpoint: fake/{name}\n",
        encoding="utf-8")
    if params is not None:
        (rd / "budget" / "model_budget.json").write_text(
            json.dumps({"models": {"M": {"parameters": {"total": params}}}}), encoding="utf-8")
    (rd / "l0" / "summary.json").write_text(
        json.dumps({"overall": [{"model": "M", "mase": mase}]}), encoding="utf-8")
    return rd


def test_a_rung_without_a_measured_parameter_count_is_excluded_not_named(tmp_path):
    """The axis is a measurement. A checkpoint name is a claim nobody checks (sec 11.34)."""
    a = _write_rung(tmp_path, "small", 46_000_000, 1.4)
    b = _write_rung(tmp_path, "large", None, 1.1)        # name says "large"; nothing measured it
    c = _write_rung(tmp_path, "base", 200_000_000, 1.2)
    assert ladder_size(b, "M") is None
    out = run_scaling_ladder([a, b, c], "M")
    assert [r["label"] for r in out["rungs"]] == ["small", "base"]
    assert len(out["excluded"]) == 1 and "budget" in out["excluded"][0]["reason"]


def test_rungs_are_ordered_by_measured_size_not_by_argument_order(tmp_path):
    big = _write_rung(tmp_path, "big", 710_000_000, 1.0)
    tiny = _write_rung(tmp_path, "tiny", 8_000_000, 2.0)
    mid = _write_rung(tmp_path, "mid", 46_000_000, 1.5)
    out = run_scaling_ladder([big, tiny, mid], "M")
    assert [r["label"] for r in out["rungs"]] == ["tiny", "mid", "big"]
    mase = out["metrics"]["mase"]
    assert mase["values"] == [2.0, 1.5, 1.0]
    assert mase["shape"] == "monotone_decreasing"   # bigger model, lower error


def test_a_metric_missing_from_one_rung_is_named_rather_than_dropped(tmp_path):
    a = _write_rung(tmp_path, "a", 8_000_000, 2.0)
    b = _write_rung(tmp_path, "b", 46_000_000, 1.5)
    c = _write_rung(tmp_path, "c", 200_000_000, 1.0)
    (b / "clustering").mkdir()
    (b / "clustering" / "comparison.json").write_text(
        json.dumps({"ami": {"value": 0.5, "lo": 0.4, "hi": 0.6}}), encoding="utf-8")
    out = run_scaling_ladder([a, b, c], "M")
    ami = out["metrics"]["clustering_ami"]
    assert ami["n_rungs"] == 1
    assert sorted(ami["missing_rungs"]) == ["a", "c"]
