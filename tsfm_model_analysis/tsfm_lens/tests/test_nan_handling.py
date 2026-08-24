"""NaN/missing-timestep handling probe unit tests (ROADMAP.md sec 16 E17).

Synthetic, planted-answer only -- constructs the four possible per-adapter
verdicts directly from hand-built scenario records, no real checkpoint I/O
(that lives in a live-run check reported separately, per CLAUDE.md sec 2.4).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.nan_handling import nan_handling_stats


def _scenario(position, raised, error_type=None, nonfinite=None):
    return {"position": position, "raised": raised, "error_type": error_type,
            "output_has_nonfinite": nonfinite}


def test_requires_at_least_one_scenario():
    try:
        nan_handling_stats([])
        assert False, "expected ValueError for an empty scenario list"
    except ValueError:
        pass
    print("empty-scenario-list rejection test passed")


def test_all_raise_gives_errors_verdict():
    results = [_scenario("front", True, "RuntimeError"),
              _scenario("middle", True, "RuntimeError"),
              _scenario("back", True, "ValueError")]
    stats = nan_handling_stats(results)
    assert stats["verdict"] == "errors"
    assert stats["n_raised"] == 3
    assert stats["n_clean"] == 0
    print("all-raise 'errors' verdict test passed")


def test_all_finite_gives_handled_verdict():
    results = [_scenario("front", False, nonfinite=False),
              _scenario("middle", False, nonfinite=False),
              _scenario("back", False, nonfinite=False)]
    stats = nan_handling_stats(results)
    assert stats["verdict"] == "handled"
    assert stats["n_clean"] == 3
    assert stats["n_propagated_nonfinite"] == 0
    print("all-finite 'handled' verdict test passed")


def test_all_nonfinite_gives_propagates_verdict():
    results = [_scenario("front", False, nonfinite=True),
              _scenario("middle", False, nonfinite=True)]
    stats = nan_handling_stats(results)
    assert stats["verdict"] == "propagates"
    assert stats["n_propagated_nonfinite"] == 2
    print("all-nonfinite 'propagates' verdict test passed")


def test_mixed_behavior_gives_mixed_verdict():
    results = [_scenario("front", False, nonfinite=False),
              _scenario("back", False, nonfinite=True)]
    stats = nan_handling_stats(results)
    assert stats["verdict"] == "mixed"
    print("mixed-behavior 'mixed' verdict test passed")


def test_one_raise_one_clean_gives_mixed_not_errors():
    results = [_scenario("front", True, "RuntimeError"),
              _scenario("back", False, nonfinite=False)]
    stats = nan_handling_stats(results)
    assert stats["verdict"] == "mixed"
    assert stats["n_raised"] == 1
    assert stats["n_clean"] == 1
    print("partial-raise 'mixed' verdict test passed")


if __name__ == "__main__":
    test_requires_at_least_one_scenario()
    test_all_raise_gives_errors_verdict()
    test_all_finite_gives_handled_verdict()
    test_all_nonfinite_gives_propagates_verdict()
    test_mixed_behavior_gives_mixed_verdict()
    test_one_raise_one_clean_gives_mixed_not_errors()
    print("All NaN handling tests passed")
