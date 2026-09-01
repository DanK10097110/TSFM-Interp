"""Tests for `report/sae_role_matching.py` (ROADMAP.md sec 25.7 part 6's
pure builder half). Synthetic `role_correspondence_table`-shaped fixtures
only -- no report.py/HTML rendering, mirroring `tests/
test_sae_roles_report.py`'s split between pure logic and I/O.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.sae_role_matching import (
    correspondence_summary_rows,
    pair_match_table,
    role_by_model_matrix,
)


def _comparable_pair(model_a="A", model_b="B", depth_a=0.5, depth_b=0.5, matches=None,
                     match_rate=0.8, quotable=True, floor=None) -> dict:
    return {
        "model_a": model_a, "model_b": model_b, "comparable": True,
        "target_a": f"{model_a}/layer", "target_b": f"{model_b}/layer",
        "depth_a": depth_a, "depth_b": depth_b,
        "n_roles_a": 2, "n_roles_b": 2,
        "matches": matches or [{"role_a": "r1", "role_b": "r1-b", "cosine": 0.9}],
        "cosine_threshold": 0.5,
        "match_rate": match_rate,
        "untrained_twin_floor": floor,
        "match_rate_quotable": quotable,
        "match_rate_quotable_reason": "reason text",
        "clears_untrained_twin_floor": (match_rate > max(floor.values())) if floor else None,
    }


def _incomparable_pair(model_a="A", model_b="C") -> dict:
    return {"model_a": model_a, "model_b": model_b, "comparable": False,
           "reason": "no target exists to check"}


# ---------------------------------------------------------------------------
# pair_match_table
# ---------------------------------------------------------------------------

def test_pair_match_table_empty_for_incomparable_pair():
    df = pair_match_table(_incomparable_pair())
    assert df.empty


def test_pair_match_table_has_one_row_per_match_with_cosine():
    pair = _comparable_pair(matches=[
        {"role_a": "r1", "role_b": "r1-b", "cosine": 0.9},
        {"role_a": "r2", "role_b": "r2-b", "cosine": 0.3},
    ])
    df = pair_match_table(pair)
    assert len(df) == 2
    assert "cosine" in df.columns
    assert list(df["cosine"]) == [0.9, 0.3]


def test_pair_match_table_includes_secondary_signals_when_present():
    pair = _comparable_pair(matches=[
        {"role_a": "r1", "role_b": "r1-b", "cosine": 0.9,
         "activation_profile_correlation": 0.4, "max_activating_series_jaccard": 0.2},
    ])
    df = pair_match_table(pair)
    assert "activation profile ρ" in df.columns
    assert "max-activating overlap (Jaccard)" in df.columns


# ---------------------------------------------------------------------------
# correspondence_summary_rows
# ---------------------------------------------------------------------------

def test_summary_rows_one_per_pair_including_incomparable():
    table = {"pairs": [_comparable_pair(), _incomparable_pair()]}
    rows = correspondence_summary_rows(table)
    assert len(rows) == 2
    assert rows[0]["comparable"] is True
    assert rows[1]["comparable"] is False
    assert rows[1]["reason"] == "no target exists to check"


def test_summary_rows_quotable_reflects_floor_availability():
    with_floor = _comparable_pair(quotable=True, floor={"A": 0.5})
    without_floor = _comparable_pair(model_a="X", model_b="Y", quotable=False, floor=None)
    rows = correspondence_summary_rows({"pairs": [with_floor, without_floor]})
    assert rows[0]["quotable"] is True
    assert rows[0]["untrained-twin floor"] != "not available"
    assert rows[1]["quotable"] is False
    assert rows[1]["untrained-twin floor"] == "not available"


# ---------------------------------------------------------------------------
# role_by_model_matrix
# ---------------------------------------------------------------------------

def test_matrix_blank_cell_is_none_not_a_string():
    """A model with no matched role at a canonical position must render a
    real `None`, never a placeholder string like 'n/a' or '0.0' -- sec
    25.6's "blank means no target at comparable depth, never absence of
    effect" contract, checked at the exact type level so a future edit
    can't silently swap in a string that reads as data.
    """
    pair = _comparable_pair(model_a="A", model_b="B",
                            matches=[{"role_a": "canon-role", "role_b": "b-role", "cosine": 0.7}])
    table = {"pairs": [pair]}
    df = role_by_model_matrix(table, ["A", "B", "C"])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["A"] is not None
    assert row["B"] is not None
    assert row["C"] is None  # C has no target in this pair at all


def test_matrix_folds_same_role_name_from_multiple_pairs_into_one_row():
    """The same canonical role name appearing as the A-side role in two
    different pairs must produce ONE row, not two -- the "canonical role"
    grouping sec 25.7 part 6 asks for.
    """
    pair_ab = _comparable_pair(model_a="A", model_b="B",
                               matches=[{"role_a": "shared-role", "role_b": "b-partner", "cosine": 0.6}])
    pair_ac = _comparable_pair(model_a="A", model_b="C",
                               matches=[{"role_a": "shared-role", "role_b": "c-partner", "cosine": 0.8}])
    table = {"pairs": [pair_ab, pair_ac]}
    df = role_by_model_matrix(table, ["A", "B", "C"])
    assert len(df) == 1
    assert df.iloc[0]["role"] == "shared-role"
    assert df.iloc[0]["A"] is not None
    assert df.iloc[0]["B"] is not None
    assert df.iloc[0]["C"] is not None


def test_matrix_keeps_best_cosine_partner_when_multiple_pairs_target_same_row():
    """When model B's cell for a canonical role could be filled from two
    different A-side pairs, the HIGHER-cosine partner must win -- not
    whichever pair happened to be processed last.
    """
    pair_low = _comparable_pair(model_a="A", model_b="B",
                                matches=[{"role_a": "canon", "role_b": "weak-match", "cosine": 0.2}])
    pair_high = _comparable_pair(model_a="X", model_b="B",
                                 matches=[{"role_a": "canon", "role_b": "strong-match", "cosine": 0.9}])
    table = {"pairs": [pair_low, pair_high]}
    df = role_by_model_matrix(table, ["A", "X", "B"])
    row = df[df["role"] == "canon"].iloc[0]
    assert "0.90" in row["B"] or "+0.90" in row["B"]
