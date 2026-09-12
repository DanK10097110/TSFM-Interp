"""A blank cell must say why it is blank, never render as `NaN`.

`df.to_html` writes a missing float as the literal string `NaN`, which reads
as a measurement that was attempted and broke. Two joins in this report have
legitimately empty cells by construction -- patching runs over its own
`layer_stride` and its own `l3.patching.corruptions` subset, so most rows of
both the corruption breakdown and the per-layer table have no restoration --
and both notes already told the reader that a blank means "this stage did not
measure this row". The rendering contradicted the note: 44 cells of a live
four-model report came back `NaN`.

That is `CLAUDE.md` sec 11.37 at a third site: absent, chosen and bad must be
three states. The load-bearing negatives here are that the reason travels
PER ROW (a corruption nobody patched and a model whose patching stage never
ran are different facts), and that a numeric cell that IS measured keeps its
own formatting rather than being stringified into something a reader cannot
compare down the column.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report import derived, report  # noqa: E402


def _write(run: Path, rel: str, payload) -> None:
    p = run / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# the helper
# ---------------------------------------------------------------------------

def test_an_absent_cell_renders_its_reason_and_a_present_one_keeps_its_format():
    df = pd.DataFrame({"x": [1.0, None, float("nan")]})
    report._absent_as_text(df, ["x"], "not patched")
    assert list(df["x"]) == ["1.000", "not patched", "not patched"]
    assert not df.isna().any().any()


def test_the_reason_can_differ_per_row():
    """The whole point at the corruption site: two absences, two causes."""
    df = pd.DataFrame({"x": [None, None, 2.5]})
    report._absent_as_text(df, ["x"], ["patching did not run", "not patched", None])
    assert list(df["x"]) == ["patching did not run", "not patched", "2.500"]


def test_a_column_that_does_not_exist_is_skipped_rather_than_created():
    df = pd.DataFrame({"x": [1.0]})
    report._absent_as_text(df, ["nope"], "why")
    assert list(df.columns) == ["x"]


def test_a_missing_reason_still_says_something_rather_than_nothing():
    """An empty reason is the blank cell this helper exists to remove."""
    df = pd.DataFrame({"x": [None]})
    report._absent_as_text(df, ["x"], [""])
    assert list(df["x"]) == ["not measured"]


# ---------------------------------------------------------------------------
# derived: the three states of a missing restoration
# ---------------------------------------------------------------------------

def _l3(tmp_path, patching: dict | None, restoration=None):
    _write(tmp_path, "l3/meta.json", {
        "corruptions": ["patched_one", "unpatched_one"],
        "layers": {"A": ["a.0", "a.1"]},
        "rel_depth": {"A": [0.0, 1.0]},
        "models": ["A"],
        "calibration": {"patched_one": {"footprint": 0.1, "energy": 1.0},
                        "unpatched_one": {"footprint": 0.2, "energy": 2.0}},
        "behavior_ci": {"A": {"patched_one": {"value": 1.0},
                              "unpatched_one": {"value": 2.0}}}})
    if patching is not None:
        _write(tmp_path, "l3/patching.json", patching)
    if restoration is not None:
        (tmp_path / "l3").mkdir(parents=True, exist_ok=True)
        np.savez(tmp_path / "l3" / "patching.npz", restoration_A=restoration)


def test_a_corruption_that_was_patched_and_one_that_was_not_are_distinguishable(tmp_path):
    _l3(tmp_path, {"A": {"corruptions": ["patched_one"], "layers": ["a.0", "a.1"]}},
        restoration=np.array([[0.2, 0.8]]))
    df = derived.corruption_breakdown(tmp_path).set_index("corruption")
    assert df.loc["patched_one", "best_restoration"] == pytest.approx(0.8)
    assert df.loc["patched_one", "best_restoration_layer"] == "a.1"
    assert df.loc["patched_one", "restoration_absent"] is None
    assert df.loc["unpatched_one", "restoration_absent"] == "not patched"


def test_a_model_whose_patching_never_ran_says_that_instead(tmp_path):
    """Distinct from "not patched" -- one is a config subset, one is a
    stage that produced nothing at all, and collapsing them asserts a
    selection that never happened (sec 11.37)."""
    _l3(tmp_path, None)
    df = derived.corruption_breakdown(tmp_path).set_index("corruption")
    assert set(df["restoration_absent"]) == {"patching did not run"}


def test_a_selected_corruption_with_no_recorded_array_is_its_own_state(tmp_path):
    _l3(tmp_path, {"A": {"corruptions": ["patched_one"], "layers": ["a.0"]}})
    df = derived.corruption_breakdown(tmp_path).set_index("corruption")
    assert df.loc["patched_one", "restoration_absent"] == "no restoration recorded"


# ---------------------------------------------------------------------------
# rendered output (sec 11.48 -- verify the render, not the diff)
# ---------------------------------------------------------------------------

def test_the_rendered_corruption_table_has_no_nan_and_names_the_reason(tmp_path):
    _l3(tmp_path, {"A": {"corruptions": ["patched_one"], "layers": ["a.0", "a.1"]}},
        restoration=np.array([[0.2, 0.8]]))
    html = report._corruption_breakdown_block(tmp_path, [])
    assert html
    assert ">NaN<" not in html
    assert "not patched" in html
    assert "0.800" in html and "a.1" in html


def test_the_rendered_layer_table_has_no_nan_and_names_the_stride(tmp_path):
    """`internals` captured two layers; patching ran at one of them."""
    _write(tmp_path, "internals/profile.json", {"A": {
        "layers": ["a.0", "a.1"], "rel_depth": [0.0, 1.0],
        "effective_dim": [4.0, 8.0]}})
    _write(tmp_path, "l3/patching.json", {"A": {"layers": ["a.1"],
                                                "corruptions": ["c"]}})
    np.savez(tmp_path / "l3" / "patching.npz", restoration_A=np.array([[0.6]]))
    html = report._layer_metrics_block(tmp_path, ["A"], [])
    assert html
    assert ">NaN<" not in html
    assert "layer not patched" in html
    assert "0.600" in html
