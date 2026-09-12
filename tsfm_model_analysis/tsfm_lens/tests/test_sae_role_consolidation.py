"""The consolidated SAE role views (user request, 2026-09-09).

The section used to render one roles table and one feature x channel heatmap
PER TARGET (thirteen of each on the four-model run), plus one matched-role
table per model PAIR (six). This covers the four reductions that replaced
them: per-model roles, per-model heatmap rows, shared-role groups, and
per-model specific roles.

Synthetic fixtures with planted answers only -- no artifacts, no HTML, no
checkpoint -- mirroring `tests/test_sae_role_matching_report.py`'s split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.sae_role_matching import (
    correspondence_overview_rows,
    correspondence_rate_rows,
    model_specific_roles_table,
    shared_roles_table,
)
from tsfm_lens.report.sae_roles import (
    model_feature_channel_matrix,
    model_roles_table,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _match(role_a, role_b, cosine, clears=True, verdict=None):
    m = {"role_a": role_a, "role_b": role_b, "cosine": cosine,
         "clears_population_null": clears}
    if verdict is not None:
        m["causal"] = {"verdict": verdict}
    return m


def _pair(model_a, model_b, matches, comparable=True, **kw):
    if not comparable:
        return {"model_a": model_a, "model_b": model_b, "comparable": False,
                "reason": "no target at a comparable depth"}
    out = {
        "model_a": model_a, "model_b": model_b, "comparable": True,
        "target_a": f"{model_a}/blk.1", "target_b": f"{model_b}/blk.2",
        "depth_a": 0.25, "depth_b": 0.5,
        "n_roles_a": 3, "n_roles_b": 3,
        "matches": matches, "cosine_threshold": 0.5,
        "match_rate": 1.0, "match_rate_quotable": False,
        "untrained_twin_floor": None,
    }
    out.update(kw)
    return out


# ---------------------------------------------------------------------------
# shared_roles_table
# ---------------------------------------------------------------------------

def test_a_role_three_models_share_is_one_row_naming_three_models():
    """The defect this replaced: six per-pair tables meant a role shared by
    three models appeared three times, once per pair, and the reader had to
    notice it was the same role. Connected components over the cleared edges
    make it one row."""
    table = {"pairs": [
        _pair("A", "B", [_match("disperser", "disperser", 0.9)]),
        _pair("B", "C", [_match("disperser", "disperser", 0.8)]),
    ]}
    df = shared_roles_table(table)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["how many models"] == 3
    assert row["models sharing it"] == "A, B, C"
    assert row["best cosine"] == "+0.900"


def test_shared_keys_on_the_population_null_not_the_cosine_threshold():
    """🔴 Load-bearing. The fixed cosine threshold reads 1.000 for four of
    six pairs on the real run while the within-run population null reads
    0.000-0.167 for those same pairs, so a "shared roles" view built on the
    threshold would call nearly everything shared. A high-cosine match that
    does NOT clear the null must not appear at all.
    """
    table = {"pairs": [_pair("A", "B", [
        _match("cleared", "cleared", 0.6, clears=True),
        _match("uncleared", "partner", 0.99, clears=False),
    ])]}
    df = shared_roles_table(table)
    assert list(df["shared role"]) == ["cleared"]


def test_shared_role_causal_verdict_keeps_not_scorable_as_itself():
    """A pair that could not be scored must not be counted as agreement
    (sec 11.37) -- and must not be reported as disagreement either."""
    unscored = shared_roles_table({"pairs": [
        _pair("A", "B", [_match("r", "r", 0.9, verdict=None)])]})
    assert unscored.iloc[0]["same causal effect?"] == "not scorable"

    differs = shared_roles_table({"pairs": [
        _pair("A", "B", [_match("r", "r", 0.9,
                                verdict="fires together, acts differently")])]})
    assert differs.iloc[0]["same causal effect?"] == "acts differently"

    alike = shared_roles_table({"pairs": [
        _pair("A", "B", [_match("r", "r", 0.9, verdict="same causal role")])]})
    assert alike.iloc[0]["same causal effect?"] == "acts alike"


def test_shared_roles_table_is_empty_when_nothing_clears():
    df = shared_roles_table({"pairs": [
        _pair("A", "B", [_match("r", "r", 0.99, clears=False)])]})
    assert df.empty


def test_shared_role_names_the_disagreeing_members():
    """Roles are named from their own dominant channel, so members of a real
    group usually agree -- where they do not, every model's own name must
    still be reachable rather than silently replaced by the majority one."""
    table = {"pairs": [_pair("A", "B", [_match("disperser", "flattener", 0.7)])]}
    row = shared_roles_table(table).iloc[0]
    assert "flattener" in row["each model's own name for it"] or \
           "disperser" in row["each model's own name for it"]


# ---------------------------------------------------------------------------
# model_specific_roles_table
# ---------------------------------------------------------------------------

def test_specific_excludes_a_role_shared_with_anyone():
    table = {"pairs": [
        _pair("A", "B", [_match("shared", "s2", 0.9, clears=True),
                         _match("mine", "m2", 0.4, clears=False)]),
    ]}
    df = model_specific_roles_table(table, "A")
    assert list(df["role only this model has"]) == ["mine"]
    assert "B: m2" in df.iloc[0]["closest thing elsewhere"]


def test_specific_counts_how_many_models_it_was_actually_compared_against():
    """"Specific" is a claim about this run's panel, so a role that only
    ever faced one counterpart must not read as broadly unique."""
    table = {"pairs": [
        _pair("A", "B", [_match("mine", "b", 0.2, clears=False)]),
        _pair("A", "C", [_match("mine", "c", 0.3, clears=False)]),
    ]}
    row = model_specific_roles_table(table, "A").iloc[0]
    assert row["compared against"].startswith("2 ")
    assert "C: c" in row["closest thing elsewhere"]  # the higher cosine wins


def test_specific_reads_the_b_side_too():
    """A model that is the B side of every pair must still get its own
    table -- otherwise the last model in the panel silently has none."""
    table = {"pairs": [_pair("A", "B", [_match("a-role", "b-role", 0.2,
                                               clears=False)])]}
    assert list(model_specific_roles_table(table, "B")["role only this model has"]) \
        == ["b-role"]


# ---------------------------------------------------------------------------
# the split correspondence views
# ---------------------------------------------------------------------------

def test_split_views_render_every_pair_with_uniform_keys_and_no_blanks():
    """Both halves must keep a non-comparable pair as a row that SAYS so --
    the seventeen-column original's two branches emitted different key sets,
    and pandas filled the difference with NaN (sec 11.37)."""
    table = {"pairs": [_pair("A", "B", [_match("r", "r", 0.9)]),
                       _pair("A", "C", [], comparable=False)]}
    for rows in (correspondence_overview_rows(table),
                 correspondence_rate_rows(table)):
        assert len(rows) == 2
        assert set(rows[0]) == set(rows[1])
        for r in rows:
            for v in r.values():
                assert isinstance(v, str) and v != ""
        # The incomparable row must SAY it was never compared, in words --
        # never leave that to a blank cell a reader reads as a zero.
        assert "never compared" in " ".join(rows[1].values()) or \
               "no target at a comparable depth" in " ".join(rows[1].values())

    # And the overview, which is the half that names the targets, carries the
    # artifact's own reason rather than a generic one.
    assert "no target at a comparable depth" in \
        correspondence_overview_rows(table)[1]["compared?"]


def test_the_uncalibrated_rate_names_itself_in_its_own_column_header():
    """🔴 A bare "1.000" in a column called "match rate" is the single most
    misreadable number this section produces, and a note the reader may not
    open is not where that belongs."""
    rows = correspondence_rate_rows({"pairs": [_pair("A", "B", [_match("r", "r", 0.9)])]})
    header = next(k for k in rows[0] if k.startswith("matched (uncalibrated"))
    assert "cosine" in header
    assert rows[0]["safe to quote?"].startswith("no")


# ---------------------------------------------------------------------------
# the per-model roles table and heatmap rows
# ---------------------------------------------------------------------------

def _role(name, clears=True, atoms=4, eff=2.0, chan="near horizon"):
    return {"name": name, "n_atoms": atoms, "clears_null": clears,
            "dominant_channel": chan, "dominant_effect_null_units": eff,
            "dominant_channel_n_clearing": 2}


def test_model_roles_table_keeps_only_causally_real_roles_and_counts_the_rest():
    all_roles = {
        "A/blk.0": {"roles": [_role("kept"), _role("inert", clears=False, atoms=9)]},
        "A/blk.1": {"roles": [_role("kept-2")]},
        "B/blk.0": {"roles": [_role("other-model")]},
    }
    df, dropped = model_roles_table(all_roles, "A", ["A/blk.0", "A/blk.1", "B/blk.0"])
    assert list(df["role"]) == ["kept", "kept-2"]
    assert list(df["layer"]) == ["blk.0", "blk.1"]
    assert dropped == [{"layer": "blk.0", "roles": 1, "atoms": 9}]
    assert df.iloc[0]["members clearing it"] == "2 of 4"


def test_model_roles_table_follows_the_callers_depth_order_not_artifact_order():
    """🔴 Under `sae.targets: auto` the artifact's keys are `layer_screen`'s
    SCORE ranking, so rows a reader scans as depth jump around -- the exact
    defect sec 26 E's second pass fixed for the structural heatmap. Pooling
    several layers into one table reintroduces it unless the caller's own
    depth-ordered list drives the loop.
    """
    all_roles = {"A/blk.9": {"roles": [_role("deep")]},
                 "A/blk.1": {"roles": [_role("shallow")]}}
    df, _ = model_roles_table(all_roles, "A", ["A/blk.1", "A/blk.9"])
    assert list(df["layer"]) == ["blk.1", "blk.9"]


def test_model_roles_table_skips_a_withheld_target_without_dropping_the_model():
    all_roles = {"A/blk.0": {"withheld": True, "roles": [_role("unreachable")]},
                 "A/blk.1": {"roles": [_role("kept")]}}
    df, _ = model_roles_table(all_roles, "A", ["A/blk.0", "A/blk.1"])
    assert list(df["role"]) == ["kept"]


def _cand(fid, clearing):
    return {"feature": fid, "clearing_channels": list(clearing),
            "up": {"channels": {"mase": {"signed_mean": 1.5},
                                "spread": {"signed_mean": -0.4}}},
            "down": {"channels": {}}}


def test_heatmap_rows_drop_silent_features_and_report_how_many():
    """A feature clearing nothing is a blank row; thirteen per-target
    heatmaps spent most of their height on them. The count must survive as
    a number, since "these were dropped" and "these did not exist" are
    different facts."""
    by_target = {"A/blk.0": [_cand(1, ["mase"]), _cand(2, [])],
                 "A/blk.1": [_cand(3, [])],
                 "B/blk.0": [_cand(4, ["mase"])]}
    labels, chans, vals, clears, n_silent = model_feature_channel_matrix(
        by_target, "A", ["A/blk.0", "A/blk.1"], ["mase", "spread"])
    assert labels == ["blk.0 · f1"]
    assert n_silent == 2
    assert vals.shape == (1, 2) and clears.shape == (1, 2)
    assert clears[0].tolist() == [True, False]


def test_heatmap_rows_return_the_same_arity_when_nothing_survives():
    """The empty branch returned a 4-tuple where the populated one returned
    5, so a model whose every feature is silent unpacked into a crash."""
    out = model_feature_channel_matrix({"A/blk.0": [_cand(1, [])]}, "A",
                                       ["A/blk.0"], ["mase"])
    assert len(out) == 5
    labels, chans, vals, clears, n_silent = out
    assert labels == [] and n_silent == 1
    assert vals.shape == (0, 1)
    assert np.asarray(clears).dtype == bool
