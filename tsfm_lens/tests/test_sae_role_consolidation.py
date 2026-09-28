"""The consolidated per-model SAE role views (user request, 2026-09-09).

The section used to render one roles table and one feature x channel heatmap
PER TARGET (thirteen of each on the four-model run). This covers the two
reductions that replaced them: per-model roles, and per-model heatmap rows.

Synthetic fixtures with planted answers only -- no artifacts, no HTML, no
checkpoint.

CLAUDE.md sec 2.9 note (2026-09-14, ROADMAP.md sec 32.9 PRUNE): this file used
to also cover `report/sae_role_matching.py`'s cross-model correspondence
tables (`shared_roles_table`, `model_specific_roles_table`,
`correspondence_overview_rows`, `correspondence_rate_rows`). That module was
the one grounded dead-code candidate CLAUDE.md sec 2.9 already named -- zero
importers in live code, orphaned when sec 30 Stage 4 superseded
`roles.json`/injection-space matching with `concepts.json`/ablation-space
clustering, kept alive only by its own tests. Deleted along with
`tests/test_sae_role_matching_report.py`; those four tests' coverage went
with it. `sae/role_matching.py` (the COMPUTATION module,
`role_correspondence_table`) is a different, still-live module -- it remains
the read path for `run_sae_compare.py`'s (study driver, dev branch) standalone
`roles_injection.json` comparison and was not touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.sae_roles import (
    model_feature_channel_matrix,
    model_roles_table,
)


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
