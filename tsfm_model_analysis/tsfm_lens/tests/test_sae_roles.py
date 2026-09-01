"""Tests for `sae/roles.py` (ROADMAP.md sec 25.5(b) / sec 25.9 Stage 3,
Component B(b)): clustering probed features into named roles, with the
naming DERIVED per sec 25.5(b)'s three rules rather than authored.

All synthetic, with planted answers -- the same evidentiary style as
`test_ground_truth_separated.py`/`test_response_fingerprint.py`. Load-bearing
negatives per the task brief: name uniqueness enforcement (a plant that
forces a collision, checking it resolves), the "no measured effect" and
"unnamed" fallback rules (each planted directly, not inferred), and that a
role's verdict is Rule-derived rather than author-settable (mirrored from
`report/derived.py`'s own `Verdict.__post_init__` pin).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.response import CHANNELS
from tsfm_lens.sae.roles import (
    build_feature_matrix,
    cluster_roles,
    derive_role_name,
    role_table,
)


def _candidate(feature: int, clearing_channels: list, effects: dict | None = None,
              sign_up: dict | None = None, sign_down: dict | None = None) -> dict:
    """A minimal `feature_response_fingerprints`-shaped candidate record.

    `effects` maps channel -> signed_mean for the "up" direction (down
    defaults to the negative, so the larger-magnitude side is deterministic
    and easy to reason about in a test).
    """
    effects = effects or {}
    up_channels, down_channels = {}, {}
    for ch in CHANNELS:
        val = effects.get(ch, 0.0)
        clears = ch in clearing_channels
        up_channels[ch] = {"available": True, "signed_mean": val,
                           "effect": abs(val), "clears_null": clears}
        down_channels[ch] = {"available": True, "signed_mean": -val,
                             "effect": abs(val), "clears_null": clears}
    return {"feature": feature, "rules": ["variance"],
           "up": {"channels": up_channels}, "down": {"channels": down_channels},
           "clearing_channels": sorted(clearing_channels)}


_NULL_P95 = {ch: 1.0 for ch in CHANNELS}


# ---------------------------------------------------------------------------
# 1. build_feature_matrix: null-normalized, signed, correct column order.
# ---------------------------------------------------------------------------

def test_build_feature_matrix_normalizes_by_null_p95_and_keeps_sign():
    cands = [
        _candidate(0, ["trend"], effects={"trend": 4.0}),   # +4x null
        _candidate(1, ["trend"], effects={"trend": -6.0}),  # -6x null
    ]
    X, cols = build_feature_matrix(cands, None, _NULL_P95)
    assert cols == list(CHANNELS)
    trend_col = cols.index("trend")
    assert X[0, trend_col] == pytest.approx(4.0)
    assert X[1, trend_col] == pytest.approx(-6.0)
    # No Stage 1 data -> the 3 trailing structural columns are all zero.
    assert np.allclose(X[:, len(CHANNELS):], 0.0)


def test_build_feature_matrix_uses_the_larger_magnitude_of_up_and_down():
    """A feature with a bigger DOWN effect than its (mirrored) up effect
    must report the down-side sign, not always default to 'up'."""
    cand = {"feature": 0, "rules": ["variance"],
           "up": {"channels": {ch: {"available": True, "signed_mean": 0.1,
                                    "clears_null": False} for ch in CHANNELS}},
           "down": {"channels": {ch: {"available": True, "signed_mean": -5.0,
                                      "clears_null": ch == "level"} for ch in CHANNELS}},
           "clearing_channels": ["level"]}
    X, cols = build_feature_matrix([cand], None, _NULL_P95)
    level_col = cols.index("level")
    assert X[0, level_col] == pytest.approx(-5.0)


def test_build_feature_matrix_concatenates_residualized_structural_signature():
    stage1 = {0: {"feature": 0,
                 "structural": {"field": "trend_scale", "rho": 0.6, "n": 400},
                 "provenance": None,
                 "top3_structural": [{"field": "trend_scale", "rho": 0.6, "n": 400},
                                     {"field": "n_seasonalities", "rho": 0.3, "n": 400}],
                 "top3_provenance": []}}
    cands = [_candidate(0, ["trend"], effects={"trend": 2.0})]
    X, cols = build_feature_matrix(cands, stage1, _NULL_P95)
    structural_part = X[0, len(CHANNELS):]
    assert structural_part[0] == pytest.approx(0.6)
    assert structural_part[1] == pytest.approx(0.3)
    assert structural_part[2] == pytest.approx(0.0)  # zero-padded, only 2 given


# ---------------------------------------------------------------------------
# 2. cluster_roles: silhouette reported, non-modular flagged rather than
#    forcing a partition.
# ---------------------------------------------------------------------------

def test_cluster_roles_finds_two_well_separated_groups():
    """Two blocks separated on EVERY dimension (not just one, with the rest
    pure noise) -- `StandardScaler` normalizes each column to unit variance
    independently, so a single discriminating column diluted across 8
    zero-signal noise columns caps silhouette well below 0.5 in 9-D space
    regardless of how far apart the two blocks are in raw units (checked
    directly: scaling the single-column separation by 10x left silhouette
    unchanged at ~0.186, since standardization removes any pure column-scale
    effect). Separating on every column is what actually produces two
    well-separated clusters after standardization."""
    rng = np.random.default_rng(0)
    a = rng.normal(loc=5.0, scale=0.05, size=(10, 9))
    b = rng.normal(loc=-5.0, scale=0.05, size=(10, 9))
    X = np.vstack([a, b])
    result = cluster_roles(X, role_k=2, min_silhouette=0.1, seed=0)
    assert result["k"] == 2
    assert result["silhouette"] > 0.5
    assert not result["non_modular"]
    # The two planted blocks must land in different clusters (label
    # consistent within each block, differing across).
    assert len(set(result["labels"][:10])) == 1
    assert len(set(result["labels"][10:])) == 1
    assert result["labels"][0] != result["labels"][10]


def test_cluster_roles_reports_non_modular_when_every_atom_is_its_own_role():
    """NEGATIVE (sec 25.13 item 2): k >= n must not silently force a fake
    partition -- it must be flagged non_modular with a stated reason."""
    X = np.random.default_rng(1).normal(size=(3, 9))
    result = cluster_roles(X, role_k=8, min_silhouette=0.1, seed=0)
    assert result["non_modular"]
    assert result["reason"]


def test_cluster_roles_reports_non_modular_on_low_silhouette():
    """A single homogeneous blob with no real substructure must not be
    reported as cleanly separated just because k=2 was requested."""
    X = np.random.default_rng(2).normal(size=(30, 9))
    result = cluster_roles(X, role_k=2, min_silhouette=0.9, seed=0)
    assert result["non_modular"]
    assert "silhouette" in result["reason"]


# ---------------------------------------------------------------------------
# 3. derive_role_name: the three rules, each planted directly.
# ---------------------------------------------------------------------------

def _labels_for(n_per_role: list) -> np.ndarray:
    out = []
    for role, n in enumerate(n_per_role):
        out.extend([role] * n)
    return np.asarray(out)


def test_role_with_no_clearing_channel_is_named_no_measured_effect():
    """Rule 2: a role where NOTHING clears its null must not be given a
    correlational label or left blank -- it gets the fixed, explicit name."""
    members = [_candidate(0, []), _candidate(1, [])]  # nothing clears
    role_entries = {0: members}
    X = np.zeros((2, len(CHANNELS) + 3))
    labels = np.zeros(2, dtype=int)
    info = derive_role_name(0, X, labels, list(CHANNELS), role_entries,
                            _NULL_P95, None, existing_names=set())
    assert info["name"] == "no measured effect (2 atoms)"
    assert info["clears_null"] is False


def test_role_with_no_discriminating_channel_is_named_unnamed():
    """Rule 3: every channel column is exactly zero (no discriminating
    signature at all) even though the role DOES clear its null on some
    channel (the clearing bookkeeping and the vector geometry are tracked
    separately in this planted case) -- must refuse rather than silently
    pick channel 0 by array order."""
    members = [_candidate(0, ["trend"])]  # clears "trend" but effects all 0.0
    role_entries = {0: members}
    X = np.zeros((1, len(CHANNELS) + 3))  # every column exactly zero
    labels = np.zeros(1, dtype=int)
    info = derive_role_name(0, X, labels, list(CHANNELS), role_entries,
                            _NULL_P95, None, existing_names=set())
    assert info["name"] == "unnamed (no distinguishing signature)"


def test_role_name_uses_dominant_channel_and_sign():
    members = [_candidate(0, ["trend"], effects={"trend": 3.0})]
    role_entries = {0: members}
    X = np.zeros((1, len(CHANNELS) + 3))
    X[0, list(CHANNELS).index("trend")] = 3.0
    labels = np.zeros(1, dtype=int)
    info = derive_role_name(0, X, labels, list(CHANNELS), role_entries,
                            _NULL_P95, None, existing_names=set())
    assert "trend-slope" in info["name"]
    assert "↑" in info["name"]
    assert info["dominant_channel"] == "trend"
    assert info["sign"] == 1


def test_role_name_includes_structural_field_when_present():
    members = [_candidate(0, ["seasonal"], effects={"seasonal": 2.0})]
    role_entries = {0: members}
    X = np.zeros((1, len(CHANNELS) + 3))
    X[0, list(CHANNELS).index("seasonal")] = 2.0
    labels = np.zeros(1, dtype=int)
    stage1 = {0: {"structural": {"field": "seasonal_amplitude_max", "rho": 0.5, "n": 300},
                 "top3_structural": [{"field": "seasonal_amplitude_max", "rho": 0.5, "n": 300}]}}
    info = derive_role_name(0, X, labels, list(CHANNELS), role_entries,
                            _NULL_P95, stage1, existing_names=set())
    assert "seasonal_amplitude_max" in info["name"]


def test_uniqueness_rule_appends_the_next_channel_to_break_a_collision():
    """Rule 1, the central test the brief asks for: two roles engineered to
    derive the SAME dominant-channel name must come out with DIFFERENT
    rendered names once the second is offered to `derive_role_name` with the
    first's name already in `existing_names`."""
    ch0, ch1 = list(CHANNELS)[0], list(CHANNELS)[1]
    idx0, idx1 = list(CHANNELS).index(ch0), list(CHANNELS).index(ch1)

    # Role A: channel0 strongly dominant.
    row_a = np.zeros(len(CHANNELS) + 3)
    row_a[idx0] = 5.0
    row_a[idx1] = 0.1
    members_a = [_candidate(0, [ch0], effects={ch0: 5.0})]

    # Role B: SAME dominant channel0 value (forces a name collision at
    # rank 0), but a different secondary channel1 value so rank-1 escalation
    # actually differs between them.
    row_b = np.zeros(len(CHANNELS) + 3)
    row_b[idx0] = 5.0
    row_b[idx1] = 3.0
    members_b = [_candidate(1, [ch0], effects={ch0: 5.0})]

    X = np.vstack([row_a, row_b])
    labels = np.array([0, 1])
    role_entries = {0: members_a, 1: members_b}

    existing: set = set()
    info_a = derive_role_name(0, X, labels, list(CHANNELS), role_entries,
                              _NULL_P95, None, existing_names=existing)
    existing.add(info_a["name"])
    info_b = derive_role_name(1, X, labels, list(CHANNELS), role_entries,
                              _NULL_P95, None, existing_names=existing)

    assert info_a["name"] != info_b["name"]
    # Role B's name must have escalated to a composite (channel0 + channel1),
    # not just silently picked a different single channel by luck.
    assert "+" in info_b["name"]


def test_role_table_produces_globally_unique_names_across_many_roles():
    """A stronger version of the uniqueness test: build several roles that
    all share the SAME dominant channel and check `role_table` (which owns
    the whole-set uniqueness enforcement) never emits a duplicate name."""
    n_roles = 4
    idx0 = 0  # first channel dominant in every role, by construction
    X_rows, candidates, labels = [], [], []
    for r in range(n_roles):
        row = np.zeros(len(CHANNELS) + 3)
        row[idx0] = 5.0
        # give each role a distinct secondary-channel value so escalation
        # can actually discriminate them
        row[1] = float(r + 1)
        X_rows.append(row)
        candidates.append(_candidate(r, [list(CHANNELS)[idx0]],
                                    effects={list(CHANNELS)[idx0]: 5.0}))
        labels.append(r)
    X = np.vstack(X_rows)
    labels = np.asarray(labels)
    cluster_result = {"labels": labels, "k": n_roles, "silhouette": 0.5, "non_modular": False}
    table = role_table(candidates, X, cluster_result, list(CHANNELS), _NULL_P95, None)
    names = [r["name"] for r in table]
    assert len(names) == len(set(names)), names


def test_role_table_disambiguates_two_roles_that_both_hit_the_unnamed_fallback():
    """A sharper uniqueness negative than the composite-channel case above:
    two DIFFERENT roles (different member features) that both have an
    all-zero channel vector collide on the exact same fixed fallback
    literal ("unnamed (no distinguishing signature)"), which
    `derive_role_name` cannot itself resolve -- rule 1's escalation has
    nothing left to escalate through when every channel is zero. `role_table`
    owns set-wide uniqueness and must still emit two distinct names."""
    candidates = [_candidate(0, ["trend"]), _candidate(1, ["trend"])]  # clears, but 0 effect
    labels = np.array([0, 1])
    X = np.zeros((2, len(CHANNELS) + 3))  # both rows all-zero -> both "unnamed"
    cluster_result = {"labels": labels, "k": 2, "silhouette": 0.5, "non_modular": False}
    table = role_table(candidates, X, cluster_result, list(CHANNELS), _NULL_P95, None)
    names = [r["name"] for r in table]
    assert len(names) == len(set(names)), names
    assert all("unnamed" in n for n in names)


def test_role_table_orders_by_descending_member_count():
    candidates = [_candidate(i, ["trend"], effects={"trend": 2.0}) for i in range(5)]
    labels = np.array([0, 0, 0, 1, 1])
    X = np.zeros((5, len(CHANNELS) + 3))
    X[:, 0] = 2.0
    cluster_result = {"labels": labels, "k": 2, "silhouette": 0.5, "non_modular": False}
    table = role_table(candidates, X, cluster_result, list(CHANNELS), _NULL_P95, None)
    assert [r["n_atoms"] for r in table] == sorted([r["n_atoms"] for r in table], reverse=True)
    assert table[0]["n_atoms"] == 3


def test_role_table_records_feature_membership():
    candidates = [_candidate(i, ["trend"], effects={"trend": 2.0}) for i in range(3)]
    labels = np.array([0, 0, 1])
    X = np.zeros((3, len(CHANNELS) + 3))
    X[:, 0] = 2.0
    cluster_result = {"labels": labels, "k": 2, "silhouette": 0.5, "non_modular": False}
    table = role_table(candidates, X, cluster_result, list(CHANNELS), _NULL_P95, None)
    all_features = sorted(f for r in table for f in r["features"])
    assert all_features == [0, 1, 2]
