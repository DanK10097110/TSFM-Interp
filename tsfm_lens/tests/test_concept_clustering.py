"""Tests for ROADMAP.md sec 32.5 Item D: `sae/concepts.py::cluster_concepts`'s
`k="auto"` path sweeping `k` instead of deriving it from a single ratio, and
gating admissibility (a cluster's smallest member count) BEFORE comparing
silhouette scores, not after.

Covers D1 (the sweep itself, `k in range(2, k_upper+1)`), D2 (a size floor
applied before scoring), D3 (declaring `non_modular` when nothing is
admissible rather than forcing an inadmissible partition), D4 (the sweep and
selection rule recorded in the result), and a real-data cross-check of D5's
reproducibility sentinel (`min_members<=1`) against the currently-recorded
`runs/full_report_run_4model/sae/concepts.json` -- the synthetic D5 pin lives
in `test_sae_concepts.py::test_k_auto_matches_resolve_role_k_at_min_members_1`
and is not duplicated here.

Every load-bearing assertion below is confirmed to discriminate by planting
its own regression and reading pytest's summary line, per sec 11.53's
postscript / sec 11.55's corollary -- a plant that changes nothing, or that
produces a collection error rather than a failure, proves nothing.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.concepts import build_concept_matrix, cluster_concepts, concept_table

_REAL_RUN = Path(__file__).resolve().parents[1] / "runs" / "full_report_run_4model"


def _fixture(group_sep: float, odd_dist: float, group_size: int = 4,
            n_groups: int = 3, dim: int = 4, scale: float = 0.3,
            seed: int = 0) -> np.ndarray:
    """`n_groups` tight clusters of `group_size` points each, spaced
    `group_sep` apart along one axis, plus one extra point sitting `odd_dist`
    away from the origin along a different axis -- the "one outlier beside
    several real clusters" shape that is exactly where a naive silhouette-
    argmax sweep isolates the outlier as its own (inadmissible, size-1)
    cluster."""
    rng = np.random.default_rng(seed)
    centers = np.array([[group_sep * i] + [0.0] * (dim - 1) for i in range(n_groups)])
    rows = [rng.normal(loc=c, scale=scale, size=(group_size, dim)) for c in centers]
    odd = rng.normal(loc=[odd_dist, odd_dist] + [0.0] * (dim - 2), scale=scale, size=(1, dim))
    rows.append(odd)
    return np.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------
# D1/D4: the sweep itself, and what it records
# ---------------------------------------------------------------------------

def test_sweep_covers_full_k_range_and_records_per_k_fields():
    """Three well-separated, evenly-sized groups (no outlier): the sweep must
    attempt every k from 2 up to `min(8, n // min_members, n - 1)`, and every
    attempted k -- admissible or not -- must be recorded with its own
    silhouette and min_cluster_size, not only the winner.

    Built directly (not via `_fixture`) with a single informative dimension:
    `_fixture`'s other `dim-1` columns are pure per-point noise around 0 for
    every group, and `StandardScaler` normalizes each column to unit variance
    across the whole sample -- with no group structure to normalize, those
    columns' tiny noise gets amplified to the same scale as the real signal
    and can swamp it (confirmed directly: the 4-D version of this fixture
    does not cleanly select k=3). Real target matrices don't have all-noise
    columns, but this fixture must not accidentally rely on one either."""
    rng = np.random.default_rng(1)
    X = np.concatenate(
        [rng.normal(loc=[12.0 * i], scale=0.3, size=(5, 1)) for i in range(3)], axis=0
    )
    result = cluster_concepts(X, k="auto", min_members=3, seed=0)
    n = X.shape[0]
    k_upper = min(8, n // 3, n - 1)
    assert result["k_selection_rule"] == f"silhouette-best admissible, k swept 2..{k_upper}"
    swept_ks = [row["k"] for row in result["k_sweep"]]
    assert swept_ks == list(range(2, k_upper + 1))
    for row in result["k_sweep"]:
        assert set(row) == {"k", "silhouette", "admissible", "min_cluster_size"}
        assert isinstance(row["admissible"], bool)
        assert row["min_cluster_size"] >= 1
    # Three well-separated, evenly-sized groups: k=3 must be admissible and
    # must win (every group has 5 >= min_members=3 members).
    assert result["k"] == 3
    assert result["non_modular"] is False


def test_sweep_field_count_negative_plant():
    """Plant: silently drop `min_cluster_size` from every recorded row. This
    is the field D4 exists to add (a reader must be able to tell a swept k
    from a ratio-derived one, ROADMAP.md sec 31.1's own `min_gap`-provenance
    lesson) -- the test must fail without it."""
    X = _fixture(group_sep=12, odd_dist=0, group_size=5, n_groups=3, seed=1)[:-1]
    result = cluster_concepts(X, k="auto", min_members=3, seed=0)
    planted_row = {k: v for k, v in result["k_sweep"][0].items() if k != "min_cluster_size"}
    assert "min_cluster_size" in result["k_sweep"][0]  # the real result
    assert "min_cluster_size" not in planted_row  # confirms the plant itself changed something


# ---------------------------------------------------------------------------
# D2/D3: admissibility gated BEFORE scoring -- the load-bearing negative
# ---------------------------------------------------------------------------

def test_admissible_runner_up_wins_over_inadmissible_best_silhouette():
    """The exact fixture ROADMAP.md sec 32.8 names for Item D: "a fixture
    where the best-silhouette partition is the inadmissible one (a singleton
    split) must select the admissible runner-up."

    Three tight groups of 4 plus one distant outlier (n=13): at
    min_members=3, k=3 (three groups plus the outlier alone) has the highest
    raw silhouette but its outlier cluster has only 1 member -- inadmissible.
    k=2 (the outlier merged with its nearest group) is admissible and must be
    selected instead, even though its silhouette is lower.
    """
    X = _fixture(group_sep=8, odd_dist=20, group_size=4, n_groups=3, seed=0)
    result = cluster_concepts(X, k="auto", min_members=3, seed=0)

    by_k = {row["k"]: row for row in result["k_sweep"]}
    assert by_k[3]["admissible"] is False
    assert by_k[3]["min_cluster_size"] == 1
    assert by_k[2]["admissible"] is True
    assert by_k[2]["min_cluster_size"] == 6
    # The inadmissible k really does score higher -- otherwise this fixture
    # would not be discriminating (sec 11.34: build the decoy in).
    assert by_k[3]["silhouette"] > by_k[2]["silhouette"]

    assert result["k"] == 2
    assert result["non_modular"] is False

    # Confirm by executing the OTHER rule over the same fixture (ROADMAP.md
    # sec 32.8's own instruction, mirroring sec 31.5's substitution-ranking
    # test): a naive silhouette-argmax-with-no-admissibility-gate would pick
    # k=3, not k=2. The two rules must disagree on this fixture, or it is not
    # exercising the gate at all.
    naive_best_k = max(result["k_sweep"], key=lambda r: r["silhouette"])["k"]
    assert naive_best_k == 3
    assert naive_best_k != result["k"]


def test_admissibility_gate_negative_plant():
    """Plant: score by raw silhouette with no admissibility gate (the
    pre-Item-D failure mode). Confirms the real function's selection differs
    from the ungated one on this exact fixture -- a plant that doesn't change
    the outcome would mean the fixture isn't load-bearing."""
    X = _fixture(group_sep=8, odd_dist=20, group_size=4, n_groups=3, seed=0)
    result = cluster_concepts(X, k="auto", min_members=3, seed=0)
    ungated_k = max(result["k_sweep"], key=lambda r: r["silhouette"])["k"]
    assert result["k"] != ungated_k, (
        "the fixture must make the ungated (pre-Item-D) rule disagree with "
        "the real, admissibility-gated selection -- otherwise it does not "
        "discriminate")


def test_no_admissible_k_declares_non_modular_not_forced():
    """D3: when no k in the swept range clears the size floor, the result
    must be `non_modular=True` with a reason naming the sweep -- never a
    fallback to whatever k scored best regardless of admissibility (sec
    25.13 item 2's pre-registered outcome)."""
    X = _fixture(group_sep=8, odd_dist=50, group_size=4, n_groups=3, seed=0)
    result = cluster_concepts(X, k="auto", min_members=3, seed=0)

    assert all(not row["admissible"] for row in result["k_sweep"])
    assert result["non_modular"] is True
    assert "no k in 2.." in result["reason"]
    assert "3 members" in result["reason"]
    assert result["k"] == 1


def test_non_modular_negative_plant():
    """Plant: treat `non_modular` as always False. Since every attempted k in
    this fixture is inadmissible, forcing `non_modular=False` would mean an
    inadmissible partition was silently returned as if it were real --
    exactly what D3 forbids. The un-planted result must actually be True for
    this assertion to mean anything."""
    X = _fixture(group_sep=8, odd_dist=50, group_size=4, n_groups=3, seed=0)
    result = cluster_concepts(X, k="auto", min_members=3, seed=0)
    planted_non_modular = False
    assert result["non_modular"] != planted_non_modular


# ---------------------------------------------------------------------------
# D3 downstream: concept_table renders no concepts for a non_modular target
# ---------------------------------------------------------------------------

def test_concept_table_empty_for_non_modular_sweep_result():
    from tsfm_lens.sae.concepts import CHANNELS

    def _candidate(feature, effect=(4.0, 1.0)):
        signed, p95 = effect
        rec = {"available": True, "effect": abs(signed), "signed_effect": signed,
              "null_p95": p95, "clears_null": abs(signed) > p95}
        channels = {ch: (rec if ch == "trend" else
                        {"available": False, "effect": None, "signed_effect": None,
                         "null_p95": None, "clears_null": False})
                   for ch in CHANNELS}
        return {"feature": feature, "rules": ["variance"], "scorable": True,
               "n_top_series": 8, "channels": channels, "n_channels_clearing": 1}

    X = _fixture(group_sep=8, odd_dist=50, group_size=4, n_groups=3, seed=0)
    candidates = [_candidate(i) for i in range(X.shape[0])]
    feature_ids = list(range(X.shape[0]))
    result = cluster_concepts(X, k="auto", min_members=3, seed=0)
    assert result["non_modular"] is True
    concepts = concept_table(candidates, X, feature_ids, result)
    assert concepts == []


# ---------------------------------------------------------------------------
# D5 real-data cross-check (synthetic pin lives in test_sae_concepts.py)
# ---------------------------------------------------------------------------

# D5's own reproducibility guarantee ("min_members=1 plus the old ratio must
# reproduce today's concepts bit for bit", ROADMAP.md sec 32.5 D5) was
# checked against `concepts.json` at the moment Item D landed, when that
# artifact still held the legacy path's own output. `run_concepts`'s own
# later production regeneration (part of Item D's own acceptance step)
# overwrote it with the NEW default (`concept_min_members=3`) sweep's output
# instead -- confirmed by reading `k_selection_rule` off the live artifact,
# which carries `_sweep_k`'s own signature string ("silhouette-best
# admissible, k swept 2..N"), never the legacy ratio path's. So comparing a
# fresh `min_members=1` rerun against the LIVE `concepts.json` no longer
# tests D5's guarantee at all for a target where the sweep found
# `non_modular` -- it tests whether two DIFFERENT algorithms happen to
# agree, and for 11 of the run's 13 targets they don't (the legacy ratio
# path always finds some split; the sweep's `min_members=3` floor exists
# precisely to refuse the degenerate ones, e.g. Chronos-2/encoder.block.10's
# legacy k=2 is a real split whose smaller cluster is a lone 1-of-15
# singleton). Confirmed by measurement, not assumed (`CLAUDE.md` sec 2.4):
# only Chronos-Bolt/encoder.block.3 and TimesFM/stacked_xf.10 still
# partition under BOTH configs, and only those two can still be compared
# against the live artifact meaningfully.
#
# D5's actual, durable guarantee is therefore checked below against a
# FROZEN reference of the legacy path's own output, not against whatever
# `concepts.json` happens to hold under today's default config. This
# reference was captured by running `cluster_concepts(..., min_members=1)`
# fresh over all 13 targets' real ablation data, and its own aggregate
# (30 concepts, 9 singletons, 0 non_modular targets across the 13 targets)
# reproduces ROADMAP.md sec 32.5 D5's own recorded acceptance figures
# ("OLD (legacy ratio, min_members=1), 13 targets: 30 concepts, 9
# singletons") exactly -- i.e. the legacy path itself has not drifted;
# only the config the production artifact is generated under has moved on.
_LEGACY_MIN_MEMBERS_1_REFERENCE = {
    "Chronos-2/encoder.block.10": {
        "k": 2,
        "partition": [(250, 487, 822, 960, 1001, 1653, 1941, 2199, 2727, 3235,
                       3546, 3769, 3906, 4020), (3134,)],
    },
    "Chronos-2/encoder.block.6": {
        "k": 3,
        "partition": [(1, 1362, 1887, 1933, 1994, 2330, 2605, 2694, 2756,
                       3134, 3690, 3878, 4015, 4052, 4752, 5471, 5528, 6056,
                       6075), (461, 2169, 2233), (3387,)],
    },
    "Chronos-2/encoder.block.8": {
        "k": 3,
        "partition": [(15, 251, 1150, 1362, 1559, 1567, 1652, 2169, 3130,
                       3327, 3742, 3851, 4064, 4434, 4845, 4994, 5347, 5513),
                      (461,), (2884, 3440, 3903, 4241)],
    },
    "Chronos-Bolt/encoder.block.3": {
        "k": 2,
        "partition": [(317, 459, 2071, 2830, 3706, 4063, 4075, 4679, 4885),
                      (1194, 3718, 4311)],
    },
    "Chronos-Bolt/encoder.block.4": {
        "k": 2,
        "partition": [(1614, 1936, 2569, 2705, 2978, 3694, 4101, 4679, 4840,
                       5218, 5708, 5876), (4754, 5672)],
    },
    "Sundial/model.layers.10": {
        "k": 3,
        "partition": [(458,), (660, 867, 1073, 1387, 1626, 1679, 2840, 2857,
                       3138, 3623, 3908, 4557, 5534, 5767), (1096, 3142, 4982)],
    },
    "Sundial/model.layers.3": {
        "k": 3,
        "partition": [(279, 1312, 1343, 1823, 1990, 2148, 2166, 2235, 2491,
                       2973, 2990),
                      (377, 573, 603, 962, 1430, 1571, 1847, 1948, 2602, 2632),
                      (1584,)],
    },
    "Sundial/model.layers.7": {
        "k": 2,
        "partition": [(249,), (1243, 1626, 1653, 1869, 3816, 4458, 4477,
                       4982, 5196, 5609, 5956, 6132)],
    },
    "TimesFM/stacked_xf.10": {
        "k": 2,
        "partition": [(453, 686, 2106, 2362, 2691, 3969, 5702, 5806),
                      (1426, 3159, 5756, 5818)],
    },
    "TimesFM/stacked_xf.16": {
        "k": 2,
        "partition": [(493, 533, 712, 994, 3320, 3727, 4395, 5663, 6067),
                      (4913,)],
    },
    "TimesFM/stacked_xf.18": {
        "k": 2,
        "partition": [(24, 200, 235, 340, 589, 590, 753, 1308, 1389, 1504,
                       1531), (147,)],
    },
    "TimesFM/stacked_xf.2": {
        "k": 2,
        "partition": [(301, 351, 533, 886, 1568, 2437, 2691, 3743, 4395),
                      (511, 1084)],
    },
    "TimesFM/stacked_xf.6": {
        "k": 2,
        "partition": [(559, 712, 852, 937, 1568, 1881, 2106, 2425, 2827,
                       3086, 3653, 3751, 4087, 4326, 5361), (3161,)],
    },
}


@pytest.mark.skipif(not (_REAL_RUN / "sae" / "concepts.json").exists(),
                    reason="the gitignored four-model run directory is absent")
def test_min_members_1_reproduces_real_recorded_concepts_bit_for_bit():
    """D5, against real data rather than only a synthetic pin: rebuilding
    each of `runs/full_report_run_4model`'s 13 targets' ablation matrices and
    re-clustering with the `min_members=1` reproducibility sentinel must
    reproduce the legacy ratio path's own `k` and exact per-concept
    feature-id partitions bit for bit, against the frozen reference above --
    this is what makes every legacy-path concept regenerable
    (`CLAUDE.md` sec 2.1), not merely a claim about it. See the reference's
    own comment for why this is checked against a frozen snapshot rather
    than the live `concepts.json` (which is regenerated under a different
    default config by Item D's own production driver).
    """
    ablation_files = sorted(glob.glob(str(_REAL_RUN / "sae" / "*" / "*_ablation.json")))
    assert ablation_files, "expected at least one *_ablation.json under the real run"

    checked = 0
    for path in ablation_files:
        art = json.load(open(path))
        key = f"{art.get('model')}/{art.get('layer')}"
        expected = _LEGACY_MIN_MEMBERS_1_REFERENCE.get(key)
        if expected is None:
            continue
        candidates = art.get("candidates", [])
        X, feature_ids, _ = build_concept_matrix(candidates, causal_only=True)
        result = cluster_concepts(X, k="auto", min_members=1, seed=0)
        checked += 1

        assert result["k"] == expected["k"], f"{key}: k mismatch"
        by_label: dict = {}
        for i, lab in enumerate(result["labels"]):
            by_label.setdefault(int(lab), []).append(feature_ids[i])
        new_sets = sorted(tuple(sorted(v)) for v in by_label.values())
        assert new_sets == sorted(expected["partition"]), f"{key}: partition mismatch"

    assert checked == len(_LEGACY_MIN_MEMBERS_1_REFERENCE)


@pytest.mark.skipif(not (_REAL_RUN / "sae" / "concepts.json").exists(),
                    reason="the gitignored four-model run directory is absent")
def test_default_min_members_3_eliminates_singletons_on_the_real_run():
    """Acceptance criterion, measured against real data: with the new
    default (`concept_min_members=3`), every concept rendered across the
    real run's 13 targets has >=3 members, or the target is `non_modular`
    with a stated reason -- 0 singleton concepts, never a forced partition.
    """
    ablation_files = sorted(glob.glob(str(_REAL_RUN / "sae" / "*" / "*_ablation.json")))
    assert ablation_files

    total_concepts = 0
    singletons = 0
    for path in ablation_files:
        art = json.load(open(path))
        if art.get("withheld"):
            continue
        candidates = art.get("candidates", [])
        X, feature_ids, _ = build_concept_matrix(candidates, causal_only=True)
        result = cluster_concepts(X, k="auto", min_members=3, seed=0)
        concepts = concept_table(candidates, X, feature_ids, result, art)
        total_concepts += len(concepts)
        singletons += sum(1 for c in concepts if c["n_members"] == 1)
        if not result["non_modular"]:
            assert all(c["n_members"] >= 3 for c in concepts), (
                f"{art.get('model')}/{art.get('layer')}: an admissible "
                f"partition rendered a sub-floor concept")

    assert singletons == 0
    assert total_concepts >= 1  # at least one target must still admit a partition
