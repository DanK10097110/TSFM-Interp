"""ROADMAP.md sec 27: the causal second opinion on a cross-model feature match.

The three signals `sae/matching.py` already had all ask "do these two
features fire on the same series". This suite pins the fourth, which asks
"and do they DO the same thing" -- the distinction the whole addition exists
for, since correlated series properties make co-firing common between atoms
with unrelated causal roles.

Three of these are load-bearing negatives:

  * per-channel null normalization is not cosmetic -- the plant makes the
    normalized and unnormalized answers DISAGREE, so an implementation that
    skipped it would pass every other assertion here;
  * a feature that cleared no channel has no causal identity, and must
    report `available: False` with a reason rather than a cosine computed
    off a noise direction (`CLAUDE.md` sec 11.37);
  * "could not be scored" is a third outcome, never folded into
    "disagrees".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.matching import (
    add_causal_agreement,
    causal_agreement,
    causal_fingerprint,
    causal_permutation_null,
)
from tsfm_lens.sae.response import CHANNELS

_CH = list(CHANNELS)


def _entry(feature: int, signed: dict, p95: dict, clearing: int | None = None) -> dict:
    """One `*_ablation.json` candidate record. `clearing` defaults to the
    number of channels whose |signed| exceeds its own p95, i.e. what the
    real pass would have counted."""
    chans = {}
    auto = 0
    for ch in CHANNELS:
        if ch not in signed:
            chans[ch] = {"available": False, "reason": "not in this fixture",
                         "effect": None, "signed_effect": None, "null_p95": None,
                         "clears_null": False, "margin": None}
            continue
        s, n = float(signed[ch]), float(p95[ch])
        clears = abs(s) > n
        auto += int(clears)
        chans[ch] = {"available": True, "effect": abs(s), "signed_effect": s,
                     "null_p95": n, "clears_null": clears,
                     "margin": abs(s) - n}
    return {"feature": feature, "scorable": True, "channels": chans,
            "n_channels_clearing": auto if clearing is None else clearing}


def _artifact(entries: list) -> dict:
    return {"withheld": False, "candidates": entries}


# ---------------------------------------------------------------------------
# 1. The fingerprint itself
# ---------------------------------------------------------------------------

def test_fingerprint_is_normalized_by_each_channels_own_null():
    e = _entry(0, {"trend": 4.0, "level": 50.0}, {"trend": 1.0, "level": 100.0})
    fp = causal_fingerprint(e)
    v = np.asarray(fp["vector"])
    assert fp["available"] is True
    assert v[CHANNELS.index("trend")] == pytest.approx(4.0)
    assert v[CHANNELS.index("level")] == pytest.approx(0.5)
    # Unmeasured channels contribute nothing rather than a fabricated zero
    # effect that happens to look the same -- the magnitude below is over
    # the two measured channels only.
    assert fp["magnitude"] == pytest.approx(np.hypot(4.0, 0.5))


def test_fingerprint_is_withheld_when_the_feature_cleared_no_channel():
    # Large raw numbers, none of which beat their own null. Direction here is
    # a direction in noise; a cosine against it would look like a finding.
    e = _entry(1, {"trend": 90.0, "level": 90.0}, {"trend": 100.0, "level": 100.0})
    fp = causal_fingerprint(e)
    assert fp["available"] is False
    assert "not distinguishable from noise" in fp["reason"]
    assert fp["magnitude"] is not None  # recorded, just not usable


def test_fingerprint_of_a_missing_or_unscorable_feature_names_why():
    assert causal_fingerprint(None)["available"] is False
    assert "no ablation measurement" in causal_fingerprint(None)["reason"]
    unscorable = {"feature": 3, "scorable": False, "reason": "fires on no series"}
    assert causal_fingerprint(unscorable)["reason"] == "fires on no series"


# ---------------------------------------------------------------------------
# 2. Normalization changes the answer -- the load-bearing negative
# ---------------------------------------------------------------------------

def test_per_channel_normalization_flips_the_verdict_against_raw_magnitudes():
    # Side A's `level` channel is 100x noisier than its `trend` channel, so a
    # level effect of 50 is SMALLER in null units than a trend effect of 4.
    # Side B has equal nulls. Raw, both look level-dominated and agree; in
    # null units A is trend-dominated and B is level-dominated, and they do
    # not.
    a = _entry(0, {"trend": 4.0, "level": 50.0}, {"trend": 1.0, "level": 100.0})
    b = _entry(0, {"trend": 0.5, "level": 5.0}, {"trend": 1.0, "level": 1.0})

    normalized = causal_agreement(causal_fingerprint(a), causal_fingerprint(b))["cosine"]

    raw_a = np.array([4.0, 50.0])
    raw_b = np.array([0.5, 5.0])
    raw = float(raw_a @ raw_b / (np.linalg.norm(raw_a) * np.linalg.norm(raw_b)))

    assert raw > 0.99          # unnormalized: near-perfect agreement
    assert normalized < 0.75   # in null units: clearly not the same role
    assert raw - normalized > 0.25


def test_opposite_causal_directions_give_a_negative_cosine():
    a = _entry(0, {"trend": 5.0, "level": 5.0}, {"trend": 1.0, "level": 1.0})
    b = _entry(0, {"trend": -5.0, "level": -5.0}, {"trend": 1.0, "level": 1.0})
    ca = causal_agreement(causal_fingerprint(a), causal_fingerprint(b))
    assert ca["available"] is True
    assert ca["cosine"] == pytest.approx(-1.0)
    # Sign is the whole point: an unsigned fingerprint would call a feature
    # that raises the level and one that lowers it identical.


def test_cosine_reports_both_magnitudes_because_it_discards_them():
    a = _entry(0, {"trend": 5.0, "level": 5.0}, {"trend": 1.0, "level": 1.0})
    b = _entry(0, {"trend": 50.0, "level": 50.0}, {"trend": 1.0, "level": 1.0})
    ca = causal_agreement(causal_fingerprint(a), causal_fingerprint(b))
    assert ca["cosine"] == pytest.approx(1.0)
    assert ca["magnitude_b"] > 9 * ca["magnitude_a"]


# ---------------------------------------------------------------------------
# 3. The null and the three-way verdict
# ---------------------------------------------------------------------------

def _pair_match(feature_a: int, feature_b: int) -> dict:
    return {"feature_a": feature_a, "feature_b": feature_b, "field_a": "x",
            "field_b": "x", "rho_a": 0.5, "rho_b": 0.5, "correlation": 0.9,
            "jaccard": 0.5, "ground_truth_agree": True, "score": 1.9}


_P1 = {ch: 1.0 for ch in CHANNELS}


def _populations():
    """Six features a side, sized so the permutation null has a pool it can
    actually resolve (see `_MIN_NULL_POOL`): 6 usable A x 5 usable B minus
    the 3 matched pairings = 27 >= 20. The fillers are deliberately non-
    parallel mixtures rather than one-hots, so no pair of them sits at
    |cosine| 1.0 and drags the p95 to the ceiling.
    """
    art_a = _artifact([
        _entry(0, {"trend": 5.0, "level": 5.0}, _P1),
        _entry(1, {"trend": 5.0, "level": -5.0}, _P1),
        _entry(2, {"spectral_centroid": 5.0, "mase": 1.5}, _P1),
        _entry(3, {"dispersion": 5.0, "flatness": 1.0}, _P1),
        _entry(4, {"mase": 5.0, "trend": 2.0}, _P1),
        _entry(5, {"flatness": 5.0, "spectral_centroid": 2.5}, _P1),
    ])
    art_b = _artifact([
        _entry(0, {"trend": 5.0, "level": 5.0}, _P1),          # identical to A0
        _entry(1, {"flatness": 5.0}, _P1),                     # orthogonal to A0
        _entry(2, {"trend": 90.0, "level": 90.0},              # clears nothing
               {ch: 100.0 for ch in CHANNELS}),
        _entry(3, {"spectral_centroid": 1.5, "mase": 5.0}, _P1),
        _entry(4, {"dispersion": 1.0, "flatness": 5.0}, _P1),
        _entry(5, {"mase": 2.0, "trend": 5.0}, _P1),
    ])
    return art_a, art_b


def test_agree_disagree_and_unscorable_are_three_separate_outcomes():
    art_a, art_b = _populations()
    result = {"matched": [_pair_match(0, 0), _pair_match(0, 1), _pair_match(0, 2)]}
    out = add_causal_agreement(result, art_a, art_b)

    verdicts = [p["causal"]["verdict"] for p in out["matched"]]
    assert verdicts == ["same causal role", "fires together, acts differently",
                        "not scorable"]
    c = out["causal"]
    assert (c["n_agree"], c["n_disagree"], c["n_unscorable"]) == (1, 1, 1)
    # The unscorable pair must not have been counted as a disagreement -- an
    # absent measurement voting against the match is exactly the failure the
    # third outcome exists to prevent.
    assert c["n_agree"] + c["n_disagree"] + c["n_unscorable"] == 3


def test_unscorable_pair_names_which_side_failed_and_why():
    art_a, art_b = _populations()
    out = add_causal_agreement({"matched": [_pair_match(0, 2)]}, art_a, art_b)
    reason = out["matched"][0]["causal"]["reason"]
    assert reason.startswith("side B:")
    assert "not distinguishable from noise" in reason


def test_permutation_null_is_estimated_from_both_populations():
    art_a, art_b = _populations()
    fps_a = [causal_fingerprint(e) for e in art_a["candidates"]]
    fps_b = [causal_fingerprint(e) for e in art_b["candidates"]]
    null = causal_permutation_null(fps_a, fps_b, n_samples=200, seed=0)
    assert 0.0 <= null["p95"] <= 1.0
    # B's third feature cleared nothing, so it is excluded from the pool:
    # 3 usable A x 2 usable B.
    # B's third feature cleared nothing and is excluded: 6 usable A x 5
    # usable B, with no matched pairs excluded in this direct call.
    assert null["n_pool"] == 30
    assert null["resolvable"] is True


def test_null_that_cannot_be_estimated_is_none_not_zero():
    art_a, _ = _populations()
    empty = _artifact([_entry(9, {"trend": 1.0}, {"trend": 100.0})])  # clears nothing
    fps_a = [causal_fingerprint(e) for e in art_a["candidates"]]
    fps_b = [causal_fingerprint(e) for e in empty["candidates"]]
    null = causal_permutation_null(fps_a, fps_b)
    assert null["p95"] is None and null["n_pool"] == 0
    assert null["resolvable"] is False


# ---------------------------------------------------------------------------
# 4. Degrading when the pass never ran
# ---------------------------------------------------------------------------

def test_missing_ablation_artifact_degrades_with_a_runnable_hint():
    out = add_causal_agreement({"matched": [_pair_match(0, 0)]}, None, None)
    assert out["causal"]["available"] is False
    assert "run_sae_ablation.py" in out["causal"]["hint"]
    assert "causal" not in out["matched"][0]


def test_withheld_target_is_reported_as_withheld_not_as_disagreement():
    art_a, _ = _populations()
    withheld = {"withheld": True, "reach": {"reason": "patch never lands"}}
    out = add_causal_agreement({"matched": [_pair_match(0, 0)]}, art_a, withheld)
    assert out["causal"]["available"] is False
    assert "patch never lands" in out["causal"]["reason"]


def test_untrained_twin_floor_is_named_as_unmeasured_rather_than_omitted():
    art_a, art_b = _populations()
    out = add_causal_agreement({"matched": [_pair_match(0, 0)]}, art_a, art_b)
    c = out["causal"]
    assert "untrained_twin_floor" in c and c["untrained_twin_floor"] is None
    assert "random_init" in c["untrained_twin_floor_reason"]


def test_a_pool_too_small_to_resolve_a_p95_says_so_instead_of_thresholding():
    # Two usable features a side minus the matched pair leaves 3 cross-pairs.
    # A 95th percentile cannot exclude the top 5% of 3, so no pair could
    # clear it at any effect size -- the same arithmetic as sec 6.6's Holm
    # p-floor. Reporting the threshold anyway would turn "we could not test
    # this" into "nothing agreed".
    tiny_a = _artifact([_entry(0, {"trend": 5.0, "level": 5.0}, _P1),
                        _entry(1, {"flatness": 5.0}, _P1)])
    tiny_b = _artifact([_entry(0, {"trend": 5.0, "level": 5.0}, _P1),
                        _entry(1, {"mase": 5.0}, _P1)])
    out = add_causal_agreement({"matched": [_pair_match(0, 0)]}, tiny_a, tiny_b)
    c = out["causal"]
    assert c["null_resolvable"] is False
    assert c["null_p95"] is None
    assert c["n_agree"] == 0 and c["n_disagree"] == 0 and c["n_unscorable"] == 1
    assert "at any effect size" in out["matched"][0]["causal"]["reason"]


def test_a_degenerate_null_channel_contributes_zero_not_infinity():
    # `null_p95` of exactly 0 is a real state on quantized decoders. Dividing
    # the signed effect by it would put an inf into the fingerprint and make
    # every cosine involving that feature NaN.
    e = _entry(0, {"trend": 4.0, "level": 3.0}, {"trend": 0.0, "level": 1.0},
               clearing=1)
    fp = causal_fingerprint(e)
    v = np.asarray(fp["vector"])
    assert np.isfinite(v).all()
    assert v[CHANNELS.index("trend")] == 0.0
    assert v[CHANNELS.index("level")] == pytest.approx(3.0)


def test_matched_pairs_are_excluded_from_the_null_they_must_beat():
    # A null estimated partly FROM the pairs under test raises the very
    # threshold those pairs have to clear -- a perfect match makes itself
    # fail. `role_matching.permutation_null_cosine` excludes the targets
    # under test for this reason; here the population IS the pair, so the
    # matched pairings are what gets excluded.
    #
    # Sized so the two answers differ: three identical matched pairs are 3 of
    # 25 cross-pairs (12%, above the 5% a p95 discards), so including them
    # puts the p95 at 1.0 and nothing can clear it. Every other cross-pair
    # here is exactly orthogonal, so excluding them puts it at 0.0.
    def side(shared_channels, own_channels):
        return _artifact([_entry(i, {ch: 5.0}, _P1)
                          for i, ch in enumerate(shared_channels + own_channels)])

    shared = ["trend", "level", "spectral_centroid"]
    art_a = side(shared, ["dispersion", "mase"])
    art_b = side(shared, ["flatness", "horizon_shape_near"])
    matched = [_pair_match(i, i) for i in range(3)]

    out = add_causal_agreement({"matched": [dict(m) for m in matched]}, art_a, art_b)
    c = out["causal"]
    assert c["null_pool_pairs"] == 22        # 5 x 5 usable, minus the 3 matched
    assert c["null_resolvable"] is True
    assert c["null_p95"] == pytest.approx(0.0)
    assert (c["n_agree"], c["n_disagree"]) == (3, 0)

    # The same three pairs, scored against a null that includes them:
    fps_a = {e["feature"]: causal_fingerprint(e) for e in art_a["candidates"]}
    fps_b = {e["feature"]: causal_fingerprint(e) for e in art_b["candidates"]}
    contaminated = causal_permutation_null(fps_a, fps_b, exclude_pairs=set(), seed=0)
    assert contaminated["p95"] == pytest.approx(1.0)
    # ...against which a cosine of exactly 1.0 does not clear, so all three
    # perfect matches would have been reported as acting differently.
    assert not (1.0 > contaminated["p95"])


# ---------------------------------------------------------------------------
# ROLE granularity (ROADMAP.md sec 27) -- the surface the report renders
# ---------------------------------------------------------------------------

def _abl_doc(*feats):
    """feats: (index, {channel: (signed, p95)})"""
    cands = []
    for f, chans in feats:
        cands.append({"feature": f, "scorable": True,
                      "n_channels_clearing": len(chans),
                      "channels": {ch: {"available": True, "clears_null": True,
                                        "null_degenerate": False,
                                        "null_p95": p95, "signed_effect": s}
                                   for ch, (s, p95) in chans.items()}})
    return {"withheld": False, "candidates": cands}


def test_an_unmeasured_member_does_not_shrink_the_role_toward_zero():
    from tsfm_lens.sae.matching import role_causal_fingerprint
    role = {"role": 0, "features": [1, 2, 3]}
    doc = _abl_doc((1, {"level": (6.0, 2.0)}), (2, {"level": (6.0, 2.0)}))
    fp = role_causal_fingerprint(role, doc)
    assert fp["available"] and fp["n_members"] == 3 and fp["n_members_measured"] == 2
    # Mean over the two MEASURED members is 3.0, not 2.0 (which is what
    # counting the third as a zero vector would give -- "this role does
    # little" where the truth is "a third of it was never scored").
    assert fp["vector"][_CH.index("level")] == pytest.approx(3.0)


def test_a_role_with_no_measured_member_is_unavailable_with_a_reason():
    from tsfm_lens.sae.matching import role_causal_fingerprint
    fp = role_causal_fingerprint({"role": 0, "features": [9]}, _abl_doc())
    assert not fp["available"] and fp["reason"]
    assert fp["vector"] is None


def _role_table(cos_pairs):
    """A 1-pair correspondence table with `cos_pairs` matched role indices."""
    return {"pairs": [{"comparable": True, "model_a": "A", "model_b": "B",
                       "target_a": "A/l0", "target_b": "B/l0",
                       "matches": [{"role_a_index": ia, "role_b_index": ib}
                                   for ia, ib in cos_pairs]}]}


def test_roles_that_fire_together_but_act_oppositely_are_flagged_as_such():
    from tsfm_lens.sae.matching import add_role_causal_agreement
    # 25 roles a side so the null pool clears _MIN_NULL_POOL. Role 0 on each
    # side is the matched pair, and they push `level` in OPPOSITE directions.
    roles = {"A/l0": {"roles": [{"role": i, "features": [i]} for i in range(25)]},
             "B/l0": {"roles": [{"role": i, "features": [i]} for i in range(25)]}}
    rng = np.random.default_rng(0)
    a_feats = [(0, {"level": (6.0, 2.0)})]
    b_feats = [(0, {"level": (-6.0, 2.0)})]
    for i in range(1, 25):
        a_feats.append((i, {"seasonal": (float(rng.normal()) * 4, 2.0),
                            "trend": (float(rng.normal()) * 4, 2.0)}))
        b_feats.append((i, {"seasonal": (float(rng.normal()) * 4, 2.0),
                            "trend": (float(rng.normal()) * 4, 2.0)}))
    abl = {"A/l0": _abl_doc(*a_feats), "B/l0": _abl_doc(*b_feats)}
    out = add_role_causal_agreement(_role_table([(0, 0)]), roles, abl)
    m = out["pairs"][0]["matches"][0]
    assert m["causal"]["verdict"] == "fires together, acts differently"
    assert m["causal"]["cosine"] < 0
    assert out["pairs"][0]["causal_summary"]["n_disagree"] == 1


def test_roles_pushing_the_same_way_agree():
    from tsfm_lens.sae.matching import add_role_causal_agreement
    roles = {"A/l0": {"roles": [{"role": i, "features": [i]} for i in range(25)]},
             "B/l0": {"roles": [{"role": i, "features": [i]} for i in range(25)]}}
    rng = np.random.default_rng(1)
    a_feats = [(0, {"level": (6.0, 2.0)})]
    b_feats = [(0, {"level": (6.0, 2.0)})]
    for i in range(1, 25):
        a_feats.append((i, {"seasonal": (float(rng.normal()) * 4, 2.0),
                            "trend": (float(rng.normal()) * 4, 2.0)}))
        b_feats.append((i, {"seasonal": (float(rng.normal()) * 4, 2.0),
                            "trend": (float(rng.normal()) * 4, 2.0)}))
    abl = {"A/l0": _abl_doc(*a_feats), "B/l0": _abl_doc(*b_feats)}
    out = add_role_causal_agreement(_role_table([(0, 0)]), roles, abl)
    assert out["pairs"][0]["matches"][0]["causal"]["verdict"] == "same causal role"


def test_a_role_rate_is_never_marked_quotable_without_its_own_twin_floor():
    from tsfm_lens.sae.matching import add_role_causal_agreement
    roles = {"A/l0": {"roles": [{"role": 0, "features": [0]}]},
             "B/l0": {"roles": [{"role": 0, "features": [0]}]}}
    abl = {"A/l0": _abl_doc((0, {"level": (6.0, 2.0)})),
           "B/l0": _abl_doc((0, {"level": (6.0, 2.0)}))}
    s = add_role_causal_agreement(_role_table([(0, 0)]), roles, abl)["pairs"][0]["causal_summary"]
    assert s["quotable"] is False and s["untrained_twin_floor"] is None
    assert "random_init" in s["quotable_reason"]


def test_a_tiny_role_population_is_not_scorable_rather_than_disagreeing():
    # Two roles a side leaves at most 3 cross-pairs after excluding the
    # matched one -- a p95 cannot exclude the top 5% of that, so the pair
    # must come back "not scorable" and NOT as a failure to clear.
    from tsfm_lens.sae.matching import add_role_causal_agreement
    roles = {"A/l0": {"roles": [{"role": i, "features": [i]} for i in range(2)]},
             "B/l0": {"roles": [{"role": i, "features": [i]} for i in range(2)]}}
    abl = {"A/l0": _abl_doc((0, {"level": (6.0, 2.0)}), (1, {"level": (4.0, 2.0)})),
           "B/l0": _abl_doc((0, {"level": (6.0, 2.0)}), (1, {"level": (4.0, 2.0)}))}
    out = add_role_causal_agreement(_role_table([(0, 0)]), roles, abl)
    m = out["pairs"][0]["matches"][0]
    assert m["causal"]["verdict"] == "not scorable"
    assert out["pairs"][0]["causal_summary"]["n_disagree"] == 0


def test_the_correlational_match_rate_is_left_untouched():
    from tsfm_lens.sae.matching import add_role_causal_agreement
    table = _role_table([(0, 0)])
    table["pairs"][0]["match_rate"] = 0.833
    table["pairs"][0]["match_rate_quotable"] = True
    roles = {"A/l0": {"roles": [{"role": 0, "features": [0]}]},
             "B/l0": {"roles": [{"role": 0, "features": [0]}]}}
    abl = {"A/l0": _abl_doc((0, {"level": (6.0, 2.0)})),
           "B/l0": _abl_doc((0, {"level": (-6.0, 2.0)}))}
    out = add_role_causal_agreement(table, roles, abl)
    # A flat causal disagreement must not move the published number.
    assert out["pairs"][0]["match_rate"] == 0.833
    assert out["pairs"][0]["match_rate_quotable"] is True


def test_a_collinear_population_saturates_the_null_and_is_named_as_such():
    # Every distractor loads on one channel, so an ARBITRARY cross-pair
    # already has |cosine| 1.0 and the p95 sits at the bound. A perfectly
    # matched pair then cannot clear it -- which must be reported as an
    # unusable null, not as the pair acting differently.
    from tsfm_lens.sae.matching import add_role_causal_agreement
    roles = {"A/l0": {"roles": [{"role": i, "features": [i]} for i in range(25)]},
             "B/l0": {"roles": [{"role": i, "features": [i]} for i in range(25)]}}
    rng = np.random.default_rng(0)
    feats = lambda: [(i, {"seasonal": (float(rng.normal()) * 4, 2.0)})
                     for i in range(1, 25)]
    abl = {"A/l0": _abl_doc((0, {"level": (6.0, 2.0)}), *feats()),
           "B/l0": _abl_doc((0, {"level": (6.0, 2.0)}), *feats())}
    out = add_role_causal_agreement(_role_table([(0, 0)]), roles, abl)
    m = out["pairs"][0]["matches"][0]
    assert m["causal"]["verdict"] == "not scorable"
    assert m["causal"]["cosine"] == pytest.approx(1.0)     # a perfect match...
    assert out["pairs"][0]["causal_summary"]["n_disagree"] == 0   # ...not a disagreement
    assert out["pairs"][0]["causal_summary"]["null"]["null_saturated"] is True


def test_matching_imports_cleanly_when_role_matching_loads_first():
    # The two modules import each other. Every test in this package imports
    # `matching` first, which resolves the cycle; `report.py` imports
    # `role_matching` first, which did not. A module-level
    # `from .role_matching import sign_aware_cosine` passed the whole suite
    # and failed the real CLI.
    import subprocess, sys as _s
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    r = subprocess.run(
        [_s.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "import tsfm_lens.sae.role_matching;"
         "import tsfm_lens.sae.matching as m;"
         "assert hasattr(m, 'add_role_causal_agreement')" % str(root)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
