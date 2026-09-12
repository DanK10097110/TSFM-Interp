"""Tests for `sae/role_matching.py` (ROADMAP.md sec 25.6 / sec 25.9 Stage 4,
Component C): cross-model role correspondence.

All synthetic, with planted answers -- the same evidentiary style as
`test_sae_roles.py`/`test_response_fingerprint.py`. Load-bearing negatives
per the task brief:
  - sign-aware cosine is NOT |cosine| -- a planted anti-correlated pair must
    score negative, not positive.
  - a match-rate number must never be rendered quotable without an
    untrained-twin floor available for at least one side.
  - a pair with no comparable-depth targets must report `comparable: False`
    with a stated reason, never a silent empty result.
  - the byte-for-byte panel/pair reproduction test, mirroring
    `tests/test_panel_pairs.py`'s own pattern: pair 0 of a 3-model panel's
    role-correspondence table must equal a direct 2-model config's table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.response import CHANNELS
from tsfm_lens.sae.role_matching import (
    greedy_match_roles,
    match_roles_across_models,
    match_roles_for_pair,
    permutation_null_cosine,
    role_correspondence_table,
    role_population_vectors,
    role_response_vector,
    shuffled_series_null_activation_profile,
    sign_aware_cosine,
    untrained_twin_role_floor,
)
from tsfm_lens.utils import save_json


def _candidate(feature: int, effects: dict | None = None) -> dict:
    """A minimal `feature_response_fingerprints`-shaped candidate record --
    mirrors `tests/test_sae_roles.py::_candidate`, kept independent (not
    imported) since the two test files must not share fixture state across a
    future edit to either.
    """
    effects = effects or {}
    up_channels, down_channels = {}, {}
    for ch in CHANNELS:
        val = effects.get(ch, 0.0)
        up_channels[ch] = {"available": True, "signed_mean": val}
        down_channels[ch] = {"available": True, "signed_mean": -val}
    return {"feature": feature, "up": {"channels": up_channels},
           "down": {"channels": down_channels}}


_NULL_P95 = {ch: 1.0 for ch in CHANNELS}


def _role(idx: int, name: str, features: list) -> dict:
    return {"role": idx, "name": name, "n_atoms": len(features), "features": features}


# ---------------------------------------------------------------------------
# 1. role_response_vector: mean signed, null-normalized effect over members.
# ---------------------------------------------------------------------------

def test_role_response_vector_averages_members_and_ignores_non_members():
    """A role with two members whose planted 'trend' effects are 2.0 and 4.0
    (null p95 = 1.0) must average to 3.0 -- and a third candidate NOT in the
    role's feature list must not move the result even though it is present
    in the passed candidate list.
    """
    candidates = [
        _candidate(0, {"trend": 2.0}),
        _candidate(1, {"trend": 4.0}),
        _candidate(2, {"trend": 100.0}),  # not a member -- must be ignored
    ]
    role = _role(0, "trend up", features=[0, 1])
    vec = role_response_vector(role, candidates, _NULL_P95)
    trend_idx = CHANNELS.index("trend")
    assert vec[trend_idx] == pytest.approx(3.0)


def test_role_response_vector_takes_larger_magnitude_direction():
    """`up`/`down` planted asymmetrically (up=1.0, down=-9.0 signed) -- the
    larger-MAGNITUDE signed value (-9.0) must be selected, not the 'up'
    value by default and not the algebraic maximum (which would wrongly
    pick +1.0).
    """
    cand = _candidate(0)
    cand["up"]["channels"]["level"] = {"available": True, "signed_mean": 1.0}
    cand["down"]["channels"]["level"] = {"available": True, "signed_mean": -9.0}
    role = _role(0, "x", features=[0])
    vec = role_response_vector(role, [cand], _NULL_P95)
    assert vec[CHANNELS.index("level")] == pytest.approx(-9.0)


# ---------------------------------------------------------------------------
# 2. sign_aware_cosine: signed, not |cosine| -- the primary load-bearing
#    negative for the whole module.
# ---------------------------------------------------------------------------

def test_sign_aware_cosine_is_negative_for_opposite_direction_roles():
    """A role pushing `trend` up by 3.0 and one pushing it down by 3.0 (same
    magnitude, opposite sign) must score cosine EXACTLY -1.0, not +1.0 --
    the whole point of 'sign-aware' (sec 25.6): these are different causal
    identities, not the same role read backwards. Any implementation using
    |cosine| would fail this test by construction.
    """
    u = np.zeros(len(CHANNELS))
    v = np.zeros(len(CHANNELS))
    u[CHANNELS.index("trend")] = 3.0
    v[CHANNELS.index("trend")] = -3.0
    assert sign_aware_cosine(u, v) == pytest.approx(-1.0)


def test_sign_aware_cosine_is_one_for_identical_vectors():
    u = np.array([1.0, 2.0, -1.0] + [0.0] * (len(CHANNELS) - 3))
    assert sign_aware_cosine(u, u.copy()) == pytest.approx(1.0)


def test_sign_aware_cosine_zero_for_degenerate_all_zero_vector():
    """A role with literally no measured effect (all-zero row) has nothing
    to correlate, positively or negatively -- must return exactly 0.0, never
    NaN (which would silently poison any downstream mean/threshold check).
    """
    u = np.zeros(len(CHANNELS))
    v = np.ones(len(CHANNELS))
    result = sign_aware_cosine(u, v)
    assert result == 0.0
    assert not np.isnan(result)


# ---------------------------------------------------------------------------
# 3. greedy_match_roles: nearest-neighbour, explicitly non-optimal.
# ---------------------------------------------------------------------------

def test_greedy_match_roles_picks_best_scoring_partner():
    """Two A-side roles, two B-side roles: A-role 0 is planted to align
    exactly with B-role 1 (cosine 1.0) and weakly with B-role 0; A-role 1
    is planted to align exactly with B-role 0. The greedy match must recover
    both correct partners.
    """
    roles_a = [_role(0, "a0", [0]), _role(1, "a1", [1])]
    roles_b = [_role(0, "b0", [0]), _role(1, "b1", [1])]
    vecs_a = [np.array([1.0, 0.0] + [0.0] * (len(CHANNELS) - 2)),
             np.array([0.0, 1.0] + [0.0] * (len(CHANNELS) - 2))]
    vecs_b = [np.array([0.0, 1.0] + [0.0] * (len(CHANNELS) - 2)),
             np.array([1.0, 0.0] + [0.0] * (len(CHANNELS) - 2))]
    matches = greedy_match_roles(roles_a, vecs_a, roles_b, vecs_b)
    assert matches[0]["role_b"] == "b1" and matches[0]["cosine"] == pytest.approx(1.0)
    assert matches[1]["role_b"] == "b0" and matches[1]["cosine"] == pytest.approx(1.0)


def test_greedy_match_is_not_an_optimal_assignment():
    """Two A-side roles that BOTH align best with the same single B-side
    role must both claim it -- greedy nearest-neighbour has no mechanism to
    prevent this, and this test pins that as the documented behavior rather
    than silently expecting a Hungarian-style one-to-one assignment.
    """
    roles_a = [_role(0, "a0", [0]), _role(1, "a1", [1])]
    roles_b = [_role(0, "b0", [0]), _role(1, "b1", [1])]
    shared_target = np.array([1.0, 0.0] + [0.0] * (len(CHANNELS) - 2))
    vecs_a = [shared_target.copy(), shared_target.copy()]
    vecs_b = [shared_target.copy(), np.zeros(len(CHANNELS))]
    matches = greedy_match_roles(roles_a, vecs_a, roles_b, vecs_b)
    assert matches[0]["role_b"] == "b0"
    assert matches[1]["role_b"] == "b0"  # both claim the same partner


# ---------------------------------------------------------------------------
# 4. shuffled_series_null_activation_profile: L1's own convention, reused.
# ---------------------------------------------------------------------------

def test_shuffled_series_null_destroys_planted_correlation():
    """Two roles' activation columns are planted to be PERFECTLY correlated
    (col_b = col_a exactly) over 50 series. The real correlation is 1.0;
    the shuffled-series null, which breaks the row correspondence, must
    come back far below 1.0 -- proving the null actually exercises the
    shuffling rather than being a no-op that would trivially reproduce 1.0.
    """
    rng = np.random.default_rng(0)
    col = rng.normal(size=50)
    role_a = _role(0, "a", [0, 1])
    role_b = _role(0, "b", [0, 1])
    features_a = np.stack([col, col], axis=1)
    features_b = np.stack([col, col], axis=1)
    real = np.corrcoef(col, col)[0, 1]
    assert real == pytest.approx(1.0)
    null = shuffled_series_null_activation_profile(role_a, features_a, role_b, features_b,
                                                    n_shuffles=5, seed=0)
    assert null is not None
    assert null["max"] < 0.9  # decisively below the real, unshuffled 1.0


def test_shuffled_series_null_returns_none_when_role_indices_out_of_range():
    """A stale `roles.json` naming a feature index beyond the dictionary's
    current size must degrade to `None`, not raise or silently slice wrong
    -- sec 2.5's degrade-loudly doctrine applied to a mismatched artifact.
    """
    role = _role(0, "a", [999])
    features = np.zeros((10, 5))
    result = shuffled_series_null_activation_profile(role, features, role, features)
    assert result is None


# ---------------------------------------------------------------------------
# 5. match_roles_for_pair: end-to-end on planted per-target role sets,
#    including the mandatory "no comparable depth" refusal.
# ---------------------------------------------------------------------------

def _write_stage2(tmp_path: Path, model: str, layer: str, candidates: list) -> Path:
    from tsfm_lens.sae.train import sanitize
    d = tmp_path / "sae" / sanitize(model)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{sanitize(layer)}_stage2_response.json"
    save_json(path, {"candidates": candidates, "null_p95": _NULL_P95})
    return path


def test_match_roles_for_pair_end_to_end_planted_match(tmp_path):
    """Model A has one role ('trend up'); model B has two roles, one planted
    to align with A's role (cosine 1.0) and one planted to be unrelated. The
    match must select the aligned one.
    """
    cand_a = [_candidate(0, {"trend": 5.0})]
    cand_b = [_candidate(0, {"trend": 5.0}), _candidate(1, {"level": 3.0})]
    _write_stage2(tmp_path, "ModelA", "layer0", cand_a)
    _write_stage2(tmp_path, "ModelB", "layer0", cand_b)

    roles_json = {
        "ModelA/layer0": {"withheld": False, "skipped": False,
                          "roles": [_role(0, "trend up (A)", [0])]},
        "ModelB/layer0": {"withheld": False, "skipped": False,
                          "roles": [_role(0, "trend up (B)", [0]),
                                    _role(1, "level up (B)", [1])]},
    }
    result = match_roles_for_pair("ModelA", "ModelB", roles_json, run_dir=tmp_path)
    assert result["comparable"] is True
    assert len(result["matches"]) == 1
    assert result["matches"][0]["role_b"] == "trend up (B)"
    assert result["matches"][0]["cosine"] == pytest.approx(1.0)


def test_match_roles_for_pair_refuses_when_no_target_at_comparable_depth():
    """Two models each with exactly one role target, planted far apart on
    the depth axis -- must return `comparable: False` with a reason naming
    both depths and the tolerance, not a silent/empty match list. This is
    the exact real-data shape found on `runs/full_report_run_large`
    (TimesFM 0.947 vs Chronos-T5-Base 0.435 on the block axis, 0.513 apart)
    -- ROADMAP.md sec 20186 states this is the EXPECTED outcome for an
    encoder-only capture surface, not a bug to be tuned away.
    """
    roles_json = {
        "ModelA/deep": {"withheld": False, "skipped": False,
                        "roles": [_role(0, "r", [0])]},
        "ModelB/shallow": {"withheld": False, "skipped": False,
                          "roles": [_role(0, "r", [0])]},
    }
    depths = {"ModelA/deep": 0.95, "ModelB/shallow": 0.40}
    result = match_roles_for_pair("ModelA", "ModelB", roles_json,
                                  depths=depths, depth_tolerance=0.15)
    assert result["comparable"] is False
    assert "0.15" in result["reason"]
    assert "comparable depth" in result["reason"]


def test_match_roles_for_pair_refuses_when_model_has_no_target():
    roles_json = {"ModelA/layer0": {"withheld": False, "skipped": False,
                                    "roles": [_role(0, "r", [0])]}}
    result = match_roles_for_pair("ModelA", "ModelB", roles_json)
    assert result["comparable"] is False
    assert "ModelB" in result["reason"]
    assert "no target exists to check" in result["reason"]


def test_match_roles_for_pair_refuses_on_withheld_target():
    roles_json = {
        "ModelA/layer0": {"withheld": False, "skipped": False, "roles": [_role(0, "r", [0])]},
        "ModelB/layer0": {"withheld": True, "reason": "reach gate failed"},
    }
    result = match_roles_for_pair("ModelA", "ModelB", roles_json)
    assert result["comparable"] is False


# ---------------------------------------------------------------------------
# 6. untrained_twin_role_floor: the mandatory null, and its own degrade path.
# ---------------------------------------------------------------------------

def test_untrained_twin_role_floor_none_when_twin_run_has_no_roles_json(tmp_path):
    result = untrained_twin_role_floor("ModelA", "layer0", tmp_path, "ModelA", "ModelA-random")
    assert result is None


def test_untrained_twin_role_floor_matches_real_against_twin(tmp_path):
    # ROADMAP.md sec 30 (Stage 4, 2026-09-11): the report's own SAE section
    # now reads `sae/concepts.json`; this function's own artifact was
    # archived under `roles_injection.json` and this test writes it there,
    # matching `untrained_twin_role_floor`'s own repointed read path.
    from tsfm_lens.utils import save_json
    twin_dir = tmp_path / "twin_run"
    cand = [_candidate(0, {"trend": 5.0})]
    _write_stage2(twin_dir, "ModelA", "layer0", cand)
    _write_stage2(twin_dir, "ModelA-random", "layer0", cand)
    roles_json = {
        "ModelA/layer0": {"withheld": False, "skipped": False, "roles": [_role(0, "r", [0])]},
        "ModelA-random/layer0": {"withheld": False, "skipped": False, "roles": [_role(0, "r", [0])]},
    }
    save_json(twin_dir / "sae" / "roles_injection.json", roles_json)
    result = untrained_twin_role_floor("ModelA", "layer0", twin_dir, "ModelA", "ModelA-random")
    assert result is not None
    assert result["comparable"] is True
    assert result["matches"][0]["cosine"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 7. role_correspondence_table: the mandatory floor-pairing enforcement --
#    sec 25.6/sec 25.9 Stage 4's second exit criterion.
# ---------------------------------------------------------------------------

def test_match_rate_not_quotable_without_any_untrained_twin_floor(tmp_path):
    """With NO untrained-twin floors supplied at all, a real, comparable
    pair's match_rate must still be computed (it is a real, cheap
    statistic) but marked `match_rate_quotable: False` with a reason -- the
    structural enforcement of sec 25.6's "never render a shared-fraction
    number without its floor" rule.
    """
    cand = [_candidate(0, {"trend": 5.0})]
    _write_stage2(tmp_path, "ModelA", "layer0", cand)
    _write_stage2(tmp_path, "ModelB", "layer0", cand)
    roles_json = {
        "ModelA/layer0": {"withheld": False, "skipped": False, "roles": [_role(0, "r", [0])]},
        "ModelB/layer0": {"withheld": False, "skipped": False, "roles": [_role(0, "r", [0])]},
    }
    table = role_correspondence_table([("ModelA", "ModelB")], roles_json, run_dir=tmp_path)
    pair = table["pairs"][0]
    assert pair["comparable"] is True
    assert pair["match_rate"] == pytest.approx(1.0)
    assert pair["match_rate_quotable"] is False
    assert "NO untrained-twin floor" in pair["match_rate_quotable_reason"]
    assert "clears_untrained_twin_floor" not in pair


def test_match_rate_quotable_when_a_floor_is_supplied(tmp_path):
    cand = [_candidate(0, {"trend": 5.0})]
    _write_stage2(tmp_path, "ModelA", "layer0", cand)
    _write_stage2(tmp_path, "ModelB", "layer0", cand)
    roles_json = {
        "ModelA/layer0": {"withheld": False, "skipped": False, "roles": [_role(0, "r", [0])]},
        "ModelB/layer0": {"withheld": False, "skipped": False, "roles": [_role(0, "r", [0])]},
    }
    floor = {"matches": [{"cosine": 0.2}]}  # a low floor -- real pair should clear it
    table = role_correspondence_table([("ModelA", "ModelB")], roles_json,
                                      untrained_twin_floors={"ModelA": floor},
                                      run_dir=tmp_path)
    pair = table["pairs"][0]
    assert pair["match_rate_quotable"] is True
    assert pair["clears_untrained_twin_floor"] is True


def test_incomparable_pair_still_produces_a_row_never_a_missing_one():
    """sec 24.3's rule for panel artifacts: every configured pair gets a row,
    even an incomparable one -- a pair silently missing from the table would
    be indistinguishable from a bug that dropped it.
    """
    roles_json = {"ModelA/layer0": {"withheld": False, "skipped": False,
                                    "roles": [_role(0, "r", [0])]}}
    table = role_correspondence_table([("ModelA", "ModelB")], roles_json)
    assert len(table["pairs"]) == 1
    assert table["pairs"][0]["comparable"] is False


# ---------------------------------------------------------------------------
# 8. Byte-for-byte panel/pair reproduction -- Stage 4's FIRST exit criterion,
#    mirroring `tests/test_panel_pairs.py`'s own pattern.
# ---------------------------------------------------------------------------

def test_panel_pair_zero_reproduces_direct_two_model_result(tmp_path):
    """A 3-model panel's pair-0 role-correspondence record must be IDENTICAL
    to running the same two models alone as a direct pair -- the byte-for-
    byte reproduction ROADMAP.md sec 25.9 Stage 4 requires. Model C (the
    third panel member) has its own role target but must have zero
    influence on the A-B pair's own match result.
    """
    cand_a = [_candidate(0, {"trend": 5.0}), _candidate(1, {"seasonal": 2.0})]
    cand_b = [_candidate(0, {"trend": 5.0}), _candidate(1, {"level": 9.0})]
    cand_c = [_candidate(0, {"mase": 7.0})]
    _write_stage2(tmp_path, "ModelA", "layer0", cand_a)
    _write_stage2(tmp_path, "ModelB", "layer0", cand_b)
    _write_stage2(tmp_path, "ModelC", "layer0", cand_c)

    roles_full = {
        "ModelA/layer0": {"withheld": False, "skipped": False,
                          "roles": [_role(0, "trend (A)", [0]), _role(1, "seasonal (A)", [1])]},
        "ModelB/layer0": {"withheld": False, "skipped": False,
                          "roles": [_role(0, "trend (B)", [0]), _role(1, "level (B)", [1])]},
        "ModelC/layer0": {"withheld": False, "skipped": False,
                          "roles": [_role(0, "mase (C)", [0])]},
    }
    roles_pair_only = {k: v for k, v in roles_full.items() if not k.startswith("ModelC/")}

    panel_pairs = [("ModelA", "ModelB"), ("ModelA", "ModelC"), ("ModelB", "ModelC")]
    panel_table = role_correspondence_table(panel_pairs, roles_full, run_dir=tmp_path)

    direct_pairs = [("ModelA", "ModelB")]
    direct_table = role_correspondence_table(direct_pairs, roles_pair_only, run_dir=tmp_path)

    panel_ab = panel_table["pairs"][0]
    direct_ab = direct_table["pairs"][0]
    assert panel_ab["model_a"] == direct_ab["model_a"] == "ModelA"
    assert panel_ab["model_b"] == direct_ab["model_b"] == "ModelB"
    assert panel_ab["matches"] == direct_ab["matches"]
    assert panel_ab["match_rate"] == direct_ab["match_rate"]
    assert panel_ab["n_roles_a"] == direct_ab["n_roles_a"]
    assert panel_ab["n_roles_b"] == direct_ab["n_roles_b"]

    # And pair-0 is genuinely the designated pair, not a relabeled one.
    assert panel_pairs[0] == ("ModelA", "ModelB")


def test_panel_generates_all_c_n_2_pairs():
    """sec 24.3's rule generalized to role matching: a 3-model panel must
    produce exactly C(3,2)=3 pairs, in the order given, never only the
    designated reference pair.
    """
    roles_json = {f"Model{c}/layer0": {"withheld": False, "skipped": False,
                                       "roles": [_role(0, "r", [0])]}
                 for c in "ABC"}
    pairs = [("ModelA", "ModelB"), ("ModelA", "ModelC"), ("ModelB", "ModelC")]
    table = role_correspondence_table(pairs, roles_json)
    assert table["n_pairs"] == 3
    assert [(p["model_a"], p["model_b"]) for p in table["pairs"]] == pairs


# ---------------------------------------------------------------------------
# 8. permutation_null_cosine / the per-role population-null split (sec 26
#    D1) -- the fix for a real, rendered defect: on a live run every
#    observed role-matching cosine cleared the fixed 0.5 threshold, forcing
#    `match_rate` to an uninformative 1.000 at every pair regardless of
#    which roles actually corresponded. These tests are the load-bearing
#    negative for THAT defect, distinct from (and not a replacement for)
#    the untrained-twin-floor tests above, which answer a different
#    question (architecture alone, not "is this cosine typical of two
#    arbitrary trained roles in this run's 9-channel response space").
# ---------------------------------------------------------------------------

def test_permutation_null_cosine_none_when_pool_too_small():
    """A population with only one vector outside the excluded targets --
    the null cannot be estimated, and the function must say so with `None`,
    not fabricate a value from a single point (sec 2.5's degrade-loudly
    doctrine, applied here to a null estimate rather than a capability).
    """
    population = {"ModelC/layer0": [(_role(0, "r", [0]), np.array([1.0, 0.0]))]}
    p95, n_pool = permutation_null_cosine(population, exclude_targets=(), n_samples=50)
    assert p95 is None
    assert n_pool == 1


def test_permutation_null_cosine_excludes_named_targets_from_the_pool():
    """The two targets being matched must not contribute to their own null
    -- a population containing ONLY the excluded targets' vectors must
    behave exactly as if the population were empty (`None`, not a null
    silently estimated from the very roles under test).
    """
    population = {
        "ModelA/layer0": [(_role(0, "a", [0]), np.array([1.0, 0.0]))],
        "ModelB/layer0": [(_role(0, "b", [0]), np.array([0.0, 1.0]))],
    }
    p95, n_pool = permutation_null_cosine(
        population, exclude_targets=("ModelA/layer0", "ModelB/layer0"), n_samples=50)
    assert p95 is None
    assert n_pool == 0


def test_permutation_null_cosine_is_exactly_one_for_a_uniform_pool():
    """Every pool vector pointing the same direction (a stand-in for "every
    role in this run pushes the same generic effect") must give a null p95
    of EXACTLY 1.0, deterministically -- any two draws from the pool are
    the same direction by construction, so there is no sampling noise to
    account for in this assertion.
    """
    v = np.array([1.0, 0.0, 0.0])
    population = {"ModelC/layer0": [(_role(0, "r", [0]), v.copy()) for _ in range(3)]}
    p95, n_pool = permutation_null_cosine(population, exclude_targets=(), n_samples=100, seed=0)
    assert p95 == pytest.approx(1.0)
    assert n_pool == 3


def test_population_null_rejects_a_match_the_fixed_threshold_accepts():
    """THE load-bearing negative for the sec 26 D1 fix. Population roles
    (ModelC, ModelD -- not the pair under test) are all built in one
    direction, so two arbitrary trained roles in this run's response space
    are, by construction, indistinguishable -- null p95 = 1.0 exactly. The
    real A/B pair's own role match ALSO scores cosine 1.0 (its two roles
    are planted to align perfectly) -- comfortably clearing the legacy
    fixed `cosine_threshold=0.5`, so `match_rate` (unaffected by this fix)
    still reads 1.0. But `clears_population_null` must be False: a real
    cosine of 1.0 does not exceed a null whose OWN ceiling is also 1.0, so
    this match cannot be told apart from chance in this run's response
    space -- exactly the defect a bare `match_rate` hid on the real run
    that motivated this fix.
    """
    cand_a = [_candidate(0, {"level": 5.0})]
    cand_b = [_candidate(0, {"level": 5.0})]
    roles_json = {
        "ModelA/layer0": {"withheld": False, "skipped": False, "roles": [_role(0, "r", [0])]},
        "ModelB/layer0": {"withheld": False, "skipped": False, "roles": [_role(0, "r", [0])]},
    }
    vec_a = role_response_vector(roles_json["ModelA/layer0"]["roles"][0], cand_a, _NULL_P95)
    vec_b = role_response_vector(roles_json["ModelB/layer0"]["roles"][0], cand_b, _NULL_P95)

    uniform = np.zeros(len(CHANNELS))
    uniform[CHANNELS.index("trend")] = 1.0
    population = {
        "ModelC/layer0": [(_role(0, "r", [0]), uniform.copy()),
                          (_role(1, "r2", [1]), uniform.copy())],
        "ModelD/layer0": [(_role(0, "r", [0]), uniform.copy()),
                          (_role(1, "r2", [1]), uniform.copy())],
    }

    import tsfm_lens.sae.role_matching as rm
    orig = rm._candidates_and_null_p95
    lookup = {("ModelA", "layer0"): (cand_a, _NULL_P95), ("ModelB", "layer0"): (cand_b, _NULL_P95)}
    rm._candidates_and_null_p95 = lambda run_dir, model, layer: lookup.get((model, layer), ([], {}))
    try:
        result = match_roles_for_pair("ModelA", "ModelB", roles_json, population=population,
                                      n_population_samples=100, null_seed=0)
    finally:
        rm._candidates_and_null_p95 = orig

    assert result["comparable"] is True
    assert result["matches"][0]["cosine"] == pytest.approx(1.0)
    assert result["population_null_p95"] == pytest.approx(1.0)
    assert result["population_null_n_pool"] == 4
    assert result["matches"][0]["clears_population_null"] is False
    assert result["frac_shared_by_population_null_a"] == pytest.approx(0.0)
    assert result["roles_specific_to_a"] == ["r"]
    assert result["roles_shared_a"] == []
    # And the pre-existing, threshold-based statistic is untouched by this
    # fix -- still reads a clean 1.0, which is exactly the number this test
    # exists to show is uninformative on its own.
    assert (result["matches"][0]["cosine"] or 0.0) >= 0.5


def test_role_correspondence_table_surfaces_null_based_match_rate(tmp_path):
    """The same scenario through the real entry point
    (`role_correspondence_table`, via `match_roles_across_models`'s default
    `use_population_null=True`) rather than by hand-building `population` --
    confirms the population is actually assembled from `roles_json` end to
    end, not only exercised through the lower-level function above.
    """
    cand_a = [_candidate(0, {"level": 5.0})]
    cand_b = [_candidate(0, {"level": 5.0})]
    uniform_cand = [_candidate(0, {"trend": 5.0}), _candidate(1, {"trend": 5.0})]
    _write_stage2(tmp_path, "ModelA", "layer0", cand_a)
    _write_stage2(tmp_path, "ModelB", "layer0", cand_b)
    _write_stage2(tmp_path, "ModelC", "layer0", uniform_cand)
    _write_stage2(tmp_path, "ModelD", "layer0", uniform_cand)
    roles_json = {
        "ModelA/layer0": {"withheld": False, "skipped": False, "roles": [_role(0, "r", [0])]},
        "ModelB/layer0": {"withheld": False, "skipped": False, "roles": [_role(0, "r", [0])]},
        "ModelC/layer0": {"withheld": False, "skipped": False,
                          "roles": [_role(0, "c0", [0]), _role(1, "c1", [1])]},
        "ModelD/layer0": {"withheld": False, "skipped": False,
                          "roles": [_role(0, "d0", [0]), _role(1, "d1", [1])]},
    }
    table = role_correspondence_table([("ModelA", "ModelB")], roles_json, run_dir=tmp_path,
                                      null_seed=0)
    pair = table["pairs"][0]
    assert pair["match_rate"] == pytest.approx(1.0)  # legacy statistic: unchanged
    assert pair["match_rate_null_based_quotable"] is True
    assert pair["population_null_n_pool"] == 4
    assert pair["match_rate_null_based"] == pytest.approx(0.0)


def test_use_population_null_false_leaves_new_fields_none():
    """The escape hatch: `use_population_null=False` must reproduce the
    pre-fix behaviour exactly (every new field `None`), never silently
    computing a null anyway.
    """
    roles_json = {
        "ModelA/layer0": {"withheld": False, "skipped": False, "roles": [_role(0, "r", [0])]},
        "ModelB/layer0": {"withheld": False, "skipped": False, "roles": [_role(0, "r", [0])]},
    }
    pairs = match_roles_across_models([("ModelA", "ModelB")], roles_json, use_population_null=False)
    pair = pairs[0]
    assert pair["population_null_p95"] is None
    assert pair["frac_shared_by_population_null_a"] is None
    assert pair["matches"][0]["clears_population_null"] is None


def test_role_population_vectors_skips_withheld_and_skipped_targets():
    roles_json = {
        "ModelA/layer0": {"withheld": False, "skipped": False, "roles": [_role(0, "r", [0])]},
        "ModelB/layer0": {"withheld": True, "roles": [_role(0, "r", [0])]},
        "ModelC/layer0": {"skipped": True, "roles": [_role(0, "r", [0])]},
    }
    population = role_population_vectors(roles_json)
    assert set(population.keys()) == {"ModelA/layer0"}
