"""The two-stage blame family (ROADMAP.md sec 39.7, R0b): `analysis/repair_blame.two_stage_blame`.

Stage 1 screens every scorable feature on half A of the pool by `|z|`; stage 2 re-blames the top
`screen_m` on the disjoint half B with a fresh null stream; BH runs over exactly those `screen_m`
features. Tests use the planted oracle dictionary of `tests/repair_fixture.py`: a harmful and a
helpful atom that must survive both stages, a decoy that must not, and inert atoms. Each
load-bearing assertion was confirmed to fail against a planted regression (see the R0b report).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from repair_fixture import HORIZON, LAYER, QUANTILES, StubData, build_world  # noqa: E402
from tsfm_lens.analysis import repair_blame as rb  # noqa: E402
from tsfm_lens.analysis import repair_known_answer as rka  # noqa: E402
from tsfm_lens.sae.transfer import benjamini_hochberg  # noqa: E402

SCREEN_M = 12


@pytest.fixture(scope="module")
def world():
    return build_world(seed=0, dose=8.0)


@pytest.fixture(scope="module")
def two_stage(world):
    """One two-stage run, with every `blame_features` call's arguments recorded."""
    w = world
    calls = []
    original = rb.blame_features

    def spy(adapter, layer, sae, data, device, activations, pool_rows, features, *a, **kw):
        calls.append({"pool": np.asarray(pool_rows).copy(), "features": list(features),
                      "null_stream": kw.get("null_stream", rb.NULL_STREAM)})
        return original(adapter, layer, sae, data, device, activations, pool_rows, features, *a, **kw)

    rb.blame_features = spy
    try:
        t = rb.two_stage_blame(w["adapter"], LAYER, w["sae"], w["data"], w["device"], w["acts"],
                               np.arange(w["data"].n), list(range(w["sae"].dict_size)), HORIZON,
                               QUANTILES, split_seed=4, screen_m=SCREEN_M, k=16, n_null=32, seed=0,
                               n_boot=200, min_rows=4)
    finally:
        rb.blame_features = original
    return t, calls


def test_halves_are_disjoint_stratified_and_the_stages_use_different_halves(world, two_stage):
    t, calls = two_stage
    assert len(calls) == 2
    a, b = calls[0]["pool"], calls[1]["pool"]
    assert np.intersect1d(a, b).size == 0
    assert np.array_equal(np.sort(np.concatenate([a, b])), np.arange(world["data"].n))
    fam = world["data"].families
    for f in np.unique(fam):
        assert abs(int((fam[a] == f).sum()) - int((fam[b] == f).sum())) <= 1
    assert t["stage1_rows_disjoint_from_stage2"] is True
    assert (t["n_stage1_series"], t["n_stage2_series"]) == (len(a), len(b))


def test_stage_two_reblames_only_the_screened_features_with_a_fresh_null_stream(two_stage):
    t, calls = two_stage
    assert len(calls[0]["features"]) > SCREEN_M
    assert calls[1]["features"] == t["stage1"]["screened"] and len(calls[1]["features"]) == SCREEN_M
    assert calls[0]["null_stream"] != calls[1]["null_stream"]
    assert [r["feature"] for r in t["features"]] == t["stage1"]["screened"]


def test_bh_family_is_exactly_the_screened_set(two_stage):
    t, _ = two_stage
    assert t["design"] == "two_stage" and t["bh_family_size"] == SCREEN_M == len(t["features"])
    p = {r["feature"]: (r["p_normal"] if r["scorable"] else 1.0) for r in t["features"]}
    bh = benjamini_hochberg(p, 0.05)
    for r in t["features"]:
        if r["scorable"]:
            assert r["p_bh"] == pytest.approx(bh[r["feature"]]["p_bh"])
            assert r["significant"] == bh[r["feature"]]["survives"]


def test_planted_answer_through_both_stages(world, two_stage):
    t, _ = two_stage
    atom = world["atom"]
    rows = {r["feature"]: r for r in t["features"]}
    for cid in ("repair_harmful", "repair_helpful"):
        r = rows[atom[cid]]
        assert r["stage1_rank"] <= SCREEN_M and r["scorable"] and r["significant"]
        assert (r["mean_delta_mase"] < 0) == (cid == "repair_harmful")
    decoy = rows.get(atom["repair_decoy"])
    assert decoy is None or not decoy["significant"]
    assert not any(r["significant"] for r in t["features"] if r["feature"] not in set(atom.values()))
    ranks = [r["stage1_rank"] for r in t["features"]]
    assert ranks == sorted(ranks) and ranks[0] == 1


def test_screening_uses_half_a_only(world):
    w = world
    data = w["data"]
    half_a, half_b = rb.split_pool(np.arange(data.n), data.families, 4)
    mirrored = data.targets().copy()
    mirrored[half_b] = 2.0 * data.contexts()[half_b, -1:] - mirrored[half_b]
    stub = StubData(data, mirrored)
    feats = [w["atom"][c] for c in ("repair_harmful", "repair_helpful", "repair_decoy",
                                    "repair_sideeffect")] + list(range(10, 40))
    kw = dict(horizon=HORIZON, quantiles=QUANTILES, split_seed=4, screen_m=6, k=16, n_null=16, seed=0,
              n_boot=100, min_rows=4)
    base = rb.two_stage_blame(w["adapter"], LAYER, w["sae"], data, w["device"], w["acts"],
                              np.arange(data.n), feats, **kw)
    moved = rb.two_stage_blame(w["adapter"], LAYER, w["sae"], stub, w["device"], w["acts"],
                               np.arange(data.n), feats, **kw)
    assert base["stage1"]["screened"] == moved["stage1"]["screened"]
    harmful = w["atom"]["repair_harmful"]
    b = next(r for r in base["features"] if r["feature"] == harmful)
    m = next(r for r in moved["features"] if r["feature"] == harmful)
    assert b["mean_delta_mase"] < 0 < m["mean_delta_mase"]


def test_a_planted_atom_outside_the_screen_fails_the_seed():
    table = {"features": [{"feature": 99, "scorable": True, "significant": True, "mean_delta_mase": -1.0,
                           "n_rows_scored": 5, "sd_delta_mase": 0.1, "ci_lo": -1.1, "ci_hi": -0.9,
                           "null_p95": 0.1, "null_normalized": -10.0, "p_empirical": 0.01,
                           "p_normal": 0.0, "p_bh": 0.0, "corr_mase": 0.1,
                           "act_weighted_excess_mase": 0.1}]}
    matches = {c: {"feature": 5, "cosine": 0.99, "recovered": True} for c in rka.CONCEPTS}
    matches["repair_harmful"]["feature"] = 99
    out = rka.score_blame(table, matches)
    assert out["harmful_correct"] is True
    assert out["helpful_correct"] is False and out["decoy_clear"] is False
    assert out["repair_helpful"]["scorable"] is False


def test_default_single_stage_null_stream_is_unchanged():
    assert rb.NULL_STREAM == 17 and rb.STAGE2_NULL_STREAM != rb.NULL_STREAM and rb.SCREEN_M == 16
