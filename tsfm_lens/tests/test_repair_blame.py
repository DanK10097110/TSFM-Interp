"""Causal per-feature MASE blame (ROADMAP.md sec 39, R0): `analysis/repair_blame.py`.

Planted answers (`tests/repair_fixture.py`): the harmful atom's ablation lowers MASE, the
helpful one raises it, the decoy fires preferentially on hard series (positive correlation
with MASE) and has no causal effect, and atoms inside the complement of the planted span are
exactly inert. The blame must be in MASE units, from the ablation, on the pool's series only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from repair_fixture import HORIZON, LAYER, QUANTILES, build_world  # noqa: E402
from tsfm_lens.analysis import repair_blame as rb  # noqa: E402
from tsfm_lens.analysis import repair_edit as re_  # noqa: E402
from tsfm_lens.analysis.stats import mase  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return build_world(seed=0, dose=8.0)


@pytest.fixture(scope="module")
def table(world):
    w = world
    pool = np.arange(w["data"].n)
    return rb.blame_features(w["adapter"], LAYER, w["sae"], w["data"], w["device"], w["acts"], pool,
                             list(range(w["sae"].dict_size)), HORIZON, QUANTILES, k=16, n_null=32,
                             seed=0, n_boot=300, min_rows=4)


def _row(table, f):
    return next(r for r in table["features"] if r["feature"] == f)


def test_blame_recovers_the_planted_signs_and_clears_the_inert_atoms(world, table):
    atom = world["atom"]
    harmful, helpful = _row(table, atom["repair_harmful"]), _row(table, atom["repair_helpful"])
    assert harmful["mean_delta_mase"] < 0 and harmful["significant"] and harmful["ci_hi"] < 0
    assert harmful["direction"] == "harmful"
    assert helpful["mean_delta_mase"] > 0 and helpful["significant"] and helpful["ci_lo"] > 0
    assert helpful["direction"] == "helpful"
    planted = set(atom.values())
    stray = [r["feature"] for r in table["features"] if r["scorable"] and r["significant"]
             and r["feature"] not in planted]
    assert stray == []


def test_decoy_correlates_with_error_but_gets_no_causal_blame(world, table):
    decoy = _row(table, world["atom"]["repair_decoy"])
    assert decoy["corr_mase"] > 0 and decoy["act_weighted_excess_mase"] > 0
    assert abs(decoy["mean_delta_mase"]) < 1e-3
    assert not decoy["significant"] and decoy["p_bh"] > 0.5


def test_blame_is_in_mase_units_and_equals_an_independent_ablation(world, table):
    w = world
    f = w["atom"]["repair_harmful"]
    row = _row(table, f)
    rows = np.argsort(-w["acts"][:, f], kind="stable")[:row["n_rows_scored"]]
    data = w["data"]
    base, ablated = re_.predict_gains(w["adapter"], LAYER, w["sae"], w["device"], data.contexts()[rows],
                                      {f: 0.0}, HORIZON, QUANTILES, 0)
    direct = mase(ablated, data.targets()[rows], data.contexts()[rows]) - mase(
        base, data.targets()[rows], data.contexts()[rows])
    assert row["mean_delta_mase"] == pytest.approx(float(direct.mean()), rel=1e-6, abs=1e-9)
    assert row["sd_delta_mase"] == pytest.approx(float(direct.std(ddof=1)), rel=1e-6)
    assert row["null_normalized"] == pytest.approx(row["mean_delta_mase"] / row["null_p95"])


def test_the_null_is_present_and_nondegenerate_for_effect_carrying_atoms(world, table):
    for cid in ("repair_harmful", "repair_helpful", "repair_decoy"):
        r = _row(table, world["atom"][cid])
        assert r["null_p95"] > 0 and not r["null_degenerate"] and r["n_null"] == 32
    assert 0 < _row(table, world["atom"]["repair_harmful"])["p_empirical"] <= 1.0


def test_bh_floor_is_reported_not_hidden(table):
    assert table["empirical_p_floor"] == pytest.approx(1 / 33)
    assert table["bh_empirical_satisfiable"] is False
    assert table["primary_p"] == "normal"


def test_null_p_values_degenerate_null_falls_back_loudly():
    p = rb.null_p_values(0.5, np.zeros(20))
    assert p["null_degenerate"] and p["p_normal"] == p["p_empirical"] == pytest.approx(1 / 21)
    q = rb.null_p_values(0.0, np.linspace(-1, 1, 21))
    assert not q["null_degenerate"] and q["p_normal"] > 0.9 and q["p_empirical"] == 1.0


def test_blame_sees_only_the_pool_and_caps_k_at_the_batch_size(world):
    w = world
    data = w["data"]
    train, test = re_.split_series(data.n, data.families, seed=2)
    seen = []
    original = w["adapter"].predict

    def spy(contexts, horizon, quantiles):
        seen.append(np.asarray(contexts))
        return original(contexts, horizon, quantiles)

    w["adapter"].predict = spy
    try:
        t = rb.blame_features(w["adapter"], LAYER, w["sae"], data, w["device"], w["acts"], train,
                              [w["atom"]["repair_harmful"]], HORIZON, QUANTILES, k=500, n_null=4, seed=0,
                              n_boot=50, min_rows=4)
    finally:
        del w["adapter"].predict
    test_rows = {row.tobytes() for row in data.contexts()[test].astype(np.float32)}
    leaked = [1 for batch in seen for row in batch.astype(np.float32) if row.tobytes() in test_rows]
    assert leaked == []
    assert t["effective_k"] == 64 and t["top_k_requested"] == 500
    assert _row(t, w["atom"]["repair_harmful"])["n_rows_scored"] <= 64


def test_a_rarely_firing_feature_is_unscorable_with_a_reason(world):
    w = world
    pool = np.arange(w["data"].n)
    acts = w["acts"].copy()
    acts[:, 5] = 0.0
    acts[:3, 5] = 1.0
    t = rb.blame_features(w["adapter"], LAYER, w["sae"], w["data"], w["device"], acts, pool, [5],
                          HORIZON, QUANTILES, k=16, n_null=4, seed=0, n_boot=50, min_rows=8)
    r = t["features"][0]
    assert not r["scorable"] and "min_rows" in r["reason"] and r["n_rows_scored"] == 3


def test_non_series_level_activations_are_refused(world):
    w = world
    with pytest.raises(ValueError, match="series-level"):
        rb.blame_features(w["adapter"], LAYER, w["sae"], w["data"], w["device"], w["acts"][:10],
                          np.arange(10), [0], HORIZON, QUANTILES)
