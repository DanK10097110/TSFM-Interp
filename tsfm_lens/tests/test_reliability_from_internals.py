"""Tests for `analysis/reliability_from_internals.py` (`ROADMAP.md` sec 38.4, K4).

Planted fixture: two models whose log MASE depends on a shared baseline
signal `x` and on an internal signal `z` with OPPOSITE signs (model A gets
worse as z rises, model B better). So
  * the `informative` internal group carries failure information that the
    baseline lacks (the gain CI must exclude 0, in both models);
  * the `decoy_dup` group duplicates the baseline column `x` (gain ~ 0);
  * the `noise` group is 60 pure-noise columns: out of fold it must not gain,
    in sample it would (the cross-fitting plant);
  * routing on z beats routing on the baseline, because the best model flips
    with the sign of z and the baseline cannot see it;
  * a fit pooled across the two models cancels the opposite signs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis import reliability_from_internals as rfi

N = 1000
N_BOOT = 300


def _fixture(n: int = N, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    strata = np.array(["fam_a", "fam_b", "fam_c", "fam_d"])[np.arange(n) % 4]
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    noise = rng.normal(size=(n, 60))
    other = rng.normal(size=(n, 3))
    baseline = np.column_stack([x, other])
    groups = {"informative": np.column_stack([z + 0.3 * rng.normal(size=n)]),
              "decoy_dup": np.column_stack([x + 0.01 * rng.normal(size=n)]),
              "noise": noise}
    models = {}
    for name, sign in (("A", 1.0), ("B", -1.0)):
        log_mase = 0.6 * x + sign * 0.9 * z + 0.3 * rng.normal(size=n)
        models[name] = {"mase": np.exp(log_mase), "baseline": baseline,
                        "baseline_names": [f"b{i}" for i in range(baseline.shape[1])],
                        "groups": {k: v.copy() for k, v in groups.items()},
                        "skipped": {}, "notes": {}}
    return {"run": "planted", "strata": strata, "reliable": np.ones(n),
            "seasonal_mase": np.full(n, np.exp(0.2)), "models": models}


@pytest.fixture(scope="module")
def fx():
    return _fixture()


@pytest.fixture(scope="module")
def result(fx):
    return rfi.run_reliability(fx, n_folds=5, n_repeats=1, n_boot=N_BOOT, seed=1)


def test_informative_group_gain_excludes_zero_in_both_models(result):
    for m in ("A", "B"):
        rec = result["models"][m]
        g = rec["u1"]["log_mase_spearman"]["gain"]
        assert g["lo"] > 0 and g["ci_excludes_zero_positive"], (m, g)
        info = rec["by_group"]["informative"]["log_mase_spearman"]
        assert info["lo"] > 0.05, (m, info)
        fa = rec["u1"]["failure_auroc"]["gain"]
        assert fa["lo"] > 0, (m, fa)


def test_decoy_duplicate_and_noise_groups_add_nothing(result):
    for m in ("A", "B"):
        by = result["models"][m]["by_group"]
        for name in ("decoy_dup", "noise"):
            g = by[name]["log_mase_spearman"]
            bound = abs(g["gain"]) if name == "decoy_dup" else g["gain"]
            assert bound < 0.02 and not g["ci_excludes_zero_positive"], (m, name, g)
            gf = by[name]["failure_auroc"]
            assert not gf["ci_excludes_zero_positive"], (m, name, gf)


def test_routing_on_the_informative_feature_beats_baseline_routing(result):
    u2 = result["u2"]
    pol = {k: v["value"] for k, v in u2["policies"].items()}
    gap = u2["gap_baseline_minus_internals"]
    assert gap["lo"] > 0, gap
    assert pol["baseline_plus_internals_routing"] < pol["baseline_routing"]
    assert pol["oracle"] <= pol["baseline_plus_internals_routing"] < pol["best_single_model"]
    assert pol["baseline_routing"] >= pol["oracle"]


def test_record_is_per_model_states_evidence_class_and_secondary_by_family(result):
    assert set(result["models"]) == {"A", "B"}
    assert "not a cause" in result["caveat"] and "predictive" in result["evidence_class"]
    fam = result["models"]["A"]["by_family"]
    assert set(fam) == {"fam_a", "fam_b", "fam_c", "fam_d"}
    assert fam["fam_a"]["log_mase_spearman"]["scorable"]
    assert result["settings"]["ci"].startswith("series cluster bootstrap")


def test_missing_internal_inputs_degrade_loudly_and_skip_the_group(fx):
    inp = {**fx, "models": {m: {**v, "groups": {"informative": v["groups"]["informative"]},
                                "skipped": {"lens_depth": "lens/convergence.json absent"}}
                            for m, v in fx["models"].items()}}
    out = rfi.run_reliability(inp, n_folds=5, n_repeats=1, n_boot=100, seed=1)
    rec = out["models"]["A"]
    assert rec["internal_groups_skipped"] == {"lens_depth": "lens/convergence.json absent"}
    assert list(rec["internal_groups"]) == ["informative"]
    bare = {**fx, "models": {m: {**v, "groups": {}, "skipped": {"sae_families": "no persisted SAE features"}}
                             for m, v in fx["models"].items()}}
    out = rfi.run_reliability(bare, n_folds=5, n_repeats=1, n_boot=100, seed=1)
    assert "no gain to report" in out["models"]["A"]["u1"]["note"]
    assert out["models"]["A"]["internal_groups_skipped"]["sae_families"]


def test_lens_group_with_partial_coverage_is_skipped_with_the_coverage_stated(fx):
    m = {k: dict(v) for k, v in fx["models"].items()}
    seen = np.zeros(N, bool)
    seen[:100] = True
    for v in m.values():
        v["groups"] = {"lens_depth": v["groups"]["informative"]}
        v["notes"] = {"_lens_seen": seen}
    out = rfi.run_reliability({**fx, "models": m}, n_folds=5, n_repeats=1, n_boot=100, seed=1)
    rec = out["models"]["A"]
    assert "covers 0.100" in rec["internal_groups_skipped"]["lens_depth"]
    assert rec["internal_groups"] == {}


def test_folds_are_stratified_and_balanced():
    strata = np.array(["a"] * 53 + ["b"] * 47 + ["c"] * 5)
    folds = rfi.stratified_folds(strata, 5, seed=0)
    for label in ("a", "b", "c"):
        counts = np.bincount(folds[strata == label], minlength=5)
        assert counts.max() - counts.min() <= 1, (label, counts)
    assert np.bincount(folds).max() - np.bincount(folds).min() <= 1
    assert not np.array_equal(folds, rfi.stratified_folds(strata, 5, seed=1))


def test_oof_predictions_never_see_their_own_series():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 80))
    y = rng.normal(size=n)
    strata = np.array(["a", "b"])[np.arange(n) % 2]
    oof = rfi.cross_fitted_predictions(X, y, strata, "ridge", 5, 1, seed=0)
    assert abs(rfi.spearman(oof, y)) < 0.3


def test_scores_rank_in_float64_not_float16():
    rng = np.random.default_rng(0)
    n = 4000
    base = rng.normal(size=n).astype(np.float64)
    y = base + 0.5 * rng.normal(size=n)
    x16 = (base * 0.02 + 1.0).astype(np.float16)
    ref = float(spearmanr(x16.astype(np.float64), y)[0])
    assert abs(rfi.spearman(x16, y) - ref) < 1e-9
    lab = (y > 0).astype(float)
    s64 = x16.astype(np.float64)
    ranks = rankdata(s64)
    k = int(lab.sum())
    ref_auc = (ranks[lab.astype(bool)].sum() - k * (k + 1) / 2) / (k * (n - k))
    assert abs(rfi.auroc(x16, lab) - ref_auc) < 1e-9
