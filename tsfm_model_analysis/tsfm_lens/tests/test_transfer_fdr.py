"""ROADMAP.md sec 37 P3: p-values, adaptive redraw,
per-pair-per-leg Benjamini-Hochberg FDR control for `sae/transfer.py`'s
cross-model transfer tests, and their extension to the pooled concept ATLAS
(`run_atlas_transfer` -> `sae/atlas_transfer.json`).

All synthetic, with planted, known answers -- the atlas-transfer fixture's
"shared" concept plants the identical top-firing series at both models
(must transfer reciprocally under FDR) beside an "unrelated" concept whose
two models' features fire on disjoint series (must not). Every load-bearing
assertion here was independently confirmed to discriminate by planting the
regression (deleting a legacy-key write, reverting the stratum-matched null
to a uniform one, swapping BH for a naive per-test threshold) and reading
pytest's own summary line, not just its exit status (`CLAUDE.md` sec 11.55).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens.config import ModelConfig, PipelineConfig  # noqa: E402
from tsfm_lens.doctor import check_transfer_fdr_budget  # noqa: E402
from tsfm_lens.extraction.store import ActivationStore, save_meta  # noqa: E402
from tsfm_lens.sae.transfer import (  # noqa: E402
    _exact_p,
    _by_stratum,
    benjamini_hochberg,
    matched_draws,
    run_atlas_transfer,
    run_transfer,
)
from tsfm_lens.utils import load_json, save_json  # noqa: E402


# ---------------------------------------------------------------------------
# p-values (item 1)
# ---------------------------------------------------------------------------

def test_p_value_matches_rank_on_hand_built_null():
    """`p = (1 + #{null >= obs}) / (1 + n)`, exactly."""
    null = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p, hits_floor = _exact_p(0.75, null)
    assert p == pytest.approx((1 + 3) / (1 + 10))  # 0.8, 0.9, 1.0 >= 0.75
    assert hits_floor is False
    p_floor, hits2 = _exact_p(1.5, null)
    assert p_floor == pytest.approx(1 / 11)
    assert hits2 is True
    # negative: a plain empirical p (no +1 floor) would read exactly 0 here,
    # which is a stronger, wrong claim than "at most 1/(n+1)".
    naive = float(np.mean(null >= 1.5))
    assert naive == 0.0
    assert p_floor != naive


def test_p_value_at_the_boundary_is_exactly_the_floor():
    null = np.array([0.5] * 50)
    p, hits = _exact_p(0.5, null)  # every null draw >= obs (tie counts as >=)
    assert hits is False
    assert p == pytest.approx(51 / 51)
    p2, hits2 = _exact_p(0.500001, null)
    assert hits2 is True
    assert p2 == pytest.approx(1 / 51)


# ---------------------------------------------------------------------------
# Adaptive redraw (item 2). The GPD tail it replaced failed on real data
# (2/16 floor-hitting legs within x2 of a 5,000-draw exact p) and was removed.
# ---------------------------------------------------------------------------

def test_matched_draws_is_prefix_stable_under_a_larger_redraw():
    """`adaptive`'s determinism claim: the first `n` rows of an `N`-draw call
    are bit-identical to a standalone `n`-draw call from the same seed, so the
    redrawn null EXTENDS the original instead of replacing it."""
    strata = np.repeat(["a", "b", "c"], 20)
    by_stratum = _by_stratum(strata)
    S = np.array([0, 1, 25, 26, 45])
    small = matched_draws(S, strata, by_stratum, 50, np.random.default_rng(11))
    big = matched_draws(S, strata, by_stratum, 500, np.random.default_rng(11))
    assert np.array_equal(small, big[:50])
    other = matched_draws(S, strata, by_stratum, 50, np.random.default_rng(12))
    assert not np.array_equal(small, other)


def test_adaptive_resolves_a_floor_hitting_leg_below_the_exact_floor(tmp_path):
    """The planted shared concept beats every one of 200 null draws, so its
    exact p is the floor 1/201; `adaptive` must redraw to `max_redraw` and
    report the finer floor 1/(max_redraw+1), labelled as such. A leg that did
    NOT hit the floor keeps its exact p and label."""
    run_dir, _shared, _by_model = _make_store(tmp_path, ("m1", "m2"), n_feat=5, seed=7, k_signal=8)
    concepts = {"targets": {
        "m1/L0": {"model": "m1", "withheld": False,
                  "concepts": [{"concept": 0, "features": [0]}, {"concept": 1, "features": [1]}]},
        "m2/L0": {"model": "m2", "withheld": False,
                  "concepts": [{"concept": 0, "features": [0]}, {"concept": 1, "features": [1]}]},
    }}
    exact = run_transfer(run_dir, concepts, _make_cfg(run_dir, ("m1", "m2"), top_k=8))
    adaptive = run_transfer(run_dir, concepts, _make_cfg(run_dir, ("m1", "m2"), top_k=8,
                                                         p_method="adaptive", max_redraw=1000))
    for pe, pa in zip(exact["pairs"], adaptive["pairs"]):
        if pe["p"] == pytest.approx(1 / 201):
            assert pa["p_method"] == "adaptive"
            assert pa["p"] < pe["p"]
        else:
            assert pa["p_method"] == "exact"
            assert pa["p"] == pe["p"]
    assert any(pa["p_method"] == "adaptive" for pa in adaptive["pairs"])
    assert any(pa["p_method"] == "exact" for pa in adaptive["pairs"])


def test_adaptive_forward_redraw_extends_the_callers_forward_null(tmp_path, monkeypatch):
    """The forward leg's redraw must be seeded from the seed the caller built
    `fwd_draws` from, so its first `n_null` draws reproduce the original
    forward null exactly. Decoy: the reverse leg's redraw extends ITS null,
    which was drawn from a different seed -- a redraw seeded from the reverse
    seed would reproduce the wrong one."""
    import tsfm_lens.sae.transfer as T
    captured = []
    real = T._resolve_p

    def spy(obs, null_vals, method, redraw_fn, max_redraw=T._MAX_REDRAW_DEFAULT):
        captured.append((np.array(null_vals), redraw_fn))
        return real(obs, null_vals, method, redraw_fn, max_redraw=max_redraw)

    monkeypatch.setattr(T, "_resolve_p", spy)
    run_dir, _shared, _by_model = _make_store(tmp_path, ("m1", "m2"), n_feat=5, seed=7, k_signal=8)
    concepts = {"targets": {
        "m1/L0": {"model": "m1", "withheld": False, "concepts": [{"concept": 0, "features": [0]}]},
        "m2/L0": {"model": "m2", "withheld": False, "concepts": [{"concept": 0, "features": [0]}]},
    }}
    run_transfer(run_dir, concepts, _make_cfg(run_dir, ("m1", "m2"), top_k=8,
                                              p_method="adaptive", max_redraw=400))
    assert len(captured) == 4
    for null_vals, redraw_fn in captured:
        assert np.array_equal(redraw_fn(null_vals.size), null_vals)


def test_unknown_p_method_fails_before_any_draw(tmp_path):
    """A removed or misspelled method (`gpd_tail`) must raise up front, not
    only on the first floor-hitting leg, which a small run may never reach."""
    run_dir, _shared, _by_model = _make_store(tmp_path, ("m1", "m2"), n_feat=5, seed=7, k_signal=8)
    concepts = {"targets": {
        "m1/L0": {"model": "m1", "withheld": False, "concepts": [{"concept": 1, "features": [1]}]},
        "m2/L0": {"model": "m2", "withheld": False, "concepts": [{"concept": 1, "features": [1]}]},
    }}
    with pytest.raises(ValueError, match="transfer_p_method"):
        run_transfer(run_dir, concepts, _make_cfg(run_dir, ("m1", "m2"), top_k=8,
                                                  p_method="gpd_tail"))


# ---------------------------------------------------------------------------
# Benjamini-Hochberg (item 4)
# ---------------------------------------------------------------------------

def test_bh_recovers_planted_mixture_across_seeds():
    """900 null p ~ Uniform(0,1) + 100 planted-strong p ~ Uniform(0, 1e-3),
    q=0.05: BH must recover >=90 of the 100 planted tests on average across
    20 seeds, with an average false-discovery share <= ~0.05 (BH's own
    guarantee)."""
    n_recovered, false_share = [], []
    planted = set(range(900, 1000))
    for seed in range(20):
        rng = np.random.default_rng(seed)
        null_p = rng.uniform(0, 1, size=900)
        strong_p = rng.uniform(0, 1e-3, size=100)
        pvals = {i: float(v) for i, v in enumerate(np.concatenate([null_p, strong_p]))}
        res = benjamini_hochberg(pvals, q=0.05)
        survivors = {k for k, v in res.items() if v["survives"]}
        n_recovered.append(len(survivors & planted))
        false_share.append(len(survivors - planted) / max(1, len(survivors)))
    assert np.mean(n_recovered) >= 90, np.mean(n_recovered)
    assert np.mean(false_share) <= 0.06, np.mean(false_share)
    # negative: a NAIVE per-test threshold (p <= q, no correction at all)
    # is strictly more liberal than BH and must show a WORSE (>=) false
    # share on the same data -- if it didn't, BH would not be doing
    # anything on this fixture.
    naive_false = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        null_p = rng.uniform(0, 1, size=900)
        strong_p = rng.uniform(0, 1e-3, size=100)
        pvals = np.concatenate([null_p, strong_p])
        survivors = set(np.flatnonzero(pvals <= 0.05))
        naive_false.append(len(survivors - planted) / max(1, len(survivors)))
    assert np.mean(naive_false) >= np.mean(false_share)


def test_bh_survivors_are_a_prefix_of_sorted_p():
    """BH's decision rule rejects `p_(1..k)` for the largest satisfying
    rank -- so the SET of survivors, sorted by p, must be a contiguous
    prefix, never a scattered subset."""
    rng = np.random.default_rng(0)
    pvals = {i: float(v) for i, v in enumerate(rng.uniform(0, 1, size=50))}
    res = benjamini_hochberg(pvals, q=0.3)
    order = sorted(pvals, key=lambda k: pvals[k])
    survive_flags = [res[k]["survives"] for k in order]
    if True in survive_flags:
        last_true = max(i for i, v in enumerate(survive_flags) if v)
        assert all(survive_flags[: last_true + 1])  # prefix, no gaps


def test_bh_empty_input_returns_empty():
    assert benjamini_hochberg({}, q=0.05) == {}


def test_reciprocal_fdr_requires_both_legs_to_survive():
    """`_apply_pairwise_fdr` must AND the two legs' BH survival, never OR
    them -- a test whose forward leg survives BH but whose reverse leg does
    not (or vice versa) is not reciprocal. None of the `run_transfer`/
    `run_atlas_transfer` fixtures elsewhere in this file happen to produce a
    single-leg-only survivor (their planted concepts are either reciprocal
    in both legs or in neither), so this hand-built family closes that gap
    directly against `_apply_pairwise_fdr`. With m=3 and q=0.05, BH's
    rank thresholds are 0.05/3, 0.10/3, 0.15/3 = 0.0167, 0.0333, 0.05; a
    p of 0.001 clears every rank up to 2, a p of 0.9 clears none."""
    from tsfm_lens.sae.transfer import _apply_pairwise_fdr

    tests = [
        {"src_model": "m1", "dst_model": "m2", "p": 0.001, "rev_p": 0.9},
        {"src_model": "m1", "dst_model": "m2", "p": 0.9, "rev_p": 0.001},
        {"src_model": "m1", "dst_model": "m2", "p": 0.001, "rev_p": 0.001},
    ]
    out = _apply_pairwise_fdr(tests, "src_model", "dst_model", q=0.05)
    assert out[0]["survives_fdr"] is True and out[0]["rev_survives_fdr"] is False
    assert out[0]["reciprocal_fdr"] is False, "forward-only survivor must not be reciprocal"
    assert out[1]["survives_fdr"] is False and out[1]["rev_survives_fdr"] is True
    assert out[1]["reciprocal_fdr"] is False, "reverse-only survivor must not be reciprocal"
    assert out[2]["survives_fdr"] is True and out[2]["rev_survives_fdr"] is True
    assert out[2]["reciprocal_fdr"] is True


# ---------------------------------------------------------------------------
# Synthetic store fixture shared by run_transfer / run_atlas_transfer tests
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path, models: tuple, n_feat: int, seed: int,
                n_strata: int = 4, n_per: int = 30, k_signal: int = 8) -> tuple:
    """A minimal real `ActivationStore` + `meta.parquet`, bypassing the full
    pipeline: `n_strata * n_per` series, two named "signal" sets planted at
    specific series -- `special_shared` (identical series at BOTH models'
    feature 0, so a concept built from it must transfer reciprocally) and
    two DISJOINT `special_<model>_only` sets (feature 1 at each model, no
    overlap, so a concept built from it must NOT transfer). Returns
    `(run_dir, special_shared, special_by_model)`.
    """
    n = n_strata * n_per
    strata = np.repeat([f"s{i}" for i in range(n_strata)], n_per)
    rng = np.random.default_rng(seed)

    special_shared = rng.choice(n, size=k_signal, replace=False)
    remaining = np.setdiff1d(np.arange(n), special_shared)
    rng.shuffle(remaining)
    special_by_model = {}
    for i, m in enumerate(models):
        special_by_model[m] = remaining[i * k_signal:(i + 1) * k_signal]

    store = ActivationStore.create(tmp_path / "activations.zarr", n_series=n,
                                   n_windows=4, window=32, context_len=128)

    def _pooled(special_shared_idx, special_only_idx):
        pooled = rng.normal(0, 0.1, size=(n, n_feat))
        pooled[special_shared_idx, 0] += 5.0
        pooled[special_only_idx, 1] += 5.0
        return pooled

    for m in models:
        pooled = _pooled(special_shared, special_by_model[m])
        store.init_sae_layer(m, "L0", n_feat)
        window_feats = np.repeat(pooled[:, None, :], 4, axis=1)
        store.write_sae_batch(m, "L0", 0, window_feats)

    save_meta(tmp_path, pd.DataFrame({"archetype": strata, "family": strata}))
    return tmp_path, special_shared, special_by_model


def _make_cfg(tmp_path: Path, models: tuple, top_k: int, n_null: int = 200,
             p_method: str = "exact", fdr_q: float = 0.05,
             max_redraw: int = 5000) -> PipelineConfig:
    cfg = PipelineConfig(models=[ModelConfig(name=m, adapter="mock") for m in models])
    cfg.run.name = "transfer_fdr_fixture"
    cfg.run.out_dir = str(tmp_path.parent)
    cfg.sae.transfer_top_k = top_k
    cfg.sae.transfer_n_null = n_null
    cfg.sae.transfer_seed = 0
    cfg.concepts.transfer_p_method = p_method
    cfg.concepts.transfer_fdr_q = fdr_q
    cfg.concepts.transfer_max_redraw = max_redraw
    return cfg


# ---------------------------------------------------------------------------
# run_transfer: byte-identical legacy keys (item 1/4's "additive" promise)
# ---------------------------------------------------------------------------

def test_uncorrected_transfer_keys_survive_pinned_values(tmp_path):
    """A fixed-seed fixture where concept 0 is planted to transfer and
    concept 1 is planted NOT to -- pins exact legacy-field values, which
    `_exact_p`'s addition must not perturb. Deleting the `"auc"`/`"clears"`/
    ... write from `transfer_one`'s returned dict (confirmed by hand) makes
    this fail on the `in` assertions below; corrupting the AUC formula
    (confirmed by hand) makes the `pytest.approx` pins fail."""
    run_dir, _shared, _by_model = _make_store(tmp_path, ("m1", "m2"), n_feat=5, seed=42, k_signal=8)
    concepts = {"targets": {
        "m1/L0": {"model": "m1", "withheld": False,
                  "concepts": [{"concept": 0, "features": [0]}, {"concept": 1, "features": [1]}]},
        "m2/L0": {"model": "m2", "withheld": False,
                  "concepts": [{"concept": 0, "features": [0]}, {"concept": 1, "features": [1]}]},
    }}
    cfg = _make_cfg(run_dir, ("m1", "m2"), top_k=8)
    out = run_transfer(run_dir, concepts, cfg)

    legacy_fields = ("auc", "feature", "null_p95", "clears", "rev_auc",
                     "rev_null_p95", "rev_clears", "reciprocal")
    assert len(out["pairs"]) == 4
    by_key = {(p["src"], p["concept"], p["dst"]): p for p in out["pairs"]}
    concept0_fwd = by_key[("m1/L0", 0, "m2/L0")]
    for f in legacy_fields:
        assert f in concept0_fwd, f
    assert concept0_fwd["auc"] == pytest.approx(1.0)
    assert concept0_fwd["clears"] is True
    assert concept0_fwd["reciprocal"] is True
    concept1_fwd = by_key[("m1/L0", 1, "m2/L0")]
    assert concept1_fwd["reciprocal"] is False

    assert out["reach"] == [
        {"src": "m1/L0", "concept": 0, "dst_model": "m2", "reciprocal": True},
        {"src": "m1/L0", "concept": 1, "dst_model": "m2", "reciprocal": False},
        {"src": "m2/L0", "concept": 0, "dst_model": "m1", "reciprocal": True},
        {"src": "m2/L0", "concept": 1, "dst_model": "m1", "reciprocal": False},
    ]
    assert out["matrix"] == {"m1": {"m2": 0.5}, "m2": {"m1": 0.5}}
    assert out["universality"] == {"1": 2, "0": 2}

    # additive keys are present and consistent with the legacy ones.
    assert concept0_fwd["p"] == pytest.approx(1 / 201)
    assert concept0_fwd["reciprocal_fdr"] is True
    assert concept1_fwd["reciprocal_fdr"] is False


def test_uncorrected_keys_identical_across_p_methods(tmp_path):
    """`p_method` must never change any legacy field's value -- run twice
    (exact vs adaptive) on the identical fixture/seed and diff every legacy
    key plus the `reach`/`matrix`/`universality` reductions built from
    them."""
    run_dir, _shared, _by_model = _make_store(tmp_path, ("m1", "m2"), n_feat=5, seed=7, k_signal=8)
    concepts = {"targets": {
        "m1/L0": {"model": "m1", "withheld": False,
                  "concepts": [{"concept": 0, "features": [0]}, {"concept": 1, "features": [1]}]},
        "m2/L0": {"model": "m2", "withheld": False,
                  "concepts": [{"concept": 0, "features": [0]}, {"concept": 1, "features": [1]}]},
    }}
    legacy_fields = ("auc", "feature", "null_p95", "clears", "rev_auc",
                     "rev_null_p95", "rev_clears", "reciprocal")
    cfg_exact = _make_cfg(run_dir, ("m1", "m2"), top_k=8, p_method="exact")
    out_exact = run_transfer(run_dir, concepts, cfg_exact)
    cfg_gpd = _make_cfg(run_dir, ("m1", "m2"), top_k=8, p_method="adaptive",
                        max_redraw=1000)
    out_gpd = run_transfer(run_dir, concepts, cfg_gpd)

    assert len(out_exact["pairs"]) == len(out_gpd["pairs"]) == 4
    for pe, pg in zip(out_exact["pairs"], out_gpd["pairs"]):
        for f in legacy_fields:
            assert pe[f] == pg[f], f
    assert out_exact["reach"] == out_gpd["reach"]
    assert out_exact["matrix"] == out_gpd["matrix"]
    assert out_exact["universality"] == out_gpd["universality"]


def test_same_model_destinations_are_skipped_source_inspection():
    """Source inspection (mirrors `test_sae_transfer.py`'s own): `run_transfer`
    must never emit a pair whose src and dst share a model."""
    src = ROOT / "tsfm_lens" / "sae" / "transfer.py"
    text = src.read_text(encoding="utf-8")
    assert "dst_model == src_model" in text and "continue" in text


# ---------------------------------------------------------------------------
# run_atlas_transfer (item 3): planted shared vs. unrelated concept
# ---------------------------------------------------------------------------

def _atlas_fixture(tmp_path: Path) -> tuple:
    run_dir, _shared, _by_model = _make_store(tmp_path, ("alpha", "beta"), n_feat=5,
                                              seed=0, k_signal=10)
    (run_dir / "sae").mkdir(exist_ok=True)
    save_json(run_dir / "sae" / "meta.json", {
        "alpha/L0": {"features_persisted": True},
        "beta/L0": {"features_persisted": True},
    })
    atlas = {
        "concepts": [
            {"concept": 0, "name": "shared", "models": {"alpha": 1, "beta": 1}},
            {"concept": 1, "name": "unrelated", "models": {"alpha": 1, "beta": 1}},
        ],
        "rows": [
            {"model": "alpha", "layer": "L0", "feature": 0, "concept": 0},
            {"model": "beta", "layer": "L0", "feature": 0, "concept": 0},
            {"model": "alpha", "layer": "L0", "feature": 1, "concept": 1},
            {"model": "beta", "layer": "L0", "feature": 1, "concept": 1},
        ],
    }
    cfg = _make_cfg(run_dir, ("alpha", "beta"), top_k=10)
    return run_dir, atlas, cfg


def test_atlas_transfer_planted_shared_concept_is_fdr_reciprocal(tmp_path):
    run_dir, atlas, cfg = _atlas_fixture(tmp_path)
    out = run_atlas_transfer(run_dir, atlas, cfg)
    assert (run_dir / "sae" / "atlas_transfer.json").exists()
    by_key = {(t["concept"], t["src_model"], t["dst_model"]): t for t in out["tests"]}
    assert by_key[(0, "alpha", "beta")]["reciprocal_fdr"] is True
    assert by_key[(0, "beta", "alpha")]["reciprocal_fdr"] is True
    assert by_key[(1, "alpha", "beta")]["reciprocal_fdr"] is False
    assert by_key[(1, "beta", "alpha")]["reciprocal_fdr"] is False


def test_atlas_transfer_unrelated_concept_is_not_fdr_reciprocal(tmp_path):
    """Companion negative to the above, isolated: the concept whose two
    models' features fire on DISJOINT series must never survive BH in
    either direction."""
    run_dir, atlas, cfg = _atlas_fixture(tmp_path)
    out = run_atlas_transfer(run_dir, atlas, cfg)
    unrelated = [t for t in out["tests"] if t["concept"] == 1]
    assert len(unrelated) == 2
    assert not any(t["reciprocal_fdr"] for t in unrelated)


def test_atlas_transfer_cross_check_2x2(tmp_path):
    """Both concepts span both models in EFFECT space (each has a member
    at each of alpha/beta), so every tested cell falls in the
    `effect_yes_*` row; the shared concept's two ordered tests land in
    `effect_yes_input_yes`, the unrelated concept's two in
    `effect_yes_input_no`."""
    run_dir, atlas, cfg = _atlas_fixture(tmp_path)
    out = run_atlas_transfer(run_dir, atlas, cfg)
    cc = out["cross_check_2x2"]
    assert cc["effect_yes_input_yes"] == 2
    assert cc["effect_yes_input_no"] == 2
    assert cc["effect_no_input_yes"] == 0
    assert cc["effect_no_input_no"] == 0

    pair_summary = {(r["src_model"], r["dst_model"]): r for r in out["pair_summary"]}
    assert pair_summary[("alpha", "beta")]["n_tests"] == 2
    assert pair_summary[("alpha", "beta")]["n_fdr_reciprocal"] == 1
    assert pair_summary[("beta", "alpha")]["n_fdr_reciprocal"] == 1


def test_atlas_transfer_concept_with_one_model_member_still_tests_but_reads_effect_no(tmp_path):
    """A concept with a member at only ONE model (say alpha) is still a real
    SOURCE: `run_atlas_transfer` tests "every atlas concept part... against
    every target of every OTHER model" unconditionally (item 3), so its
    alpha-part IS tested against beta's target even though beta holds no
    member of this concept. What changes is the 2x2 cell it lands in:
    `effect_yes = dst_model in members_models`, so a destination with no
    concept membership reads `effect_no`, not "excluded"."""
    run_dir, atlas, cfg = _atlas_fixture(tmp_path)
    atlas = dict(atlas)
    atlas["concepts"] = atlas["concepts"] + [{"concept": 2, "name": "alpha_only", "models": {"alpha": 1}}]
    atlas["rows"] = atlas["rows"] + [{"model": "alpha", "layer": "L0", "feature": 2, "concept": 2}]
    out = run_atlas_transfer(run_dir, atlas, cfg)
    tested_concepts = {t["concept"] for t in out["tests"]}
    assert 2 in tested_concepts  # alpha's part WAS tested against beta
    concept2_tests = [t for t in out["tests"] if t["concept"] == 2]
    assert len(concept2_tests) == 1
    assert concept2_tests[0]["src_model"] == "alpha" and concept2_tests[0]["dst_model"] == "beta"
    summary_by_id = {c["concept"]: c for c in out["concept_summary"]}
    concept2_pairs = summary_by_id[2]["pairs"]
    assert len(concept2_pairs) == 1
    assert concept2_pairs[0]["spans_effect_space"] is False  # beta has no member
    cc_total = sum(out["cross_check_2x2"].values())
    assert cc_total == 5  # the 4 effect-spanning cells plus this 1 effect_no cell


def test_atlas_transfer_untested_cell_from_non_persisted_target_is_excluded(tmp_path):
    """The genuine "untested" case (CLAUDE.md sec 11.37's three-states
    discipline): a concept whose only member sits at a target whose SAE
    features were never persisted (absent from `sae/meta.json`, so absent
    from `live`) contributes NO source row -- no test, no cross_check cell
    -- rather than a fabricated "No"."""
    run_dir, atlas, cfg = _atlas_fixture(tmp_path)
    atlas = dict(atlas)
    # "gamma/L9" is not a key in sae/meta.json at all -- not `live`.
    atlas["concepts"] = atlas["concepts"] + [{"concept": 3, "name": "not_persisted", "models": {"alpha": 1}}]
    atlas["rows"] = atlas["rows"] + [{"model": "alpha", "layer": "L9", "feature": 0, "concept": 3}]
    out = run_atlas_transfer(run_dir, atlas, cfg)
    tested_concepts = {t["concept"] for t in out["tests"]}
    assert 3 not in tested_concepts
    cc_total = sum(out["cross_check_2x2"].values())
    assert cc_total == 4  # unchanged from the two persisted, effect-spanning concepts only


# ---------------------------------------------------------------------------
# doctor: transfer FDR floor check (item 7)
# ---------------------------------------------------------------------------

def test_doctor_transfer_fdr_floor_warns_at_n200_m830_exact():
    check = check_transfer_fdr_budget(n_null=200, m=830, q=0.05, p_method="exact")
    assert check.status == "warn"
    assert "830" in check.detail
    # BH is step-up: 830 / (201 * 0.05) = 82.59 -> a batch of 83 floor-level
    # p-values survives, so the family is NOT unsatisfiable.
    assert ">= 83 floor-level" in check.detail
    assert "unsatisfiable" not in (check.detail + check.remediation).lower()
    # negative: the same m under adaptive routes around the floor entirely.
    check_ad = check_transfer_fdr_budget(n_null=200, m=830, q=0.05, p_method="adaptive")
    assert check_ad.status == "pass"


def test_doctor_transfer_fdr_floor_passes_when_m_is_small():
    check = check_transfer_fdr_budget(n_null=200, m=5, q=0.05, p_method="exact")
    assert check.status == "pass"


def test_doctor_transfer_fdr_floor_boundary_is_exact():
    # m/(n_null+1) > q  <=>  m > q*(n_null+1) = 0.05*201 = 10.05
    check10 = check_transfer_fdr_budget(n_null=200, m=10, q=0.05, p_method="exact")
    check11 = check_transfer_fdr_budget(n_null=200, m=11, q=0.05, p_method="exact")
    assert check10.status == "pass"
    assert check11.status == "warn"


# ---------------------------------------------------------------------------
# concept_stage.py wiring (item 6): atlas_transfer block + stale-artifact drop
# ---------------------------------------------------------------------------

def test_concept_stage_writes_atlas_transfer_block(tmp_path):
    """End-to-end through the real `concepts` pipeline stage (2 mock
    architectures) rather than calling `run_atlas_transfer` directly --
    pins that `concept_stage.py`'s wiring actually reaches it, in the
    order the module docstring states (after the atlas)."""
    sys.path.insert(0, str(ROOT))
    from tests.test_smoke import build_config
    from tsfm_lens.config import config_from_dict
    from tsfm_lens.pipeline import run_pipeline

    cfg = config_from_dict(build_config(str(tmp_path)))
    cfg.run.name = "concept_stage_atlas_transfer"
    cfg.sae.enabled = True
    cfg.sae.epochs = 3
    cfg.sae.dict_size_mult = 2
    cfg.sae.k = 4
    cfg.sae.forecast_preservation_max_series = 16
    cfg.sae.persist_features = True
    cfg.concepts.enabled = True
    cfg.sae.concept_min_members = 1
    cfg.concepts.n_features_per_rule = 6
    cfg.concepts.n_null_directions = 4
    cfg.concepts.max_series = 16
    cfg.concepts.top_k_series = 4
    run_pipeline(cfg, stages=["extract"])
    from tsfm_lens.extraction.store import ActivationStore
    store = ActivationStore(cfg.run_dir() / "activations.zarr", mode="r")
    cfg.sae.targets = [{"model": "patchy", "layer": store.layers("patchy")[-1]},
                       {"model": "steppy", "layer": store.layers("steppy")[-1]}]
    run_pipeline(cfg, stages=["sae", "concepts"])

    record = load_json(cfg.run_dir() / "sae" / "concept_stage.json")
    assert "atlas_transfer" in record
    if record["atlas"]["status"] == "ran":
        # only when the atlas itself found >=1 concept is atlas_transfer
        # expected to have run (rather than skip with a stated reason).
        assert record["atlas_transfer"]["status"] in ("ran", "skipped")
        if record["atlas_transfer"]["status"] == "ran":
            assert (cfg.run_dir() / "sae" / "atlas_transfer.json").exists()
            art = load_json(cfg.run_dir() / "sae" / "atlas_transfer.json")
            assert record["atlas_transfer"]["n_tests"] == len(art["tests"])


def test_atlas_transfer_skip_reason_solo_run():
    from tsfm_lens.sae.concept_stage import _atlas_transfer_skip_reason

    class _Cfg:
        class sae:
            transfer_enabled = True

        @staticmethod
        def run_shape():
            return "solo"

    reason = _atlas_transfer_skip_reason(_Cfg, atlas=None)
    assert reason is not None and "solo" in reason


def test_atlas_transfer_skip_reason_no_atlas():
    from tsfm_lens.sae.concept_stage import _atlas_transfer_skip_reason

    class _Cfg:
        class sae:
            transfer_enabled = True

        @staticmethod
        def run_shape():
            return "pair"

    assert _atlas_transfer_skip_reason(_Cfg, atlas=None) is not None
    assert _atlas_transfer_skip_reason(_Cfg, atlas={"concepts": []}) is not None
    assert _atlas_transfer_skip_reason(_Cfg, atlas={"concepts": [{"concept": 0}]}) is None


# ---------------------------------------------------------------------------
# report module: no hardcoded model names (mirrors test_concept_atlas.py)
# ---------------------------------------------------------------------------

def test_report_module_has_no_hardcoded_model_name():
    import re

    src = (ROOT / "tsfm_lens" / "report" / "sae_transfer_fdr.py").read_text(encoding="utf-8")
    stripped = re.sub(r'"""[\s\S]*?"""', "", src)
    banned = ("TimesFM", "Chronos", "Sundial", "patchy", "steppy")
    for name in banned:
        assert name not in stripped, name
    assert not re.search(r"models\[\d|cfg\.models\[", stripped)


def test_report_module_scan_catches_a_planted_violation():
    """Companion negative: confirm the scan above actually discriminates,
    against a deliberately-bad stand-in (not the real module)."""
    import re

    bad_src = 'def f(cfg):\n    return "TimesFM did better than Chronos"\n'
    stripped = re.sub(r'"""[\s\S]*?"""', "", bad_src)
    assert any(name in stripped for name in ("TimesFM", "Chronos", "Sundial", "patchy", "steppy"))


def test_transfer_fdr_block_renders_without_model_names(tmp_path):
    """The rendered block itself, not just the source: labels come from the
    artifact's own `src_model`/`dst_model` strings, never a hardcoded name."""
    run_dir, atlas, cfg = _atlas_fixture(tmp_path)
    run_atlas_transfer(run_dir, atlas, cfg)
    from tsfm_lens.report.sae_transfer_fdr import transfer_fdr_block

    html = transfer_fdr_block(run_dir, cfg)
    assert "alpha" in html and "beta" in html
    assert "Cross-model transfer" in html


def test_ledger_does_not_call_a_bh_transfer_family_unsatisfiable(tmp_path):
    """BH is step-up, so a family past `m/(n_null+1) > q` loses only its
    LONE effects (a 70-test real pair kept 36 survivors). The ledger must
    render it as a coarse floor with its minimum batch, never with the Holm
    "No result here can be significant" block or an L0 finding. Decoy: an
    unsatisfiable Holm family in the same ledger must still be named."""
    from tsfm_lens.report.report import _multiplicity_block
    pairs = [{"src_model": "A", "dst_model": "B"} for _ in range(70)]
    save_json(tmp_path / "sae" / "transfer.json",
              {"n_null_draws": 200, "fdr_q": 0.05, "p_method": "exact", "pairs": pairs})
    summary = {"multiplicity": {"scope": "l0.family", "method": "holm", "alpha": 0.05,
                                "n_models": 2, "n_pairs": 1, "n_tests": 18,
                                "n_boot": 150, "min_attainable_p_holm": 18 / 150,
                                "most_stringent_threshold": 0.05 / 18,
                                "designated_pair": ["A", "B"]}}
    findings: list = []
    html = _multiplicity_block(tmp_path, summary, findings)
    assert html.count("Unsatisfiable correction") == 1
    assert "L0 per-family paired tests" in html.split("Unsatisfiable correction")[1][:80]
    assert "Coarse p-floor" in html
    assert "2 of 2 BH transfer" in html
    assert "7 floor-level p-values" in html
    unsat = [f for f in findings if "UNSATISFIABLE" in f.text]
    assert len(unsat) == 1 and "Transfer" not in unsat[0].text
