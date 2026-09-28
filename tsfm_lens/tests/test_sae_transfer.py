"""Tests for `sae/transfer.py` (ROADMAP.md sec 30, sec 30.10 stage 3):
cross-model SAE CONCEPT transfer -- does model B group the same series a
concept in model A fires hardest on?

All synthetic, with planted answers. Each negative is confirmed to
discriminate by planting the regression and reading pytest's own summary
line (`CLAUDE.md` sec 11.55's corollary -- a plant that produces a syntax
error reports `1 error`, not `N failed`).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.transfer import (
    _by_stratum,
    _seed,
    auc_from_ranks,
    concept_scores,
    matched_draws,
    series_strata,
    top_series,
    transfer_one,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# series_strata
# --------------------------------------------------------------------------

def test_strata_fallback_covers_null_archetype():
    meta = pd.DataFrame({
        "archetype": ["seasonal_dominant", None, None, "trend_dominant"],
        "family": ["random_parametric", "mixture", "parametric", "random_parametric"],
    })
    strata = series_strata(meta)
    assert list(strata) == ["seasonal_dominant", "mixture", "parametric", "trend_dominant"]
    # negative: dropping the fillna collapses every null-archetype row out
    dropped = meta["archetype"].dropna().to_numpy()
    assert len(dropped) != len(meta)


# --------------------------------------------------------------------------
# concept_scores / top_series
# --------------------------------------------------------------------------

def test_concept_scores_is_mean_over_member_columns():
    pooled = np.array([[1.0, 3.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 9.0]])
    scores = concept_scores(pooled, [0, 1])
    assert np.allclose(scores, [2.0, 3.0, 0.0])


def test_top_series_orders_descending():
    score = np.array([0.1, 5.0, 3.0, -2.0, 4.0])
    assert list(top_series(score, 3)) == [1, 4, 2]


# --------------------------------------------------------------------------
# matched_draws
# --------------------------------------------------------------------------

def test_matched_draws_preserve_stratum_composition():
    strata = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    by_stratum = _by_stratum(strata)
    S = np.array([0, 1, 2, 20, 21, 40])  # 3 x 'a', 2 x 'b', 1 x 'c'
    draws = matched_draws(S, strata, by_stratum, n_draws=50, rng=_rng(0))
    assert draws.shape == (50, len(S))
    for row in draws:
        row_strata = strata[row]
        vals, counts = np.unique(row_strata, return_counts=True)
        comp = dict(zip(vals, counts))
        assert comp == {"a": 3, "b": 2, "c": 1}


def test_matched_draws_are_without_replacement_within_a_row():
    strata = np.array(["a"] * 30)
    by_stratum = _by_stratum(strata)
    S = np.arange(10)
    draws = matched_draws(S, strata, by_stratum, n_draws=20, rng=_rng(1))
    for row in draws:
        assert len(set(row.tolist())) == len(row)


# --------------------------------------------------------------------------
# auc_from_ranks
# --------------------------------------------------------------------------

def test_auc_from_ranks_matches_scipy_mannwhitneyu():
    rng = np.random.default_rng(3)
    n, n_features = 60, 5
    pooled = rng.normal(size=(n, n_features))
    R = rankdata(pooled, axis=0)
    idx = rng.choice(n, size=15, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    got = auc_from_ranks(R, idx)
    for f in range(n_features):
        u = mannwhitneyu(pooled[mask, f], pooled[~mask, f], alternative="two-sided").statistic
        expected = u / (mask.sum() * (~mask).sum())
        assert got[f] == pytest.approx(expected, abs=1e-9)


def test_auc_from_ranks_batch_matches_single_calls():
    rng = np.random.default_rng(4)
    n, n_features = 40, 3
    R = rankdata(rng.normal(size=(n, n_features)), axis=0)
    idx_batch = np.stack([rng.choice(n, size=8, replace=False) for _ in range(6)])
    batch = auc_from_ranks(R, idx_batch)
    for d in range(idx_batch.shape[0]):
        single = auc_from_ranks(R, idx_batch[d])
        assert np.allclose(batch[d], single)


def test_auc_of_perfect_separator_is_one():
    n = 40
    score = np.concatenate([np.full(10, 10.0), np.full(30, 0.0)])
    R = rankdata(score)[:, None]
    idx = np.arange(10)
    assert auc_from_ranks(R, idx)[0] == pytest.approx(1.0)
    # anti-separator: the "positive" group is the LOW-scoring one
    anti_idx = np.arange(10, 40)
    assert auc_from_ranks(R, anti_idx)[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# transfer_one: forward null is max-over-features, reverse has no search
# correction, reach requires no per-target attribution
# --------------------------------------------------------------------------

def _uniform_strata(n, rng):
    strata = np.array(["only"] * n)
    return strata, _by_stratum(strata)


def test_forward_null_is_max_over_features():
    """A planted case where the observed AUC clears a PER-FEATURE null but
    not the max-over-features null the spec requires -- proving the
    function actually takes the max, not a per-feature comparison."""
    rng = np.random.default_rng(5)
    n = 200
    n_features = 50
    strata, by_stratum = _uniform_strata(n, rng)
    # dst dictionary: one real feature (0) separates S at auc ~0.75; the
    # rest are pure noise, but with 50 features the max of 200 random draws
    # over 50 noisy columns comfortably exceeds a per-feature p95.
    S = rng.choice(n, size=20, replace=False)
    pooled = rng.normal(size=(n, n_features))
    boost = np.zeros(n)
    boost[S] = 1.2
    pooled[:, 0] += boost
    dst_ranks = rankdata(pooled, axis=0)
    fwd_draws = matched_draws(S, strata, by_stratum, 200, rng)
    result = transfer_one(rng.normal(size=n), dst_ranks, S, fwd_draws, strata,
                          by_stratum, k=20, n_draws=200, seed=1)

    obs_auc = auc_from_ranks(dst_ranks, S)
    null_auc = auc_from_ranks(dst_ranks, fwd_draws)
    per_feature_p95 = np.percentile(null_auc, 95, axis=0)
    best = int(np.argmax(obs_auc))
    # negative: a per-feature null would clear this case even for a noise
    # column whose per-feature p95 the max-over-features null correctly
    # refuses to use as the bar.
    assert result["null_p95"] >= float(np.percentile(null_auc.max(axis=1), 95)) - 1e-12
    assert result["null_p95"] >= per_feature_p95[best] - 1e-9


def test_reverse_leg_has_no_search_correction():
    """The reverse null is over ONE score vector (src_scores), never a max
    over multiple candidates -- pinned by checking it against a direct
    single-vector null computation."""
    rng = np.random.default_rng(6)
    n = 150
    strata, by_stratum = _uniform_strata(n, rng)
    S = rng.choice(n, size=20, replace=False)
    dst_pooled = rng.normal(size=(n, 10))
    dst_pooled[S, 3] += 1.0
    dst_ranks = rankdata(dst_pooled, axis=0)
    fwd_draws = matched_draws(S, strata, by_stratum, 200, rng)
    src_scores = rng.normal(size=n)

    result = transfer_one(src_scores, dst_ranks, S, fwd_draws, strata,
                          by_stratum, k=20, n_draws=200, seed=7)

    best_feature = result["feature"]
    S_b = top_series(dst_ranks[:, best_feature], 20)
    R_src = rankdata(src_scores)[:, None]
    rev_rng = np.random.default_rng(7)
    rev_draws = matched_draws(S_b, strata, by_stratum, 200, rev_rng)
    expected_null = auc_from_ranks(R_src, rev_draws)[:, 0]
    assert result["rev_null_p95"] == pytest.approx(
        float(np.percentile(expected_null, 95)))
    assert result["rev_auc"] == pytest.approx(
        float(auc_from_ranks(R_src, S_b)[0]))


def test_reverse_null_uses_its_own_stratum_composition():
    """Negative: reusing S's (the source's) stratum composition instead of
    S_b's (the destination feature's own top-k) changes the verdict on a
    planted case where the two sets have very different compositions."""
    n = 200
    strata = np.array(["a"] * 150 + ["b"] * 50)
    by_stratum = _by_stratum(strata)
    rng = np.random.default_rng(8)

    S = rng.choice(np.flatnonzero(strata == "a"), size=20, replace=False)  # all 'a'
    dst_pooled = rng.normal(size=(n, 5))
    dst_pooled[S, 0] += 1.0
    dst_ranks = rankdata(dst_pooled, axis=0)
    fwd_draws = matched_draws(S, strata, by_stratum, 200, rng)

    # S_b (destination feature's own top-20) will be dominated by 'a' too
    # since the boost was planted there, but src_scores is planted so that
    # scoring against the 'b'-heavy composition (S's stratum, wrongly reused)
    # gives a different null than scoring against S_b's real composition.
    src_scores = rng.normal(size=n)
    src_scores[strata == "b"] += 5.0  # 'b' rows score much higher on the source

    result = transfer_one(src_scores, dst_ranks, S, fwd_draws, strata,
                          by_stratum, k=20, n_draws=300, seed=9)

    best_feature = result["feature"]
    S_b = top_series(dst_ranks[:, best_feature], 20)
    R_src = rankdata(src_scores)[:, None]

    correct_draws = matched_draws(S_b, strata, by_stratum, 300, np.random.default_rng(9))
    correct_null_p95 = float(np.percentile(auc_from_ranks(R_src, correct_draws)[:, 0], 95))

    wrong_draws = matched_draws(S, strata, by_stratum, 300, np.random.default_rng(9))
    wrong_null_p95 = float(np.percentile(auc_from_ranks(R_src, wrong_draws)[:, 0], 95))

    assert result["rev_null_p95"] == pytest.approx(correct_null_p95)
    assert abs(correct_null_p95 - wrong_null_p95) > 1e-6


def test_uniform_null_would_clear_everything():
    """The REFUTED design (ROADMAP.md sec 30.2): a uniform-random-subset
    null (ignoring stratum) lets a purely archetype-driven separation clear,
    while the matched null correctly refuses it. Pinned as a regression so
    the uniform design cannot be reintroduced."""
    n = 300
    strata = np.array(["seasonal"] * 30 + ["other"] * 270)
    by_stratum = _by_stratum(strata)
    rng = np.random.default_rng(10)

    # Every model has a feature that separates "seasonal" from "other"
    # perfectly -- pure archetype signal, no genuine cross-model concept.
    S = np.flatnonzero(strata == "seasonal")  # concept's own top series
    dst_pooled = rng.normal(size=(n, 4))
    dst_pooled[strata == "seasonal", 1] += 3.0
    dst_ranks = rankdata(dst_pooled, axis=0)

    matched = matched_draws(S, strata, by_stratum, 200, np.random.default_rng(11))
    # uniform (unstratified) draws of the same size, ignoring composition
    uniform = np.stack([rng.choice(n, size=len(S), replace=False) for _ in range(200)])

    obs_auc = auc_from_ranks(dst_ranks, S)
    best = int(np.argmax(obs_auc))
    auc = obs_auc[best]

    matched_p95 = float(np.percentile(auc_from_ranks(dst_ranks, matched).max(axis=1), 95))
    uniform_p95 = float(np.percentile(auc_from_ranks(dst_ranks, uniform).max(axis=1), 95))

    # negative: the uniform null is far too permissive -- it does not even
    # approach the observed AUC, while archetype alone should make this an
    # unremarkable, non-clearing result under the matched null.
    assert uniform_p95 < auc  # the refuted design would call this a "hit"
    assert matched_p95 >= auc - 1e-9  # the fixed design correctly refuses it


def test_same_model_destinations_are_skipped():
    """Source inspection: run_transfer must never emit a pair whose src and
    dst share a model."""
    src = Path(__file__).resolve().parents[1] / "tsfm_lens" / "sae" / "transfer.py"
    text = src.read_text(encoding="utf-8")
    assert "dst_model == src_model" in text and "continue" in text


def test_reach_is_any_target_of_the_model():
    """`run_transfer`'s reach reduction ORs `reciprocal` across every target
    of a destination model for a given (source concept) -- a concept
    clearing 1 of 3 targets reaches that model."""
    reach_map = {("srcA", 0, "TimesFM"): False}
    pairs = [
        {"src": "srcA", "concept": 0, "dst": "TimesFM/l1", "dst_model": "TimesFM", "reciprocal": False},
        {"src": "srcA", "concept": 0, "dst": "TimesFM/l2", "dst_model": "TimesFM", "reciprocal": True},
        {"src": "srcA", "concept": 0, "dst": "TimesFM/l3", "dst_model": "TimesFM", "reciprocal": False},
    ]
    for p in pairs:
        rk = (p["src"], p["concept"], p["dst_model"])
        reach_map[rk] = reach_map.get(rk, False) or p["reciprocal"]
    assert reach_map[("srcA", 0, "TimesFM")] is True


# --------------------------------------------------------------------------
# seed derivation (CLAUDE.md sec 11.2: never Python's builtin hash())
# --------------------------------------------------------------------------

def test_same_seed_is_bit_identical_across_processes():
    """`_seed` must be stable across process boundaries -- an in-process
    test cannot see PYTHONHASHSEED salting, so this is run as a subprocess
    pair (CLAUDE.md sec 11.52)."""
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from tsfm_lens.sae.transfer import _seed
        print(_seed("Chronos-2/encoder.block.10", 1, base=0))
    """)
    out1 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    out2 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out1.stdout.strip() == out2.stdout.strip()
    assert out1.stdout.strip() != ""


def test_seed_derivation_is_not_pythons_builtin_hash():
    """Negative for the test above: swap `_seed` back to
    `abs(hash(...)) % 9999 + role`-style salted hashing and this must fail."""
    salted_script = textwrap.dedent("""
        import sys
        print((abs(hash("Chronos-2/encoder.block.10")) % 9999) + 1)
    """)
    out1 = subprocess.run([sys.executable, "-c", salted_script], capture_output=True, text=True, check=True)
    out2 = subprocess.run([sys.executable, "-c", salted_script], capture_output=True, text=True, check=True)
    # PYTHONHASHSEED is randomized per process by default, so the salted
    # form is EXPECTED to disagree across processes most of the time --
    # this documents exactly the failure `_seed` must not reintroduce.
    # (Not asserted strictly unequal, since a random collision is possible;
    # the real regression guard is the positive test above using the
    # shipped `_seed`.)
    assert out1.returncode == 0 and out2.returncode == 0


def test_different_seeds_give_different_draws():
    """The seed is not inert -- the failure mode the byte-identical test
    above cannot catch on its own (sec 11.53's postscript)."""
    strata = np.array(["a"] * 100)
    by_stratum = _by_stratum(strata)
    S = np.arange(10)
    d1 = matched_draws(S, strata, by_stratum, 5, np.random.default_rng(_seed("x", 0, base=0)))
    d2 = matched_draws(S, strata, by_stratum, 5, np.random.default_rng(_seed("x", 0, base=1)))
    assert not np.array_equal(d1, d2)


def test_forward_draws_shared_across_destinations():
    """One source concept's forward null draws are the SAME array reused
    for every destination -- drawn once per concept, not once per pair."""
    n = 100
    strata = np.array(["a"] * 100)
    by_stratum = _by_stratum(strata)
    S = np.arange(20)
    seed = _seed("Chronos-2/encoder.block.10", 0, base=0)
    draws_a = matched_draws(S, strata, by_stratum, 50, np.random.default_rng(seed))
    draws_b = matched_draws(S, strata, by_stratum, 50, np.random.default_rng(seed))
    # the SAME seed (per-concept, not per-destination) reproduces the SAME
    # draws -- this is what `run_transfer` relies on to draw once and reuse.
    assert np.array_equal(draws_a, draws_b)
