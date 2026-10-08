"""ROADMAP.md sec 30.4.2 -- cross-model SAE CONCEPT transfer, stage 3 of sec
30.10's five-stage build. Answers: take a concept in model A, take the series
it fires hardest on, and ask whether model B groups those same series too.

This is a test over SHARED INPUTS (every model's `sae_pooled/{model}/{layer}`
row `i` is the same corpus series, sec 30.2), so unlike the response-
fingerprint cosine matching in `role_matching.py` it needs no untrained-twin
floor -- its null is a resample of *which series were picked*, which is free
and unimpeachable as long as that resample preserves the source set's own
per-stratum (archetype-or-family) composition. Sec 30.2 measured and REFUTED
a uniform-random-subset null (30 of 30 pairs "transferred" at AUC 0.876-0.996,
because a concept whose top series are mostly one archetype is trivially
separable by any model with a feature for that archetype) -- `matched_draws`
below is the fix, and `test_uniform_null_would_clear_everything` pins the
refutation as a regression so it cannot be reintroduced.

Seeds are derived via `_seed`'s stable sha256 digest, NEVER via Python's
builtin `hash()` -- salted per process (`CLAUDE.md` sec 11.2), and exactly the
trap the sec 30.2 prototype reintroduced, which is why its own 509/830 count
could not reproduce itself (505-513 across four re-runs of the *same*
script). The real module's acceptance criterion (sec 30.9 criterion 2) is
therefore a band plus a same-seed-is-byte-identical check, not an equality.

ROADMAP.md sec 37 P3 -- p-values, FDR and a cross-model concept ATLAS
extension of this same machinery, added without moving anything the sections
above already promise stays put:

  1. `transfer_one` gains `p`/`rev_p` (a search-corrected, floor-respecting
     permutation p per leg -- `(1 + #{null >= obs}) / (1 + n)`) and
     `p_method`/`rev_p_method`, recording how each was resolved. These are
     ADDITIVE keys; `auc`/`clears`/`null_p95`/... are computed exactly as
     before and are byte-identical.
  2. At `n_null=200` (the default `sae.transfer_n_null`) the exact p floor is
     `1/201`. BH is step-up, so `k` floor-level p-values in a family of `m`
     survive together once `k >= m / ((n_null+1) q)`, but a LONE effect in a
     large family cannot (its rank-1 bar `q/m` sits below the floor --
     `doctor.py`'s `check_transfer_fdr_budget`). For any leg whose exact p
     sits exactly on that floor, the `adaptive` method redraws a larger null
     (up to `max_redraw`, default 5000) from the SAME seed -- deterministic,
     because `matched_draws`' per-draw loop only depends on the running RNG
     state, so the first `n_draws` of a `max_redraw`-draw call are bit-
     identical to a standalone `n_draws` call from the same seed. A
     generalized-Pareto tail extrapolation (Knijnenburg et al. 2009) was
     built and removed: on `runs/full_report_run_4model` only 2 of 16
     floor-hitting legs landed within a factor of 2 of a 5,000-draw exact p
     (ratios 5.26e-06 to 13.07; a 20-point tail fit at `n_null=200`), ROADMAP
     sec 37 P3.
  3. `run_atlas_transfer` tests every cross-model concept ATLAS's per-model
     PART (`sae/concept_atlas.py`'s pooled, cross-model clusters) against
     every OTHER model's targets, reusing `transfer_one`/`matched_draws`/
     `_seed` directly -- no re-derivation of the draw-building or scoring
     path `run_transfer` already uses. Writes `sae/atlas_transfer.json`.
  4. `benjamini_hochberg` applies BH-FDR control within one ordered
     (source model, destination model) pair's own family of p-values,
     SEPARATELY for the forward and reverse leg -- two independent families
     per pair, per `concepts.transfer_fdr_q`. A test is `reciprocal_fdr`
     only when BOTH legs survive BH in their own family. `run_transfer`'s
     `pairs`/`reach`/`matrix`/`universality` stay byte-identical; `p_bh`,
     `survives_fdr`, `rev_p_bh`, `rev_survives_fdr`, `reciprocal_fdr`, and
     the FDR-scored siblings `reach_fdr`/`matrix_fdr`/`universality_fdr` are
     additive.

ROADMAP.md sec 37.10 P7b -- `transfer_one_fixed_feature` is a sharper,
frozen-feature sibling of `transfer_one`: `analysis/hypotheses.py`/`analysis/
confirm.py`'s `concepts.transfer_claim_mode: "frozen"` path uses it instead
of re-deriving an argmax destination feature, so a private replication can
claim "this SPECIFIC dev feature pair selects the same series" rather than
only "the destination dictionary has some feature that does". `transfer_one`
itself is untouched by this addition.

Evidence class is unchanged by any of this: a transfer test (uncorrected or
FDR-controlled) is still a claim about shared INPUT SELECTIVITY, never about
a shared causal effect (that is `sae/concept_atlas.py`'s own, narrower claim
-- see its module docstring) and never causal in itself.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..extraction.store import ActivationStore, load_meta
from ..utils import log, save_json

__all__ = ["series_strata", "concept_scores", "top_series", "matched_draws",
          "auc_from_ranks", "transfer_one", "transfer_one_fixed_feature", "run_transfer",
          "benjamini_hochberg", "run_atlas_transfer"]

_MAX_REDRAW_DEFAULT = 5000
_P_METHODS = ("exact", "adaptive")


def series_strata(meta: pd.DataFrame) -> np.ndarray:
    """One stratum label per series: `archetype`, falling back to `family`
    for the rows with no archetype. The fallback is load-bearing -- sec
    30.2's reference run has `archetype` populated only for its 425
    `random_parametric` series (540 of 965 are null); dropping the null rows
    instead of falling back would restrict the whole test to one generator
    family.
    """
    return meta["archetype"].fillna(meta["family"]).to_numpy()


def concept_scores(pooled: np.ndarray, feature_ids: list) -> np.ndarray:
    """Per-series score for one concept: the mean of its member features'
    pooled SAE activation, `(n_series,)`."""
    return np.asarray(pooled[:, feature_ids], dtype=np.float64).mean(axis=1)


def top_series(score: np.ndarray, k: int) -> np.ndarray:
    """Index of the `k` highest-scoring series, descending. Works equally
    on raw scores or on a rank column (rank is a monotone transform of
    score, so the top-k set is identical either way) -- `transfer_one`
    relies on this to pick a destination feature's own top-k directly from
    the precomputed rank matrix, with no second array to keep in sync."""
    return np.argsort(-score)[:k]


def _by_stratum(strata: np.ndarray) -> dict:
    """Series indices grouped by stratum value, computed once per run and
    reused by every `matched_draws` call -- the whole point of hoisting this
    out of the per-concept loop."""
    return {val: np.flatnonzero(strata == val) for val in np.unique(strata)}


def matched_draws(S: np.ndarray, strata: np.ndarray, by_stratum: dict,
                  n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """`(n_draws, len(S))` index array, each row drawn WITHOUT REPLACEMENT
    from `by_stratum` so it has `S`'s own exact per-stratum composition --
    the fix for sec 30.2's refuted uniform-random-subset null. Every
    stratum value present in `S` has a pool at least as large as `S`'s own
    count in it (S is itself a subset of that pool), so sampling without
    replacement never runs short.

    Deterministic and PREFIX-STABLE in `n_draws`: because every draw only
    consumes RNG state sequentially (no state is read or reset between
    draws besides `pos`), the first `k` rows of a call with `n_draws=N` are
    bit-identical to a standalone call with `n_draws=k` given the SAME
    seed -- `_resolve_p`'s adaptive-redraw path below relies on exactly
    this to extend a null sample without re-deriving the seed.
    """
    k = len(S)
    S_strata = strata[S]
    vals, counts = np.unique(S_strata, return_counts=True)
    draws = np.empty((n_draws, k), dtype=np.int64)
    for d in range(n_draws):
        pos = 0
        for val, cnt in zip(vals, counts):
            pool = by_stratum[val]
            draws[d, pos:pos + cnt] = rng.choice(pool, size=int(cnt), replace=False)
            pos += cnt
    return draws


def auc_from_ranks(R: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUC per feature via the rank-sum formula, from a
    PRECOMPUTED rank matrix `R` (`scipy.stats.rankdata(pooled, axis=0)`,
    `(n_series, n_features)`) -- so `n_draws` null draws are a cheap
    gather-and-sum over `R`, never a re-rank.

    `AUC = (rank_sum_of_positive_group - k*(k+1)/2) / (k*(n-k))`.

    `idx` is `(k,)` for one index set -> returns `(n_features,)`; `(n_draws,
    k)` for a batch of draws -> returns `(n_draws, n_features)`.
    """
    if R.dtype.itemsize < 8:
        raise TypeError(
            f"auc_from_ranks needs float64 ranks, got {R.dtype}: scipy's rankdata "
            "keeps a float16 input's dtype, which quantizes rank sums and AUCs")
    n = R.shape[0]
    idx = np.asarray(idx)
    k = idx.shape[-1]
    rank_sum = R[idx].sum(axis=idx.ndim - 1)
    return (rank_sum - k * (k + 1) / 2.0) / (k * (n - k))


def _seed(*parts, base: int) -> int:
    """Stable per-(parts) seed via sha256 -- NEVER Python's builtin
    `hash()`, which is salted per process (`CLAUDE.md` sec 11.2) and is
    exactly what made the sec 30.2 prototype's own counts unreproducible
    run to run. `base` is `transfer_seed`, so the whole artifact moves as
    one when the config seed moves and is otherwise fixed forever."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return (base + int.from_bytes(h[:4], "big")) % (2 ** 32)


# ---------------------------------------------------------------------------
# p-values: exact and adaptive-redraw (sec 37 P3).
# ---------------------------------------------------------------------------

def _exact_p(obs: float, null_vals: np.ndarray) -> tuple:
    """`-> (p, hits_floor)`. The standard permutation-test p,
    `(1 + #{null >= obs}) / (1 + n)`, floored at `1/(n+1)` and never
    exactly 0 -- `CLAUDE.md` sec 6.6's bootstrap-p discipline applied to a
    permutation test. `hits_floor` is `True` only when NO null draw reached
    `obs` (`count == 0`), i.e. the floor is the whole story the exact test
    can tell -- adaptive only ever engages in that case."""
    n = int(null_vals.size)
    count = int(np.sum(null_vals >= obs))
    return (1 + count) / (1 + n), count == 0


def _p_method(concepts_cfg) -> str:
    """`concepts.transfer_p_method`, validated up front: an unknown method
    must fail before any draws are spent, not on the first floor-hitting leg
    (which may never occur on a small run, leaving the typo silent)."""
    method = str(getattr(concepts_cfg, "transfer_p_method", "exact") or "exact")
    if method not in _P_METHODS:
        raise ValueError(f"concepts.transfer_p_method must be one of {_P_METHODS}, "
                         f"got {method!r}")
    return method


def _resolve_p(obs: float, null_vals: np.ndarray, method: str, redraw_fn,
               max_redraw: int = _MAX_REDRAW_DEFAULT) -> tuple:
    """`-> (p, p_method_used)`. `redraw_fn(n)` draws a FRESH `n`-row null
    (from the same seed as the original, per `matched_draws`' prefix
    stability) and returns the corresponding null-statistic array; only
    called for `method == "adaptive"` and only when the exact p already
    hit the floor -- an exact p that does NOT hit the floor is left alone
    regardless of `method`, since it already has a resolvable value.
    """
    p, hits_floor = _exact_p(obs, null_vals)
    if not hits_floor or method == "exact":
        return p, "exact"
    if method == "adaptive":
        big_null = redraw_fn(max_redraw)
        big_p, _ = _exact_p(obs, big_null)
        return big_p, "adaptive"
    raise ValueError(f"unknown transfer_p_method: {method!r}")


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR control (sec 37 P3 item 4).
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvals: dict, q: float) -> dict:
    """Standard Benjamini-Hochberg step-up FDR control over ONE family of
    p-values (never pool across families -- each ordered (source model,
    destination model) pair, per leg, is its own family, `CLAUDE.md` sec
    6.6's discipline for Holm applied to BH here).

    `-> {key: {"p": float, "p_bh": float, "survives": bool}}`. `p_bh` is
    the usual monotone step-up adjusted p (`min_{j>=rank} p_(j) * m / j`);
    `survives` is whether that test is rejected at level `q` under the
    standard BH decision rule (reject `p_(1..k)` where `k` is the largest
    rank with `p_(k) <= (k/m)*q`). Empty input returns `{}`.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    if m == 0:
        return {}
    adj = [0.0] * m
    running = 1.0
    for i in range(m - 1, -1, -1):
        _key, p = items[i]
        rank = i + 1
        running = min(running, p * m / rank)
        adj[i] = running
    survive_upto = 0
    for rank, (_key, p) in enumerate(items, start=1):
        if p <= (rank / m) * q:
            survive_upto = rank
    out = {}
    for i, (key, p) in enumerate(items):
        rank = i + 1
        out[key] = {"p": float(p), "p_bh": float(min(1.0, adj[i])),
                    "survives": rank <= survive_upto}
    return out


def _apply_pairwise_fdr(tests: list, key_src: str, key_dst: str, q: float) -> list:
    """BH within each ordered `(tests[i][key_src], tests[i][key_dst])`
    group, SEPARATELY for the forward leg's `p` and the reverse leg's
    `rev_p` -- two independent correction families per pair. Mutates and
    returns `tests` in place; a test survives `reciprocal_fdr` only when
    BOTH legs survive BH in their own family. Shared by `run_transfer` and
    `run_atlas_transfer` so the grouping/BH logic exists in exactly one
    place.
    """
    groups: dict = {}
    for i, t in enumerate(tests):
        groups.setdefault((t[key_src], t[key_dst]), []).append(i)
    for idxs in groups.values():
        fwd = {i: tests[i]["p"] for i in idxs}
        rev = {i: tests[i]["rev_p"] for i in idxs}
        fwd_bh = benjamini_hochberg(fwd, q)
        rev_bh = benjamini_hochberg(rev, q)
        for i in idxs:
            f, r = fwd_bh[i], rev_bh[i]
            tests[i]["p_bh"] = f["p_bh"]
            tests[i]["survives_fdr"] = f["survives"]
            tests[i]["rev_p_bh"] = r["p_bh"]
            tests[i]["rev_survives_fdr"] = r["survives"]
            tests[i]["reciprocal_fdr"] = bool(f["survives"] and r["survives"])
    return tests


def _reach_matrix_universality(live: dict, pairs: list, field: str) -> tuple:
    """`-> (reach, matrix, universality)` -- the reduction from per-pair
    test records to "did source concept X reach destination model M" (OR
    across every target of M), then to a model x model rate matrix, then to
    a universality histogram. Parameterized on which boolean `field` to OR
    over (`"reciprocal"` for the uncorrected reduction, `"reciprocal_fdr"`
    for the FDR-controlled sibling) so `run_transfer` computes both from
    ONE implementation rather than two copies that could drift apart.
    """
    reach_map: dict = {}
    for p in pairs:
        rk = (p["src"], p["concept"], p["dst_model"])
        reach_map[rk] = reach_map.get(rk, False) or p[field]
    reach = [{"src": s, "concept": c, "dst_model": m, "reciprocal": r}
            for (s, c, m), r in sorted(reach_map.items())]

    matrix: dict = {}
    matrix_counts: dict = {}
    for (src_key, _concept, dst_model), r in reach_map.items():
        src_model = live[src_key]["model"]
        matrix_counts.setdefault(src_model, {}).setdefault(dst_model, [0, 0])
        matrix_counts[src_model][dst_model][0] += int(r)
        matrix_counts[src_model][dst_model][1] += 1
    for src_model, dsts in matrix_counts.items():
        matrix[src_model] = {dst_model: (n_hit / n_total if n_total else 0.0)
                             for dst_model, (n_hit, n_total) in dsts.items()}

    other_models = {live[k]["model"] for k in live}
    n_reached: dict = {}
    for (src_key, concept_id), _ in {(s, c): None for (s, c, _m) in reach_map}.items():
        src_model = live[src_key]["model"]
        n_others = len({m for m in other_models if m != src_model})
        reached = sum(1 for (s2, c2, _m2), r in reach_map.items()
                     if s2 == src_key and c2 == concept_id and r)
        n_reached[(src_key, concept_id)] = (reached, n_others)

    universality: dict = {}
    for reached, _n_others in n_reached.values():
        universality[str(reached)] = universality.get(str(reached), 0) + 1

    return reach, matrix, universality


def transfer_one(src_scores: np.ndarray, dst_ranks: np.ndarray, S: np.ndarray,
                 fwd_draws: np.ndarray, strata: np.ndarray, by_stratum: dict,
                 k: int = 20, n_draws: int = 200, seed: int = 0,
                 p_method: str = "exact",
                 max_redraw: int = _MAX_REDRAW_DEFAULT,
                 fwd_seed: int | None = None) -> dict:
    """Forward + reverse legs for ONE (concept, destination target) pair.

    `seed` seeds the reverse leg's null; `fwd_seed` is the seed the caller
    built `fwd_draws` from. `p_method="adaptive"` needs it, so the redrawn
    forward null EXTENDS `fwd_draws` (prefix stability of `matched_draws`)
    rather than being a fresh draw from the reverse leg's seed.

    `fwd_draws` is the `(n_draws, len(S))` matched-draw block the CALLER
    built once for this source concept and reuses across every destination
    -- only the reverse leg draws inside this function, because only its
    `S_b` (the destination feature's own top-k series) depends on the
    destination.

    forward: best feature in the destination's WHOLE dictionary by AUC on
             `S`, vs the p95 of the MAX-over-features null AUC -- a
             per-feature null would ignore the search over `dict_size`
             candidates and clear almost everything
             (`test_forward_null_is_max_over_features`). `p` uses that SAME
             max-over-features null, so it is search-corrected too.
    reverse: that feature's own top-k series `S_b`, scored by `src_scores`,
             vs the p95 over matched draws of that SAME single score -- no
             max here, since the observed statistic has no search either
             (`test_reverse_leg_has_no_search_correction`). The null is
             redrawn against `S_b`'s OWN stratum composition, never `S`'s.
             `rev_p` uses this same single-statistic null.

    `p_method` (`"exact"` | `"adaptive"`, sec 37 P3) only
    changes behavior for a leg whose EXACT p hits the floor
    `1/(n_draws+1)`; `null_p95`/`clears`/`auc`/... are computed exactly as
    before `p_method` existed and never depend on it.
    """
    obs_auc = auc_from_ranks(dst_ranks, S)
    best_feature = int(np.argmax(obs_auc))
    auc = float(obs_auc[best_feature])
    null_auc = auc_from_ranks(dst_ranks, fwd_draws)
    null_max = null_auc.max(axis=1)
    null_p95 = float(np.percentile(null_max, 95))
    clears = bool(auc > null_p95)

    if p_method == "adaptive" and fwd_seed is None:
        raise ValueError("transfer_one: p_method='adaptive' needs fwd_seed, the seed "
                         "fwd_draws was built from")

    def _fwd_redraw(n: int) -> np.ndarray:
        draws = matched_draws(S, strata, by_stratum, n, np.random.default_rng(fwd_seed))
        return auc_from_ranks(dst_ranks, draws).max(axis=1)

    p, p_used = _resolve_p(auc, null_max, p_method, _fwd_redraw, max_redraw=max_redraw)

    S_b = top_series(dst_ranks[:, best_feature], k)

    R_src = rankdata(src_scores)[:, None]
    rev_auc = float(auc_from_ranks(R_src, S_b)[0])
    rng = np.random.default_rng(seed)
    rev_draws = matched_draws(S_b, strata, by_stratum, n_draws, rng)
    rev_null = auc_from_ranks(R_src, rev_draws)[:, 0]
    rev_null_p95 = float(np.percentile(rev_null, 95))
    rev_clears = bool(rev_auc > rev_null_p95)

    def _rev_redraw(n: int) -> np.ndarray:
        draws = matched_draws(S_b, strata, by_stratum, n, np.random.default_rng(seed))
        return auc_from_ranks(R_src, draws)[:, 0]

    rev_p, rev_p_used = _resolve_p(rev_auc, rev_null, p_method, _rev_redraw,
                                   max_redraw=max_redraw)

    return {"auc": auc, "feature": best_feature, "null_p95": null_p95,
           "clears": clears, "p": p, "p_method": p_used,
           "rev_auc": rev_auc, "rev_null_p95": rev_null_p95,
           "rev_clears": rev_clears, "rev_p": rev_p, "rev_p_method": rev_p_used,
           "reciprocal": bool(clears and rev_clears)}


def transfer_one_fixed_feature(src_scores: np.ndarray, dst_ranks: np.ndarray, S: np.ndarray,
                               fwd_draws: np.ndarray, strata: np.ndarray, by_stratum: dict,
                               feature: int, k: int = 20, n_draws: int = 200, seed: int = 0,
                               p_method: str = "exact",
                               max_redraw: int = _MAX_REDRAW_DEFAULT,
                               fwd_seed: int | None = None) -> dict:
    """Forward + reverse legs for ONE (concept, destination target) pair with
    the destination FEATURE FROZEN -- ROADMAP.md sec 37.10 P7b's sharper
    claim, "this SPECIFIC dev feature pair selects the same series", as
    opposed to `transfer_one`'s "the destination dictionary has SOME feature
    that does". `transfer_one` itself is untouched (its own tests, and every
    existing caller, stay byte-identical); this is the "small helper beside
    it" the design calls for rather than a fork of the stratum machinery --
    every argument, and every returned key, matches `transfer_one`'s shape
    exactly, so a caller can dispatch on `mode` without changing anything
    else about how it reads the result.

    Only the FORWARD leg's null differs from `transfer_one`: since there is
    no search over features, `null_p95`/`p` come from `feature`'s OWN AUC
    over the SAME matched draws, never a max over the whole dictionary (no
    `.max(axis=1)` anywhere in this function) -- `test_frozen_null_is_not_
    max_over_features` pins this. The reverse leg is IDENTICAL in shape to
    `transfer_one`'s own (it never searched either); only its `best_feature`
    input is `feature` directly instead of an argmax result.
    """
    dst_col = dst_ranks[:, [feature]]
    obs_auc = auc_from_ranks(dst_col, S)
    auc = float(obs_auc[0])
    null_auc = auc_from_ranks(dst_col, fwd_draws)[:, 0]
    null_p95 = float(np.percentile(null_auc, 95))
    clears = bool(auc > null_p95)

    if p_method == "adaptive" and fwd_seed is None:
        raise ValueError("transfer_one_fixed_feature: p_method='adaptive' needs fwd_seed, "
                         "the seed fwd_draws was built from")

    def _fwd_redraw(n: int) -> np.ndarray:
        draws = matched_draws(S, strata, by_stratum, n, np.random.default_rng(fwd_seed))
        return auc_from_ranks(dst_col, draws)[:, 0]

    p, p_used = _resolve_p(auc, null_auc, p_method, _fwd_redraw, max_redraw=max_redraw)

    S_b = top_series(dst_ranks[:, feature], k)

    R_src = rankdata(src_scores)[:, None]
    rev_auc = float(auc_from_ranks(R_src, S_b)[0])
    rng = np.random.default_rng(seed)
    rev_draws = matched_draws(S_b, strata, by_stratum, n_draws, rng)
    rev_null = auc_from_ranks(R_src, rev_draws)[:, 0]
    rev_null_p95 = float(np.percentile(rev_null, 95))
    rev_clears = bool(rev_auc > rev_null_p95)

    def _rev_redraw(n: int) -> np.ndarray:
        draws = matched_draws(S_b, strata, by_stratum, n, np.random.default_rng(seed))
        return auc_from_ranks(R_src, draws)[:, 0]

    rev_p, rev_p_used = _resolve_p(rev_auc, rev_null, p_method, _rev_redraw,
                                   max_redraw=max_redraw)

    return {"auc": auc, "feature": int(feature), "null_p95": null_p95,
           "clears": clears, "p": p, "p_method": p_used,
           "rev_auc": rev_auc, "rev_null_p95": rev_null_p95,
           "rev_clears": rev_clears, "rev_p": rev_p, "rev_p_method": rev_p_used,
           "reciprocal": bool(clears and rev_clears)}


def _store_context(run_dir: Path) -> tuple:
    """`(strata, by_stratum, store)` shared by `run_transfer` and
    `run_atlas_transfer` -- one place opens the store and derives strata so
    the two functions cannot compute them differently."""
    meta = load_meta(run_dir)
    strata = series_strata(meta)
    by_stratum = _by_stratum(strata)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    return strata, by_stratum, store


def _pooled_ranks_fns(store: ActivationStore):
    """`(pooled_fn, ranks_fn)` closures caching `space="sae"` pooled
    features and their rank transform per `"model/layer"` key, shared by
    both drivers below."""
    pooled_cache: dict = {}
    ranks_cache: dict = {}

    def _pooled(key: str) -> np.ndarray:
        if key not in pooled_cache:
            model, layer = key.split("/", 1)
            pooled_cache[key] = store.load(model, layer, level="series", space="sae")
        return pooled_cache[key]

    def _ranks(key: str) -> np.ndarray:
        if key not in ranks_cache:
            ranks_cache[key] = rankdata(np.asarray(_pooled(key), dtype=np.float64), axis=0)
        return ranks_cache[key]

    return _pooled, _ranks


def run_transfer(run_dir: Path, concepts: dict, cfg) -> dict:
    """All concepts x all targets of every OTHER model. Writes
    `sae/transfer.json` (sec 30.5's schema): `pairs` is the raw evidence,
    `reach`/`matrix`/`universality` are reductions over it, written
    together so a reader can audit a rate back to the pair rows that
    produced it. `reach_fdr`/`matrix_fdr`/`universality_fdr` (sec 37 P3)
    are the SAME reductions over `reciprocal_fdr` instead of `reciprocal`;
    the uncorrected fields are computed identically to before this item and
    are byte-identical (`test_uncorrected_transfer_keys_are_byte_identical`).

    Skips same-model destinations (a concept transferring to another layer
    of its own model is a depth statement, not a cross-model one) and any
    withheld target. A source target with no concepts (non-modular, or
    fewer than the causal-candidate floor) contributes no pairs but a
    non-modular DESTINATION still participates -- its whole dictionary is
    a legitimate search space regardless of whether it partitions into
    named concepts.
    """
    run_dir = Path(run_dir)
    sae_cfg = getattr(cfg, "sae", None)
    concepts_cfg = getattr(cfg, "concepts", None)
    k_top = int(getattr(sae_cfg, "transfer_top_k", 20))
    n_null = int(getattr(sae_cfg, "transfer_n_null", 200))
    base_seed = int(getattr(sae_cfg, "transfer_seed", 0))
    p_method = _p_method(concepts_cfg)
    fdr_q = float(getattr(concepts_cfg, "transfer_fdr_q", 0.05) or 0.05)
    max_redraw = int(getattr(concepts_cfg, "transfer_max_redraw", _MAX_REDRAW_DEFAULT) or _MAX_REDRAW_DEFAULT)

    strata, by_stratum, store = _store_context(run_dir)
    _pooled, _ranks = _pooled_ranks_fns(store)

    targets = concepts.get("targets", {})
    live = {key: t for key, t in targets.items() if not t.get("withheld")}

    pairs: list = []
    for src_key, src_target in sorted(live.items()):
        src_model = src_target["model"]
        src_concepts = src_target.get("concepts", [])
        if not src_concepts:
            continue
        src_pooled = _pooled(src_key)
        for concept in src_concepts:
            feature_ids = concept["features"]
            score = concept_scores(src_pooled, feature_ids)
            S = top_series(score, k_top)
            fwd_seed = _seed(src_key, concept["concept"], base=base_seed)
            fwd_rng = np.random.default_rng(fwd_seed)
            fwd_draws = matched_draws(S, strata, by_stratum, n_null, fwd_rng)

            for dst_key, dst_target in sorted(live.items()):
                dst_model = dst_target["model"]
                if dst_model == src_model:
                    continue
                dst_ranks = _ranks(dst_key)
                rev_seed = _seed(src_key, concept["concept"], dst_key, base=base_seed)
                result = transfer_one(score, dst_ranks, S, fwd_draws, strata,
                                      by_stratum, k=k_top, n_draws=n_null,
                                      seed=rev_seed, p_method=p_method,
                                      max_redraw=max_redraw, fwd_seed=fwd_seed)
                pairs.append({"src": src_key, "src_model": src_model, "concept": concept["concept"],
                             "dst": dst_key, "dst_model": dst_model, **result})

    pairs = _apply_pairwise_fdr(pairs, "src_model", "dst_model", fdr_q)

    reach, matrix, universality = _reach_matrix_universality(live, pairs, "reciprocal")
    reach_fdr, matrix_fdr, universality_fdr = _reach_matrix_universality(
        live, pairs, "reciprocal_fdr")

    out = {"schema_version": 1, "k_top_series": k_top, "n_null_draws": n_null,
          "stratum_field": "archetype|family", "p_method": p_method, "fdr_q": fdr_q,
          "pairs": pairs, "reach": reach, "matrix": matrix, "universality": universality,
          "reach_fdr": reach_fdr, "matrix_fdr": matrix_fdr,
          "universality_fdr": universality_fdr}
    out_path = run_dir / "sae" / "transfer.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    n_clear = sum(1 for p in pairs if p["clears"])
    n_recip = sum(1 for p in pairs if p["reciprocal"])
    n_recip_fdr = sum(1 for p in pairs if p["reciprocal_fdr"])
    log.info("sae transfer: wrote %s (%d pairs, %d/%d forward clear, "
            "%d/%d reciprocal, %d/%d reciprocal under per-leg FDR q=%s)",
            out_path, len(pairs), n_clear, len(pairs), n_recip, len(pairs),
            n_recip_fdr, len(pairs), fdr_q)
    return out


# ---------------------------------------------------------------------------
# Atlas transfer (sec 37 P3 item 3): the cross-model concept ATLAS's own
# per-model parts, tested against every other model's dictionaries.
# ---------------------------------------------------------------------------

def _live_atlas_targets(meta_sae: dict) -> dict:
    """`-> {"model/layer": model}` for every SAE target whose features are
    persisted -- the destination (and eligible source) universe for atlas
    transfer, built straight from `sae/meta.json` since `run_atlas_transfer`
    is handed only `(run_dir, atlas, cfg)`, not `concepts.json`'s own
    target dict (which `run_transfer` above uses instead)."""
    out: dict = {}
    for key, rec in (meta_sae or {}).items():
        if isinstance(rec, dict) and rec.get("features_persisted"):
            model, _layer = key.split("/", 1)
            out[key] = model
    return out


def _concept_source_parts(atlas_rows: list, live: dict) -> dict:
    """`-> {(concept_id, "model/layer"): {"model": m, "features": [...]}}`
    -- every atlas concept's membership, grouped by the (concept, TARGET) it
    lives at, restricted to targets with persisted features (an atlas
    concept can have members at a target whose SAE features were never
    persisted; that part of the concept cannot be tested and is silently
    excluded from the source side here, exactly as `run_transfer` excludes
    withheld targets from its own `live` set)."""
    out: dict = {}
    for r in atlas_rows or []:
        cid = r.get("concept")
        if cid is None:
            continue
        src_key = f"{r['model']}/{r['layer']}"
        if src_key not in live:
            continue
        entry = out.setdefault((int(cid), src_key), {"model": r["model"], "features": []})
        entry["features"].append(int(r["feature"]))
    return out


def _atlas_concept_summary(atlas: dict, tests: list) -> tuple:
    """`-> (concept_summary [list], cross_check_2x2 {dict})`.

    One row per atlas concept: which models its parts transfer to under FDR
    (OR over every source target at that model, mirroring `run_transfer`'s
    own reach reduction), and, for every ordered (source model, destination
    model) pair this concept was actually TESTED against, whether the pair
    also spans both models in EFFECT space (both hold >=1 member of this
    same atlas concept) and whether it transfers in INPUT space
    (FDR-reciprocal at any tested target pair).

    The 2x2 cross-check is built ONLY from (concept, src_model, dst_model)
    triples that were actually tested -- a concept whose only members at a
    model sit at a non-live (features-not-persisted) target contributes no
    cell rather than a fabricated "No" (`CLAUDE.md` sec 11.37's
    three-states discipline: untested is not the same claim as "does not
    transfer").
    """
    concepts_by_id = {int(c["concept"]): c for c in (atlas.get("concepts") or [])}

    triple_reciprocal: dict = {}
    triple_seen: set = set()
    for t in tests:
        key = (int(t["concept"]), t["src_model"], t["dst_model"])
        triple_seen.add(key)
        triple_reciprocal[key] = triple_reciprocal.get(key, False) or bool(t["reciprocal_fdr"])

    cross_check = {"effect_yes_input_yes": 0, "effect_yes_input_no": 0,
                   "effect_no_input_yes": 0, "effect_no_input_no": 0}
    concept_summary: list = []
    for cid, crec in sorted(concepts_by_id.items()):
        members_models = set((crec.get("models") or {}).keys())
        transfers_to: dict = {}
        pairs_tested: list = []
        for (c2, src_model, dst_model) in sorted(triple_seen):
            if c2 != cid:
                continue
            input_yes = bool(triple_reciprocal[(c2, src_model, dst_model)])
            effect_yes = dst_model in members_models
            transfers_to[dst_model] = transfers_to.get(dst_model, False) or input_yes
            pairs_tested.append({"src_model": src_model, "dst_model": dst_model,
                                 "spans_effect_space": effect_yes,
                                 "transfers_input_space": input_yes})
            cell = ("effect_yes" if effect_yes else "effect_no") + "_" + \
                   ("input_yes" if input_yes else "input_no")
            cross_check[cell] += 1
        concept_summary.append({
            "concept": cid, "name": crec.get("name"),
            "models_at_concept": sorted(members_models),
            "transfers_to_models_fdr": sorted(m for m, v in transfers_to.items() if v),
            "pairs": pairs_tested,
        })
    return concept_summary, cross_check


def atlas_transfer_tests(atlas: dict, live: dict, strata: np.ndarray, by_stratum: dict,
                         pooled_fn, ranks_fn, k_top: int, n_null: int, base_seed: int,
                         p_method: str, max_redraw: int,
                         extra_destinations: dict | None = None,
                         only_extra: bool = False) -> list:
    """The uncorrected test list of `run_atlas_transfer`: every atlas concept
    part (source = concept x one live target) against every live target of
    another model.

    Factored out so a caller with a DIFFERENT set of pooled features (the
    window-sensitivity recomputation, `analysis/window_sensitivity.py`) or an
    extra destination (`sae/control_transfer.py`) runs the identical test with
    the identical seeds. `pooled_fn(key)` and `ranks_fn(key)` give a target's
    series-level features and their column ranks. `extra_destinations` is
    `{dst_key: (dst_model, dst_ranks)}`, tested in addition to `live` (a
    destination there is skipped for a source of the same model); `only_extra` tests
    the extra destinations alone, `live` then supplying sources only.
    """
    by_concept_src = _concept_source_parts(atlas.get("rows") or [], live)
    tests: list = []
    for (cid, src_key), rec in sorted(by_concept_src.items()):
        src_model = rec["model"]
        feature_ids = sorted(rec["features"])
        score = concept_scores(pooled_fn(src_key), feature_ids)
        S = top_series(score, k_top)
        fwd_seed = _seed("atlas", src_key, cid, base=base_seed)
        fwd_rng = np.random.default_rng(fwd_seed)
        fwd_draws = matched_draws(S, strata, by_stratum, n_null, fwd_rng)

        dests = [] if only_extra else [(k, m, None) for k, m in sorted(live.items())]
        dests += [(k, m, r) for k, (m, r) in sorted((extra_destinations or {}).items())]
        for dst_key, dst_model, dst_ranks in dests:
            if dst_model == src_model:
                continue
            if dst_ranks is None:
                dst_ranks = ranks_fn(dst_key)
            rev_seed = _seed("atlas", src_key, cid, dst_key, base=base_seed)
            result = transfer_one(score, dst_ranks, S, fwd_draws, strata, by_stratum,
                                  k=k_top, n_draws=n_null, seed=rev_seed, p_method=p_method,
                                  max_redraw=max_redraw, fwd_seed=fwd_seed)
            tests.append({"concept": cid, "src_target": src_key, "src_model": src_model,
                          "dst_target": dst_key, "dst_model": dst_model, **result})
    return tests


def run_atlas_transfer(run_dir: Path, atlas: dict, cfg) -> dict:
    """Every cross-model ATLAS concept's per-model PART, tested against
    every OTHER model's targets, reusing `transfer_one`/`matched_draws`/
    `_seed` directly (never re-derived). Writes `sae/atlas_transfer.json`.

    A concept's "part at model A" is the pooled mean of every one of its
    member features that live at a target of A -- `concept_scores` applied
    across possibly several of A's own targets at once, mirroring how
    `run_transfer` scores one target's own per-target concept. Unlike
    `run_transfer`, the SOURCE unit here is (concept, one target of A): a
    concept spanning two of A's own layers contributes two source rows
    (one per target), each scored on ITS OWN target's pooled features --
    concatenating features across differently-scaled targets into one
    score would conflate two different SAE dictionaries' activation
    scales, which nothing in this build justifies.
    """
    run_dir = Path(run_dir)
    sae_cfg = getattr(cfg, "sae", None)
    concepts_cfg = getattr(cfg, "concepts", None)
    k_top = int(getattr(sae_cfg, "transfer_top_k", 20))
    n_null = int(getattr(sae_cfg, "transfer_n_null", 200))
    base_seed = int(getattr(sae_cfg, "transfer_seed", 0))
    p_method = _p_method(concepts_cfg)
    fdr_q = float(getattr(concepts_cfg, "transfer_fdr_q", 0.05) or 0.05)
    max_redraw = int(getattr(concepts_cfg, "transfer_max_redraw", _MAX_REDRAW_DEFAULT) or _MAX_REDRAW_DEFAULT)

    meta_path = run_dir / "sae" / "meta.json"
    meta_sae = {}
    if meta_path.exists():
        import json
        meta_sae = json.loads(meta_path.read_text(encoding="utf-8"))
    live = _live_atlas_targets(meta_sae)

    strata, by_stratum, store = _store_context(run_dir)
    _pooled, _ranks = _pooled_ranks_fns(store)

    tests = atlas_transfer_tests(atlas, live, strata, by_stratum, _pooled, _ranks,
                                 k_top=k_top, n_null=n_null, base_seed=base_seed,
                                 p_method=p_method, max_redraw=max_redraw)

    tests = _apply_pairwise_fdr(tests, "src_model", "dst_model", fdr_q)
    concept_summary, cross_check = _atlas_concept_summary(atlas, tests)

    pair_summary: dict = {}
    for t in tests:
        key = (t["src_model"], t["dst_model"])
        rec = pair_summary.setdefault(key, {"src_model": t["src_model"],
                                            "dst_model": t["dst_model"],
                                            "n_tests": 0, "n_uncorrected_reciprocal": 0,
                                            "n_fdr_reciprocal": 0})
        rec["n_tests"] += 1
        rec["n_uncorrected_reciprocal"] += int(bool(t["reciprocal"]))
        rec["n_fdr_reciprocal"] += int(bool(t["reciprocal_fdr"]))
    pair_summary_list = [pair_summary[k] for k in sorted(pair_summary)]

    out = {"schema_version": 1, "k_top_series": k_top, "n_null_draws": n_null,
          "stratum_field": "archetype|family", "p_method": p_method, "fdr_q": fdr_q,
          "tests": tests, "pair_summary": pair_summary_list,
          "concept_summary": concept_summary, "cross_check_2x2": cross_check}
    out_path = run_dir / "sae" / "atlas_transfer.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    n_recip_fdr = sum(1 for t in tests if t["reciprocal_fdr"])
    log.info("sae atlas transfer: wrote %s (%d test(s) across %d ordered model pair(s), "
            "%d/%d reciprocal under per-leg FDR q=%s)", out_path, len(tests),
            len(pair_summary_list), n_recip_fdr, len(tests), fdr_q)
    return out
