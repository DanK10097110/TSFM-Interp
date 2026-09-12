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
          "auc_from_ranks", "transfer_one", "run_transfer"]


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
    return pooled[:, feature_ids].mean(axis=1)


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


def transfer_one(src_scores: np.ndarray, dst_ranks: np.ndarray, S: np.ndarray,
                 fwd_draws: np.ndarray, strata: np.ndarray, by_stratum: dict,
                 k: int = 20, n_draws: int = 200, seed: int = 0) -> dict:
    """Forward + reverse legs for ONE (concept, destination target) pair.

    `fwd_draws` is the `(n_draws, len(S))` matched-draw block the CALLER
    built once for this source concept and reuses across every destination
    -- only the reverse leg draws inside this function, because only its
    `S_b` (the destination feature's own top-k series) depends on the
    destination.

    forward: best feature in the destination's WHOLE dictionary by AUC on
             `S`, vs the p95 of the MAX-over-features null AUC -- a
             per-feature null would ignore the search over `dict_size`
             candidates and clear almost everything
             (`test_forward_null_is_max_over_features`).
    reverse: that feature's own top-k series `S_b`, scored by `src_scores`,
             vs the p95 over matched draws of that SAME single score -- no
             max here, since the observed statistic has no search either
             (`test_reverse_leg_has_no_search_correction`). The null is
             redrawn against `S_b`'s OWN stratum composition, never `S`'s.
    """
    obs_auc = auc_from_ranks(dst_ranks, S)
    best_feature = int(np.argmax(obs_auc))
    auc = float(obs_auc[best_feature])
    null_auc = auc_from_ranks(dst_ranks, fwd_draws)
    null_p95 = float(np.percentile(null_auc.max(axis=1), 95))
    clears = bool(auc > null_p95)

    S_b = top_series(dst_ranks[:, best_feature], k)

    R_src = rankdata(src_scores)[:, None]
    rev_auc = float(auc_from_ranks(R_src, S_b)[0])
    rng = np.random.default_rng(seed)
    rev_draws = matched_draws(S_b, strata, by_stratum, n_draws, rng)
    rev_null_p95 = float(np.percentile(auc_from_ranks(R_src, rev_draws)[:, 0], 95))
    rev_clears = bool(rev_auc > rev_null_p95)

    return {"auc": auc, "feature": best_feature, "null_p95": null_p95,
           "clears": clears, "rev_auc": rev_auc, "rev_null_p95": rev_null_p95,
           "rev_clears": rev_clears, "reciprocal": bool(clears and rev_clears)}


def run_transfer(run_dir: Path, concepts: dict, cfg) -> dict:
    """All concepts x all targets of every OTHER model. Writes
    `sae/transfer.json` (sec 30.5's schema): `pairs` is the raw evidence,
    `reach`/`matrix`/`universality` are reductions over it, written
    together so a reader can audit a rate back to the pair rows that
    produced it.

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
    k_top = int(getattr(sae_cfg, "transfer_top_k", 20))
    n_null = int(getattr(sae_cfg, "transfer_n_null", 200))
    base_seed = int(getattr(sae_cfg, "transfer_seed", 0))

    meta = load_meta(run_dir)
    strata = series_strata(meta)
    by_stratum = _by_stratum(strata)

    store = ActivationStore(run_dir / "activations.zarr", mode="r")

    targets = concepts.get("targets", {})
    live = {key: t for key, t in targets.items() if not t.get("withheld")}

    pooled_cache: dict = {}
    ranks_cache: dict = {}

    def _pooled(key: str) -> np.ndarray:
        if key not in pooled_cache:
            model, layer = key.split("/", 1)
            pooled_cache[key] = store.load(model, layer, level="series", space="sae")
        return pooled_cache[key]

    def _ranks(key: str) -> np.ndarray:
        if key not in ranks_cache:
            ranks_cache[key] = rankdata(_pooled(key), axis=0)
        return ranks_cache[key]

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
                                      by_stratum, k=k_top, n_draws=n_null, seed=rev_seed)
                pairs.append({"src": src_key, "concept": concept["concept"],
                             "dst": dst_key, "dst_model": dst_model, **result})

    reach_map: dict = {}
    for p in pairs:
        rk = (p["src"], p["concept"], p["dst_model"])
        reach_map[rk] = reach_map.get(rk, False) or p["reciprocal"]
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

    out = {"schema_version": 1, "k_top_series": k_top, "n_null_draws": n_null,
          "stratum_field": "archetype|family", "pairs": pairs, "reach": reach,
          "matrix": matrix, "universality": universality}
    out_path = run_dir / "sae" / "transfer.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    n_clear = sum(1 for p in pairs if p["clears"])
    n_recip = sum(1 for p in pairs if p["reciprocal"])
    log.info("sae transfer: wrote %s (%d pairs, %d/%d forward clear, "
            "%d/%d reciprocal)", out_path, len(pairs), n_clear, len(pairs),
            n_recip, len(pairs))
    return out
