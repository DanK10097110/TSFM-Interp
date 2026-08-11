"""Cross-model matching of independently-trained SAE dictionaries (ROADMAP.md sec 16 E16).

Complements the crosscoder (sec 6.2 item 1)'s joint-training answer to
cross-model SAE comparability: this module matches two *independently*
trained per-model, per-layer dictionaries after the fact, so the founding
question ("does each model learn the same features, or different ones?")
gets an answer even for a model pair the crosscoder was never jointly
trained on.

Both targets' `ground_truth_alignment` results already sample the exact
same series (same `cfg.run.seed + 12` and `cfg.sae.ground_truth_max_series`
for every target in one run -- see `ground_truth.py::ground_truth_alignment`)
and record that sample as `result["rows"]`. That shared sample gives three
matching signals in a space that is actually comparable across
architectures with different hidden dims, without inventing any new
cross-model index:

1. **Activation-profile correlation** -- Pearson correlation between two
   features' own per-series activation values across the shared sample.
   Deliberately *not* a literal decoder-weight correlation: decoder vectors
   live in each model's own, differently-sized hidden space and cannot be
   compared directly. This is the honest cross-model analog -- "do these two
   features fire on the same series, in the same relative strength" -- and
   is invariant to each SAE's own arbitrary feature scale (Pearson, not raw
   dot product).
2. **Max-activating-series overlap** -- Jaccard of each feature's own top-k
   most-activating series. Series-level, matching `ground_truth.py`'s own
   series-level scope (its docstring already names a window-level version as
   a natural, not-yet-built extension -- the same caveat applies here for
   the same reason).
3. **Ground-truth-field agreement** -- do the two features' own
   best-matched ground-truth field (`best_field`) agree, with the same sign
   of `rho`? A weaker, more literal echo of `CLAUDE.md` sec 6.3's founding
   question ("do features that align well with 'trend order' actually
   matter for the same reason in both models") at the matching-signal level
   rather than the causal-ablation level sec 7 bullet 3 already covers.

Candidates on each side are restricted to that target's own
ground-truth-matched features (`best_field is not None`) --
`best_ground_truth_matches`'s own truncation to the top 50 by `|rho|`
bounds this to at most 50x50 pairs regardless of dictionary size, so this
stays cheap even for a 10k-atom dictionary and needs no new sampling or
truncation logic of its own.

Matching is greedy nearest-neighbor (for each candidate on side A, the
single best-scoring candidate on side B), not an optimal one-to-one
assignment (a Hungarian-algorithm version is a natural improvement, not
built this session) -- so more than one A-side feature can nominate the
same B-side partner. This is stated as a scope limit, not hidden: the
`matched` list's own `feature_b` column can repeat.
"""

from __future__ import annotations

import numpy as np


def matched_candidates(gt: dict) -> list[dict]:
    """Features from a `ground_truth_alignment` result that matched some field."""
    return [f for f in gt.get("features", []) if f.get("best_field") is not None]


def activation_profile_correlation(col_a: np.ndarray, col_b: np.ndarray) -> float:
    """Pearson correlation between two features' per-series activation vectors.

    Zero (not NaN) for a constant column -- a feature that never fires (or
    always fires identically) across the shared sample has no profile to
    correlate, and should score as "no signal," not propagate a NaN into the
    combined score downstream.
    """
    if np.std(col_a) == 0 or np.std(col_b) == 0:
        return 0.0
    return float(np.corrcoef(col_a, col_b)[0, 1])


def top_k_series(col: np.ndarray, k: int) -> set:
    """Row indices of the `k` most-activating series for one feature column."""
    k = min(k, len(col))
    return set(np.argsort(-col)[:k].tolist())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def ground_truth_agreement(feat_a: dict, feat_b: dict) -> bool:
    # bool(...) matters: `and` on a numpy comparison returns numpy.bool_, not
    # python bool, which fails an `is True` check downstream (same class of
    # bug as CLAUDE.md sec 9's capability-matrix numpy.bool_/`is` finding).
    return bool(feat_a["best_field"] == feat_b["best_field"]
                and np.sign(feat_a["rho"]) == np.sign(feat_b["rho"]))


def match_cross_model_features(features_a: np.ndarray, gt_a: dict,
                               features_b: np.ndarray, gt_b: dict,
                               top_k: int = 10, corr_threshold: float = 0.3) -> dict:
    """Greedy nearest-neighbor match of A-side onto B-side ground-truth-matched features.

    `features_a`/`features_b` are `[N, dict_size]` series-level SAE encodings
    over the SAME shared series sample -- the caller must verify
    `gt_a["rows"] == gt_b["rows"]` before calling this (not re-checked here,
    since this function has no access to the `rows` lists themselves, only
    the already-sliced feature matrices and match records).

    A pair is reported in `matched` only if `|correlation| >= corr_threshold`
    -- Jaccard and ground-truth agreement are informative context on a
    match, not gates on their own, since a feature pair can correlate
    strongly in activation while disagreeing on which single ground-truth
    field is its *best* match (ties, or two correlated-but-distinct fields).
    """
    cand_a = matched_candidates(gt_a)
    cand_b = matched_candidates(gt_b)
    pairs = []
    for fa in cand_a:
        col_a = features_a[:, fa["feature"]]
        top_a = top_k_series(col_a, top_k)
        best = None
        for fb in cand_b:
            col_b = features_b[:, fb["feature"]]
            corr = activation_profile_correlation(col_a, col_b)
            jac = jaccard(top_a, top_k_series(col_b, top_k))
            agree = ground_truth_agreement(fa, fb)
            score = abs(corr) + jac + (0.5 if agree else 0.0)
            if best is None or score > best["score"]:
                best = {"feature_a": fa["feature"], "field_a": fa["best_field"], "rho_a": fa["rho"],
                       "feature_b": fb["feature"], "field_b": fb["best_field"], "rho_b": fb["rho"],
                       "correlation": corr, "jaccard": jac,
                       "ground_truth_agree": agree, "score": score}
        if best is not None:
            pairs.append(best)
    matched = [p for p in pairs if abs(p["correlation"]) >= corr_threshold]
    return {
        "n_candidates_a": len(cand_a), "n_candidates_b": len(cand_b),
        "n_matched": len(matched),
        "n_unmatched_a": len(cand_a) - len(matched),
        "matched": sorted(matched, key=lambda p: -p["score"]),
        "corr_threshold": corr_threshold, "top_k": top_k,
    }
