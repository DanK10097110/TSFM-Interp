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


# ---------------------------------------------------------------------------
# The causal signal (ROADMAP.md sec 27): a SECOND opinion on a match made in
# activation space.
#
# The three signals above all ask variants of "do these two features fire on
# the same series". None of them can tell apart two features that fire
# together for DIFFERENT reasons -- which is common rather than exotic,
# because the properties that make a series distinctive are correlated with
# each other (a strongly seasonal series is usually also low-noise, so an
# atom keyed on either fires on much the same rows).
#
# Ablating each feature on its own top-firing series and reading the
# forecast's response across the nine channels gives a signature of what the
# feature DOES, in a space both models already share (the forecast), and the
# cosine between two such signatures is a second, orthogonal opinion on a
# match the activation profile proposed.
#
# It is deliberately NOT folded into the greedy match score. Matching stays
# an activation-space operation with an activation-space threshold, and the
# causal column is reported beside it -- so "these two fire together but act
# differently" is a visible, countable outcome rather than a pair that
# quietly drops out of the table for a reason the reader cannot see.
# ---------------------------------------------------------------------------

from .response import CHANNELS as _CHANNELS  # noqa: E402


def _cos():
    """`role_matching.sign_aware_cosine`, imported lazily.

    `role_matching` imports THIS module at its own line 96 and defines
    `sign_aware_cosine` below that, so a module-level import here is an
    ImportError whenever `role_matching` is the first of the two to load --
    which is exactly what `report.py` does. It passed every test in this
    package, because a test importing `matching` first resolves the cycle the
    other way round. Found by running the real CLI, not by reading the diff
    (`CLAUDE.md` sec 11.48).
    """
    from .role_matching import sign_aware_cosine
    return sign_aware_cosine


def causal_fingerprint(entry: dict | None, channels: tuple = _CHANNELS) -> dict:
    """One feature's null-normalized SIGNED causal vector, from its entry in
    a `*_ablation.json` candidate list.

    Each channel contributes `signed_effect / null_p95` -- normalizing by
    that channel's own row-matched null is what makes the vector comparable
    across two models whose forecasts live on entirely different scales
    (`CLAUDE.md` sec 11.33: a raw-magnitude comparison would rank by
    forecast scale, not by causal role).

    `available` is False -- with a stated reason, never a zero vector passed
    off as a measurement -- when the feature was never scored, cleared no
    channel at all, or normalizes to nothing. A feature whose ablation moved
    nothing has no causal identity, and its cosine against ANY other vector
    is a direction picked out of noise; reporting one would be sec 11.37's
    degenerate baseline again, where absent reads as a confident answer.
    """
    if not entry or not entry.get("scorable"):
        return {"available": False, "vector": None, "magnitude": None,
                "reason": "this feature has no ablation measurement"
                          if not entry else
                          (entry.get("reason") or "not scorable"),
                "n_channels_clearing": 0}
    chans = entry.get("channels", {})
    vec, n_used = [], 0
    for ch in channels:
        rec = chans.get(ch) or {}
        p95, signed = rec.get("null_p95"), rec.get("signed_effect")
        if not rec.get("available") or not p95 or signed is None:
            vec.append(0.0)
            continue
        vec.append(float(signed) / float(p95))
        n_used += 1
    v = np.asarray(vec, dtype=np.float64)
    clearing = int(entry.get("n_channels_clearing", 0))
    if n_used == 0:
        return {"available": False, "vector": None, "magnitude": None,
                "n_channels_clearing": clearing,
                "reason": "no channel of this feature was scorable against a null"}
    if clearing == 0:
        return {"available": False, "vector": v.tolist(),
                "magnitude": float(np.linalg.norm(v)), "n_channels_clearing": 0,
                "reason": "ablating this feature cleared no channel's null, so "
                          "its response direction is not distinguishable from noise"}
    return {"available": True, "vector": v.tolist(),
            "magnitude": float(np.linalg.norm(v)),
            "n_channels_clearing": clearing, "reason": ""}


def causal_agreement(fp_a: dict, fp_b: dict) -> dict:
    """Cosine between two causal fingerprints, with BOTH magnitudes beside it.

    Magnitude is reported because cosine deliberately discards it: two
    features can push the forecast in exactly the same direction while one
    does so an order of magnitude harder, which is a real difference between
    the models and is invisible in the cosine alone.
    """
    if not fp_a.get("available") or not fp_b.get("available"):
        which = "A" if not fp_a.get("available") else "B"
        return {"available": False, "cosine": None,
                "reason": f"side {which}: " + (fp_a if which == "A" else fp_b)["reason"],
                "magnitude_a": fp_a.get("magnitude"), "magnitude_b": fp_b.get("magnitude")}
    cos = _cos()(np.asarray(fp_a["vector"]), np.asarray(fp_b["vector"]))
    return {"available": True, "cosine": cos, "reason": "",
            "magnitude_a": fp_a["magnitude"], "magnitude_b": fp_b["magnitude"]}


# A p95 cannot exclude the top 5% of a pool it is estimated FROM. With fewer
# than 20 available cross-pairs, the 95th percentile sits at or above the
# largest one, so NO pair can clear it at any effect size -- the same
# arithmetic as sec 6.6's Holm p-floor, in a different statistic. Below this
# the null is reported as unresolvable rather than as a threshold nothing
# happened to beat.
_MIN_NULL_POOL = 20


def causal_permutation_null(fps_a: dict | list, fps_b: dict | list,
                            exclude_pairs: set | None = None,
                            n_samples: int = 500, seed: int = 0) -> dict:
    """`|cosine|` p95 over random A-side x B-side fingerprint pairs -- how
    similar two ARBITRARY features from these two populations look.

    This is the reference a matched pair's cosine must clear, so the matched
    pairings themselves are EXCLUDED from it (`exclude_pairs`, a set of
    `(feature_a, feature_b)`). Without that exclusion the null is estimated
    partly from the very pairs under test, and a perfect match raises the
    threshold it then has to beat -- which makes it fail. That is
    `role_matching.permutation_null_cosine`'s own reason for excluding the
    targets under test, applied where the population IS the pair.

    Pass dicts (`{feature: fingerprint}`) to use `exclude_pairs`; lists are
    accepted for the unfiltered case. Returns a dict rather than a bare
    number so `resolvable` -- whether the pool is even large enough for the
    p95 to be clearable -- travels with it.
    """
    if isinstance(fps_a, dict):
        A = [(f, np.asarray(v["vector"])) for f, v in fps_a.items() if v.get("available")]
    else:
        A = [(i, np.asarray(v["vector"])) for i, v in enumerate(fps_a) if v.get("available")]
    if isinstance(fps_b, dict):
        B = [(f, np.asarray(v["vector"])) for f, v in fps_b.items() if v.get("available")]
    else:
        B = [(i, np.asarray(v["vector"])) for i, v in enumerate(fps_b) if v.get("available")]

    exclude = exclude_pairs or set()
    pairs = [(u, v) for fa, u in A for fb, v in B if (fa, fb) not in exclude]
    if not pairs:
        return {"p95": None, "n_pool": 0, "resolvable": False,
                "reason": "no cross-model pair has a usable fingerprint on both sides"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), size=n_samples)
    _sc = _cos()
    cos = np.array([abs(_sc(*pairs[i])) for i in idx])
    p95 = float(np.percentile(cos, 95))
    # A cosine is bounded by 1, so a null p95 AT that bound is unclearable at
    # any effect size -- the same arithmetic as sec 6.6's Holm p-floor and as
    # `_MIN_NULL_POOL` above, arrived at from the other direction. It happens
    # whenever the distractor population is collinear (every role loading on
    # one channel), and it is not a small population: found by a fixture in
    # which a PERFECTLY matched pair came back "fires together, acts
    # differently". Reporting that as a substantive disagreement is sec
    # 11.37's degenerate baseline, in the direction that manufactures a
    # confident negative rather than a confident positive.
    saturated = p95 >= 1.0 - 1e-9
    resolvable = len(pairs) >= _MIN_NULL_POOL and not saturated
    if saturated:
        reason = ("the null p95 is at the maximum a cosine can take, so no pair "
                  "could clear it at any effect size -- these two role "
                  "populations are collinear enough that an arbitrary "
                  "cross-pair already looks identical")
    elif len(pairs) < _MIN_NULL_POOL:
        reason = (f"only {len(pairs)} usable cross-pairs after excluding the "
                  f"matched ones; a p95 cannot exclude the top 5% of a pool "
                  f"smaller than {_MIN_NULL_POOL}, so no pair could clear it "
                  f"at any effect size")
    else:
        reason = ""
    return {"p95": p95, "n_pool": len(pairs), "resolvable": resolvable,
            "null_saturated": bool(saturated), "reason": reason}


def add_causal_agreement(match_result: dict, ablation_a: dict | None,
                         ablation_b: dict | None, seed: int = 0) -> dict:
    """Annotate an activation-space `match_cross_model_features` result with
    the causal second opinion, in place, and summarize it.

    `ablation_a`/`ablation_b` are `*_ablation.json` artifacts (or None). Each
    matched pair gains a `causal` block; the summary gains counts of pairs
    that agree causally, that DISAGREE (fire together, act differently --
    the outcome this signal exists to make visible), and that could not be
    scored at all.

    A pair whose causal cosine cannot be computed is counted separately and
    never folded into either verdict: absent is a third outcome, not a
    quiet vote for disagreement (`CLAUDE.md` sec 11.37).
    """
    def _index(art):
        if not art or art.get("withheld"):
            return None, (art or {}).get("reason") or (
                (art or {}).get("reach") or {}).get("reason") or "no ablation artifact"
        return {int(c["feature"]): c for c in art.get("candidates", [])}, ""

    idx_a, why_a = _index(ablation_a)
    idx_b, why_b = _index(ablation_b)
    if idx_a is None or idx_b is None:
        match_result["causal"] = {
            "available": False,
            "reason": f"A: {why_a or 'ok'}; B: {why_b or 'ok'}",
            "hint": "run run_sae_ablation.py for both targets"}
        return match_result

    fps_a = {f: causal_fingerprint(e) for f, e in idx_a.items()}
    fps_b = {f: causal_fingerprint(e) for f, e in idx_b.items()}
    matched_pairs = {(int(p["feature_a"]), int(p["feature_b"]))
                     for p in match_result.get("matched", [])}
    null = causal_permutation_null(fps_a, fps_b, exclude_pairs=matched_pairs, seed=seed)
    null_p95 = null["p95"] if null["resolvable"] else None

    n_agree = n_disagree = n_unscorable = 0
    cosines = []
    for pair in match_result.get("matched", []):
        fp_a = fps_a.get(int(pair["feature_a"]),
                         {"available": False, "reason": "not an ablation candidate"})
        fp_b = fps_b.get(int(pair["feature_b"]),
                         {"available": False, "reason": "not an ablation candidate"})
        ca = causal_agreement(fp_a, fp_b)
        if ca["available"] and null_p95 is not None:
            cosines.append(ca["cosine"])
            if abs(ca["cosine"]) > null_p95:
                ca["verdict"] = "same causal role"
                n_agree += 1
            else:
                ca["verdict"] = "fires together, acts differently"
                n_disagree += 1
        else:
            ca["verdict"] = "not scorable"
            if ca["available"] and null_p95 is None:
                ca["reason"] = null["reason"]
            n_unscorable += 1
        pair["causal"] = ca

    match_result["causal"] = {
        "available": True,
        "null_p95": null_p95, "null_pool_pairs": null["n_pool"],
        "null_resolvable": null["resolvable"], "null_reason": null["reason"],
        "n_agree": n_agree, "n_disagree": n_disagree, "n_unscorable": n_unscorable,
        "mean_abs_cosine": float(np.mean(np.abs(cosines))) if cosines else None,
        # Named, not omitted: the same untrained-twin floor discipline every
        # other cross-model number in this repo carries (sec 6.2 item 3's
        # role matching, the crosscoder's `frac_shared`). The permutation
        # null above answers "do two ARBITRARY features look this alike";
        # it does NOT answer "would two untrained networks of these shapes
        # look this alike", which needs an ablation pass over a random_init
        # twin run and has not been measured.
        "untrained_twin_floor": None,
        "untrained_twin_floor_reason":
            "not measured -- needs run_sae_ablation.py against a random_init "
            "twin of each model; read the agree/disagree split as a "
            "within-run contrast until it is",
    }
    return match_result


# ---------------------------------------------------------------------------
# The same causal second opinion, at ROLE granularity (ROADMAP.md sec 27).
#
# `match_cross_model_features` above has no caller outside
# `crosscoder_eval.py` (study driver, dev branch); the surface the report
# actually renders is
# `role_matching.py`'s role correspondence table. So the causal check is
# attached there too rather than only at a granularity nothing displays --
# and by reusing `causal_agreement`/`causal_permutation_null` unchanged,
# since both are already generic over `{id: fingerprint}` (sec 2.2: a second
# implementation of a cosine and a percentile is how the two drift).
# ---------------------------------------------------------------------------

def role_causal_fingerprint(role: dict, ablation_doc: dict | None,
                            channels: tuple = _CHANNELS) -> dict:
    """A role's causal fingerprint: the mean of its members' own.

    A role IS a set of features clustered for sharing a response signature,
    so the mean of their vectors is that signature. Averaging is done over
    the members with a usable fingerprint only -- treating an unmeasured
    member as a zero vector would shrink the role's magnitude toward zero in
    proportion to how much of it was never scored, which reads as "this role
    does little" rather than "this role was mostly not measured"
    (`CLAUDE.md` sec 11.37).

    `n_members_measured` travels with the result for the same reason: a role
    whose signature rests on 1 of 16 atoms and one resting on 16 of 16 are
    not the same evidence, and the cosine alone cannot tell them apart.
    """
    members = [int(f) for f in (role.get("features") or [])]
    by_feature = {int(c["feature"]): c
                  for c in ((ablation_doc or {}).get("candidates") or [])}
    vecs, clearing = [], 0
    for f in members:
        fp = causal_fingerprint(by_feature.get(f), channels)
        if fp.get("available"):
            vecs.append(np.asarray(fp["vector"], dtype=np.float64))
            clearing += int(fp.get("n_channels_clearing") or 0)
    if not vecs:
        return {"available": False, "vector": None, "magnitude": None,
                "n_members": len(members), "n_members_measured": 0,
                "n_channels_clearing": 0,
                "reason": ("no member of this role has an ablation measurement"
                           if ablation_doc is None else
                           "no member of this role cleared any channel's null, "
                           "so the role has no causal direction to compare")}
    v = np.mean(np.stack(vecs, axis=0), axis=0)
    return {"available": True, "vector": v.tolist(),
            "magnitude": float(np.linalg.norm(v)),
            "n_members": len(members), "n_members_measured": len(vecs),
            "n_channels_clearing": clearing, "reason": ""}


def add_role_causal_agreement(table: dict, roles_json: dict,
                              ablation_by_target: dict, seed: int = 0) -> dict:
    """Annotate `role_matching.role_correspondence_table`'s output with the
    causal second opinion, in place.

    Every matched pair gains a `causal` block, and each pair record gains a
    `causal_summary`. The point of the whole exercise is the DISAGREEMENT
    count: two roles can match on activation profile -- they fire on the
    same series -- while pushing the forecast in different directions, which
    is precisely the case the user's request names and which no
    correlational statistic in this repo can currently separate.

    🔴 This is deliberately NOT folded into `match_rate` or into the greedy
    matching score. The correlational match is a published number with
    recorded values across several runs; silently redefining it would
    rewrite them (`CLAUDE.md` sec 2.1). It is a second opinion rendered
    beside the first, and the two disagreeing is the finding.
    """
    for rec in table.get("pairs", []):
        if not rec.get("comparable"):
            continue
        ta, tb = rec.get("target_a"), rec.get("target_b")
        roles_a = {int(r.get("role")): r
                   for r in ((roles_json.get(ta) or {}).get("roles") or [])}
        roles_b = {int(r.get("role")): r
                   for r in ((roles_json.get(tb) or {}).get("roles") or [])}
        abl_a = ablation_by_target.get(ta)
        abl_b = ablation_by_target.get(tb)
        fps_a = {i: role_causal_fingerprint(r, abl_a) for i, r in roles_a.items()}
        fps_b = {i: role_causal_fingerprint(r, abl_b) for i, r in roles_b.items()}

        matched = set()
        for m in rec.get("matches", []):
            ia, ib = m.get("role_a_index"), m.get("role_b_index")
            if ia is None or ib is None:
                continue
            matched.add((int(ia), int(ib)))
        null = causal_permutation_null(fps_a, fps_b, exclude_pairs=matched,
                                       seed=seed)

        agree = disagree = unscorable = 0
        for m in rec.get("matches", []):
            ia, ib = m.get("role_a_index"), m.get("role_b_index")
            if ia is None or ib is None:
                m["causal"] = {"available": False, "verdict": "not scorable",
                               "reason": "this row is not a matched pair"}
                unscorable += 1
                continue
            ag = causal_agreement(fps_a.get(int(ia), {"available": False,
                                                      "reason": "no such role"}),
                                  fps_b.get(int(ib), {"available": False,
                                                      "reason": "no such role"}))
            ag["n_members_measured_a"] = fps_a.get(int(ia), {}).get("n_members_measured")
            ag["n_members_measured_b"] = fps_b.get(int(ib), {}).get("n_members_measured")
            ag["null_p95"] = null.get("p95")
            ag["null_resolvable"] = null.get("resolvable")
            if not ag.get("available") or not null.get("resolvable") or null.get("p95") is None:
                ag["verdict"] = "not scorable"
                ag["reason"] = ag.get("reason") or null.get("reason")
                unscorable += 1
            elif abs(float(ag["cosine"])) > float(null["p95"]) and float(ag["cosine"]) > 0:
                ag["verdict"] = "same causal role"
                agree += 1
            else:
                ag["verdict"] = "fires together, acts differently"
                disagree += 1
            m["causal"] = ag

        rec["causal_summary"] = {
            "n_agree": agree, "n_disagree": disagree, "n_not_scorable": unscorable,
            "null": null,
            # Same discipline the correlational `match_rate` above is held
            # to (sec 25.6): a cross-model rate is not quotable without its
            # own untrained-twin floor, and it does not have one yet.
            "untrained_twin_floor": None,
            "quotable": False,
            "quotable_reason":
                "not measured -- needs run_sae_ablation.py against a "
                "random_init twin of each model; read the agree/disagree "
                "split as a within-run contrast until it is",
        }
    return table
