"""Component C -- cross-model role correspondence (ROADMAP.md sec 25.6,
sec 25.9 Stage 4).

`sae/roles.py` (Component B(b)) gives every (model, layer) target a small
set of named ROLES, each with a mean signed, null-normalized response
vector over `sae/response.py::CHANNELS`. This module matches roles ACROSS
models, for every model pair a run configures (`cfg.comparison_pairs()`,
`ROADMAP.md` sec 24.3's rule -- all C(n,2) pairs on a panel, not only the
designated reference pair).

Why the response fingerprint is the right space, restated from sec 25.6
--------------------------------------------------------------------------
`sae/matching.py` (sec 16 E16) already argues correctly that decoder
vectors cannot be compared across models -- different hidden dims,
arbitrary bases -- and falls back to activation-profile correlation, which
needs both models to fire on the SAME series in the SAME relative order.
The response fingerprint needs neither: it lives in forecast space, which
both models share exactly (same horizon, same series, same units),
regardless of width, tokenizer, patch size, depth or capture surface. It
is the only genuinely architecture-neutral feature space this repo has
produced (`sae/response.py`'s own docstring), and it is a free by-product
of Component A -- this module adds no new forward pass.

Primary signal: sign-aware cosine
----------------------------------
"Sign-aware" means plain cosine similarity (range [-1, 1]), not `|cosine|`.
Taking the absolute value would match a role that pushes `trend` UP against
one that pushes it DOWN just because both dominate on the `trend` channel
-- exactly the kind of collapse sec 25.1 already diagnosed for the old
argmax-over-30-fields labelling scheme, one level up. A negative score is a
real, informative anti-correlation (two roles with opposite causal
identities), not a failure to match.

Only the CHANNELS half of a role's clustering vector is used here, never
the residualized-structural half `sae/roles.py::build_feature_matrix` also
carries -- the structural signature is per-corpus and per-provenance-scheme
(sec 25.1's own indicted labelling), not architecture-neutral, and mixing
it into the primary cross-model distance would silently re-import exactly
the defect sec 25 exists to remove.

Secondary signals, reused not duplicated
------------------------------------------
`sae/matching.py::activation_profile_correlation`/`top_k_series`/`jaccard`
are called here, pooled to the role level by averaging over each role's
member atoms' encoded activation columns on the SAME shared series sample
both targets' `ground_truth_alignment` already draws
(`cfg.run.seed + 12`, `cfg.sae.ground_truth_max_series` -- see
`sae/matching.py`'s own module docstring). These need both targets to
belong to the SAME run (so the encoded features are on the identical row
order) -- true for two targets in one panel run, and NOT assumed true
across two different run directories (e.g. real vs. an untrained-twin RUN,
which has its own, unrelated corpus/seed unless the twin was extracted
inside the same run as its real counterpart, which sec 16 E9's convention
is not). So the activation-profile signal is computed only within one run;
the untrained-twin null (below) uses the response-fingerprint signal alone,
which needs no shared series sample at all (each run's own reach/battery is
self-contained).

The two required nulls
-----------------------
1. Shuffled-series null for the activation-profile signal: L1's own
   convention (`analysis/l1_geometry.py::_shuffled_null_cka`) -- break which
   series lines up with which before recomputing the correlation, so
   leftover similarity from "any two reasonable encoders of the same
   structured input agree somewhat" is measured and can be subtracted out
   rather than mistaken for real correspondence.
2. Untrained-twin null: if a `random_init` run exists for either model
   (`ModelConfig.random_init`, sec 16 E9), match real-vs-twin roles the
   same way and report that score beside the real cross-model score. This
   is not optional -- sec 6.2.1 Stage 1 measured the crosscoder's
   `frac_shared` at 0.845 against an untrained-twin floor of 0.974
   (TimesFM read as MORE shared with a random copy of itself than with
   Chronos), and `match_roles_across_models`/`role_correspondence_table`
   below refuse to report a bare match-rate number without this floor
   sitting beside it in the same record (sec 25.6, sec 25.9 Stage 4's exit
   criterion).

Honest blank-cell semantics
------------------------------
A model pair with no comparable targets (neither model has an SAE role
target within `depth_tolerance` of the other's, on the `block` depth axis)
produces an entry with `comparable: False` and a stated reason -- NEVER a
silent absence. This is the distinction sec 25.6 requires: a blank cell in
the eventual display means "no target at a comparable depth", not "this
model lacks this role".
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..utils import load_json, log
from .matching import activation_profile_correlation, jaccard, top_k_series
from .response import CHANNELS
from .train import sanitize

__all__ = [
    "role_response_vector",
    "sign_aware_cosine",
    "greedy_match_roles",
    "shuffled_series_null_activation_profile",
    "role_population_vectors",
    "permutation_null_cosine",
    "match_roles_for_pair",
    "match_roles_across_models",
    "untrained_twin_role_floor",
    "role_correspondence_table",
]


# ---------------------------------------------------------------------------
# The primary signal: response-fingerprint cosine similarity.
# ---------------------------------------------------------------------------

def role_response_vector(role: dict, candidates: list, null_p95: dict,
                         channel_columns: tuple = CHANNELS) -> np.ndarray:
    """A role's mean signed, null-normalized response vector over `channel_columns`.

    Recomputes the exact per-channel normalization
    `sae/roles.py::build_feature_matrix` uses (larger-magnitude signed
    effect of the two steering directions, divided by that channel's own
    null p95) for just this role's member candidates, then averages over
    members -- so this is the same number `derive_role_name`'s
    `mean_row[:len(channel_columns)]` slice already computes internally,
    exposed here as its own function because Component C needs it
    independently of clustering (matching happens across two SEPARATE
    clustering runs, one per model, whose candidate index spaces share
    nothing).

    `candidates` is this target's full `*_stage2_response.json` candidate
    list (not just the role's members pre-filtered) -- callers pass
    `sae.roles.role_member_candidates`-style filtered lists in practice, but
    this function does not require it: any candidate whose `feature` is not
    in `role["features"]` is simply ignored, so passing the full list is
    always safe.
    """
    wanted = set(int(f) for f in role.get("features", []))
    members = [c for c in candidates if int(c["feature"]) in wanted]
    if not members:
        return np.zeros(len(channel_columns), dtype=np.float64)
    rows = []
    for c in members:
        row = []
        for ch in channel_columns:
            p95 = null_p95.get(ch)
            up = ((c.get("up") or {}).get("channels") or {}).get(ch, {})
            down = ((c.get("down") or {}).get("channels") or {}).get(ch, {})
            signed_up = up.get("signed_mean")
            signed_down = down.get("signed_mean")
            signed = [v for v in (signed_up, signed_down) if v is not None]
            if not signed or not p95:
                row.append(0.0)
                continue
            best = max(signed, key=abs)
            row.append(float(best) / float(p95))
        rows.append(row)
    return np.asarray(rows, dtype=np.float64).mean(axis=0)


def sign_aware_cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Plain cosine similarity, range [-1, 1] -- "sign-aware" means this is
    NOT `|cosine|` (sec 25.6). Zero (not NaN) when either vector has no
    measured effect at all (all-zero row -- e.g. a role whose members never
    cleared any null), since a role with literally no response fingerprint
    has nothing to correlate, positively or negatively.
    """
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


# ---------------------------------------------------------------------------
# Secondary signals, pooled to the role level (sec 25.6: "free from E16").
# ---------------------------------------------------------------------------

def _role_activation_columns(role: dict, features: np.ndarray) -> Optional[np.ndarray]:
    """Mean per-series encoded activation across a role's member atoms.

    `features` is `[n_series, dict_size]` (this target's own
    `encode_series_level` output over the shared row sample). Returns
    `None` when the role's member feature indices are out of range for
    this dictionary (defensive -- should not happen for a properly-built
    `roles.json`, but a stale artifact must degrade rather than crash a
    report-only pass).
    """
    idx = [int(f) for f in role.get("features", [])]
    if not idx or max(idx, default=-1) >= features.shape[1]:
        return None
    return features[:, idx].mean(axis=1)


def role_activation_profile_signals(role_a: dict, features_a: np.ndarray,
                                    role_b: dict, features_b: np.ndarray,
                                    top_k: int = 10) -> Optional[dict]:
    """Signals 2 and 3 (sec 25.6): pooled activation-profile correlation and
    max-activating-series Jaccard, between two roles' MEAN member activation
    columns over the shared series sample. `None` when either role's member
    indices don't resolve against the supplied feature matrices (see
    `_role_activation_columns`).
    """
    col_a = _role_activation_columns(role_a, features_a)
    col_b = _role_activation_columns(role_b, features_b)
    if col_a is None or col_b is None:
        return None
    corr = activation_profile_correlation(col_a, col_b)
    jac = jaccard(top_k_series(col_a, top_k), top_k_series(col_b, top_k))
    return {"activation_profile_correlation": corr, "max_activating_series_jaccard": jac}


def shuffled_series_null_activation_profile(role_a: dict, features_a: np.ndarray,
                                            role_b: dict, features_b: np.ndarray,
                                            n_shuffles: int = 5, seed: int = 0,
                                            top_k: int = 10) -> Optional[dict]:
    """L1's own shuffled-series null (`analysis/l1_geometry.py
    ::_shuffled_null_cka`), applied to the activation-profile correlation
    signal: break which series lines up with which on the B side before
    recomputing the correlation, `n_shuffles` times, and report the mean and
    max -- the max is the more conservative reference for a "does the real
    correlation clear the null" check, since it is the null's own best case
    rather than its average.
    """
    col_a = _role_activation_columns(role_a, features_a)
    col_b = _role_activation_columns(role_b, features_b)
    if col_a is None or col_b is None:
        return None
    rng = np.random.default_rng(seed)
    n = len(col_a)
    vals = []
    for _ in range(n_shuffles):
        perm = rng.permutation(n)
        vals.append(abs(activation_profile_correlation(col_a, col_b[perm])))
    return {"mean": float(np.mean(vals)), "max": float(np.max(vals)),
           "n_shuffles": int(n_shuffles)}


# ---------------------------------------------------------------------------
# Greedy nearest-neighbour matching (sec 25.6: "the same honesty matching.py
# already practises about greedy nearest-neighbour not being an assignment").
# ---------------------------------------------------------------------------

def greedy_match_roles(roles_a: list, vectors_a: list, roles_b: list, vectors_b: list,
                       secondary: Optional[list] = None) -> list:
    """For each role on side A, its single best-scoring role on side B by
    sign-aware cosine over the response fingerprint.

    NOT an optimal one-to-one assignment (a Hungarian-algorithm version is a
    natural improvement, not built here) -- `sae/matching.py`'s own stated
    scope limit, restated: more than one A-side role can nominate the same
    B-side partner, and this function does not prevent it. `secondary`, when
    supplied, is `[[dict|None, ...] for each (i,j) pair]` -- the
    activation-profile signals for role i (A) vs role j (B), in the same
    `len(roles_a) x len(roles_b)` row-major order cosine is computed in --
    attached to each match record as informational context, never used to
    break the primary cosine ranking (sec 25.6: cosine over the response
    fingerprint IS the primary signal; activation-profile/Jaccard are
    "free" secondary evidence, not a second vote).
    """
    out = []
    for i, ra in enumerate(roles_a):
        best_j, best_score = None, None
        for j, rb in enumerate(roles_b):
            score = sign_aware_cosine(vectors_a[i], vectors_b[j])
            if best_score is None or score > best_score:
                best_j, best_score = j, score
        rec = {"role_a": ra["name"], "role_a_index": ra.get("role", i),
              "role_b": roles_b[best_j]["name"], "role_b_index": roles_b[best_j].get("role", best_j),
              "cosine": best_score}
        if secondary is not None:
            sec = secondary[i][best_j]
            if sec is not None:
                rec.update(sec)
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# One model pair.
# ---------------------------------------------------------------------------

def _candidates_and_null_p95(run_dir: Optional[Path], model: str, layer: str) -> tuple:
    """Load a target's Stage 2 artifact for its raw candidate list and
    per-channel null p95 -- `roles.json` deliberately keeps only the
    role-level REDUCTION (`sae/roles.py::role_table`'s output), not the
    per-candidate detail `role_response_vector` needs to recompute a role's
    mean response vector. Returns `([], {})` when `run_dir` is `None` or the
    Stage 2 file is missing, so a caller with no store access still gets a
    well-formed (empty) result rather than an exception.
    """
    if run_dir is None:
        return [], {}
    path = Path(run_dir) / "sae" / sanitize(model) / f"{sanitize(layer)}_stage2_response.json"
    if not path.exists():
        return [], {}
    stage2 = load_json(path)
    return stage2.get("candidates", []), stage2.get("null_p95", {})


# ---------------------------------------------------------------------------
# The within-run population null for the PRIMARY cosine signal itself.
#
# The two nulls above answer "does architecture alone explain this" (the
# untrained-twin floor) and "does input-statistics-alone explain the
# ACTIVATION-PROFILE signal" (the shuffled-series null). Neither answers a
# third, narrower question the response fingerprint's own low dimension
# raises: `CHANNELS` has only 9 entries, so two roles that both push a
# generic effect (e.g. "dispersion up") can score a high cosine purely from
# sharing that one axis, with no correspondence implied. `_match_rate`'s
# fixed `cosine_threshold` (default 0.5) has no such calibration -- on a
# real run every observed cosine cleared it, forcing `match_rate` to a
# content-free 1.0 regardless of which roles matched which (found by
# reading the rendered report, not the code -- sec 2.4/11.48's practice).
# `permutation_null_cosine` below answers "how similar do two ARBITRARY
# trained roles from elsewhere in this run look, absent any claim they
# correspond" -- a population-level reference computed from roles this run
# already built, needing no extra forward pass or twin run.
# ---------------------------------------------------------------------------

def role_population_vectors(roles_json: dict, run_dir: Optional[Path] = None) -> dict:
    """`{target: [(role, vector), ...]}` for every non-skipped, non-withheld
    role target in this run's `roles.json` -- the pool `permutation_null_cosine`
    draws from. Computed once per run (not once per pair) so
    `match_roles_across_models` can share a single population across every
    pair it evaluates.
    """
    out = {}
    for target, rec in roles_json.items():
        if not isinstance(rec, dict) or rec.get("skipped") or rec.get("withheld"):
            continue
        roles = rec.get("roles") or []
        if not roles:
            continue
        model, _, layer = target.partition("/")
        cand, p95 = _candidates_and_null_p95(run_dir, model, layer)
        out[target] = [(r, role_response_vector(r, cand, p95)) for r in roles]
    return out


def permutation_null_cosine(population: dict, exclude_targets: tuple = (),
                            n_samples: int = 200, seed: int = 0) -> tuple:
    """`|sign-aware cosine|` null over `n_samples` random PAIRS of role
    vectors drawn (with replacement) from every target in `population`
    EXCEPT `exclude_targets` -- normally the two targets being matched, so
    the null is estimated from roles elsewhere in the run rather than from
    the very roles under test. Absolute value because the null asks "how
    large can a chance cosine get", and a real match's own sign is judged
    against that magnitude regardless of direction (sec 25.6's sign-aware
    cosine stays the primary signal; only the null's own spread is
    unsigned).

    Returns `(p95, n_pool)`. `p95` is `None` when fewer than 2 vectors are
    available outside `exclude_targets` -- sec 2.5's degrade-loudly
    doctrine: a caller must be able to tell "the null could not be
    estimated" apart from "the null is zero". `n_pool` is recorded even
    when `p95` is `None` (0 or 1), and always when it succeeds, so a reader
    can judge the estimate's quality -- a run with only two SAE role
    targets has a pool this small by construction, not by a bug, and the
    artifact should say so rather than hide it behind a number.
    """
    pool = [v for t, rvs in population.items() if t not in exclude_targets for _, v in rvs]
    if len(pool) < 2:
        return None, len(pool)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pool), size=(n_samples, 2))
    cos = np.array([abs(sign_aware_cosine(pool[i], pool[j])) for i, j in idx])
    return float(np.percentile(cos, 95)), len(pool)


def match_roles_for_pair(model_a: str, model_b: str, roles_json: dict,
                         depths: Optional[dict] = None,
                         depth_tolerance: float = 0.15,
                         features_by_target: Optional[dict] = None,
                         null_seed: int = 0,
                         run_dir: Optional[Path] = None,
                         population: Optional[dict] = None,
                         n_population_samples: int = 200) -> dict:
    """The full Component C record for one model pair.

    `depths` (optional): `{f"{model}/{layer}": float}` on the `block` axis
    for every role target in this run -- when supplied, and when either
    model has MULTIPLE role targets, the pair with depths closest together
    is chosen and the chosen depths are recorded (sec 25.6: "a role's depth
    coordinate carries the same coverage qualifier every other
    depth-located claim does"). When omitted, or when a model has exactly
    one role target, that one target is used unconditionally.

    `features_by_target` (optional): `{f"{model}/{layer}": np.ndarray[n, dict_size]}`
    of series-level encoded activations over the SAME shared row sample for
    every role target in THIS run -- enables the activation-profile
    secondary signal and its shuffled-series null. `None` skips both,
    degrading to the response-fingerprint signal alone (still the primary
    one, per sec 25.6).

    `population` (optional): `role_population_vectors(roles_json, run_dir)`'s
    output, shared across every pair `match_roles_across_models` evaluates so
    it is computed once per run, not once per pair -- when supplied, each
    match (both A->B and the reverse B->A pass this function also computes)
    is checked against `permutation_null_cosine`'s within-run population
    null (see that function's docstring), attaching `clears_population_null`
    per match and `population_null_p95`/`population_null_n_pool`/
    `frac_shared_by_population_null_a`/`_b` at the pair level -- the
    per-role-group shared/specific split `ROADMAP.md` sec 26 D1 named as the
    correct shape, in place of one pooled `match_rate` number. `None` skips
    this (both null fields become `None`), never silently substituting a
    threshold-only verdict for a null-calibrated one.

    Returns `{"model_a", "model_b", "comparable": bool, "reason": str, ...}`
    -- `comparable=False` with a stated reason is the honest "no target at a
    comparable depth" case (sec 25.6), never a silent omission.
    """
    keys_a = [k for k in roles_json if k.startswith(f"{model_a}/")
             and not (roles_json[k].get("skipped") or roles_json[k].get("withheld"))]
    keys_b = [k for k in roles_json if k.startswith(f"{model_b}/")
             and not (roles_json[k].get("skipped") or roles_json[k].get("withheld"))]
    if not keys_a or not keys_b:
        missing = model_a if not keys_a else model_b
        return {"model_a": model_a, "model_b": model_b, "comparable": False,
               "reason": f"{missing} has no usable SAE role target in this run "
                         f"(withheld, skipped, or never built) -- not \"this model "
                         f"lacks these roles\", but \"no target exists to check\""}

    key_a, key_b, chosen_depth_a, chosen_depth_b = keys_a[0], keys_b[0], None, None
    if depths:
        best_pair, best_d = None, None
        for ka in keys_a:
            da = depths.get(ka)
            if da is None:
                continue
            for kb in keys_b:
                db = depths.get(kb)
                if db is None:
                    continue
                d = abs(da - db)
                if best_d is None or d < best_d:
                    best_pair, best_d = (ka, kb), d
        if best_pair is not None:
            if best_d is not None and best_d > depth_tolerance:
                return {"model_a": model_a, "model_b": model_b, "comparable": False,
                       "reason": f"closest role targets are {best_pair[0]} and "
                                 f"{best_pair[1]} at depths {depths[best_pair[0]]:.3f}/"
                                 f"{depths[best_pair[1]]:.3f} (block axis) -- "
                                 f"{best_d:.3f} apart, beyond the {depth_tolerance:g} "
                                 f"tolerance. Not \"this model lacks this role\": no "
                                 f"target exists at a comparable depth.",
                       "depth_a": depths[best_pair[0]], "depth_b": depths[best_pair[1]],
                       "depth_gap": best_d}
            key_a, key_b = best_pair
            chosen_depth_a, chosen_depth_b = depths.get(key_a), depths.get(key_b)

    rec_a, rec_b = roles_json[key_a], roles_json[key_b]
    roles_a, roles_b = rec_a.get("roles", []), rec_b.get("roles", [])
    if not roles_a or not roles_b:
        return {"model_a": model_a, "model_b": model_b, "comparable": False,
               "reason": f"{key_a if not roles_a else key_b} built zero roles "
                         f"(too few candidates to cluster)",
               "layer_a": key_a.split('/', 1)[1], "layer_b": key_b.split('/', 1)[1]}

    cand_a, p95_a = _candidates_and_null_p95(run_dir, model_a, key_a.split("/", 1)[1])
    cand_b, p95_b = _candidates_and_null_p95(run_dir, model_b, key_b.split("/", 1)[1])
    vecs_a = [role_response_vector(r, cand_a, p95_a) for r in roles_a]
    vecs_b = [role_response_vector(r, cand_b, p95_b) for r in roles_b]

    secondary = None
    null_result = None
    if features_by_target and key_a in features_by_target and key_b in features_by_target:
        fa, fb = features_by_target[key_a], features_by_target[key_b]
        secondary = [[role_activation_profile_signals(ra, fa, rb, fb) for rb in roles_b]
                    for ra in roles_a]
        null_vals = []
        for i, ra in enumerate(roles_a):
            for j, rb in enumerate(roles_b):
                if secondary[i][j] is not None:
                    n = shuffled_series_null_activation_profile(ra, fa, rb, fb, seed=null_seed)
                    if n is not None:
                        null_vals.append(n["max"])
        if null_vals:
            null_result = {"shuffled_series_null_max_mean": float(np.mean(null_vals)),
                          "shuffled_series_null_max_p95": float(np.percentile(null_vals, 95)),
                          "n_role_pairs_nulled": len(null_vals)}

    matches = greedy_match_roles(roles_a, vecs_a, roles_b, vecs_b, secondary=secondary)
    matches_b_to_a_raw = greedy_match_roles(roles_b, vecs_b, roles_a, vecs_a)
    # Relabel so "role_a"/"role_b" always name a model_a/model_b role
    # respectively, regardless of which side was the query -- the raw
    # greedy_match_roles output always calls its query side "role_a", which
    # would otherwise make a B-side query read as an A-side role.
    matches_b_to_a = [
        {"role_b": m["role_a"], "role_b_index": m["role_a_index"],
        "role_a": m["role_b"], "role_a_index": m["role_b_index"],
        "cosine": m["cosine"]}
        for m in matches_b_to_a_raw]

    pop_null_p95, pop_n_pool = (None, 0)
    if population is not None:
        pop_null_p95, pop_n_pool = permutation_null_cosine(
            population, exclude_targets=(key_a, key_b),
            n_samples=n_population_samples, seed=null_seed)
    if pop_null_p95 is not None:
        for m in matches:
            m["clears_population_null"] = bool(abs(m["cosine"]) > pop_null_p95)
        for m in matches_b_to_a:
            m["clears_population_null"] = bool(abs(m["cosine"]) > pop_null_p95)
        frac_shared_a = sum(1 for m in matches if m["clears_population_null"]) / len(matches)
        frac_shared_b = sum(1 for m in matches_b_to_a if m["clears_population_null"]) / len(matches_b_to_a)
        roles_shared_a = [m["role_a"] for m in matches if m["clears_population_null"]]
        roles_specific_a = [m["role_a"] for m in matches if not m["clears_population_null"]]
        roles_shared_b = [m["role_b"] for m in matches_b_to_a if m["clears_population_null"]]
        roles_specific_b = [m["role_b"] for m in matches_b_to_a if not m["clears_population_null"]]
    else:
        for m in matches:
            m["clears_population_null"] = None
        for m in matches_b_to_a:
            m["clears_population_null"] = None
        frac_shared_a = frac_shared_b = None
        roles_shared_a = roles_specific_a = roles_shared_b = roles_specific_b = None

    return {"model_a": model_a, "model_b": model_b, "comparable": True,
           "target_a": key_a, "target_b": key_b,
           "layer_a": key_a.split("/", 1)[1], "layer_b": key_b.split("/", 1)[1],
           "depth_a": chosen_depth_a, "depth_b": chosen_depth_b,
           "depth_axis": "block" if depths else None,
           "n_roles_a": len(roles_a), "n_roles_b": len(roles_b),
           "matches": matches,
           "matches_b_to_a": matches_b_to_a,
           "population_null_p95": pop_null_p95,
           "population_null_n_pool": pop_n_pool,
           "frac_shared_by_population_null_a": frac_shared_a,
           "frac_shared_by_population_null_b": frac_shared_b,
           "roles_shared_a": roles_shared_a,
           "roles_specific_to_a": roles_specific_a,
           "roles_shared_b": roles_shared_b,
           "roles_specific_to_b": roles_specific_b,
           "activation_profile_shuffled_series_null": null_result,
           "note": "primary signal is sign-aware cosine over the null-normalized "
                   "response fingerprint (sec 25.6); activation-profile "
                   "correlation and max-activating-series Jaccard are secondary, "
                   "informational context computed only when both targets share "
                   "one run's row sample, never used to re-rank the primary match. "
                   "Matching is greedy nearest-neighbour, not an optimal assignment "
                   "-- a role_b may be claimed by more than one role_a. "
                   "roles_shared_a/roles_specific_to_a (and the _b mirror) are the "
                   "per-role-group split gated on the within-run population null, "
                   "not on the fixed cosine_threshold match_rate uses -- None "
                   "throughout when population was not supplied."}


# ---------------------------------------------------------------------------
# All pairs in a run.
# ---------------------------------------------------------------------------

def match_roles_across_models(model_pairs: list, roles_json: dict,
                              depths: Optional[dict] = None,
                              depth_tolerance: float = 0.15,
                              features_by_target: Optional[dict] = None,
                              null_seed: int = 0,
                              run_dir: Optional[Path] = None,
                              use_population_null: bool = True) -> list:
    """`match_roles_for_pair` for every `(model_a, model_b)` in `model_pairs`
    -- the generalization to `cfg.comparison_pairs()` (sec 24.3's rule: all
    C(n,2) pairs on a panel, pair 0 always the designated reference pair).
    Order is preserved so pair 0 of the output is always the designated
    pair, matching every other panel-pairs artifact in this repo
    (`ROADMAP.md` sec 24.3 sub-item 3's convention, `tests/
    test_panel_pairs.py`'s own load-bearing assertion about pair ordering).

    `use_population_null` (default `True`): builds `role_population_vectors`
    once from `roles_json` and shares it across every pair, so the
    within-run population null (see `permutation_null_cosine`) is computed
    from one pass over the run's roles rather than once per pair. `False`
    reproduces the pre-population-null behaviour exactly (all the new
    fields on each pair become `None`) -- kept for callers that only want
    the cheap path.
    """
    population = role_population_vectors(roles_json, run_dir) if use_population_null else None
    return [match_roles_for_pair(a, b, roles_json, depths=depths,
                                 depth_tolerance=depth_tolerance,
                                 features_by_target=features_by_target,
                                 null_seed=null_seed, run_dir=run_dir,
                                 population=population)
           for a, b in model_pairs]


# ---------------------------------------------------------------------------
# The untrained-twin null (sec 25.6, mandatory before any shared-fraction claim).
# ---------------------------------------------------------------------------

def untrained_twin_role_floor(real_model: str, real_layer: str,
                              twin_run_dir: Path, real_name_in_twin_run: str,
                              twin_name_in_twin_run: str) -> Optional[dict]:
    """Match `real_model`'s roles (from the TWIN run's own copy of that
    model, at `real_layer`) against its `random_init` twin's roles, both
    read from `<twin_run_dir>/sae/roles_injection.json` (ROADMAP.md sec 30,
    Stage 4, 2026-09-11: `sae/roles.json` is superseded by `sae/
    concepts.json` throughout the report, and archived under this name --
    this untrained-twin floor is a Component C mechanism unaffected by that
    supersession and simply reads the archived artifact).

    This is the sec 6.2.1 Stage 1 precedent applied to roles instead of the
    crosscoder's `frac_shared`: a real model's SAE roles matched against an
    ARCHITECTURE-MATCHED but UNTRAINED copy of itself. If real-vs-twin
    scores as high as (or higher than) real-vs-other-model, the cross-model
    match is not evidence of shared LEARNED structure -- it could be
    architecture alone.

    Returns `None` (not a fabricated floor) when `twin_run_dir` has no
    `sae/roles_injection.json`, or when either name's role target is
    missing/withheld there -- sec 2.5's degrade-loudly doctrine: a caller
    must be able to tell "the floor could not be computed" apart from "the
    floor is zero". `real_name_in_twin_run`/`twin_name_in_twin_run` are the
    model names AS CONFIGURED IN THE TWIN RUN (e.g. "TimesFM" and
    "TimesFM-random") -- deliberately separate from `real_model` (the name
    in the CALLER's run, which may differ if the two runs configure the pair
    under different names, though in this repo's convention they are the
    same string).
    """
    roles_path = Path(twin_run_dir) / "sae" / "roles_injection.json"
    if not roles_path.exists():
        log.info(f"role matching: no untrained-twin floor available -- "
                 f"{roles_path} does not exist (run run_sae_roles.py against "
                 f"{twin_run_dir} first, after Stage 1/2 there)")
        return None
    twin_roles = load_json(roles_path)
    result = match_roles_for_pair(real_name_in_twin_run, twin_name_in_twin_run, twin_roles,
                                  run_dir=twin_run_dir)
    if not result.get("comparable"):
        log.info(f"role matching: untrained-twin floor not comparable for "
                 f"{real_model}/{real_layer}: {result.get('reason')}")
        return None
    result["real_model"] = real_model
    result["real_layer"] = real_layer
    result["twin_run_dir"] = str(twin_run_dir)
    return result


def _match_rate(matches: list, cosine_threshold: float = 0.5) -> float:
    """Fraction of A-side roles whose best B-side match clears `cosine_threshold`.

    This is the "fraction of roles shared" quantity sec 25.6 explicitly
    forbids quoting without its untrained-twin floor beside it -- kept as
    its own small function so both the real score and the floor's score are
    computed by literally the same code, never two hand-written reductions
    that could silently diverge.
    """
    if not matches:
        return 0.0
    return sum(1 for m in matches if (m.get("cosine") or 0.0) >= cosine_threshold) / len(matches)


def role_correspondence_table(model_pairs: list, roles_json: dict,
                              depths: Optional[dict] = None,
                              depth_tolerance: float = 0.15,
                              features_by_target: Optional[dict] = None,
                              null_seed: int = 0,
                              untrained_twin_floors: Optional[dict] = None,
                              cosine_threshold: float = 0.5,
                              run_dir: Optional[Path] = None) -> dict:
    """The Stage 4 exit-criterion artifact: every pair's role matches, each
    with its match-rate NEVER rendered without an untrained-twin floor
    alongside it (sec 25.6, sec 25.9 Stage 4).

    `untrained_twin_floors`: optional `{model_name: result_dict_or_None}`,
    typically built by calling `untrained_twin_role_floor` once per model
    that has a known twin run. A pair's `match_rate` is computed
    unconditionally (it is a real, cheap statistic), but
    `match_rate_quotable` is `False` -- and a `withheld_reason` is attached
    -- whenever NEITHER side of the pair has an available floor. This is the
    structural enforcement sec 25.9 Stage 4's second exit criterion asks
    for: the number exists in the artifact (so a future floor can be joined
    against it without recomputing anything), but any renderer that skips
    the `match_rate_quotable` check would be doing so against this
    function's own stated contract, not against a silent absence.
    """
    pairs = match_roles_across_models(model_pairs, roles_json, depths=depths,
                                      depth_tolerance=depth_tolerance,
                                      features_by_target=features_by_target,
                                      null_seed=null_seed, run_dir=run_dir)
    twins = untrained_twin_floors or {}
    out_pairs = []
    for rec in pairs:
        if not rec.get("comparable"):
            out_pairs.append(rec)
            continue
        rate = _match_rate(rec["matches"], cosine_threshold=cosine_threshold)
        floor_a = twins.get(rec["model_a"])
        floor_b = twins.get(rec["model_b"])
        floor_rate_a = _match_rate(floor_a["matches"], cosine_threshold=cosine_threshold) if floor_a else None
        floor_rate_b = _match_rate(floor_b["matches"], cosine_threshold=cosine_threshold) if floor_b else None
        available_floors = [(rec["model_a"], floor_rate_a), (rec["model_b"], floor_rate_b)]
        available_floors = [(name, v) for name, v in available_floors if v is not None]
        rec = {**rec,
              "cosine_threshold": cosine_threshold,
              "match_rate": rate,
              "untrained_twin_floor": {name: v for name, v in available_floors} or None,
              "match_rate_quotable": bool(available_floors),
              "match_rate_quotable_reason": (
                  "at least one side's untrained-twin floor is available"
                  if available_floors else
                  "NO untrained-twin floor is available for either "
                  f"{rec['model_a']} or {rec['model_b']} at this target -- per "
                  "ROADMAP.md sec 25.6/6.2.1 Stage 1's precedent (frac_shared "
                  "0.845 read as high until its own floor of 0.974 showed a "
                  "real model reads as MORE shared with a random copy of "
                  "itself than with another model), this match_rate MUST NOT "
                  "be quoted as evidence of shared structure until a "
                  "random_init twin run exists for at least one side and its "
                  "Stage 0-2 artifacts are built")}
        if available_floors:
            best_floor = max(v for _, v in available_floors)
            rec["clears_untrained_twin_floor"] = rate > best_floor
        # Null-calibrated alternative to the fixed-threshold `match_rate`
        # above (sec 26 D1): quotable whenever the within-run population
        # pool was large enough to estimate a null from at all -- unlike
        # `match_rate_quotable`, this does NOT need a random_init twin run,
        # since it answers a different question (is this cosine typical
        # among arbitrary trained roles in this run, not whether
        # architecture alone explains it). Both gates are independent and
        # both must be checked; neither substitutes for the other.
        n_pool = rec.get("population_null_n_pool") or 0
        rec["match_rate_null_based"] = rec.get("frac_shared_by_population_null_a")
        rec["match_rate_null_based_quotable"] = n_pool >= 4
        rec["match_rate_null_based_reason"] = (
            f"population null estimated from {n_pool} role vectors drawn from "
            "this run's other SAE targets"
            if n_pool >= 4 else
            f"only {n_pool} role vectors available outside this pair's own two "
            "targets -- too small a population in this run to estimate a "
            "meaningful chance level from")
        out_pairs.append(rec)
    return {"pairs": out_pairs, "cosine_threshold": cosine_threshold,
           "n_pairs": len(out_pairs)}
