"""ROADMAP.md sec 30 -- re-cluster SAE features on their ABLATION fingerprint
(sec 27, `sae/response.py::feature_ablation_fingerprints`) instead of the
INJECTION fingerprint `sae/roles.py` clusters on (sec 25.23,
`feature_response_fingerprints`). Sec 30.1's gate measurement (a read-only
reduction over already-committed artifacts, no training, no new forward
passes) found the injection-space partition anti-correlated with ablation
behavior at every one of 13 real targets (mean silhouette OLD -0.235 vs NEW
+0.450) -- this module is that re-clustering, made real.

Stage 1 of sec 30.10's five-stage build (this module's core clustering
functions + `run_concepts`'s artifact write), verified against sec 30.1's own
table. Stage 2 (this update) adds `compose_name` -- the deterministic,
no-LLM naming composer (sec 30.4.1, sec 30.7's BUILD half) -- and wires it
into `run_concepts` via `assign_concept_names`, whose peer set is EVERY
concept across the WHOLE RUN (sec 30.7: "concept cards rank by interest
across the whole run, so the peer set is every concept, not just a
target's"). Cross-model transfer (`transfer.py`, stage 3) and the misfits
section (`misfits.py`, stage 5) are elsewhere; `concept_table` above still
leaves `misfits` as an explicit placeholder for that stage, and
`description`/`description_generated` for sec 30.7's Qwen narrator, which is
DESIGN ONLY and must not be implemented.

`supersede_roles_artifact` (added once report wiring made `concepts.json`
the thing the report actually reads, per this module's own earlier note
that the rename must be "gated on the report actually reading concepts.json
first" -- sec 11.39's lesson: a rename with no consumer yet is a silent
breakage waiting for the next reader) is the IRREVERSIBLE half of stage 4:
renaming one run's `sae/roles.json` to `sae/roles_injection.json`, carrying
the sec 30.1 gate measurement that justifies the supersession. It is not
part of `run_concepts` and is never invoked automatically by any pipeline
stage or CLI driver -- see `retire_roles.py` for the one-shot script that
calls it against a single, explicitly-named run directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils import load_json, log, save_json
from .response import CHANNELS
from .roles import cluster_roles

__all__ = ["CHANNELS", "ablation_vector", "build_concept_matrix",
          "cluster_concepts", "concept_table", "compose_name",
          "assign_concept_names", "run_concepts", "supersede_roles_artifact",
          "SUPERSEDED_REASON"]

SUPERSEDED_REASON = (
    "ROADMAP.md sec 30.1's gate measurement (a read-only reduction over "
    "already-committed causal-ablation artifacts, no training, no new "
    "forward passes) found this file's injection-space role partition "
    "anti-correlated with ablation behavior at every one of 13 real "
    "targets checked: mean silhouette -0.235 (this partition, scored in "
    "ablation space) vs +0.450 for `concepts.json`'s own ablation-space "
    "clustering (the same targets, fit directly in the space the "
    "comparison is made in) -- the new clustering beat this one at 13 of "
    "13 targets. `sae/concepts.json` is the report's replacement "
    "throughout (`report/sae_concepts.py::sae_concepts_block`); this file "
    "is kept, not deleted, because `run_sae_describe.py`, "
    "`run_sae_compare.py` and `sae/role_matching.py`'s untrained-twin floor "
    "all still read the injection-space evidence it carries.")

_MIN_CAUSAL_CANDIDATES = 4
_MAX_CLAUSES = 3
_TIER_CUTS = (("dominant", 5.0), ("strong", 2.0), ("mild", 1.0))


def ablation_vector(candidate: dict) -> np.ndarray | None:
    """One candidate's 9-vector of `signed_effect / null_p95`, in `CHANNELS`
    order, read from an ablation-battery candidate record
    (`sae/response.py::feature_ablation_fingerprints`'s `candidates[i]`).

    A channel with a falsy `null_p95` or a `None` `signed_effect` contributes
    0.0 -- it was not scorable, which is not the same as "no effect"
    (`CLAUDE.md` sec 11.37) -- and the count of such channels is recorded
    **on the candidate dict itself** as `n_channels_unscored`, so a caller
    iterating `candidates` afterward can read it without a second pass.

    Returns `None` when EVERY channel is unscorable, so the caller drops the
    row rather than clustering an all-zero vector into an arbitrary
    neighbourhood (an all-zero row is not "no effect on any channel", it is
    "we measured nothing about this candidate at all").
    """
    channels = candidate.get("channels") or {}
    row = np.zeros(len(CHANNELS), dtype=np.float64)
    n_unscored = 0
    for i, ch in enumerate(CHANNELS):
        rec = channels.get(ch) or {}
        p95 = rec.get("null_p95")
        signed = rec.get("signed_effect")
        if not p95 or signed is None:
            n_unscored += 1
            continue
        row[i] = float(signed) / float(p95)
    candidate["n_channels_unscored"] = n_unscored
    if n_unscored >= len(CHANNELS):
        return None
    return row


def build_concept_matrix(candidates: list, causal_only: bool = True
                        ) -> tuple[np.ndarray, list[int], dict]:
    """`-> (X [n, 9], feature_ids, diagnostics)`.

    `causal_only=True` (the default, and what sec 30.1 measured) keeps only
    candidates with `n_channels_clearing > 0`: a feature that moves nothing
    has no causal profile to cluster, and including it drags every centroid
    toward the origin. Candidates dropped for that reason are counted in
    `diagnostics["n_dropped_no_cleared_channel"]`, never silently discarded;
    a non-scorable candidate (this atom fires on no series, sec 27) is
    counted separately as `n_dropped_not_scorable`, and an all-unscored
    candidate (`ablation_vector` returning `None`) as
    `n_dropped_all_unscored`.
    """
    diagnostics = {
        "n_candidates": len(candidates),
        "n_dropped_not_scorable": 0,
        "n_dropped_no_cleared_channel": 0,
        "n_dropped_all_unscored": 0,
    }
    rows: list[np.ndarray] = []
    feature_ids: list[int] = []
    for c in candidates:
        if not c.get("scorable"):
            diagnostics["n_dropped_not_scorable"] += 1
            continue
        if causal_only and not c.get("n_channels_clearing"):
            diagnostics["n_dropped_no_cleared_channel"] += 1
            continue
        vec = ablation_vector(c)
        if vec is None:
            diagnostics["n_dropped_all_unscored"] += 1
            continue
        rows.append(vec)
        feature_ids.append(int(c["feature"]))
    X = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(CHANNELS)))
    return X, feature_ids, diagnostics


def cluster_concepts(X: np.ndarray, k="auto", min_silhouette: float = 0.1,
                     min_members: int = 3, seed: int = 0) -> dict:
    """`StandardScaler` + `KMeans`. An explicit `k` (int) is IDENTICAL to
    `roles.py::cluster_roles` -- reused by import, never reimplemented, so
    sec 30.1's NEW-vs-OLD comparison is a statement about the FEATURE SPACE
    rather than about two different clusterers (`CLAUDE.md` sec 11.41).

    `k="auto"` no longer resolves through `_resolve_role_k`'s single ratio-
    derived value (`clip(max(2, n // 6), 2, 8)`) -- `ROADMAP.md` Item D
    (sec 32.5/32.13) found that ratio's silhouette was frequently negative
    (mean -0.235 across the 13 real targets sec 30.1 measured) and, worse,
    that the metric meant to catch a bad partition REWARDS the degenerate
    one: a singleton's silhouette contribution is maximal by construction,
    so 9 of 30 concepts recorded before this fix were single-feature
    k-means artifacts the old ratio could not see as a problem.

    D1/D2: `k="auto"` now SWEEPS `k in range(2, k_upper+1)` where
    `k_upper = min(8, n // min_members, n - 1)` (`n - 1` because a `k >= n`
    partition is degenerate -- every point its own cluster -- and
    `silhouette_score` is undefined there), scoring every attempted k by
    silhouette, and marks a k `admissible` only when its SMALLEST cluster
    has at least `min_members` rows -- checked BEFORE comparing scores,
    never after, because a singleton's silhouette is exactly the number
    this admissibility gate exists to keep from winning
    (`CLAUDE.md` sec 11.48: gate before scoring, not after). The best-
    silhouette ADMISSIBLE k is selected; the OLD `min_silhouette` floor
    still applies to whatever that selection is (a partition that clears
    the size floor but not the quality floor is still `non_modular`).

    D3: when no k in the swept range is admissible, that is a recorded
    result, not a fallback to an inadmissible partition (sec 25.13 item 2's
    pre-registered outcome, echoed here for a different metric) --
    `non_modular=True` with `reason` naming the sweep.

    D4: every attempted k (whether or not it was admissible) is recorded in
    `k_sweep: [{k, silhouette, admissible, min_cluster_size}, ...]`, plus
    `k_selection_rule` describing which rule produced the returned `k` --
    "silhouette-best admissible, k swept 2..N" for the new path, or
    "explicit k=N" / "ratio (min_members<=1, legacy reproduction)" for the
    two paths that bypass the sweep (below), so a reader can always tell a
    swept k from a ratio-derived one (`ROADMAP.md` sec 31.1's own
    `min_gap`-provenance lesson, applied here to k).

    D5: `min_members<=1` is a SENTINEL, not just a lenient floor -- it
    bypasses the sweep entirely and delegates straight to `cluster_roles`
    with its own `_resolve_role_k` ratio, exactly the pre-D code path, so
    every already-recorded concept stays regenerable BY CONSTRUCTION rather
    than by the sweep coincidentally re-discovering the same k
    (`CLAUDE.md` sec 2.1/sec 7 invariant 1's precedent: freeze the old
    behavior behind an explicit, reproducible knob rather than only behind
    a default). An explicit (non-`"auto"`) `k` takes the same legacy path
    for the same reason -- `test_cluster_concepts_is_roles_cluster_roles`
    pins that this is bit-for-bit unchanged.

    A target with fewer than 4 causal candidates is skipped with a recorded
    reason rather than clustered -- three of a kind is not a partition.

    Returns `{"labels", "k", "silhouette", "non_modular", "reason",
    "k_sweep", "k_selection_rule"}`.
    """
    n = X.shape[0]
    if n < _MIN_CAUSAL_CANDIDATES:
        return {"labels": np.zeros(n, dtype=int), "k": 1, "silhouette": float("nan"),
               "non_modular": True,
               "reason": f"only {n} causal candidate(s) -- fewer than "
                         f"{_MIN_CAUSAL_CANDIDATES}, not clustered",
               "k_sweep": None, "k_selection_rule": "too few candidates"}

    if k != "auto" or int(min_members) <= 1:
        result = dict(cluster_roles(X, role_k=k, min_silhouette=min_silhouette, seed=seed))
        result["k_sweep"] = None
        result["k_selection_rule"] = (
            f"explicit k={k}" if k != "auto"
            else "ratio (min_members<=1, legacy reproduction)")
        return result

    return _sweep_k(X, min_silhouette=min_silhouette, min_members=int(min_members),
                    seed=seed)


def _sweep_k(X: np.ndarray, min_silhouette: float, min_members: int, seed: int) -> dict:
    """The D1/D2/D3/D4 sweep itself, factored out of `cluster_concepts` so
    the legacy/sweep branch point above stays readable. Only called for
    `k="auto"` with `min_members >= 2`."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    n = X.shape[0]
    k_upper = min(8, n // min_members, n - 1)
    Xs = StandardScaler().fit_transform(X)

    k_sweep: list[dict] = []
    best = None  # (silhouette, k, labels)
    for kk in range(2, k_upper + 1):
        labels = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(Xs)
        sizes = np.bincount(labels, minlength=kk)
        min_cluster_size = int(sizes.min())
        admissible = min_cluster_size >= min_members
        sil = float(silhouette_score(Xs, labels))
        k_sweep.append({"k": kk, "silhouette": sil, "admissible": admissible,
                        "min_cluster_size": min_cluster_size})
        if admissible and (best is None or sil > best[0]):
            best = (sil, kk, labels)

    if best is None:
        return {"labels": np.zeros(n, dtype=int), "k": 1, "silhouette": float("nan"),
               "non_modular": True,
               "reason": (f"no k in 2..{k_upper} partitions {n} candidates into "
                          f"clusters of at least {min_members} members"),
               "k_sweep": k_sweep,
               "k_selection_rule": f"silhouette-best admissible, k swept 2..{k_upper}"}

    sil, kk, labels = best
    non_modular = bool(np.isfinite(sil) and sil < min_silhouette)
    reason = "" if not non_modular else f"silhouette {sil:.3f} below the {min_silhouette:g} floor"
    return {"labels": labels, "k": kk, "silhouette": sil,
           "non_modular": non_modular, "reason": reason,
           "k_sweep": k_sweep,
           "k_selection_rule": f"silhouette-best admissible, k swept 2..{k_upper}"}


def _dominant_channel_index(mean_row: np.ndarray, channel_columns: list,
                            rank: int = 0) -> tuple:
    """The `rank`-th most discriminating channel by `|mean|`, and its sign.

    Originally documented here as "must NEVER be rendered as the concept's
    name" per sec 30.1 measurement 2 (30.3% member-agrees-with-name against
    a 30.3% permutation null). **Corrected (ROADMAP.md sec 32.4, Item C):**
    measurement 2's finding is about naming by **argmax alone** -- picking
    a name from raw magnitude with no contrast against the peer population,
    which is what produced chance-level agreement. It is a claim about HOW
    the channel was chosen as the WHOLE name, not a blanket ban on this
    value ever appearing in a composed name. `_compose_batch`'s design (c)
    renders this channel as the name's LEAD clause specifically because it
    is `concept_table`'s own `dominant_channel` -- what the concept's
    largest measured effect actually is -- and pairs it with a SEPARATE,
    statistically-validated contrast clause (the sec 30.7 z-score mechanism)
    that answers a different question. That composition is not naming by
    argmax; sec 32.9's own correction (search "no single-channel concept
    name anywhere") already draws this exact line for a different
    acceptance-criterion draft. `concept_table` still also stores this as
    the standalone `dominant_channel` field for downstream sort/filter use.
    """
    order = np.argsort(-np.abs(mean_row))
    if rank >= len(order):
        return None, 0
    idx = int(order[rank])
    val = float(mean_row[idx])
    if val == 0.0:
        return None, 0
    return channel_columns[idx], (1 if val > 0 else -1)


def _within_cosine_mean(rows: np.ndarray) -> float:
    """Mean pairwise cosine similarity among a concept's own member rows --
    sec 30.1's secondary check (corroborating the silhouette from a
    different statistic, since KMeans optimizes standardized Euclidean
    distance, not cosine). `NaN` for a singleton (no pair to compare)."""
    n = rows.shape[0]
    if n < 2:
        return float("nan")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = rows / norms
    sims = unit @ unit.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(sims[iu]))


def _centroid_cosine_mean(rows: np.ndarray, centroid: np.ndarray) -> float:
    """Mean cosine similarity between each member row and the concept's own
    CENTROID -- distinct from `_within_cosine_mean`'s mean PAIRWISE cosine,
    and the two are not interchangeable (`ROADMAP.md` sec 32.13b/Item L): a
    centroid sits closer to every member than members sit to one another by
    construction, so `misfits.py::misfit_rows` used to compare a member's
    cosine-to-centroid against a bar built from the pairwise mean -- two
    different statistics, measured +0.248 apart on average, which is why the
    detector produced zero misfits site-wide. This is the bar-setting
    statistic Item L calls for: same "distance from centroid" quantity that
    gets scored, so the threshold and the score are finally like-for-like.
    `NaN` for a singleton, matching `_within_cosine_mean`'s own contract."""
    n = rows.shape[0]
    if n < 2:
        return float("nan")
    c_norm = float(np.linalg.norm(centroid))
    if c_norm == 0.0:
        return float("nan")
    row_norms = np.linalg.norm(rows, axis=1)
    valid = row_norms > 0
    if not np.any(valid):
        return float("nan")
    cos = (rows[valid] @ centroid) / (row_norms[valid] * c_norm)
    return float(np.mean(cos))


def _magnitude_tier(val: float) -> str | None:
    """`dominant` >=5, `strong` >=2, `mild` >=1 null units, else `None` --
    boundaries inclusive, checked largest-first (sec 30.7)."""
    a = abs(val)
    for tier, cut in _TIER_CUTS:
        if a >= cut:
            return tier
    return None


def _verb(val: float) -> str:
    """`raises`/`lowers` at |val| >= 1.0 null unit, else `moves` -- a
    sub-null mean's sign is not resolvable (sec 11.54's second site;
    `test_sub_null_channel_loses_its_direction`)."""
    if val >= 1.0:
        return "raises"
    if val <= -1.0:
        return "lowers"
    return "moves"


def _rank_candidates(row: int, cleared: np.ndarray, z: np.ndarray) -> list:
    """Row `row`'s cleared channel indices, ranked by descending |z| --
    channel index (a fixed constant, never row-dependent) breaks ties, so
    the ranking depends only on this row's own content, never on where it
    sits in the input arrays (sec 11.2/sec 11.55: array position must never
    decide an outcome that is supposed to be reproducible)."""
    idxs = [c for c in range(cleared.shape[1]) if cleared[row, c]]
    idxs.sort(key=lambda c: (-abs(z[row, c]), c))
    return idxs


def _compose_batch(profiles: np.ndarray, cleared: np.ndarray, peers: list,
                   tags: list | None = None) -> list[dict]:
    """One `{"name": str, "lead_diversified": bool}` per row of `profiles`,
    applying `compose_name`'s mechanisms across the WHOLE peer population at
    once -- mechanism 2 needs every row's claim to decide the next row's,
    and mechanism 3 needs every row's candidate name at each length k.
    `compose_name` is a thin per-row wrapper over this; production callers
    (`assign_concept_names`) call this directly rather than paying O(n)
    repeated O(n^2) batch recomputes.

    ROADMAP.md sec 32.4, Item C, design (c). The name's FIRST clause is
    always `{tier(dominant)} {verb(dominant)} {dominant_channel}` -- the
    same channel `concept_table` persists as `dominant_channel`
    (`_dominant_channel_index`, ranked by raw |mean|), so a name can never
    assert a magnitude tier for a channel that isn't the one carrying it.
    The pre-Item-C design picked its LEAD clause by z-score contrast and
    read the tier word off THAT channel's own raw value -- conflating "what
    is statistically unusual about this concept among its peers" with "how
    large is this concept's actual effect" (sec 11.54's pattern: two correct
    fields read together as one claim).

    `tags` (ROADMAP.md sec 37.7 P4, optional, `None` reproduces the pre-P4
    behavior bit for bit): one `"level carrier"`/`"shape-causal"`/`"no
    measured effect"` string per row (`sae/concept_atlas.py::
    _tag_causal_effect`). A row tagged `"level carrier"` is withheld from
    BOTH the dominant-lead and the contrast-clause competition -- its
    `dominant_idx` is forced `None` and its contrast candidates forced empty
    -- and its rendered name is unconditionally `"shifts the level"`,
    overriding whatever `render` would otherwise have produced from its RAW
    (level-confounded) centroid. This is the fix P0/P4 exist for: a level
    carrier's RAW channel profile routinely has a large `mase` or
    `horizon_shape` value that is entirely level bleed-through (every raw
    channel moves when the forecast's level does), and naming it `"dominant
    raises horizon_shape_far"` would assert a shape claim the level-removed
    battery does not support. Withholding it from the clause competition
    (rather than only overriding its OWN name afterward) also means a level
    carrier's confounded channel can never occupy a `used_leads` dedup key
    (mechanism 2) that a genuinely shape-causal peer might otherwise want --
    it simply never enters that bookkeeping at all.
    """
    n = profiles.shape[0]
    if n == 0:
        return []

    is_level_carrier = [bool(tags) and tags[i] == "level carrier" for i in range(n)]
    n_channels = profiles.shape[1]

    # The dominant channel: identical computation to `concept_table`'s own
    # `dominant_channel` field (rank-0 by raw |mean|, via the shared
    # `_dominant_channel_index` helper) so the name's lead and the
    # persisted `dominant_channel` field can never disagree about which
    # channel is meant.
    dominant_idx: list = []
    for i in range(n):
        if is_level_carrier[i]:
            dominant_idx.append(None)
            continue
        name, _sign = _dominant_channel_index(profiles[i], CHANNELS)
        dominant_idx.append(CHANNELS.index(name) if name is not None else None)

    # Mechanism 1: contrastive z-score, computed over ALL peers (not just
    # those clearing a given channel) -- it characterizes what is TYPICAL
    # for that channel across the population, which is what makes an
    # outlier row's own clearing value stand out.
    mean = profiles.mean(axis=0)
    std = profiles.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    z = (profiles - mean) / std_safe

    own_ranked = [_rank_candidates(i, cleared, z) for i in range(n)]
    # The CONTRAST clause is the z-selected channel, distinct from the
    # dominant one -- design (c): "omitted when it is the same channel".
    # Filtering it out of the candidate list up front (rather than picking
    # the top z-ranked candidate and discarding the row's whole contrast
    # clause whenever it collides with dominant) means a row whose top
    # z-candidate happens to be its own dominant channel still gets a
    # genuinely distinct second clause when one of its other cleared
    # channels can supply it -- no diversity signal is thrown away that
    # doesn't have to be.
    contrast_candidates = [[] if is_level_carrier[i] else
                          [c for c in own_ranked[i] if c != dominant_idx[i]]
                          for i in range(n)]
    top_abs_z = [abs(z[i, contrast_candidates[i][0]]) if contrast_candidates[i] else -np.inf
                for i in range(n)]

    # Mechanism 2: leading-CONTRAST-clause diversification. Process rows in
    # descending top-|z|, tie-broken by `peers[i]` (a (target, concept_id)
    # -shaped key) -- NEVER by array position, which is what makes the
    # result invariant to how the caller ordered its rows.
    #
    # The dedup key is the RENDERED contrast clause -- (channel, verb). No
    # tier here: a contrast clause never carries a tier word (design (c)
    # renders it as "unusually {verb} {channel}"), so two rows differing
    # only in magnitude at the same channel/sign render IDENTICAL contrast
    # text and must be deduped as such, unlike the old tier-inclusive key
    # this replaces (which was keying the LEAD clause, which did carry a
    # tier). Deduping on bare channel index (an earlier version of this
    # function, pre-Item-C) is strictly more conservative than the spec
    # requires; keying on the rendered clause reaches a higher distinct-name
    # count -- see ROADMAP.md sec 30's dated Stage 2 update for the
    # (pre-Item-C) diagnosis of this same tradeoff.
    def _contrast_clause_key(row: int, ch: int) -> tuple:
        return (ch, _verb(float(profiles[row, ch])))

    order = sorted(range(n), key=lambda i: (-top_abs_z[i], peers[i]))
    used_leads: set = set()
    contrast_lead: list = [None] * n
    for i in order:
        candidates = contrast_candidates[i]
        if not candidates:
            continue
        chosen = next((c for c in candidates
                       if _contrast_clause_key(i, c) not in used_leads),
                      candidates[0])
        contrast_lead[i] = chosen
        used_leads.add(_contrast_clause_key(i, chosen))

    clause_order: list[list[int]] = []
    lead_diversified = [False] * n
    for i in range(n):
        candidates = contrast_candidates[i]
        if not candidates:
            clause_order.append([])
            continue
        lead = contrast_lead[i]
        clause_order.append([lead] + [c for c in candidates if c != lead])
        lead_diversified[i] = (lead != candidates[0])

    def _dominant_text(i: int) -> str | None:
        dom = dominant_idx[i]
        if dom is None:
            return None
        val = float(profiles[i, dom])
        tier = _magnitude_tier(val)
        clause = f"{_verb(val)} {CHANNELS[dom]}"
        return f"{tier} {clause}" if tier else clause

    def render(i: int, k: int) -> str:
        dom_text = _dominant_text(i)
        contrast_chans = clause_order[i][:k]
        contrast_clauses = [f"unusually {_verb(float(profiles[i, c]))} {CHANNELS[c]}"
                           for c in contrast_chans]
        if dom_text is None:
            if not contrast_clauses:
                return "no channel clears its own null"
            return " · ".join(contrast_clauses)
        if not contrast_clauses:
            return dom_text
        return dom_text + " · " + " · ".join(contrast_clauses)

    # Mechanism 3: shortest unique prefix, over the CONTRAST clauses only --
    # the dominant clause is fixed and always present (k=0 means "dominant
    # clause alone"). A peer with fewer contrast clauses than k is compared
    # at ITS OWN max length, since it cannot render further.
    max_contrast = max(_MAX_CLAUSES - 1, 0)
    names: list = [None] * n
    for i in range(n):
        max_k = min(len(clause_order[i]), max_contrast)
        chosen = render(i, max_k)  # fallback if every k collides: the cap
        for k in range(0, max_k + 1):
            candidate = render(i, k)
            collides = any(
                render(j, min(k, len(clause_order[j]))) == candidate
                for j in range(n) if j != i)
            if not collides:
                chosen = candidate
                break
        names[i] = chosen

    # ROADMAP.md sec 37.7 P4: a level carrier's name is fixed and honest
    # rather than composed from its (level-confounded) raw centroid -- see
    # this function's own docstring. Two level carriers sharing this exact
    # name is not a naming defect: it is the correct statement that the
    # level-removed battery cannot tell them apart by SHAPE, which is the
    # only thing the rest of this composer's vocabulary describes.
    for i in range(n):
        if is_level_carrier[i]:
            names[i] = "shifts the level"

    return [{"name": names[i], "lead_diversified": lead_diversified[i]} for i in range(n)]


def compose_name(i: int, profiles: np.ndarray, cleared: np.ndarray, peers: list) -> str:
    """The deterministic name for peer `i`. Short, specific, and distinct
    from its peers -- no LLM, no weights. See sec 30.4.1/sec 30.7,
    corrected by sec 32.4 (Item C).

    `profiles`/`cleared` are `[n_peers, n_channels]`, row-aligned with
    `peers` (a list of ids used only as a stable tie-break, e.g.
    `(target, concept_id)`). The name has two parts:

    1. DOMINANT LEAD, always present and always first: the channel
       `concept_table` itself names `dominant_channel` (rank-0 by raw
       |mean|), rendered `{tier} {verb} {channel}` -- a name can never omit
       or misstate the concept's own largest measured effect.
    2. CONTRAST CLAUSE, present only when it differs from the dominant
       channel: the most STATISTICALLY UNUSUAL cleared channel among this
       row's peers (z-score contrast, sec 30.7's original mechanism 1),
       rendered `unusually {verb} {channel}` with NO tier word -- it answers
       "what marks this concept out from its peers", a different question
       from the lead's "how large is this effect", and the wording makes
       that difference legible rather than implying a magnitude it may not
       have.

    Two sub-mechanisms operate on the contrast clause only (never on the
    now-fixed dominant lead):

    a. LEADING-CONTRAST DIVERSIFICATION. Process rows in descending top-|z|
       among each row's own (dominant-excluded) candidates, tie-broken by
       `peers[i]` so the pass is reproducible (sec 11.2). Each row takes its
       most contrastive channel whose contrast clause no earlier row has
       taken, falling back to its own best when all are taken.
    b. SHORTEST UNIQUE PREFIX. Emit 0, 1, 2... contrast clauses (0 means
       "dominant clause alone") and stop at the first count whose full
       rendered name no peer shares at that same count (a peer with fewer
       available contrast clauses is compared at its own max). Capped at
       `_MAX_CLAUSES - 1` contrast clauses even if a collision remains at
       the cap.

    Clause wording: `raises`/`lowers` <channel>, downgraded to `moves` below
    1.0 null unit -- a sub-null mean's sign is not resolvable (sec 11.54's
    second site).

    A row with no dominant channel at all (every raw value exactly 0.0) and
    no cleared channel to fall back to names as "no channel clears its own
    null" rather than fabricating one.
    """
    return _compose_batch(profiles, cleared, peers)[int(i)]["name"]


def assign_concept_names(targets: dict) -> None:
    """Mutates `targets` (`run_concepts`'s own artifact structure) in
    place, setting `name`/`name_lead_diversified` on every concept record.

    `name_lead_diversified` (sec 32.4, Item C): since the name's FIRST
    clause is now always the fixed `dominant_channel` lead, this field no
    longer describes the lead clause -- it describes whether the CONTRAST
    clause (the second, z-selected clause) is this row's own top-|z|
    candidate or was displaced from it by mechanism 2's diversification
    pass. `True` only when a row has a contrast candidate at all and
    diversification actually moved it off that row's own best pick.

    The peer set is EVERY concept across the WHOLE RUN, not one target's
    own concepts (sec 30.7: "concept cards rank by interest across the
    whole run, so the peer set is every concept"). `profiles`/`cleared` are
    built from each concept's own `centroid_null_units`/`profile` -- the
    same channel set `concept_table` already decided clears at the concept
    level, so naming never re-derives a different notion of "cleared" than
    what the card itself renders.

    Iteration order is `sorted(targets)` then each target's own `concepts`
    list order -- a fixed, content-derived order, not insertion order, so
    the peer tie-break key (`(target, concept_id)`) is what actually
    decides ties, never array position (sec 11.2/sec 11.55).
    """
    peers: list = []
    rows: list = []
    cleared_rows: list = []
    refs: list = []
    for target_key in sorted(targets):
        target = targets[target_key]
        for concept in target.get("concepts", []):
            peers.append((target_key, concept["concept"]))
            centroid = concept["centroid_null_units"]
            rows.append([float(centroid.get(ch, 0.0)) for ch in CHANNELS])
            cleared_channels = {p["channel"] for p in concept["profile"]}
            cleared_rows.append([ch in cleared_channels for ch in CHANNELS])
            refs.append(concept)

    if not peers:
        return

    profiles = np.asarray(rows, dtype=np.float64)
    cleared = np.asarray(cleared_rows, dtype=bool)
    batch = _compose_batch(profiles, cleared, peers)
    for ref, info in zip(refs, batch):
        ref["name"] = info["name"]
        ref["name_lead_diversified"] = info["lead_diversified"]

    names_seen: dict = {}
    for target_key, concept in ((p[0], r) for p, r in zip(peers, refs)):
        names_seen.setdefault(concept["name"], []).append((target_key, concept["concept"]))
    dupes = {n: ids for n, ids in names_seen.items() if len(ids) > 1}
    if dupes:
        log.warning("sae concepts: %d name(s) shared by >1 concept after the "
                    "3-clause cap: %s", len(dupes), dupes)


def concept_table(candidates: list, X: np.ndarray, feature_ids: list,
                  cluster_result: dict, ablation_art: dict | None = None) -> list[dict]:
    """One record per concept -- sec 30.5's schema, minus the fields later
    stages own: `name`/`name_lead_diversified` are left `None` HERE (this
    function does not name its own output) -- naming is a separate,
    whole-run pass (`assign_concept_names`, sec 30.10 stage 2's
    `compose_name`) run once ALL targets' concepts exist, since the peer
    set for concept naming is every concept in the run, not one target's
    (sec 30.7). `misfits` (sec 30.10 stage 5's `misfits.py`) is left `[]`.
    `description`/`description_generated` are sec 30.7's Qwen narrator,
    design-only per the user's own instruction and correctly absent from
    every stage of this build; left `None`/`False`.

    `candidates` is the target's FULL candidate list (as read from
    `sae/<model>/<layer>_ablation.json`); `feature_ids`/`X` are
    `build_concept_matrix`'s own output, so row `i` of `X` is feature
    `feature_ids[i]` and `cluster_result["labels"][i]` is that row's concept.
    Returns `[]` when `cluster_result["non_modular"]` is True -- a
    non-partitioning dictionary renders no concepts as if real.
    """
    if cluster_result.get("non_modular"):
        return []
    by_feature = {int(c["feature"]): c for c in candidates}
    labels = cluster_result["labels"]
    channel_columns = list(CHANNELS)

    groups: dict[int, list[int]] = {}
    for row_idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(row_idx)

    out = []
    for concept_id in sorted(groups, key=lambda g: -len(groups[g])):
        row_idxs = groups[concept_id]
        members_X = X[row_idxs]
        member_features = [feature_ids[i] for i in row_idxs]
        member_candidates = [by_feature[f] for f in member_features]
        centroid = members_X.mean(axis=0)

        profile = []
        for ci, ch in enumerate(channel_columns):
            centroid_val = float(centroid[ci])
            if abs(centroid_val) < 1.0:
                continue
            n_clearing = sum(
                1 for c in member_candidates
                if ((c.get("channels") or {}).get(ch) or {}).get("clears_null"))
            profile.append({"channel": ch, "signed_null_units": centroid_val,
                            "n_members_clearing": n_clearing})
        profile.sort(key=lambda p: -abs(p["signed_null_units"]))

        dominant_channel, _sign = _dominant_channel_index(centroid, channel_columns)
        n_members_clearing = sum(1 for c in member_candidates
                                 if c.get("n_channels_clearing", 0) > 0)

        out.append({
            "concept": concept_id,
            "features": member_features,
            "n_members": len(member_features),
            "n_members_clearing": n_members_clearing,
            "centroid_null_units": {ch: float(centroid[ci])
                                    for ci, ch in enumerate(channel_columns)},
            "profile": profile,
            "dominant_channel": dominant_channel,
            "name": None,
            "name_lead_diversified": None,
            "within_cosine_mean": _within_cosine_mean(members_X),
            "centroid_cosine_mean": _centroid_cosine_mean(members_X, centroid),
            "misfits": [],
            "description": None,
            "description_generated": False,
        })
    return out


def run_concepts(run_dir: Path, cfg) -> dict:
    """Driver. Reads every `sae/<model>/<layer>_ablation.json` under
    `run_dir`, re-clusters each target in ablation space, names every
    concept (`assign_concept_names`, whole-run peer set) and writes
    `sae/concepts.json` (sec 30.5's schema).

    Stage 1+2 scope only (sec 30.10): does **not** rename `roles.json` to
    `roles_injection.json` -- that supersession is stage 4, gated on the
    report actually reading `concepts.json` first (sec 11.39: a rename with
    no consumer yet is a silent breakage waiting for the next reader).
    """
    run_dir = Path(run_dir)
    sae_cfg = getattr(cfg, "sae", None)
    causal_only = bool(getattr(sae_cfg, "concept_causal_only", True))
    k = getattr(sae_cfg, "concept_k", "auto")
    min_silhouette = float(getattr(sae_cfg, "concept_min_silhouette", 0.1))
    min_members = int(getattr(sae_cfg, "concept_min_members", 3))
    seed = int(getattr(getattr(cfg, "run", None), "seed", 0) or 0)

    targets: dict[str, dict] = {}
    for ablation_file in sorted(run_dir.glob("sae/*/*_ablation.json")):
        art = load_json(ablation_file)
        model = art.get("model", ablation_file.parent.name)
        layer = art.get("layer", ablation_file.name[: -len("_ablation.json")])
        key = f"{model}/{layer}"

        if art.get("withheld"):
            targets[key] = {"model": model, "layer": layer, "withheld": True,
                            "reason": art.get("reach", {}).get("reason", "")}
            log.info("sae concepts: %s withheld (%s)", key, targets[key]["reason"])
            continue

        candidates = art.get("candidates", [])
        X, feature_ids, diagnostics = build_concept_matrix(candidates, causal_only=causal_only)
        cluster_result = cluster_concepts(X, k=k, min_silhouette=min_silhouette,
                                          min_members=min_members, seed=seed)
        concepts = concept_table(candidates, X, feature_ids, cluster_result, art)

        targets[key] = {
            "model": model, "layer": layer,
            "n_candidates": diagnostics["n_candidates"],
            "n_causal": len(feature_ids),
            "n_dropped_no_cleared_channel": diagnostics["n_dropped_no_cleared_channel"],
            "n_dropped_not_scorable": diagnostics["n_dropped_not_scorable"],
            "n_dropped_all_unscored": diagnostics["n_dropped_all_unscored"],
            "k": cluster_result["k"], "silhouette": cluster_result["silhouette"],
            "non_modular": cluster_result["non_modular"],
            "reason": cluster_result["reason"],
            "k_sweep": cluster_result.get("k_sweep"),
            "k_selection_rule": cluster_result.get("k_selection_rule"),
            "channel_columns": list(CHANNELS),
            "concepts": concepts,
        }
        log.info("sae concepts: %s: k=%d silhouette=%.3f non_modular=%s "
                 "(%d causal of %d candidates)", key, cluster_result["k"],
                 cluster_result["silhouette"] if np.isfinite(cluster_result["silhouette"]) else float("nan"),
                 cluster_result["non_modular"], len(feature_ids), len(candidates))

    assign_concept_names(targets)

    out = {"schema_version": 1, "space": "ablation",
          "supersedes": "roles_injection.json", "targets": targets}
    out_path = run_dir / "sae" / "concepts.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    log.info("sae concepts: wrote %s (%d targets)", out_path, len(targets))
    return out


def supersede_roles_artifact(run_dir: Path,
                             superseded_by: str = "concepts.json",
                             superseded_reason: str = SUPERSEDED_REASON,
                             ) -> Path:
    """The IRREVERSIBLE half of ROADMAP.md sec 30's stage 4: rename one run's
    `sae/roles.json` to `sae/roles_injection.json`, carrying two new
    top-level keys -- `superseded_by` and `superseded_reason` -- alongside
    every original per-target record, which are left byte-for-byte
    unchanged (never deleted, never re-derived: this is a rename plus an
    annotation, not a re-clustering).

    Preconditions, each checked and raised on rather than silently
    skipped or worked around (sec 2.5):
      - `sae/concepts.json` must already exist -- this module's own
        `run_concepts` docstring states the rename is "gated on the report
        actually reading concepts.json first" (sec 11.39: a rename with no
        consumer yet is a silent breakage waiting for the next reader).
      - `sae/roles.json` must exist -- nothing to rename otherwise.
      - `sae/roles_injection.json` must NOT already exist -- this function
        is meant to run exactly once per run directory; re-running it
        against an already-migrated run would either silently no-op (if it
        skipped) or silently clobber a previously-archived copy (if it
        overwrote), and either is worse than refusing with a clear reason.

    The two new keys are added at the TOP LEVEL, beside the existing
    per-target keys (each of the form "Model/layer"), rather than nested
    inside every record -- mirroring `concepts.json`'s own top-level
    `schema_version`/`space`/`supersedes` keys. This is safe for every
    consumer of `roles_injection.json` checked at the time this function
    was written (`run_sae_describe.py`, `sae/role_matching.py`'s
    `untrained_twin_role_floor`/`role_population_vectors`, all of which key
    or filter by dict-ness/prefix before dereferencing) except
    `run_sae_compare.py::load_inputs`, which was fixed in the same change
    to filter to dict-valued entries before iterating. A future consumer
    that iterates every top-level key/value assuming each is a target
    record must apply the same filter.

    Writes the new file, reads it back and verifies every original
    target's record is present and unchanged, and ONLY THEN removes the
    original `roles.json` -- so a write failure or a verification mismatch
    leaves `roles.json` untouched rather than losing data between two
    partially-completed steps. Returns the new file's path.
    """
    run_dir = Path(run_dir)
    concepts_path = run_dir / "sae" / "concepts.json"
    roles_path = run_dir / "sae" / "roles.json"
    archived_path = run_dir / "sae" / "roles_injection.json"

    if not concepts_path.exists():
        raise FileNotFoundError(
            f"{concepts_path} does not exist -- the rename is gated on the "
            f"report actually reading concepts.json first (this module's "
            f"own `run_concepts` docstring); run `run_concepts`/the sec 30 "
            f"build against {run_dir} before superseding its roles.json")
    if not roles_path.exists():
        raise FileNotFoundError(
            f"{roles_path} does not exist -- nothing to rename. If this run "
            f"was already superseded, {archived_path} should exist instead")
    if archived_path.exists():
        raise FileExistsError(
            f"{archived_path} already exists -- this run appears to have "
            f"already been superseded; refusing to overwrite an existing "
            f"archived artifact. Delete it by hand first if this is a "
            f"deliberate re-run")

    original = load_json(roles_path)
    if not isinstance(original, dict):
        raise TypeError(
            f"{roles_path} does not parse to a JSON object at the top "
            f"level (got {type(original).__name__}) -- refusing to guess "
            f"how to annotate it")

    archived = {"superseded_by": superseded_by,
               "superseded_reason": superseded_reason, **original}
    save_json(archived_path, archived)

    # Verify before removing the original -- a write or round-trip failure
    # must leave roles.json exactly as it was.
    reread = load_json(archived_path)
    for key, rec in original.items():
        if reread.get(key) != rec:
            raise RuntimeError(
                f"round-trip verification failed for target {key!r} after "
                f"writing {archived_path} -- {roles_path} was NOT removed. "
                f"Inspect both files by hand before retrying")
    if reread.get("superseded_by") != superseded_by:
        raise RuntimeError(
            f"round-trip verification failed for 'superseded_by' after "
            f"writing {archived_path} -- {roles_path} was NOT removed")

    roles_path.unlink()
    log.info("sae concepts: superseded %s -> %s (%d target(s) carried over "
             "unchanged, original removed)", roles_path, archived_path,
             len(original))
    return archived_path
