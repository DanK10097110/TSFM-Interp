"""ROADMAP.md sec 37 (concept FAMILIES) -- a general, human-readable layer
above `sae/concept_atlas.py`'s tight concepts.

USER COMPLAINT this module answers (verbatim, orchestrator diagnosis on
`runs/full_report_run_4model/sae/concept_atlas.json`): the atlas's own
complete-linkage clustering at `min_cosine: 0.9` yields 19 concepts of only
3-4 members each, with 138 of 201 causal features (69%) left unassigned --
because complete linkage requires EVERY pair inside a concept to clear 0.9,
so a feature at cosine 0.89 to a concept's whole membership is excluded, even
though it is a near-miss, not an unrelated feature. The sharing map (a UMAP
projection) then makes such excluded neighbours look like they sit inside
the concept they were refused from. Several tight concepts are also
near-duplicates of each other (five names starting "strong raises
horizon_shape_near"), and every concept's machine name ("strong raises
horizon_shape_near . unusually lowers level . unusually raises trend") is
unreadable to anyone who did not write `sae/concepts.py::_compose_batch`.

ADDITIVE throughout (`CLAUDE.md` invariant 13): `sae/concept_atlas.py`,
`concept_atlas.json` and every one of its existing keys are untouched by
this module. This module re-derives the SAME pooled causal-feature matrix
(`concept_atlas.py::pooled_features`, pure I/O + reduction, deterministic --
two calls against the same run directory produce byte-identical output) and
writes its own artifact, `sae/concept_families.json`. Nothing that reads
`concept_atlas.json` (transfer, P5b, profiles, P7 confirmation, registered
claims) needs to change; this is a new, coarser view on top.

WHY A SEPARATE ARTIFACT rather than new keys on `concept_atlas.json`: the
family layer has its own null procedures, its own config knobs and its own
staleness lifecycle (`sae/concept_stage.py`'s `_drop_stale` pattern), and
keeping it out of `concept_atlas.py`'s own writer is the cheapest way to
guarantee that writer's legacy keys stay byte-identical -- there is no
touched code path in `run_concept_atlas` at all, so nothing there can drift.

THE FAMILY, in one paragraph: average-linkage cosine hierarchical clustering
of the same L2-normalized 9-channel ablation vectors the atlas clusters,
cut at whichever cosine threshold in a small grid (0.50-0.85) gives the best
silhouette among cuts that admit at least two clusters of
>= `atlas_family_min_members` (default 5) members ("families" -- everything
smaller at that cut is a draft leftover, not a family). Family CENTROIDS are
then computed from those draft families, and EVERY pooled feature --
including the atlas's own tight-concept members and every feature the tight
atlas left unassigned -- is assigned to its nearest centroid by cosine, if
that cosine clears `atlas_family_assign_min` (default 0.5), else left
"idiosyncratic". Membership is therefore explainable by one rule (nearest
centroid) applied uniformly to every feature, and a decoy sitting at cosine
~0.88 to a tight concept (excluded from that concept exactly as today, since
this module never touches `min_cosine: 0.9`/complete linkage) is exactly the
kind of near-miss this looser grid is built to recover into the right
family -- by the loose average-linkage draft clustering itself when the
decoy's cosine already clears the chosen grid threshold (as low as 0.5), and
by the nearest-centroid step for anything that misses every draft cluster
outright (e.g. too few similar rows to reach `atlas_family_min_members` on
its own): `tests/test_concept_families.py::
test_nearest_centroid_step_recovers_an_orphan_the_draft_clustering_drops`
isolates the second mechanism specifically.

SIGN CONVENTION, stated once because every title/description this module
writes depends on it (`CLAUDE.md` sec 8: "labels are claims"). `response.py`
records `signed_effect` as the effect OF ABLATING (removing) a feature:
positive means the channel rose when the feature was zeroed out. This module
follows `sae/describe.py`'s own already-audited convention (`ablation_channels`
there, sec 32.7c) rather than inventing a second one: every profile value
this module renders is NEGATED before it reaches a title or a sentence, so a
title/description states what the feature's presence DOES to the forecast,
not what removing it does. "Ablating these features lowers seasonality"
(the raw, recorded `signed_effect`) is rendered here as "these features
increase seasonality" -- the plain-language claim a reader would actually
want. `test_concept_families.py::test_sign_convention_planted_flip_is_caught`
plants the un-negated (ablation-direction) convention and confirms the
resulting title's direction disagrees with the recorded channel sign.

Reuses, never re-derives: `concept_atlas.py::pooled_features` (the pooled
matrix), `_right_tail_p`/`_left_tail_p`/`_mean_purity`/`cross_model_verdict`/
`_cross_model_null` (both null procedures at the family level are the exact
same permutation machinery, just handed family labels instead of concept
labels), `sae/describe.py::CHANNEL_GLOSS`/`CHANNEL_VERB`/`_DEFAULT_VERB` (the
same audited plain-language vocabulary the per-feature narrator already
uses, rather than a fourth hand-written vocabulary table -- `CLAUDE.md` sec
11.53's "two hand-maintained lists of the same vocabulary" defect shape).

SEED STABILITY (ROADMAP sec 37 P2's family-level extension, item 3 of the
spec this module implements): reported only as a CHEAP aggregate over
`sae/concept_stability.json`'s already-computed per-concept reciprocal-
transfer stability -- "what fraction of this family's member CONCEPTS are
themselves seed-stable". A literal "does a replicate SAE's own feature land
in a matching family" would need the ablation battery re-run on every
replicate dictionary (an expensive forward-pass battery per replicate, never
computed today: only the primary dictionary at each target has an
`_ablation.json`), so that question is explicitly OUT of scope here and the
artifact says so rather than fabricating a cheaper proxy as if it answered
the same question.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from ..utils import load_json, log, save_json
from .concept_atlas import (_cross_model_null, _left_tail_p, _mean_purity,
                            _right_tail_p, cross_model_verdict, pooled_features)
from .describe import CHANNEL_GLOSS, CHANNEL_VERB, _DEFAULT_VERB
from .plain_text import cleared_ranked as _cleared_ranked
from .plain_text import compose_title as _family_title
from .plain_text import directed_profile as _directed_profile
from .response import CHANNELS

__all__ = ["cluster_families", "run_concept_families", "load_families"]

_FAMILY_LINKAGE_METHOD = "average"
_FAMILY_LINKAGE_METRIC = "cosine"
DEFAULT_FAMILY_MIN_MEMBERS = 5
DEFAULT_FAMILY_ASSIGN_MIN = 0.5
DEFAULT_FAMILY_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)

_CLEAR_UNITS = 1.0  # same "cleared its own null" cut concepts.py/concept_atlas.py use.

_VERB_PLURAL = {"increases": "increase", "decreases": "decrease", "moves": "move"}

# The title vocabulary (`_family_title`/`_directed_profile`/`_cleared_ranked`)
# now lives in `sae/plain_text.py`, shared with `concepts.py`'s `plain_name`
# (ROADMAP.md sec 37 F1 item 4) -- see that module's docstring for why this
# had to move out of here rather than `concepts.py` importing this module
# directly (an import cycle: `concept_atlas.py` already imports `concepts.py`,
# and this module imports `concept_atlas.py`). Re-exported under their
# original private names here so nothing else in this file (or its tests)
# has to change.


# ---------------------------------------------------------------------------
# Clustering: draft hierarchical families, then nearest-centroid assignment
# of every pooled feature.
# ---------------------------------------------------------------------------

def cluster_families(X: np.ndarray, min_members: int = DEFAULT_FAMILY_MIN_MEMBERS,
                     assign_min: float = DEFAULT_FAMILY_ASSIGN_MIN,
                     grid=DEFAULT_FAMILY_GRID) -> dict:
    """`-> {"labels" [n] (-1 idiosyncratic), "threshold", "grid", "selection_rule",
    "centroids" {family_id(str): [9 floats]}, "draft_labels" [n]}`.

    Two stages, both documented in the module docstring: (1) average-linkage
    cosine hierarchical clustering, cut at the grid threshold whose kept
    (>= `min_members`) clusters give the best silhouette among cuts
    admitting at least two such clusters (mirrors `concepts.py::_sweep_k`'s
    own admit-then-score discipline, generalized from a k-sweep to a
    threshold-sweep); (2) nearest-centroid re-assignment of EVERY row
    (valid or not, member of a kept cluster or not) against those clusters'
    own centroids, at `assign_min`. A zero-norm row can never be assigned
    (no direction to compare) and is `-1` directly, matching
    `cluster_atlas`'s own convention for the same case.
    """
    n = int(X.shape[0])
    labels = np.full(n, -1, dtype=int)
    empty = {"labels": labels, "threshold": None, "grid": [],
            "selection_rule": "no rows", "centroids": {}, "draft_labels": labels.copy()}
    if n == 0:
        return empty
    norms = np.linalg.norm(X, axis=1)
    valid_idx = np.where(norms > 0)[0]
    if valid_idx.size < int(min_members):
        return {**empty, "selection_rule": f"fewer than {min_members} nonzero-norm row(s)"}

    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import silhouette_score

    unit = X[valid_idx] / norms[valid_idx, None]
    Z = linkage(unit, method=_FAMILY_LINKAGE_METHOD, metric=_FAMILY_LINKAGE_METRIC)

    grid_records: list = []
    best = None  # (silhouette, threshold, kept-dict)
    for t in grid:
        raw = fcluster(Z, t=max(1.0 - float(t), 0.0), criterion="distance")
        groups: dict = {}
        for local_i, lab in enumerate(raw):
            groups.setdefault(int(lab), []).append(local_i)
        kept = {lab: idxs for lab, idxs in groups.items() if len(idxs) >= int(min_members)}
        n_families = len(kept)
        n_kept_members = sum(len(v) for v in kept.values())
        sil = None
        if n_families >= 2:
            lab_of: dict = {}
            for new_id, (_lab, idxs) in enumerate(sorted(kept.items())):
                for i in idxs:
                    lab_of[i] = new_id
            keep_local = sorted(lab_of)
            sil = float(silhouette_score(unit[keep_local],
                                         [lab_of[i] for i in keep_local], metric="cosine"))
        admissible = n_families >= 2
        grid_records.append({"min_cosine": float(t), "n_families": n_families,
                             "silhouette": sil, "admissible": admissible,
                             "n_kept_members": n_kept_members})
        if admissible and (best is None or sil > best[0]):
            best = (sil, float(t), kept)

    if best is None:
        return {"labels": labels, "threshold": None, "grid": grid_records,
               "selection_rule": (f"no threshold in {list(grid)} yields >=2 families of "
                                  f">= {min_members} members"),
               "centroids": {}, "draft_labels": labels.copy()}

    _sil, t_star, kept = best
    ordered = sorted(kept.items(), key=lambda kv: (-len(kv[1]), min(kv[1])))
    draft_labels = np.full(n, -1, dtype=int)
    for new_id, (_lab, idxs) in enumerate(ordered):
        for local_i in idxs:
            draft_labels[int(valid_idx[local_i])] = new_id

    n_fam = len(ordered)
    centroids = np.zeros((n_fam, X.shape[1]), dtype=np.float64)
    for fid in range(n_fam):
        centroids[fid] = X[draft_labels == fid].mean(axis=0)
    cen_norms = np.linalg.norm(centroids, axis=1)

    final_labels = np.full(n, -1, dtype=int)
    for i in range(n):
        if norms[i] <= 0.0:
            continue
        best_fid, best_sim = -1, -np.inf
        for fid in range(n_fam):
            if cen_norms[fid] <= 0.0:
                continue
            sim = float(np.dot(X[i], centroids[fid]) / (norms[i] * cen_norms[fid]))
            if sim > best_sim:
                best_fid, best_sim = fid, sim
        if best_fid >= 0 and best_sim >= float(assign_min):
            final_labels[i] = best_fid

    return {"labels": final_labels, "threshold": t_star, "grid": grid_records,
           "selection_rule": f"silhouette-best admissible cosine threshold, grid {list(grid)}",
           "centroids": {str(fid): centroids[fid].tolist() for fid in range(n_fam)},
           "draft_labels": draft_labels}


# ---------------------------------------------------------------------------
# Structure null (family count / silhouette / assigned fraction vs chance).
# ---------------------------------------------------------------------------

def _structure_null_family(X: np.ndarray, min_members: int, assign_min: float, grid,
                           rng: np.random.Generator, n_null: int) -> tuple:
    """The same channel-permutation null `concept_atlas.py::_structure_null`
    runs for the tight atlas, generalized to the family clusterer.
    `-> (n_families_null [n_null], frac_assigned_null [n_null], silhouette_null
    [<=n_null], the last only over draws that had an admissible threshold)`.
    """
    n = int(X.shape[0])
    n_families_null: list = []
    frac_assigned_null: list = []
    silhouette_null: list = []
    for _ in range(int(n_null)):
        Xp = np.empty_like(X)
        for c in range(X.shape[1]):
            Xp[:, c] = X[rng.permutation(n), c]
        result = cluster_families(Xp, min_members=min_members, assign_min=assign_min, grid=grid)
        lbl = result["labels"]
        n_families_null.append(int(lbl.max() + 1) if lbl.size and lbl.max() >= 0 else 0)
        frac_assigned_null.append(float(np.mean(lbl >= 0)) if n else 0.0)
        if result["threshold"] is not None:
            rec = next((g for g in result["grid"]
                       if abs(g["min_cosine"] - result["threshold"]) < 1e-12), None)
            if rec is not None and rec["silhouette"] is not None:
                silhouette_null.append(rec["silhouette"])
    return n_families_null, frac_assigned_null, silhouette_null


# ---------------------------------------------------------------------------
# Tight-concept -> family bookkeeping.
# ---------------------------------------------------------------------------

def _concept_family_majority(atlas_concepts: list, labels: np.ndarray) -> dict:
    """`{concept_id: {"family", "split", "member_family_counts",
    "n_distinct_families"}}`. Majority over each concept's OWN member row
    indices' assigned family (ties broken by smallest family id --
    deterministic, never by dict/array order). A concept whose members are
    entirely idiosyncratic gets `family: None`; this is recorded, never
    silently dropped."""
    out: dict = {}
    for c in atlas_concepts:
        cid = int(c["concept"])
        member_idx = c.get("members", [])
        counts = Counter(int(labels[i]) for i in member_idx)
        non_idio = {k: v for k, v in counts.items() if k != -1}
        if non_idio:
            best = sorted(non_idio.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        else:
            best = None
        out[cid] = {"family": best, "split": len(non_idio) > 1,
                   "member_family_counts": {str(k): v for k, v in counts.items()},
                   "n_distinct_families": len(non_idio)}
    return out


# ---------------------------------------------------------------------------
# Human-readable title + description.
# ---------------------------------------------------------------------------

def _family_fires_on(concept_ids: list, profiles_by_concept: dict | None):
    """Top archetype/generator labels this family's features fire on most,
    from `concept_profiles.json`'s own per-part hypergeometric enrichment
    (never re-derived here). `None` when no `concept_profiles.json` is
    available for this run (not the same as "measured, nothing survived");
    `[]` when it is available but no member part's top enrichment label
    exists."""
    if profiles_by_concept is None:
        return None
    votes: Counter = Counter()
    for cid in concept_ids:
        entry = profiles_by_concept.get(cid)
        if not entry:
            continue
        for part in entry.get("parts", []):
            ip = part.get("input_profile") or {}
            enrichment = ip.get("enrichment") or []
            if enrichment:
                votes[str(enrichment[0]["label"])] += 1
    if not votes:
        return []
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"label": lbl, "count": cnt} for lbl, cnt in ranked[:3]]


_MAX_DESC_CLAUSES = 3


def _render_effect_clauses(directed_vec: np.ndarray, cleared_subset: list) -> str:
    """Sentence 1 for one family at one candidate CLAUSE COUNT (0..3
    channels from its own `cleared` list, in order). Pure rendering, no
    uniqueness logic -- `_compose_family_sentence1` below is what picks the
    count."""
    if not cleared_subset:
        return "No channel in this family clears its own null on average."
    clauses = []
    for ci in cleared_subset:
        ch = CHANNELS[ci]
        val = float(directed_vec[ci])
        if ch in CHANNEL_VERB:
            verb = CHANNEL_VERB[ch][0]  # undirected: both entries identical ("moves")
        else:
            verb = _DEFAULT_VERB[0] if val > 0 else _DEFAULT_VERB[1]
        verb = _VERB_PLURAL.get(verb, verb)
        gloss = CHANNEL_GLOSS.get(ch, ch)
        clauses.append(f"{verb} {gloss}")
    if len(clauses) == 1:
        return f"These features {clauses[0]}."
    return "These features " + ", ".join(clauses[:-1]) + f", and {clauses[-1]}."


def _compose_family_sentence1(all_directed: list, all_cleared: list,
                              max_clauses: int = _MAX_DESC_CLAUSES) -> list:
    """One sentence-1 string per family, batched across the WHOLE run so
    each can be compared against every other -- the same SHORTEST UNIQUE
    PREFIX mechanism `concepts.py::_compose_batch` already uses for tight-
    concept names (mechanism 3 there), applied here to a plain-language
    sentence instead of a machine name. A fixed top-2 clause count made two
    to three families that both lead with `horizon_shape_near`/`_far` (the
    two channels least likely to sit below their own null, since they are
    mean-ABSOLUTE deviations rather than signed effects -- almost every
    causal feature moves the forecast SOME amount near and far) render
    byte-identical sentence-1 text on the real run (measured: 3 families
    read "moves the far horizon of the forecast and moves the near horizon
    of the forecast" verbatim before this fix). Extending only as far as
    needed to become unique, capped at `max_clauses`, fixes that without
    padding every family's sentence to the cap."""
    n = len(all_cleared)
    out = [None] * n
    for i in range(n):
        max_k = min(len(all_cleared[i]), max_clauses)
        chosen = _render_effect_clauses(all_directed[i], all_cleared[i][:max_k])
        for k in range(1, max_k + 1):
            candidate = _render_effect_clauses(all_directed[i], all_cleared[i][:k])
            collides = any(
                _render_effect_clauses(all_directed[j], all_cleared[j][:min(k, len(all_cleared[j]))])
                == candidate
                for j in range(n) if j != i)
            if not collides:
                chosen = candidate
                break
        out[i] = chosen
    return out


def _family_second_sentence(models: dict, fires_on) -> str:
    model_list = ", ".join(sorted(models))
    n_models = len(models)
    if fires_on:
        top = fires_on[0]["label"]
        return (f"Its features come from {n_models} model(s) ({model_list}) and fire most "
                f"often on {top}-type series.")
    return f"Its features come from {n_models} model(s) ({model_list})."


# ---------------------------------------------------------------------------
# Stability (cheap aggregate over `concept_stability.json`; see module docstring).
# ---------------------------------------------------------------------------

def _family_stability(concept_ids: list, stability_doc: dict | None) -> dict:
    if not stability_doc or not stability_doc.get("measured"):
        reason = ("concept_stability.json absent for this run" if not stability_doc else
                  stability_doc.get("reason", "concepts.n_sae_seeds < 2 -- not measured"))
        return {"status": "skipped", "reason": reason}
    by_concept = {int(c["concept"]): c["stability"] for c in stability_doc.get("concepts", [])}
    n_scored, n_stable = 0, 0
    for cid in concept_ids:
        st = (by_concept.get(cid) or {}).get("stable")
        if st is not None:
            n_scored += 1
            n_stable += int(bool(st))
    return {"status": "ran (concept-level aggregate)",
           "note": ("aggregated from concept_stability.json's per-concept reciprocal-transfer "
                    "stability; a replicate SAE's own dictionary is never re-ablated into the "
                    "family space (that needs a full ablation battery per replicate, out of "
                    "scope here) -- this is 'are this family's member concepts individually "
                    "seed-stable', not 'does a replicate feature land in this family'"),
           "n_concepts_scored": n_scored, "n_concepts_stable": n_stable,
           "frac_concepts_stable": (n_stable / n_scored) if n_scored else None}


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def run_concept_families(run_dir, atlas: dict, cfg, profiles: dict | None = None,
                         stability: dict | None = None) -> dict:
    """Pool, cluster into families, null-test, name/describe and write
    `sae/concept_families.json`. `atlas` is an already-loaded
    `concept_atlas.json` dict (mirrors `stability.py::concept_stability`'s
    own signature); `profiles`/`stability` are already-loaded
    `concept_profiles.json`/`concept_stability.json` dicts, both optional
    and used only for the "what does it fire on" clause and the seed-
    stability summary respectively (module docstring)."""
    run_dir = Path(run_dir)
    concepts_cfg = cfg.concepts
    causal_only = bool(getattr(getattr(cfg, "sae", None), "concept_causal_only", True))
    min_members = int(getattr(concepts_cfg, "atlas_family_min_members", DEFAULT_FAMILY_MIN_MEMBERS))
    assign_min = float(getattr(concepts_cfg, "atlas_family_assign_min", DEFAULT_FAMILY_ASSIGN_MIN))
    n_null = int(getattr(concepts_cfg, "atlas_n_null", 200))
    seed = int(getattr(getattr(cfg, "run", None), "seed", 0) or 0)
    grid = DEFAULT_FAMILY_GRID

    out_path = run_dir / "sae" / "concept_families.json"
    params = {"min_members": min_members, "assign_min": assign_min, "grid": list(grid),
             "linkage": _FAMILY_LINKAGE_METHOD, "n_null": n_null}

    X, rows = pooled_features(run_dir, causal_only=causal_only)
    n_features = int(X.shape[0])
    atlas_concepts = atlas.get("concepts") or []

    if n_features == 0 or not atlas_concepts:
        out = {"schema_version": 1, "measured": False,
              "reason": ("no pooled causal features across any target" if n_features == 0
                        else "the atlas has no tight concepts to organize into families"),
              "params": params, "n_features": n_features, "n_assigned": 0, "frac_assigned": 0.0,
              "threshold": None, "grid": [], "selection_rule": None,
              "families": [], "rows": [], "concept_family": {},
              "n_concepts_split": 0, "null": {"structure": None, "cross_model": None}}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(out_path, out)
        log.warning("concept families: %s; wrote empty %s", out["reason"], out_path)
        return out

    atlas_rows = atlas.get("rows") or []
    if len(atlas_rows) != len(rows):
        raise RuntimeError(
            f"concept families: re-pooled {len(rows)} row(s) from {run_dir} but "
            f"{out_path.parent / 'concept_atlas.json'} recorded {len(atlas_rows)} -- the atlas "
            f"artifact does not match this run's current ablation files (stale atlas?); refusing "
            f"to build families against a mismatched row set")

    result = cluster_families(X, min_members=min_members, assign_min=assign_min, grid=grid)
    labels = result["labels"]
    models_arr = np.array([r["model"] for r in rows])

    real_n_families = int(labels.max() + 1) if labels.size and labels.max() >= 0 else 0
    real_frac_assigned = float(np.mean(labels >= 0))
    real_sil = None
    if result["threshold"] is not None:
        rec = next((g for g in result["grid"]
                   if abs(g["min_cosine"] - result["threshold"]) < 1e-12), None)
        real_sil = rec["silhouette"] if rec else None

    rng_struct = np.random.default_rng(seed + 1000)
    n_fam_null, frac_null, sil_null = _structure_null_family(
        X, min_members, assign_min, grid, rng_struct, n_null)
    structure_null = {
        "n_null": n_null,
        "n_families_real": real_n_families,
        "n_families_null_mean": float(np.mean(n_fam_null)) if n_fam_null else float("nan"),
        "n_families_null_p95": float(np.quantile(n_fam_null, 0.95)) if n_fam_null else float("nan"),
        "p_n_families": _right_tail_p(real_n_families, n_fam_null),
        "frac_assigned_real": real_frac_assigned,
        "frac_assigned_null_mean": float(np.mean(frac_null)) if frac_null else float("nan"),
        "frac_assigned_null_p95": float(np.quantile(frac_null, 0.95)) if frac_null else float("nan"),
        "p_frac_assigned": _right_tail_p(real_frac_assigned, frac_null),
        "silhouette_real": real_sil,
        "n_null_with_admissible_threshold": len(sil_null),
        "silhouette_null_mean": float(np.mean(sil_null)) if sil_null else None,
        "p_silhouette": (_right_tail_p(real_sil, sil_null)
                        if (real_sil is not None and sil_null) else None),
    }

    rng_cross = np.random.default_rng(seed + 1001)
    concept_ids_sorted = sorted({int(c) for c in labels.tolist() if c >= 0})
    (real_multi, null_multi, real_n_models_by_c,
     null_n_models_by_c, fam_ids) = _cross_model_null(labels, models_arr, rng_cross, n_null)
    rng_purity = np.random.default_rng(seed + 1002)
    purity_null = [_mean_purity(labels, models_arr[rng_purity.permutation(models_arr.size)], fam_ids)
                  for _ in range(int(n_null))]
    real_purity = _mean_purity(labels, models_arr, fam_ids)
    finite_null = [v for v in purity_null if np.isfinite(v)]
    cross_model_null = {
        "n_null": n_null, "n_multi_model_real": int(real_multi),
        "n_multi_model_null_mean": float(np.mean(null_multi)) if null_multi else float("nan"),
        "p_n_multi_model": _right_tail_p(real_multi, null_multi),
        "p_n_multi_model_below": _left_tail_p(real_multi, null_multi),
        "mean_purity_real": real_purity,
        "mean_purity_null_mean": float(np.mean(finite_null)) if finite_null else float("nan"),
        "p_purity_above": _right_tail_p(real_purity, finite_null),
        "p_purity_below": _left_tail_p(real_purity, finite_null),
    }
    cross_model_null["verdict"] = cross_model_verdict(
        cross_model_null["p_purity_below"], cross_model_null["p_purity_above"])
    per_family_cross_p = {c: _right_tail_p(real_n_models_by_c[c], null_n_models_by_c[c])
                          for c in fam_ids}

    concept_family_map = _concept_family_majority(atlas_concepts, labels)
    n_concepts_split = sum(1 for rec in concept_family_map.values() if rec["split"])

    profiles_by_concept = None
    if profiles and profiles.get("concepts"):
        profiles_by_concept = {int(c["concept"]): c for c in profiles["concepts"]}

    # Pass 1: every family's own geometry/membership -- gathered fully before
    # any text is composed, since both the title's collision check and
    # sentence 1's shortest-unique-prefix (`_compose_family_sentence1`) need
    # every OTHER family's data to decide any one family's rendering.
    prepared = []
    for fid in concept_ids_sorted:
        idx = np.where(labels == fid)[0]
        members_X = X[idx]
        mean_profile_vec = members_X.mean(axis=0)
        directed_vec = _directed_profile(mean_profile_vec)
        cleared = _cleared_ranked(directed_vec)
        models: dict = {}
        for i in idx:
            m = rows[i]["model"]
            models[m] = models.get(m, 0) + 1
        concept_ids = sorted(cid for cid, rec in concept_family_map.items() if rec["family"] == fid)
        fires_on = _family_fires_on(concept_ids, profiles_by_concept)
        prepared.append({"fid": fid, "idx": idx, "mean_profile_vec": mean_profile_vec,
                        "directed_vec": directed_vec, "cleared": cleared, "models": models,
                        "concept_ids": concept_ids, "fires_on": fires_on})

    sentence1s = _compose_family_sentence1([p["directed_vec"] for p in prepared],
                                           [p["cleared"] for p in prepared])

    used_titles: set = set()
    families_out = []
    for p, s1 in zip(prepared, sentence1s):
        title = _family_title(p["directed_vec"], p["cleared"], used_titles)
        s2 = _family_second_sentence(p["models"], p["fires_on"])
        description = f"{s1} {s2}"
        families_out.append({
            "family": int(p["fid"]), "title": title, "description": description,
            "n_members": int(p["idx"].size), "models": p["models"], "n_models": len(p["models"]),
            "mean_profile_ablation_units": {ch: float(v) for ch, v in zip(CHANNELS, p["mean_profile_vec"])},
            "directed_profile_null_units": {ch: float(v) for ch, v in zip(CHANNELS, p["directed_vec"])},
            "cleared_channels": [CHANNELS[i] for i in p["cleared"]],
            "concept_ids": p["concept_ids"],
            "members": [int(i) for i in p["idx"]],
            "fires_on": p["fires_on"],
            "cross_model_p": per_family_cross_p.get(p["fid"]),
            "stability": _family_stability(p["concept_ids"], stability),
        })

    rows_out = [{"model": r["model"], "layer": r["layer"], "feature": int(r["feature"]),
                "family": (int(labels[i]) if labels[i] >= 0 else None)}
               for i, r in enumerate(rows)]

    out = {
        "schema_version": 1, "measured": True, "space": "ablation", "params": params,
        "n_features": n_features, "n_assigned": int(np.sum(labels >= 0)),
        "frac_assigned": real_frac_assigned,
        "threshold": result["threshold"], "grid": result["grid"],
        "selection_rule": result["selection_rule"],
        "families": families_out, "rows": rows_out,
        "concept_family": {str(cid): rec for cid, rec in concept_family_map.items()},
        "n_concepts_split": n_concepts_split,
        "null": {"structure": structure_null, "cross_model": cross_model_null},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    log.info("concept families: %d feature(s) -> %d famil(y/ies) at cosine %s "
            "(%d assigned, %.1f%%, %d concept(s) split across families)",
            n_features, len(families_out), result["threshold"],
            out["n_assigned"], 100.0 * real_frac_assigned, n_concepts_split)
    return out


# ---------------------------------------------------------------------------
# Report hook (ROADMAP.md spec item 6) -- a pure read, no clustering.
# ---------------------------------------------------------------------------

def load_families(run_dir) -> list:
    """`-> families` list from `sae/concept_families.json`, or `[]` if the
    artifact is absent/not measured. A pure I/O read for the report to call;
    it never re-clusters (`run_concept_families` is the only writer)."""
    path = Path(run_dir) / "sae" / "concept_families.json"
    if not path.exists():
        return []
    doc = load_json(path)
    if not doc.get("measured"):
        return []
    return doc.get("families", [])
