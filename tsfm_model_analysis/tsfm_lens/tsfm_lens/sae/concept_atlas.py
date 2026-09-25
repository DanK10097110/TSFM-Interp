"""ROADMAP.md sec 37.15 q6 decision (option b) -- a cross-model CONCEPT ATLAS.

`sae/concepts.py::run_concepts` clusters each SAE target's own causal
candidates *separately*: a concept is a group of one model's own features.
On the one real panel measured so far (`runs/full_report_run_4model`) that
leaves 11 of 13 targets `non_modular` -- too few causal candidates, or no
admissible k -- and even where a target IS modular, nothing in that
per-target partition can ever say "TimesFM's concept 2 and Chronos-2's
concept 0 are the SAME thing", because each target is clustered against its
own peers alone. This module answers a different question: pool EVERY
model's causal features into one space and cluster across the whole run.

Pooling is meaningful here in a way it would not be for, say, raw activation
vectors: the unit is `sae/concepts.py::ablation_vector`'s 9-channel profile
(signed_effect / null_p95, `sae/response.py::CHANNELS`), and every one of
those nine channels is a property of the FORECAST (trend, seasonality,
level, dispersion, ...), not of any model's internal representation. A
"raises trend, lowers dispersion" causal fingerprint means the same thing
whether it came from a TimesFM feature or a Chronos-Bolt one, because the
forecast it moved is the same kind of object either way -- unlike hidden
states, which live in different, differently-sized, differently-trained
spaces per model and are never directly comparable (`CLAUDE.md` sec 6.3's
whole alignment apparatus exists because of exactly that asymmetry).

EVIDENCE CLASS, stated once here because every consumer of this artifact
inherits it: a shared atlas concept means a shared *causal effect profile on
the forecast* -- these features move the same channels the same way when
removed. It does NOT mean the features fire on the same series, respond to
the same input property, or were learned the same way. That is a different,
narrower claim `sae/transfer.py` already tests (shared top-firing series,
checked reciprocally against a matched-stratum null) and this module never
re-derives it.

ADDITIVE throughout: `sae/concepts.py::run_concepts`, `concepts.json`,
`transfer.json` and their existing report blocks are untouched. This module
reads the SAME raw `sae/<model>/<layer>_ablation.json` files `run_concepts`
reads (via `build_concept_matrix`, never re-derived) and writes its own
artifact, `sae/concept_atlas.json`.

Four public entry points:

  `pooled_features(run_dir, causal_only=True)` -- pure I/O + reduction. Every
  non-withheld, non-skipped target's causal candidates, pooled into one
  `(n, 9)` matrix via `concepts.py::build_concept_matrix` (no independent
  re-derivation of the 9-channel vector), with a parallel `rows` list naming
  each row's `model`/`layer`/`feature`. Targets are visited in
  `sorted(glob(...))` order and each target's own candidates in the order
  its ablation file already lists them, so two calls against the same run
  directory produce byte-identical output.

  `cluster_atlas(X, min_cosine, min_members=3)` -- L2-normalize every row,
  then `scipy.cluster.hierarchy.linkage(method="complete", metric="cosine")`
  + `fcluster(t=1-min_cosine, criterion="distance")`. COMPLETE linkage is
  the load-bearing choice, not an arbitrary one: it is the only common
  linkage rule under which cutting the dendrogram at height `t` guarantees
  every pair of points placed in the same flat cluster has cophenetic
  distance <= `t` -- i.e. **every pair inside a concept has cosine >=
  min_cosine**, which is exactly this build's "similar enough" requirement
  (see the user decision at the top of the spec this module implements).
  Average or single linkage do not carry that guarantee -- a chain of
  pairwise-similar-but-transitively-dissimilar points can end up in one
  cluster under either. Clusters smaller than `min_members` are labelled
  `-1` (never silently dropped: `n_assigned` in the artifact counts what
  survives). A zero-norm row (every channel unscored/zero) cannot be
  L2-normalized and is labelled `-1` directly. Renumbered by size
  descending, ties broken by each cluster's own smallest original row
  index -- deterministic, and independent of dict/set iteration order.

  `run_concept_atlas(run_dir, cfg) -> dict` -- the driver. Builds the pooled
  matrix, clusters it, runs both null procedures below, names every concept
  (reusing `concepts.py::_compose_batch` directly -- the same batch-call
  shape `assign_concept_names` already uses, with every atlas concept as
  every other atlas concept's peer, rather than looping the thin
  per-concept `compose_name` wrapper and paying its own documented O(n)
  repeated O(n^2) cost), fits one 2-D PCA on the L2-normalized (clustered)
  geometry, optionally adds a seeded UMAP projection when `umap-learn`
  imports (best-effort; the report always plots PCA regardless), and writes
  `sae/concept_atlas.json`.

Two null procedures, both seeded from `cfg.run.seed` (never Python's
builtin `hash()`, per `CLAUDE.md` sec 11.2/`sae/transfer.py`'s own
precedent) so a run is reproducible run to run:

  STRUCTURE null: permute each of the 9 channel columns independently
  across rows. This keeps every channel's own marginal distribution intact
  (so a channel that is rarely non-zero stays rarely non-zero) while
  destroying the JOINT profile -- a row's trend value no longer has
  anything to do with its own dispersion value. Re-running `cluster_atlas`
  on the permuted matrix answers "how many concepts, and what fraction of
  features assigned, would this same clustering procedure find if the nine
  channels carried no joint structure at all". Real n_concepts / real
  frac_assigned are compared against this null's own mean/p95, with a
  one-sided p (`_right_tail_p`, floored at `1/(n_null+1)`) for "the real
  count is at least this large by chance".

  CROSS-MODEL null: keep the REAL cluster labels fixed (this null asks
  nothing about whether clustering itself is real -- the structure null
  already covers that) and permute the `model` label attached to each row,
  preserving the run's own per-model row counts exactly (a permutation of
  the model array, not an independent resample). For each real concept,
  recount how many distinct models its (fixed) member set would have under
  the permuted labels; the aggregate is how many concepts would span >=2
  models by chance alone, given the ACTUAL clusters this run found and the
  ACTUAL mix of models in the pooled matrix. Both directions of the
  resulting p-value are informative and neither is privileged here: real
  clearly *above* the null says the models are drawn together more than
  chance would predict (shared structure); real clearly *below* it says the
  opposite -- the models are more segregated into their own concepts than a
  random relabelling would be, which is itself a finding about how
  separable the four architectures' causal profiles are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils import load_json, log, save_json
from .concepts import CHANNELS, _compose_batch, build_concept_matrix

__all__ = ["pooled_features", "cluster_atlas", "run_concept_atlas"]

_LINKAGE_METHOD = "complete"
_LINKAGE_METRIC = "cosine"


# ---------------------------------------------------------------------------
# Pooling.
# ---------------------------------------------------------------------------

def pooled_features(run_dir, causal_only: bool = True) -> tuple:
    """`-> (X [n, 9], rows)`, `rows[i] = {"model", "layer", "feature"}`.

    Reads every `sae/<model>/<layer>_ablation.json` under `run_dir`,
    skipping a target that is `withheld` (its `reach` probe found nothing
    reachable, sec 27) or `skipped` (no trained SAE checkpoint) -- neither
    carries a `candidates` list, so `art.get("candidates", [])` would
    silently contribute zero rows for them anyway; the explicit skip is
    for the reader, not the arithmetic. Row vectors and their `feature` ids
    come from `concepts.py::build_concept_matrix` UNCHANGED -- this
    function does no independent re-derivation of the ablation-vector
    arithmetic, per this module's own docstring.
    """
    run_dir = Path(run_dir)
    X_rows: list = []
    meta_rows: list = []
    for ablation_file in sorted(run_dir.glob("sae/*/*_ablation.json")):
        art = load_json(ablation_file)
        if art.get("withheld") or art.get("skipped"):
            continue
        model = art.get("model", ablation_file.parent.name)
        layer = art.get("layer", ablation_file.name[: -len("_ablation.json")])
        candidates = art.get("candidates", [])
        X, feature_ids, _diag = build_concept_matrix(candidates, causal_only=causal_only)
        for vec, fid in zip(X, feature_ids):
            X_rows.append(vec)
            meta_rows.append({"model": str(model), "layer": str(layer), "feature": int(fid)})
    if not X_rows:
        return np.zeros((0, len(CHANNELS)), dtype=np.float64), []
    return np.asarray(X_rows, dtype=np.float64), meta_rows


# ---------------------------------------------------------------------------
# Clustering.
# ---------------------------------------------------------------------------

def cluster_atlas(X: np.ndarray, min_cosine: float, min_members: int = 3) -> np.ndarray:
    """`-> labels [n]`, `-1` for unassigned. See the module docstring for
    why complete linkage is the load-bearing choice: it is what makes
    "every pair inside a concept has cosine >= min_cosine" true by
    construction of the dendrogram cut, not merely typical of it.
    """
    n = int(X.shape[0])
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels
    norms = np.linalg.norm(X, axis=1)
    valid_idx = np.where(norms > 0)[0]
    if valid_idx.size < 2:
        return labels

    from scipy.cluster.hierarchy import fcluster, linkage

    unit = X[valid_idx] / norms[valid_idx, None]
    Z = linkage(unit, method=_LINKAGE_METHOD, metric=_LINKAGE_METRIC)
    t = max(1.0 - float(min_cosine), 0.0)
    raw = fcluster(Z, t=t, criterion="distance")

    groups: dict = {}
    for local_i, lab in enumerate(raw):
        groups.setdefault(int(lab), []).append(local_i)
    kept = [(lab, idxs) for lab, idxs in groups.items() if len(idxs) >= int(min_members)]
    # Renumber by size descending; tie-break on each cluster's own smallest
    # ORIGINAL row index, never on dict-iteration or label order
    # (`CLAUDE.md` sec 11.2/11.55: array position must never decide an
    # outcome that is supposed to be reproducible).
    kept.sort(key=lambda kv: (-len(kv[1]), min(int(valid_idx[i]) for i in kv[1])))
    for new_id, (_lab, idxs) in enumerate(kept):
        for local_i in idxs:
            labels[int(valid_idx[local_i])] = new_id
    return labels


# ---------------------------------------------------------------------------
# Nulls.
# ---------------------------------------------------------------------------

def _right_tail_p(real_value: float, null_values) -> float:
    """One-sided permutation p: fraction of null draws >= `real_value`,
    floored at `1/(n_null+1)` (the standard `(1 + count) / (n + 1)` form,
    which cannot return exactly 0 no matter how extreme the real value is
    -- `CLAUDE.md` sec 6.6's "bootstrap p-values are approximate and floored"
    discipline, applied to a permutation test here)."""
    vals = np.asarray(list(null_values), dtype=np.float64)
    n = int(vals.size)
    if n == 0:
        return float("nan")
    count = int(np.sum(vals >= float(real_value)))
    return (1 + count) / (n + 1)


def _left_tail_p(real_value: float, null_values) -> float:
    """The mirror of `_right_tail_p`: fraction of null draws <= `real_value`,
    with the same `(1 + count) / (n + 1)` floor. The cross-model null is
    informative in BOTH directions, and a right-tail p of 1.0 alone reads as
    "not significant" when it can be the opposite extreme -- models more
    segregated than chance -- which only this tail detects."""
    vals = np.asarray(list(null_values), dtype=np.float64)
    n = int(vals.size)
    if n == 0:
        return float("nan")
    return (1 + int(np.sum(vals <= float(real_value)))) / (n + 1)


def _mean_purity(labels: np.ndarray, models: np.ndarray, concept_ids: list) -> float:
    """Mean over concepts of the largest single model's share of that
    concept's members: 1.0 = every concept is one model's, lower = mixed.
    A finer segregation statistic than the count of concepts touching >=2
    models, which saturates once concepts hold a few members each."""
    if not concept_ids:
        return float("nan")
    shares = []
    for c in concept_ids:
        _, counts = np.unique(models[labels == c], return_counts=True)
        shares.append(counts.max() / counts.sum())
    return float(np.mean(shares))


def cross_model_verdict(p_above: float, p_below: float, alpha: float = 0.05) -> str:
    """`drawn together` / `segregated by model` / `consistent with chance`,
    from the two tails of a mixing statistic oriented so that larger means
    more mixed."""
    if np.isfinite(p_above) and p_above < alpha:
        return "drawn together"
    if np.isfinite(p_below) and p_below < alpha:
        return "segregated by model"
    return "consistent with chance"


def _structure_null(X: np.ndarray, min_cosine: float, min_members: int,
                    rng: np.random.Generator, n_null: int) -> tuple:
    """Permute each channel column independently (marginals kept, joint
    profile destroyed), recluster, and record `n_concepts`/`frac_assigned`
    per draw. `-> (n_concepts_null [n_null], frac_assigned_null [n_null])`."""
    n = int(X.shape[0])
    n_concepts_null: list = []
    frac_assigned_null: list = []
    for _ in range(int(n_null)):
        Xp = np.empty_like(X)
        for c in range(X.shape[1]):
            Xp[:, c] = X[rng.permutation(n), c]
        labels = cluster_atlas(Xp, min_cosine=min_cosine, min_members=min_members)
        n_concepts_null.append(int(labels.max() + 1) if labels.size and labels.max() >= 0 else 0)
        frac_assigned_null.append(float(np.mean(labels >= 0)) if n else 0.0)
    return n_concepts_null, frac_assigned_null


def _cross_model_null(labels: np.ndarray, models: np.ndarray,
                      rng: np.random.Generator, n_null: int) -> tuple:
    """Keep the REAL cluster labels fixed; permute the `model` array
    (a permutation, so every draw reproduces the run's own exact per-model
    counts). Returns `(real_n_multi_model, null_n_multi_model [n_null],
    real_n_models_by_concept {cid: int}, null_n_models_by_concept
    {cid: [n_null floats]}, concept_ids [sorted])`."""
    models = np.asarray(models)
    n = int(models.size)
    concept_ids = sorted({int(c) for c in labels.tolist() if c >= 0})
    member_idx = {c: np.where(labels == c)[0] for c in concept_ids}
    real_n_models = {c: len(set(models[member_idx[c]].tolist())) for c in concept_ids}
    real_multi = sum(1 for c in concept_ids if real_n_models[c] >= 2)

    null_multi_counts: list = []
    null_n_models_by_concept = {c: [] for c in concept_ids}
    for _ in range(int(n_null)):
        pm = models[rng.permutation(n)]
        multi = 0
        for c in concept_ids:
            k = len(set(pm[member_idx[c]].tolist()))
            null_n_models_by_concept[c].append(k)
            if k >= 2:
                multi += 1
        null_multi_counts.append(multi)
    return real_multi, null_multi_counts, real_n_models, null_n_models_by_concept, concept_ids


# ---------------------------------------------------------------------------
# Concept records + naming + projection.
# ---------------------------------------------------------------------------

def _concept_records(X: np.ndarray, rows: list, labels: np.ndarray,
                     per_concept_p: dict) -> list:
    channel_cols = list(CHANNELS)
    concept_ids = sorted({int(c) for c in labels.tolist() if c >= 0})
    out: list = []
    for cid in concept_ids:
        idx = np.where(labels == cid)[0]
        members_X = X[idx]
        norms = np.linalg.norm(members_X, axis=1)
        mean_profile_vec = members_X.mean(axis=0)

        models: dict = {}
        layers: dict = {}
        for i in idx:
            m, l = rows[i]["model"], rows[i]["layer"]
            models[m] = models.get(m, 0) + 1
            key = f"{m}/{l}"
            layers[key] = layers.get(key, 0) + 1

        if idx.size >= 2:
            safe = np.where(norms == 0, 1.0, norms)
            unit = members_X / safe[:, None]
            sims = unit @ unit.T
            iu = np.triu_indices(idx.size, k=1)
            pair = sims[iu]
            min_pair_cosine = float(pair.min())
            mean_pair_cosine = float(pair.mean())
        else:
            min_pair_cosine = float("nan")
            mean_pair_cosine = float("nan")

        out.append({
            "concept": int(cid), "name": None,
            "n_members": int(idx.size),
            "models": models, "n_models": len(models),
            "layers": layers,
            "mean_profile": {ch: float(v) for ch, v in zip(channel_cols, mean_profile_vec)},
            "mean_norm": float(norms.mean()),
            "min_pair_cosine": min_pair_cosine, "mean_pair_cosine": mean_pair_cosine,
            "members": [int(i) for i in idx],
            "cross_model_p": per_concept_p.get(cid),
        })
    return out


def _candidate_lookup(run_dir) -> dict:
    """`(model, layer, feature) -> candidate dict`, read from the SAME raw
    `sae/<model>/<layer>_ablation.json` files `pooled_features` reads
    (ROADMAP.md sec 37.7 P4) -- one extra pass over already-small JSON files,
    not a re-derivation of anything `pooled_features`/`build_concept_matrix`
    compute. Used only to look up `level_share`/`shape_channels` for features
    that are already atlas members (`_concept_records`' own `rows`), so this
    does not need to reproduce `build_concept_matrix`'s causal-only filter --
    every scorable candidate is indexed, and a caller looks up only the
    members it already knows about. `withheld`/`skipped` targets have no
    `candidates` list and contribute nothing, matching `pooled_features`.
    """
    run_dir = Path(run_dir)
    lookup: dict = {}
    for ablation_file in sorted(run_dir.glob("sae/*/*_ablation.json")):
        art = load_json(ablation_file)
        if art.get("withheld") or art.get("skipped"):
            continue
        model = str(art.get("model", ablation_file.parent.name))
        layer = str(art.get("layer", ablation_file.name[: -len("_ablation.json")]))
        for c in art.get("candidates", []):
            if not c.get("scorable"):
                continue
            lookup[(model, layer, int(c["feature"]))] = c
    return lookup


def _tag_causal_effect(concepts: list, rows: list, lookup: dict, level_share_threshold: float) -> None:
    """Mutates `concepts` in place, adding `causal_tag`, `level_share_median`
    and `n_members_with_level_share` (ROADMAP.md sec 37.7 P4, additive --
    CLAUDE.md invariant 13). Per item 4's rule, checked in this order (a
    concept with a real shape effect is `shape-causal` regardless of how much
    of its members' effect is ALSO level, since some real, non-level movement
    was still measured):

      `shape-causal`        -- at least one member has `n_shape_channels_
                               clearing > 0` (its level-removed battery
                               cleared some channel on its own row-matched
                               null, sec 37.7 item 1).
      `level carrier`       -- no member clears a level-removed channel, and
                               the members' MEDIAN `level_share` (over
                               members where it is defined; sec 37.3 P0's
                               ratio is undefined, not zero, when a
                               candidate's own top-firing series show no
                               movement at all) is >= `level_share_threshold`.
      `no measured effect`  -- neither of the above: no shape channel
                               cleared and the level evidence is either
                               absent or below the threshold.

    A member whose `(model, layer, feature)` is not in `lookup` (an older
    ablation artifact predating this field) is skipped for BOTH the level-
    share median and the shape-clearing check, rather than raising -- the
    concept is then tagged from whatever members do carry the new fields,
    and `n_members_with_level_share` records how many that was so a reader
    can see when the tag rests on a partial population.
    """
    for rec in concepts:
        level_shares: list = []
        any_shape_clears = False
        for i in rec["members"]:
            r = rows[i]
            cand = lookup.get((str(r["model"]), str(r["layer"]), int(r["feature"])))
            if cand is None:
                continue
            ls = cand.get("level_share")
            if ls is not None:
                level_shares.append(float(ls))
            if cand.get("n_shape_channels_clearing", 0) > 0:
                any_shape_clears = True
        median_ls = float(np.median(level_shares)) if level_shares else None
        if any_shape_clears:
            tag = "shape-causal"
        elif median_ls is not None and median_ls >= level_share_threshold:
            tag = "level carrier"
        else:
            tag = "no measured effect"
        rec["causal_tag"] = tag
        rec["level_share_median"] = median_ls
        rec["n_members_with_level_share"] = len(level_shares)


def _name_concepts(concepts: list) -> None:
    """Mutates `concepts` in place, setting `name`. Reuses
    `concepts.py::_compose_batch` DIRECTLY (not the thin per-concept
    `compose_name` wrapper -- see module docstring): every atlas concept's
    `mean_profile` becomes one row of `profiles`, a channel counts as
    "cleared" for naming purposes at `|mean| >= 1.0` null units -- the SAME
    cut `concepts.py::concept_table` already uses to decide which channels
    populate a (per-target) concept's own `profile` list, so an atlas
    concept's notion of "cleared" is not a second, independently-invented
    one. Peer keys are each concept's own (already unique, already
    deterministic) `concept` id.

    `causal_tag` (ROADMAP.md sec 37.7 P4, set by `_tag_causal_effect`, which
    must run first) is passed through to `_compose_batch` so a `level
    carrier` concept is named "shifts the level" rather than a raw,
    possibly level-confounded channel word -- see that function's own
    docstring. Concepts this run's `_tag_causal_effect` never saw (`causal_
    tag` absent, e.g. a caller that builds `concepts` by hand) are treated
    as untagged, reproducing the pre-P4 naming exactly.
    """
    if not concepts:
        return
    channel_cols = list(CHANNELS)
    profiles = np.array([[c["mean_profile"][ch] for ch in channel_cols] for c in concepts])
    cleared = np.abs(profiles) >= 1.0
    peers = [c["concept"] for c in concepts]
    tags = [c.get("causal_tag") for c in concepts]
    batch = _compose_batch(profiles, cleared, peers, tags=tags)
    for rec, info in zip(concepts, batch):
        rec["name"] = info["name"]


def _pca_projection(X: np.ndarray, seed: int = 0) -> tuple:
    """2-D linear PCA on the L2-normalized rows -- the SAME geometry
    `cluster_atlas` clustered, unlike `report/sae_concept_map.py`'s
    per-target map (which standardizes first): here the whole point is that
    the projection agrees with what decided cluster membership. `-> (pc1
    [n], pc2 [n], explained_variance_ratio [<=2])`."""
    n = int(X.shape[0])
    if n < 2:
        z = np.zeros(n, dtype=np.float64)
        return z, z.copy(), [0.0, 0.0]

    from sklearn.decomposition import PCA

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit = X / norms
    n_comp = max(1, min(2, n, X.shape[1]))
    pca = PCA(n_components=n_comp, random_state=int(seed))
    proj = pca.fit_transform(unit)
    pc1 = proj[:, 0]
    pc2 = proj[:, 1] if n_comp >= 2 else np.zeros(n, dtype=np.float64)
    ev = [float(v) for v in pca.explained_variance_ratio_]
    if n_comp < 2:
        ev = ev + [0.0]
    return pc1, pc2, ev


def _umap_projection(X: np.ndarray, seed: int = 0):
    """Best-effort seeded UMAP on the same L2-normalized rows, or `None`
    when `umap-learn` is not importable or fitting fails (too few points
    for its own neighbour graph, etc.) -- optional throughout, never blocks
    the atlas (module docstring; the report figure always plots PCA)."""
    n = int(X.shape[0])
    if n < 4:
        return None
    try:
        import umap
    except ImportError:
        return None
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit = X / norms
    n_neighbors = max(2, min(15, n - 1))
    try:
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=int(seed))
        proj = reducer.fit_transform(unit)
    except Exception as exc:  # pragma: no cover -- environment-dependent
        log.warning("concept atlas: UMAP projection failed (%s); PCA-only", exc)
        return None
    return proj[:, 0].astype(float), proj[:, 1].astype(float)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def run_concept_atlas(run_dir, cfg) -> dict:
    """Pool, cluster, null-test, name and project every causal SAE feature
    across the WHOLE run, and write `sae/concept_atlas.json`. Always writes
    -- including the degenerate `n_features == 0` case (no non-withheld
    target has any causal candidate), as an honest empty artifact rather
    than raising, so this function is safe to call directly (as the
    calibration/CLI path does) independent of whether the `concepts` stage
    itself decided the run was worth persisting (`concept_stage.py` makes
    that call and drops a stale file instead of invoking this function at
    all -- see its own module docstring).
    """
    run_dir = Path(run_dir)
    concepts_cfg = cfg.concepts
    causal_only = bool(getattr(getattr(cfg, "sae", None), "concept_causal_only", True))
    min_cosine = float(concepts_cfg.atlas_min_cosine)
    min_members = int(concepts_cfg.atlas_min_members)
    n_null = int(concepts_cfg.atlas_n_null)
    level_share_threshold = float(getattr(concepts_cfg, "level_share_threshold", 0.9))
    seed = int(getattr(getattr(cfg, "run", None), "seed", 0) or 0)

    out_path = run_dir / "sae" / "concept_atlas.json"
    params = {"min_cosine": min_cosine, "min_members": min_members,
             "linkage": _LINKAGE_METHOD, "n_null": n_null,
             # ROADMAP.md sec 37.7 P4 -- additive.
             "level_share_threshold": level_share_threshold}

    X, rows = pooled_features(run_dir, causal_only=causal_only)
    n_features = int(X.shape[0])

    if n_features == 0:
        out = {"schema_version": 1, "space": "ablation", "params": params,
              "n_features": 0, "n_assigned": 0, "rows": [], "concepts": [],
              "null": {"structure": None, "cross_model": None},
              "pca": {"explained_variance_ratio": []}, "projection": "pca",
              "naming": "_compose_batch"}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(out_path, out)
        log.warning("concept atlas: no pooled causal features across any target; "
                   "wrote empty %s", out_path)
        return out

    labels = cluster_atlas(X, min_cosine=min_cosine, min_members=min_members)
    models_arr = np.array([r["model"] for r in rows])

    rng_struct = np.random.default_rng(seed)
    n_concepts_null, frac_assigned_null = _structure_null(
        X, min_cosine, min_members, rng_struct, n_null)
    real_n_concepts = int(labels.max() + 1) if labels.size and labels.max() >= 0 else 0
    real_frac_assigned = float(np.mean(labels >= 0))
    structure_null = {
        "n_null": n_null,
        "n_concepts_real": real_n_concepts,
        "n_concepts_null_mean": float(np.mean(n_concepts_null)),
        "n_concepts_null_p95": float(np.quantile(n_concepts_null, 0.95)),
        "p_n_concepts": _right_tail_p(real_n_concepts, n_concepts_null),
        "frac_assigned_real": real_frac_assigned,
        "frac_assigned_null_mean": float(np.mean(frac_assigned_null)),
        "frac_assigned_null_p95": float(np.quantile(frac_assigned_null, 0.95)),
        "p_frac_assigned": _right_tail_p(real_frac_assigned, frac_assigned_null),
    }

    rng_cross = np.random.default_rng(seed + 1)
    (real_multi, null_multi_counts, real_n_models_by_c,
     null_n_models_by_c, concept_ids) = _cross_model_null(labels, models_arr, rng_cross, n_null)
    cross_model_null = {
        "n_null": n_null,
        "n_multi_model_real": int(real_multi),
        "n_multi_model_null_mean": (float(np.mean(null_multi_counts))
                                    if null_multi_counts else float("nan")),
        "n_multi_model_null_p95": (float(np.quantile(null_multi_counts, 0.95))
                                   if null_multi_counts else float("nan")),
        "p_n_multi_model": _right_tail_p(real_multi, null_multi_counts),
        "p_n_multi_model_below": _left_tail_p(real_multi, null_multi_counts),
    }
    rng_purity = np.random.default_rng(seed + 2)
    purity_null = [_mean_purity(labels, models_arr[rng_purity.permutation(models_arr.size)],
                                concept_ids) for _ in range(int(n_null))]
    real_purity = _mean_purity(labels, models_arr, concept_ids)
    finite_null = [v for v in purity_null if np.isfinite(v)]
    cross_model_null.update({
        "mean_purity_real": real_purity,
        "mean_purity_null_mean": float(np.mean(finite_null)) if finite_null else float("nan"),
        "p_purity_above": _right_tail_p(real_purity, finite_null),
        "p_purity_below": _left_tail_p(real_purity, finite_null),
    })
    cross_model_null["verdict"] = cross_model_verdict(
        cross_model_null["p_purity_below"], cross_model_null["p_purity_above"])
    per_concept_p = {c: _right_tail_p(real_n_models_by_c[c], null_n_models_by_c[c])
                     for c in concept_ids}

    concepts = _concept_records(X, rows, labels, per_concept_p)
    _tag_causal_effect(concepts, rows, _candidate_lookup(run_dir), level_share_threshold)
    _name_concepts(concepts)

    pc1, pc2, ev = _pca_projection(X, seed=seed)
    umap_xy = _umap_projection(X, seed=seed)

    rows_out = []
    for i, r in enumerate(rows):
        row_out = {"model": r["model"], "layer": r["layer"], "feature": int(r["feature"]),
                  "concept": (int(labels[i]) if labels[i] >= 0 else None),
                  "pc1": float(pc1[i]), "pc2": float(pc2[i])}
        if umap_xy is not None:
            row_out["umap1"] = float(umap_xy[0][i])
            row_out["umap2"] = float(umap_xy[1][i])
        rows_out.append(row_out)

    out = {
        "schema_version": 1, "space": "ablation", "params": params,
        "n_features": n_features, "n_assigned": int(np.sum(labels >= 0)),
        "rows": rows_out, "concepts": concepts,
        "null": {"structure": structure_null, "cross_model": cross_model_null},
        "pca": {"explained_variance_ratio": ev},
        "projection": "umap" if umap_xy is not None else "pca",
        "naming": "_compose_batch",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    n_targets = len({(r["model"], r["layer"]) for r in rows})
    log.info("concept atlas: %d feature(s) across %d target(s) -> %d concept(s) "
            "(%d assigned, %d spanning >=2 models)", n_features, n_targets,
            len(concepts), out["n_assigned"], real_multi)
    return out
