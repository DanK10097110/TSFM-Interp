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
                     seed: int = 0) -> dict:
    """`StandardScaler` + `KMeans`, IDENTICAL to `roles.py::cluster_roles` --
    reused by import, never reimplemented, so sec 30.1's NEW-vs-OLD
    comparison is a statement about the FEATURE SPACE rather than about two
    different clusterers (`CLAUDE.md` sec 11.41). `k="auto"` resolves through
    that same function's `_resolve_role_k` (`clip(max(2, n // 6), 2, 8)`).

    A target with fewer than 4 causal candidates is skipped with a recorded
    reason rather than clustered -- three of a kind is not a partition.

    Returns `{"labels", "k", "silhouette", "non_modular", "reason"}`.
    """
    n = X.shape[0]
    if n < _MIN_CAUSAL_CANDIDATES:
        return {"labels": np.zeros(n, dtype=int), "k": 1, "silhouette": float("nan"),
               "non_modular": True,
               "reason": f"only {n} causal candidate(s) -- fewer than "
                         f"{_MIN_CAUSAL_CANDIDATES}, not clustered"}
    return cluster_roles(X, role_k=k, min_silhouette=min_silhouette, seed=seed)


def _dominant_channel_index(mean_row: np.ndarray, channel_columns: list,
                            rank: int = 0) -> tuple:
    """The `rank`-th most discriminating channel by `|mean|`, and its sign.

    A sort/lookup handle only -- per sec 30.1 measurement 2 (30.3% member-
    agrees-with-name against a 30.3% permutation null), this must NEVER be
    rendered as the concept's name. `concept_table` stores it as
    `dominant_channel` for that reason: a scalar downstream code can sort or
    filter on, not a claim about what the concept "is."
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


def _compose_batch(profiles: np.ndarray, cleared: np.ndarray, peers: list) -> list[dict]:
    """One `{"name": str, "lead_diversified": bool}` per row of `profiles`,
    applying `compose_name`'s three mechanisms across the WHOLE peer
    population at once -- mechanism 2 needs every row's claim to decide the
    next row's, and mechanism 3 needs every row's candidate name at each
    length k. `compose_name` is a thin per-row wrapper over this; production
    callers (`assign_concept_names`) call this directly rather than paying
    O(n) repeated O(n^2) batch recomputes.
    """
    n = profiles.shape[0]
    if n == 0:
        return []

    # Mechanism 1: contrastive z-score, computed over ALL peers (not just
    # those clearing a given channel) -- it characterizes what is TYPICAL
    # for that channel across the population, which is what makes an
    # outlier row's own clearing value stand out.
    mean = profiles.mean(axis=0)
    std = profiles.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    z = (profiles - mean) / std_safe

    own_ranked = [_rank_candidates(i, cleared, z) for i in range(n)]
    top_abs_z = [abs(z[i, own_ranked[i][0]]) if own_ranked[i] else -np.inf
                for i in range(n)]

    # Mechanism 2: leading-clause diversification. Process rows in
    # descending top-|z|, tie-broken by `peers[i]` (a (target, concept_id)
    # -shaped key) -- NEVER by array position, which is what makes the
    # result invariant to how the caller ordered its rows.
    #
    # The dedup key is the RENDERED leading clause -- (channel, verb, tier),
    # matching the docstring's literal "whose leading clause no earlier row
    # has taken" (not "whose channel"). Two rows both clearing `trend` but
    # with opposite signs render "raises trend" / "lowers trend" -- distinct
    # text, so they must not block each other; two rows with the same sign
    # but different magnitude tiers ("mild raises X" / "dominant raises X")
    # are equally distinct. Deduping on bare channel index (an earlier
    # version of this function) is strictly more conservative than the
    # spec requires and measurably so: on `runs/full_report_run_4model`'s
    # real 30 concepts it caps distinct leading clauses at 11 of 30 against
    # a proven-optimal ceiling of 26 (a bipartite-matching computation, not
    # a guess); keying on the rendered clause instead reaches 25 of 30 --
    # see ROADMAP.md sec 30's dated Stage 2 update for the full diagnosis.
    def _leading_clause_key(row: int, ch: int) -> tuple:
        val = float(profiles[row, ch])
        return (ch, _verb(val), _magnitude_tier(val))

    order = sorted(range(n), key=lambda i: (-top_abs_z[i], peers[i]))
    used_leads: set = set()
    lead_channel: list = [None] * n
    for i in order:
        candidates = own_ranked[i]
        if not candidates:
            continue
        chosen = next((c for c in candidates
                       if _leading_clause_key(i, c) not in used_leads),
                      candidates[0])
        lead_channel[i] = chosen
        used_leads.add(_leading_clause_key(i, chosen))

    clause_order: list[list[int]] = []
    lead_diversified = [False] * n
    for i in range(n):
        candidates = own_ranked[i]
        if not candidates:
            clause_order.append([])
            continue
        lead = lead_channel[i]
        clause_order.append([lead] + [c for c in candidates if c != lead])
        lead_diversified[i] = (lead != candidates[0])

    def render(i: int, k: int) -> str:
        chans = clause_order[i][:k]
        if not chans:
            return "no channel clears its own null"
        tier = _magnitude_tier(float(profiles[i, chans[0]]))
        clauses = [f"{_verb(float(profiles[i, c]))} {CHANNELS[c]}" for c in chans]
        body = " · ".join(clauses)
        return f"{tier} {body}" if tier else body

    # Mechanism 3: shortest unique prefix. A peer with fewer clauses than k
    # is compared at ITS OWN max length, since it cannot render further.
    names: list = [None] * n
    for i in range(n):
        max_k = min(len(clause_order[i]), _MAX_CLAUSES)
        if max_k == 0:
            names[i] = render(i, 0)
            continue
        chosen = render(i, max_k)  # fallback if every k collides: the cap
        for k in range(1, max_k + 1):
            candidate = render(i, k)
            collides = any(
                render(j, min(k, len(clause_order[j]))) == candidate
                for j in range(n) if j != i and clause_order[j])
            if not collides:
                chosen = candidate
                break
        names[i] = chosen

    return [{"name": names[i], "lead_diversified": lead_diversified[i]} for i in range(n)]


def compose_name(i: int, profiles: np.ndarray, cleared: np.ndarray, peers: list) -> str:
    """The deterministic name for peer `i`. Short, specific, and distinct
    from its peers -- no LLM, no weights. See sec 30.4.1/sec 30.7.

    `profiles`/`cleared` are `[n_peers, n_channels]`, row-aligned with
    `peers` (a list of ids used only as a stable tie-break, e.g.
    `(target, concept_id)`). Three mechanisms, in this order:

    1. CONTRASTIVE ORDER. Z-score each channel across `peers`, then rank
       this row's cleared channels by |z| -- names a row by what is
       unusual about it among its peers, not by its own largest absolute
       effect (which is what makes today's argmax naming collapse onto a
       handful of population-wide-dominant channels).
    2. LEADING-CLAUSE DIVERSIFICATION. Process rows in descending top-|z|,
       tie-broken by `peers[i]` so the pass is reproducible (sec 11.2).
       Each row takes its most contrastive channel whose leading clause no
       earlier row has taken, falling back to its own best when all are
       taken.
    3. SHORTEST UNIQUE PREFIX. Emit clauses k=1,2,3... and stop at the
       first k whose rendered name no peer shares at that same k (a peer
       with fewer available clauses is compared at its own max). Capped at
       3 clauses even if a collision remains at the cap.

    Clause wording: `raises`/`lowers` <channel>, downgraded to `moves`
    below 1.0 null unit -- a sub-null mean's sign is not resolvable
    (sec 11.54's second site). Tier prefix from the leading channel only:
    `dominant` >=5, `strong` >=2, `mild` >=1 null units.

    A row with no candidate channel (`cleared[i]` all False) names as
    "no channel clears its own null" rather than fabricating one.
    """
    return _compose_batch(profiles, cleared, peers)[int(i)]["name"]


def assign_concept_names(targets: dict) -> None:
    """Mutates `targets` (`run_concepts`'s own artifact structure) in
    place, setting `name`/`name_lead_diversified` on every concept record.

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
        cluster_result = cluster_concepts(X, k=k, min_silhouette=min_silhouette, seed=seed)
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
