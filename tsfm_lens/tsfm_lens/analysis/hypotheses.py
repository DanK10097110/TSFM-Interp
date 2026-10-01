"""Hypothesis registry — the pre-registration seam `confirm` tests against
(`ROADMAP.md` sec 15 A15).

`confirm.py`'s discipline (`CLAUDE.md` §6.7) is the repo's strongest
epistemic claim: everything on the dev corpus is exploratory, and `confirm`
tests dev hypotheses exactly once against a sealed private corpus. Before
this module, that discipline was enforced by convention alone — `confirm`
read `l0/summary.json` directly at the moment it ran, a mutable dev
artifact with no record of what was "pre-registered" versus re-derived
after a second look.

This module writes `hypotheses.json`: one entry per confirmable dev claim,
each carrying the exact artifact path and content hash it was derived
from. `confirm` requires this file, verifies every referenced hash still
matches before running, and tests exactly the registered set — no more, no
fewer.

Not every stage's statistic is replicated by `confirm` yet. `l0` family
strengths, `l1`'s peak-CKA pair, `l3`'s per-corruption fingerprint agreement
(`_replicate_registered_l3`) and the `concept_transfer` claims are replicated.
L2's stitching gain, L4's clustering AMI and the per-archetype L0 strengths are
still registered here (so they count in the multiplicity ledger and their
own hash is pinned), but marked `replicable: False` with a stated reason —
a real, stated gap, not a silent omission (`CLAUDE.md` §2.5).

`ROADMAP.md` sec 38.2 (K2) adds four opt-in causal-concept claim types
(`concept_causal`, `concept_atlas`, `shared_input_agreement`,
`concept_structure`), enabled by `confirm.register_concept_claims`. With it
off (the default) this module registers exactly what it registered before.
Each such claim is frozen from dev artifacts (dev SAE checkpoints, dev
feature ids, dev channel and sign) and pins the sha256 of every artifact it
reads, `.pt` checkpoints included (`artifacts`).

`confirm.register_requires_target_significance` (ROADMAP.md sec 38.3.4, also
opt-in) additionally restricts those claims to targets whose battery clears
are BH-significant against their own empirical chance
(`analysis/target_significance.py`); every claim it removes is recorded in the
registry's `concept_claim_candidates` with its target, p and q, and a target
without `empirical_chance` makes registration refuse.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..config import PipelineConfig
from ..utils import load_json, log, save_json


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _l0_entries(run_dir: Path) -> list:
    """One entry per dev family-level "strength" claim (the original, and
    still the only, source `confirm.py` tested before this module existed).
    """
    path = run_dir / "l0" / "summary.json"
    if not path.exists():
        return []
    l0 = load_json(path)
    art_hash = _sha256_file(path)
    ratio = l0.get("mase_ratio", {})
    entries = []
    for model, fams in (l0.get("strengths") or {}).items():
        for fam in fams:
            entries.append({
                "id": f"l0_family::{fam}::{model}", "stage": "l0",
                "statistic": "paired_mase_ratio", "family": fam, "favored": model,
                "artifact": "l0/summary.json", "artifact_sha256": art_hash,
                "statement": f"{model} is significantly stronger than its comparison "
                            f"partner on family '{fam}' (dev MASE ratio "
                            f"{ratio.get(fam)}).",
                "replicable": True,
            })
    # Per-archetype strengths (sec 15 A9) are registered for the multiplicity
    # ledger but not yet replicated -- `confirm`'s private-corpus pivot is
    # family-keyed, and archetype labels don't necessarily survive onto a
    # freshly-generated private epoch the same way (a fresh `random_
    # parametric` draw gets its own fresh archetype sample). Stated gap, not
    # silently skipped.
    per_arch = l0.get("per_archetype") or {}
    for model, arches in (per_arch.get("strengths") or {}).items():
        for arch in arches:
            entries.append({
                "id": f"l0_archetype::{arch}::{model}", "stage": "l0_archetype",
                "statistic": "paired_mase_ratio", "archetype": arch, "favored": model,
                "artifact": "l0/summary.json", "artifact_sha256": art_hash,
                "statement": f"{model} is significantly stronger than its comparison "
                            f"partner on archetype '{arch}'.",
                "replicable": False,
                "not_replicable_reason": "archetype-level replication not yet implemented "
                                        "(sec 15 A15); private corpora don't guarantee the "
                                        "same archetype composition as dev.",
            })
    return entries


def _l1_entries(run_dir: Path) -> list:
    """The dev peak-CKA pair -- `confirm.py`'s other pre-existing replication target."""
    path = run_dir / "l1" / "meta.json"
    if not path.exists():
        return []
    l1 = load_json(path)
    best = l1.get("best_pair")
    if not best:
        return []
    return [{
        "id": f"l1_peak_cka::{best['layer_a']}::{best['layer_b']}", "stage": "l1",
        "statistic": "linear_cka", "layer_a": best["layer_a"], "layer_b": best["layer_b"],
        "artifact": "l1/meta.json", "artifact_sha256": _sha256_file(path),
        "statement": f"Peak CKA={best['cka']:.3f} between {best['layer_a']} and "
                    f"{best['layer_b']} replicates out of sample.",
        "replicable": True,
    }]


def _l2_entries(run_dir: Path) -> list:
    path = run_dir / "l2" / "stitching.json"
    if not path.exists():
        return []
    l2 = load_json(path)
    art_hash = _sha256_file(path)
    entries = []
    for direction, payload in (l2.get("directions") or {}).items():
        if payload.get("best_gain") is not None:
            entries.append({
                "id": f"l2_gain::{direction}", "stage": "l2", "statistic": "stitching_gain",
                "direction": direction, "artifact": "l2/stitching.json",
                "artifact_sha256": art_hash,
                "statement": f"L2 stitching gain over the input-feature baseline "
                            f"({direction}) = {payload['best_gain']:.3f}.",
                "replicable": False,
                "not_replicable_reason": "L2 replication (re-fitting stitching probes on "
                                        "private data) not yet implemented (sec 15 A15).",
            })
    return entries


def _l3_entries(run_dir: Path) -> list:
    path = run_dir / "l3" / "meta.json"
    if not path.exists():
        return []
    l3 = load_json(path)
    art_hash = _sha256_file(path)
    entries = []
    for corruption, val in (l3.get("agreement", {}).get("per_corruption") or {}).items():
        if val.get("value") is None:
            continue
        entries.append({
            "id": f"l3_agreement::{corruption}", "stage": "l3",
            "statistic": "fingerprint_agreement_rho", "corruption": corruption,
            "artifact": "l3/meta.json", "artifact_sha256": art_hash,
            "statement": f"Cross-model fingerprint agreement for corruption "
                        f"'{corruption}' = {val.get('value')}.",
            "replicable": True,
        })
    return entries


def _clustering_entries(run_dir: Path) -> list:
    path = run_dir / "clustering" / "comparison.json"
    if not path.exists():
        return []
    c = load_json(path)
    ami = (c.get("ami") or {}).get("value")
    if ami is None:
        return []
    return [{
        "id": "clustering_ami", "stage": "clustering", "statistic": "ami",
        "artifact": "clustering/comparison.json", "artifact_sha256": _sha256_file(path),
        "statement": f"Cross-model clustering AMI = {ami:.3f}.",
        "replicable": False,
        "not_replicable_reason": "Clustering replication (re-fitting cluster labels on "
                                "private data) not yet implemented (sec 15 A15).",
    }]


# ---------------------------------------------------------------------------
# ROADMAP.md sec 37.10 P7 -- `concept_transfer` claims, registered from dev
# atlas artifacts only, before the private epoch is ever touched. The knob
# family (`concept_knob::`) is a separate, EMPTY family: sec 37.9's P6a
# go/no-go (0 of 98 tests survived per-target BH) and its pre-registered
# retry (0 of 151 survived pooled BH) both found nothing to register, so
# there is no dev response to freeze -- this is recorded as an explicit empty
# family with its reason, never silently absent (`CLAUDE.md` sec 2.5).
# ---------------------------------------------------------------------------

_KNOB_FAMILY_EMPTY_REASON = (
    "no (concept, knob) input response survived BH after correction "
    "(ROADMAP.md sec 37.9 P6a go/no-go: 0 of 98 tests, per-target BH; the "
    "registered P6a retry: 0 of 151 tests, pooled BH over 9 more targets). "
    "The knob family is empty by design (sec 37.10 P7 decision item 3), not "
    "by omission: no private counterfactual path is built.")


def _concept_transfer_candidates(run_dir: Path, concepts_cfg) -> dict:
    """Rank every reciprocal-FDR, seed-stable atlas-transfer test by its dev
    AUC margin, and record the full ranking plus the registration cut.

    Candidates are `sae/atlas_transfer.json` tests with `reciprocal_fdr`
    True whose atlas concept is `stable` in `sae/concept_stability.json`
    (ROADMAP.md sec 37.10 design item 2). The margin is `min(forward AUC -
    forward null p95, reverse AUC - reverse null p95)` -- the WEAKER leg,
    since a reciprocal test needs both legs to clear their own null and the
    weaker one is what a private re-test is most likely to lose. Ties
    (not expected with float margins, but not assumed impossible) break on
    `(concept, src_target, dst_target)` for a deterministic cut regardless
    of dict/JSON iteration order.

    Degrades to an empty candidate list with a stated reason when the
    `concepts` stage's artifacts are absent -- this module must not require
    `concepts.enabled` (a confirm-only run, e.g. `configs/smoke.yaml`,
    registers its l0/l1/l2/l3/clustering claims regardless, CLAUDE.md sec
    2.5/sec 11.35's false-refusal shape).
    """
    at_path = run_dir / "sae" / "atlas_transfer.json"
    atlas_path = run_dir / "sae" / "concept_atlas.json"
    stab_path = run_dir / "sae" / "concept_stability.json"
    missing = [str(p.relative_to(run_dir)) for p in (at_path, atlas_path, stab_path)
              if not p.exists()]
    if missing:
        return {"candidates": [], "cut": 0,
               "reason": f"concept atlas artifacts not found: {', '.join(missing)}"}

    at = load_json(at_path)
    atlas = load_json(atlas_path)
    stability = load_json(stab_path)
    stable_ids = {int(c["concept"]) for c in (stability.get("concepts") or [])
                 if (c.get("stability") or {}).get("stable") is True}

    parts: dict = {}
    for r in atlas.get("rows") or []:
        cid = r.get("concept")
        if cid is None:
            continue
        key = (int(cid), f"{r['model']}/{r['layer']}")
        entry = parts.setdefault(key, {"model": r["model"], "features": set()})
        entry["features"].add(int(r["feature"]))

    n_reg = int(getattr(concepts_cfg, "n_registered", 20) or 20)
    candidates = []
    for t in at.get("tests") or []:
        if not t.get("reciprocal_fdr"):
            continue
        cid = int(t["concept"])
        if cid not in stable_ids:
            continue
        part = parts.get((cid, t["src_target"]))
        if part is None:
            continue
        fwd_margin = float(t["auc"]) - float(t["null_p95"])
        rev_margin = float(t["rev_auc"]) - float(t["rev_null_p95"])
        candidates.append({
            "concept": cid, "src_target": t["src_target"], "src_model": t["src_model"],
            "src_features": sorted(part["features"]),
            "dst_target": t["dst_target"], "dst_model": t["dst_model"],
            "dst_feature": int(t["feature"]),
            "dev_auc": t["auc"], "dev_null_p95": t["null_p95"],
            "dev_rev_auc": t["rev_auc"], "dev_rev_null_p95": t["rev_null_p95"],
            "dev_auc_margin": min(fwd_margin, rev_margin),
            "k_top_series": at.get("k_top_series"),
        })
    candidates.sort(key=lambda c: (-c["dev_auc_margin"], c["concept"],
                                   c["src_target"], c["dst_target"]))
    cut = min(n_reg, len(candidates))
    return {"candidates": candidates, "cut": cut, "n_registered_cfg": n_reg}


def _concept_transfer_frozen_entries(run_dir: Path, cfg: PipelineConfig, ranking: dict) -> list:
    """Frozen-feature dev statistics for the SAME top-`cut` candidates
    `_concept_transfer_candidates` already ranked (identical candidate pool
    and ranking to `"search"` mode -- ROADMAP.md sec 37.10 P7b decision item
    2 says so explicitly: only the CLAIM differs, not which claims are
    picked). Freezes BOTH `src_features` and `dst_feature` (dev's own best
    destination feature, i.e. the argmax `"search"` mode already found), and
    records the frozen claim's OWN dev statistics computed with a
    single-feature stratum-matched null (`sae/transfer.py::
    transfer_one_fixed_feature`, reusing the SAME `fwd_seed`/`rev_seed`
    convention `run_atlas_transfer` used to build the search-mode dev
    statistics, so this is a re-slice of the SAME matched draws through one
    feature's own column, not a fresh random draw): the forward leg's dev
    AUC is unchanged from the search `dev_auc` (same feature, same top-`k`
    set `S`), but its `null_p95`/`p` differ, because the search null maxes
    over the WHOLE destination dictionary and the frozen null does not.

    Dev-only I/O: reads the run's OWN zarr store (`space="sae"`) and
    `meta.parquet`, both already-written dev artifacts, never the private
    corpus -- registration still precedes private access.
    """
    if not ranking["candidates"][:ranking["cut"]]:
        return []
    from ..extraction.store import ActivationStore, load_meta
    from ..sae.transfer import (_by_stratum, _pooled_ranks_fns, _seed, concept_scores,
                                matched_draws, series_strata, top_series,
                                transfer_one_fixed_feature)

    candidates = ranking["candidates"][:ranking["cut"]]
    at_path = run_dir / "sae" / "atlas_transfer.json"
    art_hash = _sha256_file(at_path) if at_path.exists() else None

    meta = load_meta(run_dir)
    strata = series_strata(meta)
    by_stratum = _by_stratum(strata)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    _pooled, _ranks = _pooled_ranks_fns(store)

    sae_cfg, concepts_cfg = cfg.sae, cfg.concepts
    n_null = int(getattr(sae_cfg, "transfer_n_null", 200) or 200)
    base_seed = int(getattr(sae_cfg, "transfer_seed", 0) or 0)
    p_method = str(getattr(concepts_cfg, "transfer_p_method", "exact") or "exact")
    max_redraw = int(getattr(concepts_cfg, "transfer_max_redraw", 5000) or 5000)

    entries = []
    for c in candidates:
        src_pooled = _pooled(c["src_target"])
        score_src = concept_scores(src_pooled, c["src_features"])
        k = min(int(c.get("k_top_series") or 20), max(1, src_pooled.shape[0] - 1))
        S = top_series(score_src, k)
        fwd_seed = _seed("atlas", c["src_target"], c["concept"], base=base_seed)
        fwd_rng = np.random.default_rng(fwd_seed)
        fwd_draws = matched_draws(S, strata, by_stratum, n_null, fwd_rng)
        dst_ranks = _ranks(c["dst_target"])
        rev_seed = _seed("atlas", c["src_target"], c["concept"], c["dst_target"], base=base_seed)
        dev_stat = transfer_one_fixed_feature(
            score_src, dst_ranks, S, fwd_draws, strata, by_stratum,
            feature=int(c["dst_feature"]), k=k, n_draws=n_null, seed=rev_seed,
            p_method=p_method, max_redraw=max_redraw, fwd_seed=fwd_seed)
        entries.append({
            "id": f"concept_transfer_frozen::{c['src_target']}::{c['concept']}::"
                 f"{c['dst_target']}::f{c['dst_feature']}",
            "stage": "concept_transfer", "family": "concept_transfer", "mode": "frozen",
            "concept": c["concept"], "src_target": c["src_target"], "src_model": c["src_model"],
            "src_features": c["src_features"], "dst_target": c["dst_target"],
            "dst_model": c["dst_model"], "dst_feature": c["dst_feature"],
            "dev_auc": dev_stat["auc"], "dev_null_p95": dev_stat["null_p95"],
            "dev_p": dev_stat["p"],
            "dev_rev_auc": dev_stat["rev_auc"], "dev_rev_null_p95": dev_stat["rev_null_p95"],
            "dev_rev_p": dev_stat["rev_p"],
            "dev_auc_margin": c["dev_auc_margin"], "k_top_series": k,
            "artifact": "sae/atlas_transfer.json", "artifact_sha256": art_hash,
            "statement": (f"Concept {c['concept']} at {c['src_target']} ({c['src_model']}) "
                         f"transfers to {c['dst_model']} ({c['dst_target']}), FROZEN feature "
                         f"{c['dst_feature']} (dev AUC {dev_stat['auc']!r} against a "
                         f"single-feature null p95 {dev_stat['null_p95']!r}, dev p "
                         f"{dev_stat['p']!r})."),
            "replicable": True,
        })
    return entries


def _concept_transfer_entries(run_dir: Path, cfg: PipelineConfig) -> tuple:
    """`-> (hypothesis entries, ranking dict)`. One entry per candidate in
    the top `cut`, each carrying everything `_replicate_registered_concepts`
    needs to re-test the FROZEN claim on private data without re-reading
    `sae/concept_atlas.json` (registration precedes private access,
    ROADMAP.md sec 37.10 design item 4). `src_features` are frozen here;
    `dst_feature` is recorded for reference only, because the forward leg
    re-searches the destination dictionary on private data against a
    max-over-features null, exactly as the dev test did.

    The id names the destination TARGET, not only its model: one source
    concept routinely transfers to several layers of the same model, and an
    id that collapses them made `holm()` (keyed by id) silently shrink the
    family (20 registered claims, 9 distinct ids on the reference run).

    ROADMAP.md sec 37.10 P7b -- `concepts.transfer_claim_mode` ("search" |
    "frozen") switches which CLAIM is registered for the SAME ranked
    candidate pool (`_concept_transfer_candidates` never branches on mode).
    `"search"` (the default) is this function's original body, UNCHANGED, so
    it reproduces P7's registration byte-for-byte (`test_search_mode_
    unchanged`); `"frozen"` dispatches to `_concept_transfer_frozen_entries`
    instead, which returns entries carrying a `"mode": "frozen"` key -- an
    ADDITIVE marker the search-mode entries above never gain (`CLAUDE.md`
    sec 7 invariant 13), so a reader (or `confirm.py`) can tell the two
    apart, and a pre-P7b consumer that only ever saw `"search"` entries sees
    nothing new.
    """
    ranking = _concept_transfer_candidates(run_dir, cfg.concepts)
    mode = str(getattr(cfg.concepts, "transfer_claim_mode", "search") or "search")
    if mode not in ("search", "frozen"):
        raise ValueError(f"concepts.transfer_claim_mode must be 'search' or 'frozen', "
                         f"got {mode!r}")
    if mode == "frozen":
        return _concept_transfer_frozen_entries(run_dir, cfg, ranking), ranking

    at_path = run_dir / "sae" / "atlas_transfer.json"
    art_hash = _sha256_file(at_path) if at_path.exists() else None
    entries = []
    for c in ranking["candidates"][:ranking["cut"]]:
        entries.append({
            "id": f"concept_transfer::{c['src_target']}::{c['concept']}::{c['dst_target']}",
            "stage": "concept_transfer", "family": "concept_transfer",
            "concept": c["concept"], "src_target": c["src_target"], "src_model": c["src_model"],
            "src_features": c["src_features"], "dst_target": c["dst_target"],
            "dst_model": c["dst_model"], "dst_feature": c["dst_feature"],
            "dev_auc": c["dev_auc"], "dev_null_p95": c["dev_null_p95"],
            "dev_rev_auc": c["dev_rev_auc"], "dev_rev_null_p95": c["dev_rev_null_p95"],
            "dev_auc_margin": c["dev_auc_margin"], "k_top_series": c["k_top_series"],
            "artifact": "sae/atlas_transfer.json", "artifact_sha256": art_hash,
            "statement": (f"Concept {c['concept']} at {c['src_target']} ({c['src_model']}) "
                         f"transfers to {c['dst_model']} ({c['dst_target']}, feature "
                         f"{c['dst_feature']}); dev AUC margin "
                         f"{c['dev_auc_margin']!r} (weaker leg of the reciprocal test)."),
            "replicable": True,
        })
    return entries, ranking


# ---------------------------------------------------------------------------
# ROADMAP.md sec 38.2 (K2) -- registration of the CAUSAL concept claims.
#
# Opt-in behind `confirm.register_concept_claims`: with it off, `build_registry`
# returns exactly what it returned before this section existed (same entries,
# same keys), so every existing registry and its `registry_sha256` stay
# byte-identical (`CLAUDE.md` sec 2.1, invariant 13). Every claim below is
# FROZEN from dev artifacts only -- dev SAE checkpoints, dev feature ids, dev
# channel and sign -- and records the sha256 of EVERY dev artifact it reads
# (`artifacts`, checked by `check_registry_freshness`), the `.pt` checkpoints
# included, so a retrained dictionary refuses to be confirmed against.
# ---------------------------------------------------------------------------

CONCEPT_CLAIM_STAGES = ("concept_causal", "concept_atlas", "shared_input_agreement",
                        "concept_structure")

_NOT_REGISTERED_FAMILIES_REASON = (
    "'families do not beat the shuffle null' is a NEGATIVE structure result: "
    "confirming a null needs an equivalence margin that nobody has justified "
    "(sec 38.2.2 item 5). It stays descriptive.")
_STRUCTURE_C_REASON = (
    "'convergent concepts outnumber shared ones among multi-model concepts' is "
    "exploratory, not registered: it is low-powered at ~16 multi-model "
    "concepts and its classes come from a union-find over input-agreement "
    "pairs that has no private counterpart (sec 38.2.2 item 4c).")


def _rel(run_dir: Path, path: Path) -> str:
    return Path(path).relative_to(run_dir).as_posix()


class _Hasher:
    """`{run-relative path: sha256}` for the dev artifacts one registration
    reads, hashed once per file however many claims cite it."""

    def __init__(self, run_dir: Path):
        self.run_dir, self._cache = run_dir, {}

    def __call__(self, path: Path) -> tuple:
        rel = _rel(self.run_dir, path)
        if rel not in self._cache:
            self._cache[rel] = _sha256_file(path)
        return rel, self._cache[rel]

    def many(self, paths) -> dict:
        return dict(self(p) for p in paths if Path(p).exists())


def _ablation_null_mode(cfg) -> str:
    """The ablation null the dev battery used (`sae.ablation_null`; the field
    may not exist on an older checkout, in which case it is the legacy
    `mean_magnitude`). Recorded on every causal claim: `confirm` refuses to
    test it against a different null."""
    return str(getattr(cfg.sae, "ablation_null", "mean_magnitude") or "mean_magnitude")


def _dev_ablation_targets(run_dir: Path) -> list:
    """`[(model, layer, artifact dict, path)]` for every non-withheld,
    non-skipped target. Paths come from `ablation_run.ablation_path`, never
    by hand; a glob hit whose helper path does not exist is dropped rather
    than guessed at."""
    from ..sae.ablation_run import ablation_path

    out = []
    for f in sorted(run_dir.glob("sae/*/*_ablation.json")):
        art = load_json(f)
        if art.get("withheld") or art.get("skipped"):
            continue
        model = art.get("model", f.parent.name)
        layer = art.get("layer", f.name[: -len("_ablation.json")])
        path = ablation_path(run_dir, model, layer)
        if path.exists():
            out.append((str(model), str(layer), art, path))
    return out


def _stable_atlas_members(run_dir: Path) -> tuple:
    """`(set of (model, layer, feature) in seed-stable concepts, {member:
    concept id}, paths read)`. Empty when the atlas or the stability artifact
    is absent -- the preference for stable members then simply does not
    apply, and the ranking says so."""
    atlas_p = run_dir / "sae" / "concept_atlas.json"
    stab_p = run_dir / "sae" / "concept_stability.json"
    if not atlas_p.exists() or not stab_p.exists():
        return set(), {}, []
    atlas, stab = load_json(atlas_p), load_json(stab_p)
    stable_ids = {int(c["concept"]) for c in (stab.get("concepts") or [])
                  if (c.get("stability") or {}).get("stable") is True}
    members, concept_of = set(), {}
    for r in atlas.get("rows") or []:
        cid = r.get("concept")
        if cid is not None and int(cid) in stable_ids:
            key = (str(r["model"]), str(r["layer"]), int(r["feature"]))
            members.add(key)
            concept_of[key] = int(cid)
    return members, concept_of, [atlas_p, stab_p]


def _target_gate_table(run_dir: Path, cfg: PipelineConfig) -> dict | None:
    """The per-target significance table when
    `confirm.register_requires_target_significance` is on, else `None` (the
    byte-identical default). Computed once per registration from every
    measured target's own `empirical_chance`
    (`analysis/target_significance.py::target_significance`), so all four
    claim builders gate against the same BH family. Raises
    `TargetChanceMissing` rather than falling back to the nominal 0.05."""
    if not bool(getattr(cfg.confirm, "register_requires_target_significance", False)):
        return None
    from .target_significance import target_significance

    return target_significance(
        [(m, l, art) for m, l, art, _p in _dev_ablation_targets(run_dir)])


def _target_ok(sig: dict | None, target: str) -> bool:
    """`True` when no gate is on or `target` is BH-significant. A target the
    table does not know (never measured) is not significant."""
    return sig is None or bool((sig["targets"].get(target) or {}).get("significant"))


def _target_record(sig: dict, target: str) -> dict:
    """`{target, p, q_value}` of one target (nulls for an unmeasured one),
    the reason fields every excluded claim carries."""
    t = sig["targets"].get(target) or {}
    return {"target": target, "clearing_cells": t.get("clearing_cells"),
            "p": t.get("p"), "q_value": t.get("q_value"), "q": sig["q"]}


def _exclusion(sig: dict, claim_id: str, claim_type: str, targets: list) -> dict:
    """The ledger row for a candidate the gate removed: its id, type, the
    offending target(s) with p and BH q, and a stated reason."""
    recs = [_target_record(sig, t) for t in sorted(set(targets)) if not _target_ok(sig, t)]
    return {"id": claim_id, "claim_type": claim_type, "targets": recs,
            "reason": ("target's battery clears are not BH-significant against its own "
                       f"empirical chance at q={sig['q']} (ROADMAP.md sec 38.3.4): "
                       + "; ".join(f"{r['target']} p={r['p']!r} q={r['q_value']!r}"
                                   for r in recs))}


def _causal_id(c: dict) -> str:
    return f"concept_causal::{c['model']}::{c['layer']}::f{c['feature']}::{c['channel']}"


def _agreement_id(t: dict) -> str:
    return (f"shared_input_agreement::{t['src_target']}::{t['dst_target']}::"
            f"c{t['concept']}::f{t['dst_feature']}::{t['verdict']}")


def _best_channel(candidate: dict) -> tuple | None:
    """`(channel, effect / null_p95, signed_effect, rec)` for the channel a
    dev candidate clears by the widest effect / q95 margin, or `None`. A
    channel with a degenerate null, an unavailable value or an exactly-zero
    signed effect (no sign to freeze) is never chosen."""
    best = None
    for ch, rec in (candidate.get("channels") or {}).items():
        if not rec.get("available", True) or not rec.get("clears_null"):
            continue
        p95, eff, signed = rec.get("null_p95"), rec.get("effect"), rec.get("signed_effect")
        if not p95 or eff is None or signed is None or float(signed) == 0.0:
            continue
        ratio = float(eff) / float(p95)
        if best is None or ratio > best[1]:
            best = (ch, ratio, float(signed), rec)
    return best


def _select_causal(cands: list, n_total: int, min_per_model: int = 4) -> list:
    """The registration cut: >= `min_per_model` per model, then the best of
    the rest, seed-stable atlas members first at both steps.

    Ranking key is `(not stable, -effect/q95 ratio, model, layer, feature,
    channel)` -- fully ordered, so the cut does not depend on dict or JSON
    iteration order (`CLAUDE.md` sec 11.55). A model with fewer than
    `min_per_model` candidates contributes what it has.
    """
    def _key(c):
        return (not c["stable"], -c["ratio"], c["model"], c["layer"], c["feature"], c["channel"])

    ranked = sorted(cands, key=_key)
    chosen, seen = [], set()
    for model in sorted({c["model"] for c in ranked}):
        for c in [c for c in ranked if c["model"] == model][:min_per_model]:
            chosen.append(c)
            seen.add(id(c))
    for c in ranked:
        if len(chosen) >= n_total:
            break
        if id(c) not in seen:
            chosen.append(c)
            seen.add(id(c))
    return sorted(chosen, key=_key)


def _concept_causal_candidates(run_dir: Path, concepts_cfg, sig: dict | None = None) -> dict:
    """Every dev feature that clears its random-direction null on some
    channel, ranked, and the registration cut.

    Reads each target's `sae/<model>/<layer>_ablation.json` (path from
    `ablation_run.ablation_path`). One candidate per (target, feature): its
    strongest channel by `effect / null_p95` and that channel's dev sign.
    `concepts.n_registered_causal` (default 32, a judgment count, sec
    38.2.2) is the cut, with at least 4 per model and a preference for
    members of seed-stable atlas concepts. Degrades to an empty list with a
    stated reason when no ablation artifact exists, so a confirm-only run
    (e.g. `configs/smoke.yaml`) still registers its other claims.

    `sig` (the per-target gate table, `None` = no gate) removes candidates on
    non-significant targets BEFORE the cut, so the cut is filled from
    significant targets; the claims the ungated cut would have registered on
    a removed target come back under `excluded` with their reason, and
    `n_pool_excluded_by_gate` counts the whole removed pool.
    """
    targets = _dev_ablation_targets(run_dir)
    n_reg = int(getattr(concepts_cfg, "n_registered_causal", 32) or 32)
    if not targets:
        return {"candidates": [], "cut": 0, "n_registered_cfg": n_reg,
                "reason": "no sae/<model>/<layer>_ablation.json artifact found"}
    stable_members, concept_of, _paths = _stable_atlas_members(run_dir)
    cands = []
    for model, layer, art, path in targets:
        for c in art.get("candidates") or []:
            if not c.get("scorable") or not c.get("n_channels_clearing"):
                continue
            best = _best_channel(c)
            if best is None:
                continue
            ch, ratio, signed, rec = best
            key = (model, layer, int(c["feature"]))
            cands.append({
                "model": model, "layer": layer, "feature": int(c["feature"]),
                "channel": ch, "ratio": ratio, "sign": 1 if signed > 0 else -1,
                "effect": float(rec["effect"]), "signed_effect": signed,
                "null_p95": float(rec["null_p95"]),
                "n_top_series": int(c.get("n_top_series") or 0),
                "top_k_series": art.get("top_k_series"),
                "n_null_directions": art.get("n_null_directions"),
                "stable": key in stable_members, "atlas_concept": concept_of.get(key),
                "path": path})
    cut = _select_causal(cands, n_reg)
    gate = {}
    if sig is not None:
        kept = [c for c in cands if _target_ok(sig, f"{c['model']}/{c['layer']}")]
        gated_cut = _select_causal(kept, n_reg)
        gate = {"n_before_gate": len(cut), "n_after_gate": len(gated_cut),
                "n_pool_excluded_by_gate": len(cands) - len(kept),
                "excluded": [_exclusion(sig, _causal_id(c), "concept_causal",
                                        [f"{c['model']}/{c['layer']}"])
                             for c in cut if not _target_ok(sig, f"{c['model']}/{c['layer']}")]}
        cands, cut = kept, gated_cut
    return {"candidates": cands, "cut": cut, "n_registered_cfg": n_reg,
            "n_stable_preferred": sum(1 for c in cands if c["stable"]),
            "stable_preference_applied": bool(stable_members), **gate}


def _concept_causal_entries(run_dir: Path, cfg: PipelineConfig,
                            sig: dict | None = None) -> tuple:
    """`-> (entries, ranking summary)`. Claim
    `concept_causal::{model}::{layer}::f{feature}::{channel}`: "ablating this
    feature on its private top-k firing series moves this channel with this
    sign beyond the row-matched random-direction null". The id names model,
    layer, feature AND channel: an id missing any of them silently merges
    distinct claims and shrinks the Holm family (`CLAUDE.md` sec 8, "Keys
    that collapse")."""
    from ..sae.ablation_run import checkpoint_path

    ranking = _concept_causal_candidates(run_dir, cfg.concepts, sig)
    hasher = _Hasher(run_dir)
    null_mode = _ablation_null_mode(cfg)
    entries = []
    for c in ranking["cut"] if ranking["cut"] else []:
        art_rel, art_sha = hasher(c["path"])
        arts = hasher.many([c["path"], checkpoint_path(run_dir, c["model"], c["layer"])])
        entries.append({
            "id": _causal_id(c),
            "stage": "concept_causal", "family": "concept_causal",
            "statistic": "ablation_channel_effect_vs_random_direction_null",
            "model": c["model"], "layer": c["layer"], "target": f"{c['model']}/{c['layer']}",
            "feature": c["feature"], "channel": c["channel"], "sign": c["sign"],
            "dev_effect": c["effect"], "dev_signed_effect": c["signed_effect"],
            "dev_null_p95": c["null_p95"], "dev_effect_over_null_p95": c["ratio"],
            "dev_n_top_series": c["n_top_series"],
            "k_top_series": c["top_k_series"] or 8,
            "dev_n_null_directions": c["n_null_directions"],
            "ablation_null": null_mode,
            "seed_stable_atlas_member": c["stable"], "atlas_concept": c["atlas_concept"],
            "artifact": art_rel, "artifact_sha256": art_sha, "artifacts": arts,
            "statement": (f"Ablating feature {c['feature']} of {c['model']}/{c['layer']} "
                         f"on its top-firing series moves the '{c['channel']}' channel "
                         f"{'up' if c['sign'] > 0 else 'down'} beyond a random-direction "
                         f"null (dev effect {c['effect']!r} vs null p95 {c['null_p95']!r})."),
            "replicable": True,
        })
    summary = {k: v for k, v in ranking.items() if k not in ("candidates", "cut")}
    summary.update({"n_candidates": len(ranking["candidates"]), "cut": len(entries),
                    "cut_ids": [e["id"] for e in entries]})
    return entries, summary


def _dev_vectors(run_dir: Path) -> dict:
    """`{(model, layer, feature): 9-vector}` of the dev ablation vectors
    (`sae/concepts.py::ablation_vector`, never re-derived here)."""
    from ..sae.concepts import ablation_vector

    out = {}
    for model, layer, art, _path in _dev_ablation_targets(run_dir):
        for c in art.get("candidates") or []:
            if c.get("scorable"):
                vec = ablation_vector(c)
                if vec is not None:
                    out[(model, layer, int(c["feature"]))] = [float(v) for v in vec]
    return out


def _atlas_candidates(run_dir: Path, cfg: PipelineConfig, sig: dict | None = None) -> tuple:
    """`concept_atlas::{concept}` -- "this seed-stable atlas concept's
    members still form a concept on private data". One claim per stable
    dev concept (an atlas concept id is global across models, so it is
    unique by construction). Members, their dev 9-vectors, the dev centroid
    and the clustering thresholds are all frozen here.

    With the per-target gate (`sig`), a concept with ANY member on a
    non-significant target is not registered and is listed under `excluded`:
    the claim is that the whole member set re-forms, so one chance-level
    member makes it a claim about possibly-chance features."""
    from ..sae.ablation_run import ablation_path, checkpoint_path

    atlas_p = run_dir / "sae" / "concept_atlas.json"
    stab_p = run_dir / "sae" / "concept_stability.json"
    if not atlas_p.exists() or not stab_p.exists():
        return [], {"reason": "concept atlas artifacts not found"}
    atlas, stab = load_json(atlas_p), load_json(stab_p)
    stable_ids = sorted({int(c["concept"]) for c in (stab.get("concepts") or [])
                        if (c.get("stability") or {}).get("stable") is True})
    vectors = _dev_vectors(run_dir)
    params = atlas.get("params") or {}
    hasher = _Hasher(run_dir)
    entries, excluded = [], []
    for cid in stable_ids:
        members = [{"model": str(r["model"]), "layer": str(r["layer"]),
                    "feature": int(r["feature"])}
                   for r in atlas.get("rows") or [] if r.get("concept") == cid]
        members.sort(key=lambda m: (m["model"], m["layer"], m["feature"]))
        for m in members:
            m["dev_vector"] = vectors.get((m["model"], m["layer"], m["feature"]))
        have = [np.asarray(m["dev_vector"], dtype=np.float64) for m in members
                if m["dev_vector"] is not None]
        if len(have) < 2:
            continue
        if sig is not None and not all(_target_ok(sig, f"{m['model']}/{m['layer']}")
                                       for m in members):
            excluded.append(_exclusion(sig, f"concept_atlas::{cid}", "concept_atlas",
                                       [f"{m['model']}/{m['layer']}" for m in members]))
            continue
        unit = np.stack([v / np.linalg.norm(v) for v in have if np.linalg.norm(v) > 0])
        centroid = unit.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        target_paths = []
        for t in sorted({(m["model"], m["layer"]) for m in members}):
            target_paths += [ablation_path(run_dir, *t), checkpoint_path(run_dir, *t)]
        arts = hasher.many([atlas_p, stab_p] + target_paths)
        models = sorted({m["model"] for m in members})
        entries.append({
            "id": f"concept_atlas::{cid}", "stage": "concept_atlas", "family": "concept_atlas",
            "statistic": "member_pair_cosine_fraction_and_centroid_cosine",
            "concept": cid, "members": members, "models": models,
            "dev_centroid": [float(v) for v in centroid],
            "min_cosine": float(params.get("min_cosine", 0.9)),
            "min_members": int(params.get("min_members", 3)),
            "artifact": "sae/concept_atlas.json", "artifact_sha256": arts["sae/concept_atlas.json"],
            "artifacts": arts,
            "statement": (f"Seed-stable atlas concept {cid} ({len(members)} member feature(s) "
                         f"across {', '.join(models)}) still forms a concept on private data."),
            "replicable": True,
        })
    summary = {"n_stable_concepts": len(stable_ids), "n_registered": len(entries)}
    if sig is not None:
        summary.update({"n_before_gate": len(entries) + len(excluded),
                        "n_after_gate": len(entries), "excluded": excluded})
    return entries, summary


def _agreement_candidates(run_dir: Path, concepts_cfg, sig: dict | None = None) -> dict:
    """Dev shared-input agreement tests with a DEFINITE verdict, ranked, and
    the registration cut: every `same causal effect` test, plus
    `concepts.n_registered_agreement_differs` (default 30, judgment) of the
    `acts differently` ones, round-robin across ordered model pairs and
    deepest-below-floor first within a pair. `not scorable`, `no specific
    agreement`, `level only` and `shape only` are not claims of agreement or
    disagreement and are not registered.

    With the per-target gate (`sig`), a test whose source OR destination
    target is non-significant is removed before the cut (so the cut is
    filled from eligible tests) and listed under `excluded`."""
    p = run_dir / "sae" / "shared_input_agreement.json"
    at_p = run_dir / "sae" / "atlas_transfer.json"
    n_diff = int(getattr(concepts_cfg, "n_registered_agreement_differs", 30) or 30)
    if not p.exists():
        return {"candidates": [], "cut": [], "reason": "sae/shared_input_agreement.json not found"}
    doc = load_json(p)
    k_top = int(load_json(at_p).get("k_top_series", 20)) if at_p.exists() else 20
    same_all, differs_all = [], []
    for t in doc.get("tests") or []:
        v = t.get("verdict")
        if v not in ("same causal effect", "acts differently"):
            continue
        depths = []
        for key in ("statistic_i", "statistic_ii"):
            st = t.get(key) or {}
            if st.get("observed") is not None and st.get("floor_p05_src") is not None:
                depths.append(float(st["observed"]) - min(float(st["floor_p05_src"]),
                                                          float(st["floor_p05_dst"])))
        rec = dict(t, k_top_series=k_top, depth=min(depths) if depths else 0.0)
        (same_all if v == "same causal effect" else differs_all).append(rec)

    def _ord(r):
        return (r["src_target"], r["dst_target"], int(r["concept"]), int(r["dst_feature"]))

    def _pick(same_in, differs_in):
        same = sorted(same_in, key=_ord)
        by_pair: dict = {}
        for r in sorted(differs_in, key=lambda r: (r["depth"],) + _ord(r)):
            by_pair.setdefault((r["src_model"], r["dst_model"]), []).append(r)
        picked = []
        while len(picked) < n_diff and any(by_pair.values()):
            for pair in sorted(by_pair):
                if by_pair[pair] and len(picked) < n_diff:
                    picked.append(by_pair[pair].pop(0))
        return same, picked

    def _eligible(r):
        return _target_ok(sig, r["src_target"]) and _target_ok(sig, r["dst_target"])

    gate = {}
    if sig is None:
        same, differs = same_all, differs_all
    else:
        same = [r for r in same_all if _eligible(r)]
        differs = [r for r in differs_all if _eligible(r)]
        ungated_same, ungated_picked = _pick(same_all, differs_all)
        gate = {"n_before_gate": len(ungated_same) + len(ungated_picked),
                "excluded": [_exclusion(sig, _agreement_id(r), "shared_input_agreement",
                                        [r["src_target"], r["dst_target"]])
                             for r in sorted(same_all + differs_all, key=_ord)
                             if not _eligible(r)]}
    same, picked = _pick(same, differs)
    cut = same + sorted(picked, key=_ord)
    if sig is not None:
        gate["n_after_gate"] = len(cut)
    return {"candidates": same + differs, "cut": cut,
            "n_same": len(same), "n_differs_available": len(differs),
            "n_differs_registered": len(picked), "n_registered_differs_cfg": n_diff, **gate}


def _agreement_entries(run_dir: Path, cfg: PipelineConfig, sig: dict | None = None) -> tuple:
    """`shared_input_agreement::{src target}::{dst target}::c{concept}::
    f{dst feature}::{verdict}`. The spec's `{src}::{dst}::{verdict}` id
    under-specifies the claim (one target pair carries many concepts), so
    concept and destination feature are part of it. Source set, destination
    set and `k` are frozen here."""
    from ..sae.ablation_run import checkpoint_path

    ranking = _agreement_candidates(run_dir, cfg.concepts, sig)
    hasher = _Hasher(run_dir)
    entries = []
    for t in ranking["cut"]:
        paths = [run_dir / "sae" / "shared_input_agreement.json",
                 run_dir / "sae" / "atlas_transfer.json",
                 checkpoint_path(run_dir, *t["src_target"].split("/", 1)),
                 checkpoint_path(run_dir, *t["dst_target"].split("/", 1))]
        arts = hasher.many(paths)
        stats = {k: {f: (t.get(k) or {}).get(f) for f in
                     ("observed", "floor_p95_src", "floor_p95_dst", "floor_p05_src",
                      "floor_p05_dst", "clears", "below_floor")}
                 for k in ("statistic_i", "statistic_ii")}
        entries.append({
            "id": _agreement_id(t),
            "stage": "shared_input_agreement", "family": "shared_input_agreement",
            "statistic": "shared_input_causal_agreement",
            "dev_verdict": t["verdict"], "concept": int(t["concept"]),
            "src_target": t["src_target"], "src_model": t["src_model"],
            "src_features": [int(f) for f in t["src_features"]],
            "dst_target": t["dst_target"], "dst_model": t["dst_model"],
            "dst_feature": int(t["dst_feature"]),
            "dst_features": [int(f) for f in t["dst_features"]],
            "dst_set_kind": t.get("dst_set_kind"), "k_top_series": int(t["k_top_series"]),
            "dev_statistics": stats,
            "ablation_null": _ablation_null_mode(cfg),
            "artifact": "sae/shared_input_agreement.json",
            "artifact_sha256": arts["sae/shared_input_agreement.json"], "artifacts": arts,
            "statement": (f"Concept {t['concept']}: {t['src_model']} ({t['src_target']}) and "
                         f"{t['dst_model']} ({t['dst_target']}, feature {t['dst_feature']}) "
                         f"'{t['verdict']}' on their shared top series."),
            "replicable": True,
        })
    summary = {k: v for k, v in ranking.items() if k not in ("candidates", "cut")}
    summary.update({"n_candidates": len(ranking["candidates"]), "cut": len(entries)})
    return entries, summary


_STRUCTURE_TARGET_GATE_REASON = (
    "not applied: the structure claims are panel-level aggregates, not claims about "
    "any one target's features. (b) is a rate over ALL scorable dev candidates at every "
    "measured target, so restricting it to targets selected for having MORE clears than "
    "chance would condition on the outcome and bias the causally-null rate downward; (a) "
    "is a claim about the frozen pooled atlas as a whole, and dropping a target's features "
    "from that pool would change the claim, not test it. Both stay as registered; the "
    "per-target table is still computed (and a missing `empirical_chance` still refuses) "
    "because the gate is a property of the whole registration.")


def _structure_candidates(run_dir: Path, cfg: PipelineConfig,
                          sig: dict | None = None) -> tuple:
    """The two registered directional aggregates (sec 38.2.2 item 4), each
    registered only when it HOLDS on dev (registering a claim dev already
    contradicts would spend the look on nothing):

    (a) `concept_structure::no_concept_in_all_models` -- no atlas concept has
        causal members in every model of the pooled panel. Rule-based (no
        p-value): confirmed if the atlas recomputed on the private battery
        vectors of the SAME dev causal pool also has none.
    (b) `concept_structure::majority_prominent_features_causally_null` --
        among the dev ablation candidates (activation-prominent by
        construction), more than half clear no channel (dev 64.4%, FINDINGS
        MN-14). Confirmed if the private rate's one-sided lower 95% bound is
        above 0.5; carries a bootstrap p, so it is the family's only
        p-valued claim.

    (c) is exploratory and not registered (`_STRUCTURE_C_REASON`).

    The per-target significance gate (`sig`) is deliberately NOT applied to
    these claims; the reasoning is `_STRUCTURE_TARGET_GATE_REASON`, recorded
    in the notes when the gate is on.
    """
    from ..sae.ablation_run import ablation_path, checkpoint_path

    atlas_p = run_dir / "sae" / "concept_atlas.json"
    targets = _dev_ablation_targets(run_dir)
    hasher = _Hasher(run_dir)
    entries, notes = [], {"exploratory_not_registered": _STRUCTURE_C_REASON}
    if sig is not None:
        notes["target_gate"] = _STRUCTURE_TARGET_GATE_REASON

    if atlas_p.exists():
        atlas = load_json(atlas_p)
        rows = [{"model": str(r["model"]), "layer": str(r["layer"]),
                 "feature": int(r["feature"])} for r in atlas.get("rows") or []]
        models = sorted({r["model"] for r in rows})
        max_models = max((int(c.get("n_models") or 0) for c in atlas.get("concepts") or []),
                         default=0)
        if len(models) >= 2 and max_models < len(models):
            params = atlas.get("params") or {}
            tpaths = []
            for t in sorted({(r["model"], r["layer"]) for r in rows}):
                tpaths += [ablation_path(run_dir, *t), checkpoint_path(run_dir, *t)]
            arts = hasher.many([atlas_p] + tpaths)
            entries.append({
                "id": "concept_structure::no_concept_in_all_models", "stage": "concept_structure",
                "family": "concept_structure", "statistic": "atlas_models_per_concept",
                "p_valued": False, "dev_models": models, "dev_max_models_per_concept": max_models,
                "pool": rows, "min_cosine": float(params.get("min_cosine", 0.9)),
                "min_members": int(params.get("min_members", 3)),
                "artifact": "sae/concept_atlas.json",
                "artifact_sha256": arts["sae/concept_atlas.json"], "artifacts": arts,
                "statement": (f"No atlas concept has causal members in all {len(models)} "
                             f"models (dev maximum {max_models})."),
                "replicable": True})
        else:
            notes["no_concept_in_all_models"] = (
                f"not registered: it does not hold on dev (models in pool {len(models)}, "
                f"maximum models per concept {max_models})")
    else:
        notes["no_concept_in_all_models"] = "not registered: sae/concept_atlas.json not found"

    n_scorable = n_null = 0
    per_target = {}
    for model, layer, art, path in targets:
        feats = [int(c["feature"]) for c in art.get("candidates") or [] if c.get("scorable")]
        nulls = sum(1 for c in art.get("candidates") or []
                    if c.get("scorable") and not c.get("n_channels_clearing"))
        if feats:
            per_target[f"{model}/{layer}"] = feats
        n_scorable += len(feats)
        n_null += nulls
    if n_scorable and n_null / n_scorable > 0.5:
        tpaths = []
        for model, layer, _art, path in targets:
            tpaths += [path, checkpoint_path(run_dir, model, layer)]
        arts = hasher.many(tpaths)
        entries.append({
            "id": "concept_structure::majority_prominent_features_causally_null",
            "stage": "concept_structure", "family": "concept_structure",
            "statistic": "causally_null_rate", "p_valued": True,
            "dev_n_features": n_scorable, "dev_n_null": n_null, "dev_rate": n_null / n_scorable,
            "candidates": per_target,
            "artifact": _rel(run_dir, targets[0][3]),
            "artifact_sha256": arts[_rel(run_dir, targets[0][3])], "artifacts": arts,
            "statement": (f"More than half of the activation-prominent dev features are "
                         f"causally null (dev {n_null}/{n_scorable} = "
                         f"{n_null / n_scorable:.3f})."),
            "replicable": True})
    else:
        notes["majority_prominent_features_causally_null"] = (
            "not registered: no scorable dev candidates" if not n_scorable else
            f"not registered: the dev null rate {n_null / n_scorable:.3f} is not above 0.5")
    return entries, notes


def _concept_claim_entries(run_dir: Path, cfg: PipelineConfig) -> tuple:
    """All K2 entries, or `([], {})` when `confirm.register_concept_claims` is
    off (the byte-identical default)."""
    if not bool(getattr(cfg.confirm, "register_concept_claims", False)):
        if bool(getattr(cfg.confirm, "register_requires_target_significance", False)):
            raise ValueError(
                "confirm.register_requires_target_significance is on but "
                "confirm.register_concept_claims is off, so no concept claim would be "
                "registered and the gate would silently do nothing; turn on "
                "register_concept_claims or turn the gate off")
        return [], {}
    sig = _target_gate_table(run_dir, cfg)
    causal, causal_sum = _concept_causal_entries(run_dir, cfg, sig)
    atlas, atlas_sum = _atlas_candidates(run_dir, cfg, sig)
    agree, agree_sum = _agreement_entries(run_dir, cfg, sig)
    struct, struct_notes = _structure_candidates(run_dir, cfg, sig)
    summary = {
        "concept_causal": causal_sum, "concept_atlas": atlas_sum,
        "shared_input_agreement": agree_sum, "concept_structure": struct_notes,
        "not_registered": {"families_beat_shuffle_null": _NOT_REGISTERED_FAMILIES_REASON},
        "ablation_null": _ablation_null_mode(cfg)}
    if sig is not None:
        summary["target_significance"] = sig
    return causal + atlas + agree + struct, summary


# One Holm family per claim type. `n` is the size of the null each family's
# p-values are resolved against, so the smallest attainable Holm-adjusted p
# is `m / (n + 1)`; a family is unsatisfiable once that exceeds alpha.
_FAMILY_NULL_FIELD = {"concept_causal": "causal_max_null", "concept_atlas": "atlas_n_null",
                      "shared_input_agreement": "agreement_n_null",
                      "concept_structure": "structure_n_boot"}


def claim_family_budget(cfg: PipelineConfig, registry: dict | None = None) -> list:
    """`[{"family", "m", "n_null", "alpha", "min_attainable_p_holm",
    "satisfiable", "basis"}]` for the four K2 claim families.

    With a registry, `m` is the number of registered claims of that type
    (only the p-valued ones for `concept_structure`). Without one (a
    `--doctor` run before `register`), `m` is the config's own upper bound
    where the config has one (`concepts.n_registered_causal`), and families
    whose count is a property of the dev artifacts are reported as
    `m: None` with `basis: "unknown before register"` rather than guessed
    (`CLAUDE.md` sec 11.34). One function feeds both `confirm`'s refusal and
    the doctor row, so the two cannot disagree.
    """
    alpha = float(cfg.confirm.alpha)
    rows = []
    for family, field_name in _FAMILY_NULL_FIELD.items():
        n_null = int(getattr(cfg.confirm, field_name))
        if registry is not None:
            hyps = [h for h in registry["hypotheses"] if h["stage"] == family]
            if family == "concept_structure":
                hyps = [h for h in hyps if h.get("p_valued")]
            m, basis = len(hyps), "registered claims"
        elif family == "concept_causal":
            m, basis = int(cfg.concepts.n_registered_causal), "upper bound: n_registered_causal"
        elif family == "concept_structure":
            m, basis = 1, "at most one p-valued structure claim"
        else:
            m, basis = None, "unknown before register"
        floor = (m / (n_null + 1)) if m else None
        rows.append({"family": family, "m": m, "n_null": n_null, "alpha": alpha,
                     "min_attainable_p_holm": floor,
                     "satisfiable": (floor <= alpha) if floor is not None else None,
                     "basis": basis})
    return rows


def build_registry(cfg: PipelineConfig) -> dict:
    """Pure assembly (no I/O beyond reading already-written dev artifacts)."""
    run_dir = cfg.run_dir()
    concept_transfer_entries, ct_ranking = _concept_transfer_entries(run_dir, cfg)
    causal_entries, causal_summary = _concept_claim_entries(run_dir, cfg)
    hypotheses = (_l0_entries(run_dir) + _l1_entries(run_dir) + _l2_entries(run_dir)
                 + _l3_entries(run_dir) + _clustering_entries(run_dir)
                 + concept_transfer_entries + causal_entries)
    ids = [h["id"] for h in hypotheses]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"hypothesis registry has duplicate ids {dup}; every claim "
                         f"must be individually addressable (Holm is keyed by id)")
    registry = {"hypotheses": hypotheses,
               "n_replicable": sum(1 for h in hypotheses if h["replicable"]),
               "concept_transfer_candidates": ct_ranking,
               "concept_knob_candidates": {"candidates": [], "cut": 0, "n_registered": 0,
                                           "reason": _KNOB_FAMILY_EMPTY_REASON}}
    if causal_summary:
        registry["concept_claim_candidates"] = causal_summary
    return registry


def run_register(cfg: PipelineConfig) -> None:
    """Write `hypotheses.json` at the run root (sibling to `config_resolved.yaml`)."""
    registry = build_registry(cfg)
    save_json(cfg.run_dir() / "hypotheses.json", registry)
    log.info("register: %d hypotheses registered (%d replicable) from dev artifacts",
             len(registry["hypotheses"]), registry["n_replicable"])


def check_registry_freshness(cfg: PipelineConfig, registry: dict) -> None:
    """Raise loudly, naming the drifted artifact(s), if any registered hash no longer matches.

    This is the actual enforcement mechanism behind `CLAUDE.md` §6.7's
    discipline: re-running a dev stage (even accidentally) after
    registration changes the artifact `confirm` would otherwise silently
    re-read, which is exactly the "repeated peeking" that discipline exists
    to prevent (sec 15 A15 fix item 5).
    """
    run_dir = cfg.run_dir()
    drifted = []
    for h in registry["hypotheses"]:
        pinned = {h["artifact"]: h["artifact_sha256"]}
        pinned.update(h.get("artifacts") or {})
        for rel, registered in pinned.items():
            art_path = run_dir / rel
            if not art_path.exists():
                drifted.append(f"{rel} (no longer exists)")
                continue
            current = _sha256_file(art_path)
            if current != registered:
                drifted.append(f"{rel} (hash changed: registered "
                               f"{registered[:12]}, now {current[:12]})")
    if drifted:
        raise RuntimeError(
            "confirm refuses to run: hypotheses.json was registered against artifacts that "
            "have since changed -- " + "; ".join(sorted(set(drifted))) + ". This means a dev "
            "stage ran again after registration, exactly the repeated-peeking discipline "
            "confirm exists to prevent (CLAUDE.md sec 6.7, ROADMAP.md sec 15 A15). Re-run "
            "the 'register' stage to re-register against the current artifacts:\n  "
            "--stages register,confirm,report --force register,confirm,report\n"
            "-- understanding that this starts a fresh pre-registration, not a "
            "continuation of the old one -- or restore the original dev artifacts. "
            "(--force register ALONE does nothing: force bypasses a SELECTED stage's "
            "skip, so 'register' has to be in --stages too.)")
