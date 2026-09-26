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

Not every stage's statistic is replicated by `confirm` yet (only `l0`
family strengths and `l1`'s peak-CKA pair, matching the replication
`confirm.py` already implemented before this module existed). L2's
stitching gain, L3's fingerprint agreement, and L4's clustering AMI are
still registered here (so they count in the multiplicity ledger and their
own hash is pinned), but marked `replicable: False` with a stated reason —
a real, stated gap, not a silent omission (`CLAUDE.md` §2.5).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

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


def _concept_transfer_entries(run_dir: Path, cfg: PipelineConfig) -> tuple:
    """`-> (hypothesis entries, ranking dict)`. One entry per candidate in
    the top `cut`, each carrying everything `_replicate_registered_concepts`
    needs to re-test the FROZEN claim on private data without re-reading
    `sae/concept_atlas.json` (registration precedes private access; freezing
    `src_features`/`dst_feature` here, not re-deriving them at confirm time,
    is the mechanism, ROADMAP.md sec 37.10 design item 4).
    """
    ranking = _concept_transfer_candidates(run_dir, cfg.concepts)
    at_path = run_dir / "sae" / "atlas_transfer.json"
    art_hash = _sha256_file(at_path) if at_path.exists() else None
    entries = []
    for c in ranking["candidates"][:ranking["cut"]]:
        entries.append({
            "id": f"concept_transfer::{c['src_target']}::{c['concept']}::{c['dst_model']}",
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


def build_registry(cfg: PipelineConfig) -> dict:
    """Pure assembly (no I/O beyond reading already-written dev artifacts)."""
    run_dir = cfg.run_dir()
    concept_transfer_entries, ct_ranking = _concept_transfer_entries(run_dir, cfg)
    hypotheses = (_l0_entries(run_dir) + _l1_entries(run_dir) + _l2_entries(run_dir)
                 + _l3_entries(run_dir) + _clustering_entries(run_dir)
                 + concept_transfer_entries)
    return {"hypotheses": hypotheses,
           "n_replicable": sum(1 for h in hypotheses if h["replicable"]),
           "concept_transfer_candidates": ct_ranking,
           "concept_knob_candidates": {"candidates": [], "cut": 0, "n_registered": 0,
                                       "reason": _KNOB_FAMILY_EMPTY_REASON}}


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
        art_path = run_dir / h["artifact"]
        if not art_path.exists():
            drifted.append(f"{h['artifact']} (no longer exists)")
            continue
        current = _sha256_file(art_path)
        if current != h["artifact_sha256"]:
            drifted.append(f"{h['artifact']} (hash changed: registered "
                           f"{h['artifact_sha256'][:12]}, now {current[:12]})")
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
