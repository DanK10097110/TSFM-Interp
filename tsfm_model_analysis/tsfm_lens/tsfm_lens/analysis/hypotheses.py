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


def build_registry(cfg: PipelineConfig) -> dict:
    """Pure assembly (no I/O beyond reading already-written dev artifacts)."""
    run_dir = cfg.run_dir()
    hypotheses = (_l0_entries(run_dir) + _l1_entries(run_dir) + _l2_entries(run_dir)
                 + _l3_entries(run_dir) + _clustering_entries(run_dir))
    return {"hypotheses": hypotheses,
           "n_replicable": sum(1 for h in hypotheses if h["replicable"])}


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
