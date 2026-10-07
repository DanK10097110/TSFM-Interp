"""Confirmation on a held-out private benchmark — the pipeline's gold standard.

Everything upstream (L0-L4) is exploratory analysis on the development
corpus: many comparisons were looked at, so its significant results are
hypotheses, not conclusions. This stage tests those hypotheses exactly once
on a sealed private corpus the models and the analysis never touched:

1. Behavioral hypotheses — each dev family strength is re-tested on private
   series with a paired bootstrap and Holm correction across hypotheses,
   plus the overall paired comparison.
2. Representational spot-check — window-level CKA at the dev best pair is
   recomputed on private contexts with a cluster-bootstrap CI, checking that
   the geometric result replicates out of sample.
3. Concept replication (ROADMAP.md sec 37.10 P7) — each registered
   `concept_transfer` claim is re-tested on private data:
   `_replicate_registered_concepts` captures private activations at the
   claim's own frozen (src, dst) targets, encodes them with the SAME saved
   SAE checkpoint dev trained (never retrained), and reruns the
   stratum-matched transfer test with `sae/transfer.py`'s own
   `matched_draws`/`transfer_one` on PRIVATE strata. The `concept_knob`
   family is registered empty (sec 37.9's P6a found nothing to freeze) and
   is recorded that way, never silently dropped.

4. U1 reliability replication (ROADMAP.md sec 38.4, opt-in): each registered
   `reliability_u1` claim REFITS the frozen K4 procedure on PRIVATE series
   (`_confirm_reliability_u1`): frozen SAE checkpoints (never retrained),
   frozen family definitions, the same folds, ridge grid and seed, asking
   whether the series-bootstrap gain CI lower bound of baseline+internals
   over the free baseline is > 0, at Holm across the family. A frozen dev
   model scored on private data would add domain shift to the question,
   which is why it is a refit.

Discipline matters more than machinery here: run this once, at the end.
Repeated peeking consumes the private benchmark (regenerate a fresh epoch
via tsfm_benchmark if that happens). The stage refuses to overwrite existing
confirmation artifacts unless explicitly forced, and by default refuses
unverifiable private corpora.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from ..config import PipelineConfig
from ..data import BenchmarkData, load_benchmark
from ..extraction.alignment import align, pooling_matrix
from ..extraction.hooks import ActivationCatcher
from ..utils import batch_slices, load_json, log, save_json
from .corpus_card import audit_state, read_cross_split, read_manifest
from .hypotheses import check_registry_freshness
from .l0_behavioral import _predict_all, _score
from .l1_geometry import linear_cka
from .power import mde_paired_bootstrap
from .stats import bootstrap_ci, holm, paired_bootstrap
from .hypotheses import CONCEPT_CLAIM_STAGES, RELIABILITY_STAGE, V3_CLAIM_STAGES

# Every claim stage that needs the shared private capture and the K2 pass:
# the four K2 types plus the two V3-B types (ROADMAP.md sec 41).
_PRIVATE_PASS_STAGES = CONCEPT_CLAIM_STAGES + V3_CLAIM_STAGES

# Corpus identity (ROADMAP.md sec 34 item B5): a manifest-derived field
# degrades to this reason, never a bare `None`, when the confirm.source is
# `smoke` or an unverified jsonl -- both skip `load_sealed` entirely, so
# there is no manifest.json to read (CLAUDE.md sec 2.5, sec 6.7's own
# "smoke-mode confirm is mechanically real but distributionally identical to
# dev" warning, which this is what makes visible per-run rather than only
# documented).
_NO_MANIFEST_REASON = "no sealed manifest (smoke source, or an unverified jsonl load)"


def confirmation_complete(cfg: PipelineConfig) -> bool:
    """The `confirm` stage's skip predicate: `confirmation.json` exists AND, when
    `confirm.external_path` is set, it already holds `external_replication`.
    A confirmation made before an external corpus was configured is therefore
    not complete, and re-running the stage adds ONLY the external leg (the
    private split is not looked at again: `run_confirm`'s `external_only`)."""
    path = cfg.run_dir() / "confirm" / "confirmation.json"
    if not path.exists():
        return False
    if not getattr(cfg.confirm, "external_path", ""):
        return True
    return "external_replication" in load_json(path)


def run_confirm(cfg: PipelineConfig, hub, forced: bool = False) -> None:
    """Load the private corpus, re-test exactly the registered hypotheses, spot-check the CKA peak.

    Requires `hypotheses.json` (the `register` stage, which runs
    automatically as this stage's dependency) and refuses if any artifact
    it was registered against has since changed (`ROADMAP.md` sec 15 A15) --
    the actual enforcement behind `CLAUDE.md` §6.7's "everything on dev is
    exploratory, confirm tests it exactly once" discipline, previously
    enforced by convention alone.

    Also refuses (`ROADMAP.md` sec 34 item B5) if `confirm.path`'s own
    manifest declares a visibility other than `private` -- a second, earlier
    enforcement point for the same discipline, since a confirmation run
    against the public split would otherwise complete and look identical to
    a real one. `confirmation.json` records which corpus (digest, epoch,
    visibility, B1's audit block) this run actually consumed.
    """
    out_dir = cfg.run_dir() / "confirm"
    repeated = (out_dir / "confirmation.json").exists()
    external_wanted = bool(getattr(cfg.confirm, "external_path", ""))
    prior = load_json(out_dir / "confirmation.json") if repeated else {}
    external_done = "external_replication" in prior
    external_only = bool(repeated and not forced and external_wanted and not external_done)
    if repeated and not external_only:
        if not forced:
            raise RuntimeError(
                "confirmation artifacts already exist; the private benchmark is meant to be "
                "consumed once. Rerun with --force confirm only if you understand that this "
                "constitutes a second look (and consider a fresh private epoch).")
        # `--force confirm` was passed deliberately. Proceed, but say so at
        # WARNING: the resource being spent is the *independence* of the
        # verdicts, and nothing downstream can tell a first look from a
        # second one once the artifact is overwritten.
        log.warning(
            "CONFIRM: overwriting existing confirmation artifacts because --force confirm "
            "was passed. This is a SECOND look at the private benchmark -- the verdicts "
            "below are no longer a one-shot confirmation. Regenerate a fresh private epoch "
            "before treating them as such.")
    registry_path = cfg.run_dir() / "hypotheses.json"
    if not registry_path.exists():
        raise RuntimeError(
            "confirm requires a hypothesis registry (ROADMAP.md sec 15 A15) -- no "
            "hypotheses.json found at the run root. It should have run automatically as "
            "confirm's dependency; if you selected stages explicitly with --stages, include "
            "'register'.")
    registry = load_json(registry_path)
    check_registry_freshness(cfg, registry)
    # ROADMAP.md sec 38.2: refuse an unsatisfiable Holm family (or a null-mode
    # mismatch) BEFORE the private split is opened. A no-op when no K2 claim
    # was registered.
    if any(h["stage"] in _PRIVATE_PASS_STAGES or h["stage"] == RELIABILITY_STAGE
           for h in registry["hypotheses"]):
        check_concept_claims_before_opening(cfg, registry)

    log.warning("CONFIRM: running the one-shot private-benchmark confirmation; "
                "avoid re-running against the same private epoch")
    out_dir.mkdir(parents=True, exist_ok=True)

    private = _load_private(cfg)
    dev_manifest, private_manifest = _read_provenance_manifests(cfg)
    # Checked before any private-corpus analysis runs, not after (sec 34
    # B5.2) -- a public/private mixup should fail before spending compute
    # on the very evidence its own discipline would then be invalidating.
    _check_private_provenance(private_manifest, dev_manifest)
    provenance = _private_provenance(private, cfg.confirm.path, dev_manifest, private_manifest)
    external = (_prepare_external(cfg, private, dev_manifest, private_manifest)
                if external_wanted else None)
    if external_only:
        block = _external_leg(cfg, hub, registry, external, repeated_look=False)
        save_json(out_dir / "confirmation.json", {**prior, "external_replication": block})
        log.info("confirm: external_real leg added to the existing confirmation "
                 "(not counted in the confirm verdict)")
        return
    a, b = cfg.comparison_pair()
    metrics = _private_behavioral(cfg, hub, private, out_dir)
    hypotheses = _test_registered_hypotheses(cfg, metrics, registry, a.name, b.name)
    replication = _replicate_registered_cka(cfg, hub, private, registry)
    l3_replication = _replicate_registered_l3(cfg, hub, private, registry)
    k2_claims = any(h["stage"] in _PRIVATE_PASS_STAGES for h in registry["hypotheses"])
    acts_cache, capture_errors = ({}, {})
    if k2_claims:
        acts_cache, capture_errors = _capture_private_targets(
            cfg, hub, private, sorted(set(_needed_targets(registry)) | _transfer_targets(registry)))
    concept_replication = _replicate_registered_concepts(cfg, hub, private, registry,
                                                         acts_cache=acts_cache or None)
    if k2_claims:
        concept_replication = _replicate_causal_concept_claims(
            cfg, hub, private, registry, concept_replication, acts_cache, capture_errors)
    if any(h["stage"] == RELIABILITY_STAGE for h in registry["hypotheses"]):
        concept_replication = _replicate_reliability_u1(
            cfg, hub, private, registry, concept_replication, acts_cache, capture_errors)

    confirmed = sum(1 for h in hypotheses["tests"] if h["confirmed"])
    n_replicable = sum(1 for h in registry["hypotheses"] if h["replicable"])
    external_block = ({"external_replication": _external_leg(
        cfg, hub, registry, external, repeated_look=bool(forced and external_done))}
        if external is not None else {})
    save_json(out_dir / "confirmation.json", {
        "model_a": a.name, "model_b": b.name,
        "n_private_series": private.n,
        "alpha": cfg.confirm.alpha,
        "n_registered": len(registry["hypotheses"]),
        "n_replicable": n_replicable,
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        # False for the one-shot confirmation this stage is designed around;
        # True when --force confirm overwrote an earlier one. Recorded rather
        # than only logged, so a reader of the artifact (or the report) can
        # tell the two apart after the log has scrolled away.
        "repeated_look": bool(forced and repeated),
        **hypotheses,
        **provenance,
        "cka_replication": replication,
        "l3_replication": l3_replication,
        "concept_replication": concept_replication,
        **external_block,
    })
    log.info("confirm complete: %d/%d dev hypotheses confirmed on private data "
            "(%d registered, %d replicable)", confirmed, len(hypotheses["tests"]),
            len(registry["hypotheses"]), n_replicable)


def _load_private(cfg: PipelineConfig) -> BenchmarkData:
    """Load the private corpus with seal verification enforced by default."""
    if cfg.confirm.source == "sealed" and cfg.confirm.require_seal:
        try:
            import tsfm_benchmark  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "confirm.require_seal is true but tsfm_benchmark is not installed; "
                "private results without seal verification are not trustworthy. "
                "Install it or explicitly set confirm.require_seal: false.") from e
    data_cfg = dataclasses.replace(cfg.data, source=cfg.confirm.source,
                                   path=cfg.confirm.path,
                                   max_series=cfg.confirm.max_series)
    seed = cfg.run.seed + 1000 if cfg.confirm.source == "smoke" else cfg.run.seed
    private = load_benchmark(data_cfg, seed)
    log.info("confirm: private corpus loaded (%d series, %d families)",
             private.n, private.meta["family"].nunique())
    return private


_EXTERNAL_ROLE = "external_real"
_EXTERNAL_EVIDENCE = ("external validity of the registered transfer and family-presence claims "
                      "on raw real windows, reported apart from the confirm verdict: it is not "
                      "registered-hypothesis confirmation and is never counted in its totals")


def _uncapped_hashes(cfg: PipelineConfig, source: str, path: str, seed: int) -> set:
    """Value hashes of EVERY series in one corpus (no `max_series` cap), the
    fallback identity when a sealed manifest lists no sample hashes."""
    from .family_claims import series_value_hashes

    data = load_benchmark(dataclasses.replace(cfg.data, source=source, path=path,
                                              max_series=None), seed)
    return series_value_hashes(data.contexts(), data.targets())


def _prepare_external(cfg: PipelineConfig, private: BenchmarkData, dev_manifest,
                      private_manifest) -> dict:
    """Load the `external_real` corpus and prove it is disjoint from dev and
    private BEFORE any analysis runs (ROADMAP.md sec 41, V3-B item c).

    Refuses (raises) when the external path is the dev or private path, when
    its manifest declares a role other than `external_real`, or when it shares
    a sample hash with either consumed corpus. Hashes are the sealed
    manifests' per-sample content hashes when all three corpora list them,
    otherwise a sha256 of every series' values (exact match only: a re-cropped
    copy of a series is not caught, and the artifact records which method was
    used). The external corpus is loaded with seal verification under the same
    rule as the private one."""
    from .family_claims import check_external_disjoint, corpus_hashes

    c = cfg.confirm
    ext_path = str(c.external_path)
    for name, other in (("confirm.path", c.path), ("data.path", cfg.data.path)):
        if other and Path(other).resolve() == Path(ext_path).resolve():
            raise RuntimeError(f"confirm.external_path is the same corpus as {name} "
                               f"({other}); the external leg needs a corpus no other stage "
                               f"has read")
    if c.external_source == "sealed" and c.require_seal:
        try:
            import tsfm_benchmark  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "confirm.require_seal is true but tsfm_benchmark is not installed; an "
                "external corpus without seal verification is not trustworthy.") from e
    ext_manifest = read_manifest(ext_path) if c.external_source != "smoke" else None
    declared = ((ext_manifest or {}).get("role")
                or ((ext_manifest or {}).get("extra") or {}).get("role"))
    if declared is not None and declared != _EXTERNAL_ROLE:
        raise RuntimeError(f"confirm.external_path's manifest declares role {declared!r}, not "
                           f"{_EXTERNAL_ROLE!r}")
    seed = cfg.run.seed + 2000 if c.external_source == "smoke" else cfg.run.seed
    ext = load_benchmark(dataclasses.replace(cfg.data, source=c.external_source, path=ext_path,
                                             max_series=int(c.external_max_series)), seed)
    manifests = {"dev": dev_manifest, "private": private_manifest, "external": ext_manifest}
    if all((m or {}).get("sample_hashes") for m in manifests.values()):
        hashed = {k: corpus_hashes(m, None) for k, m in manifests.items()}
    else:
        priv_seed = cfg.run.seed + 1000 if c.source == "smoke" else cfg.run.seed
        hashed = {
            "dev": (_uncapped_hashes(cfg, cfg.data.source, cfg.data.path, cfg.run.seed),
                    "series value sha256"),
            "private": (_uncapped_hashes(cfg, c.source, c.path, priv_seed),
                        "series value sha256"),
            "external": (_uncapped_hashes(cfg, c.external_source, ext_path, seed),
                         "series value sha256")}
    overlap = check_external_disjoint(hashed["external"],
                                      {"dev": hashed["dev"], "private": hashed["private"]})
    log.info("confirm: external_real corpus loaded (%d series), disjoint from dev and private "
             "by %s", ext.n, overlap["method"])
    return {"data": ext, "manifest": ext_manifest, "overlap": overlap,
            "role_declared": declared}


def _external_leg(cfg: PipelineConfig, hub, registry: dict, external: dict,
                  repeated_look: bool) -> dict:
    """The `external_replication` block: the registered `concept_transfer` and
    `family_presence` claims re-run on the external corpus, each with its own
    Holm family over ITS tests, none of it entering the confirm verdict, the
    registered-claim totals or the ledger (ROADMAP.md sec 41, V3-B item c).

    Everything is the private pass's own machinery on a different `data`:
    frozen SAE checkpoints, frozen claims, the claim's registered null. The
    ground-truth table is `""` (raw real windows have no recipe), so the
    seasonal channel is unavailable here for every feature, which the vector
    convention already treats as an unscored channel. A claim type with no
    registered claim is recorded `skipped`, not absent."""
    ext: BenchmarkData = external["data"]
    run_dir = cfg.run_dir()
    transfer_hyps = [h for h in registry["hypotheses"] if h["stage"] == "concept_transfer"]
    family_hyps = _concept_claims(registry, "family_presence")
    targets = sorted(_transfer_targets(registry)
                     | {t for h in family_hyps for t in h["targets"]})
    acts, errors = ({}, {})
    if targets:
        acts, errors = _capture_private_targets(cfg, hub, ext, targets)
    if transfer_hyps:
        transfer = _replicate_registered_concepts(cfg, hub, ext, registry,
                                                  acts_cache=acts or None)["transfer"]
    else:
        transfer = {"status": "skipped", "tests": [],
                    "reason": "no registered concept_transfer hypotheses"}
    if family_hyps:
        feats, ferrors = _encode_frozen_targets(run_dir, acts, errors, _device_for(cfg))
        family = _confirm_family_presence(
            cfg, registry, ferrors,
            _family_presence_batteries(cfg, hub, ext, run_dir, family_hyps, feats, gt_path=""),
            tag="external")
    else:
        family = {"status": "skipped", "tests": [],
                  "reason": "no registered family_presence claims"}
    return {
        "status": "tested" if (transfer_hyps or family_hyps) else "skipped",
        "role": _EXTERNAL_ROLE, "counted_in_confirm_verdict": False,
        "evidence_class": _EXTERNAL_EVIDENCE, "repeated_look": bool(repeated_look),
        "corpus_path": str(cfg.confirm.external_path),
        "corpus_digest": getattr(ext, "corpus_digest", None), "n_series": int(ext.n),
        "manifest_role_declared": external["role_declared"],
        "overlap_check": external["overlap"],
        "transfer": transfer, "family_presence": family,
    }


def _read_provenance_manifests(cfg: PipelineConfig) -> tuple:
    """One cheap JSON read per side -- shared by `_private_provenance` (which
    renders them) and `_check_private_provenance` (which gates on them), so
    neither has to re-read the other's copy."""
    dev_manifest = read_manifest(cfg.data.path) if cfg.data.source != "smoke" else None
    private_manifest = read_manifest(cfg.confirm.path) if cfg.confirm.source != "smoke" else None
    return dev_manifest, private_manifest


def _private_provenance(private: BenchmarkData, confirm_path: str,
                        dev_manifest: dict, private_manifest: dict) -> dict:
    """B5.1: which corpus (and which of possibly several private epochs,
    `CLAUDE.md` sec 4.5) this confirmation actually consumed.

    `private.corpus_digest` already exists upstream (sec 15 A7) and was
    simply dropped here -- that is the whole gap this closes for the digest
    field. `manifest.json`'s other fields (`visibility`, `epoch`, and B1's
    `extra.audit` block) are not carried on `BenchmarkData` at all, so they
    come from `_read_provenance_manifests`'s direct read via
    `corpus_card.read_manifest` -- the identical cheap, no-reverification
    read `corpus_card.py` already does for the dev corpus (reused rather
    than duplicated, `CLAUDE.md` sec 2.2/11.24), not a second reload of the
    dev corpus's full row set just to reach two scalar fields off its
    manifest.

    `private_composition` (counts by tier/family) is NOT manifest-derived --
    it comes straight from `private.meta`, which is already in memory
    regardless of whether a manifest exists -- so it is always populated,
    unlike the manifest-derived fields below it, which share one explicit
    `manifest_reason` when there is no manifest to read (the failure mode
    B5's own text names: "all fields null with reason: 'no sealed
    manifest'"). `exchangeability` (ROADMAP.md sec 34 item B3.4) comes from
    `corpus_card.read_cross_split`, which reads an optional, plain sibling
    `cross_split.json` next to the private corpus (written by
    `benchmark_validation`'s `run_validation.py --compare-splits ...
    --persist-into-private`, entirely outside this pipeline) -- degrading
    to an explicit, actionable reason (naming the exact CLI invocation)
    rather than the earlier placeholder's bare "not yet implemented", since
    B3 has been implemented since this field was first written and the
    honest gap is now "not yet run for this corpus", not "does not exist".
    """
    has_manifest = private_manifest is not None or dev_manifest is not None
    meta = private.meta
    return {
        "private_corpus_path": confirm_path,
        "private_corpus_digest": private.corpus_digest,
        "private_manifest_epoch": (private_manifest or {}).get("epoch"),
        "private_visibility": (private_manifest or {}).get("visibility"),
        "private_composition": {
            "by_tier": {str(k): int(v) for k, v in meta["tier"].value_counts().items()},
            "by_family": {str(k): int(v) for k, v in meta["family"].value_counts().items()},
        },
        "dev_corpus_digest": (dev_manifest or {}).get("global_digest"),
        "dev_manifest_epoch": (dev_manifest or {}).get("epoch"),
        "private_audit": audit_state(private_manifest),
        "exchangeability": read_cross_split(confirm_path, private.corpus_digest),
        # Not itself one of B5.1's named fields, but required by its own
        # failure-mode text: the five manifest-derived scalars above render
        # as bare `None` with no way to tell "checked, absent" apart from
        # "never checked" unless this is read alongside them.
        "private_manifest_reason": None if has_manifest else _NO_MANIFEST_REASON,
    }


def _check_private_provenance(private_manifest: dict, dev_manifest: dict) -> None:
    """B5.2's checks -- one that must fail loudly, one that must only warn,
    and (by its absence here) one that must never run at all.

    Visibility is checked whenever a manifest was actually read, regardless
    of `confirm.require_seal` -- `read_manifest` needs no `tsfm_benchmark`
    import, so this catches a public/private mixup even when seal
    verification itself was explicitly disabled. Epoch drift across sec
    4.5's several possible private epochs is legitimate and only logged.

    Deliberately absent: any comparison of `private_corpus_digest` to
    `dev_corpus_digest`. They are different splits and MUST differ; an
    equality check here would fire on every correct run (sec 11.35's false-
    refusal shape) -- ROADMAP.md sec 34 B5.2 names this as the check to NOT
    write, not merely one to skip by omission.
    """
    if private_manifest is not None:
        visibility = private_manifest.get("visibility")
        if visibility != "private":
            raise RuntimeError(
                f"confirm.path points at a corpus whose manifest declares "
                f"visibility={visibility!r}, not 'private'. Running the one-shot "
                f"confirmation stage against a non-private corpus silently invalidates "
                f"CLAUDE.md sec 6.7's exploration-vs-confirmation discipline -- point "
                f"confirm.path at the sealed private split.")
    if private_manifest is not None and dev_manifest is not None:
        priv_epoch, dev_epoch = private_manifest.get("epoch"), dev_manifest.get("epoch")
        if priv_epoch != dev_epoch:
            log.warning(
                "CONFIRM: private corpus epoch (%r) differs from dev corpus epoch (%r) -- "
                "legitimate under CLAUDE.md sec 4.5's regeneration protocol, but the reader "
                "should know this confirmation is not against the same epoch dev findings "
                "were explored on", priv_epoch, dev_epoch)


def _private_behavioral(cfg: PipelineConfig, hub, private: BenchmarkData,
                        out_dir) -> pd.DataFrame:
    """Score both models on the private corpus with the same L0 metric definitions."""
    contexts, targets = private.contexts(), private.targets()
    frames = []
    for mcfg in cfg.comparison_pair():
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        point, quants = _predict_all(adapter, contexts, private.horizon, cfg.l0.quantiles)
        frames.append(_score(mcfg.name, point, quants, contexts, targets,
                             cfg.l0.quantiles, private.meta))
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_parquet(out_dir / "behavioral.parquet")
    return metrics


def _test_registered_hypotheses(cfg: PipelineConfig, metrics: pd.DataFrame, registry: dict,
                                name_a: str, name_b: str) -> dict:
    """Re-test exactly the registered, replicable `l0` family claims on private series.

    Sourced from `hypotheses.json`, not a fresh re-read of `l0/summary.json`
    (`ROADMAP.md` sec 15 A15) -- `check_registry_freshness` has already
    proven that file matches what was registered, so reading it here for
    the descriptive `dev_ratio` display value is safe, but *which* claims
    get tested comes from the registry, not from re-deriving the claim list.

    Records `mde_private` for every claim -- the private-n minimum detectable
    effect, computed before that claim's own test runs (`ROADMAP.md` §34
    A1.3) -- alongside, never as a gate: a hypothesis whose private-n MDE
    exceeds its own dev effect size is one this confirmation cannot confirm
    even in principle, and the reader is entitled to that fact regardless of
    which way the test itself comes out. `min_attainable_p_holm` (§6.6's
    p-floor) is computed once, from the count of claims this private run can
    actually test, and threaded into every claim's MDE call unchanged --
    never re-derived per claim, matching `l0_behavioral.py`'s own discipline.
    """
    sc = cfg.stats
    dev = load_json(cfg.run_dir() / "l0" / "summary.json")
    dev_ratio = dev.get("mase_ratio", {})
    claims = [(h["family"], h["favored"]) for h in registry["hypotheses"]
             if h["stage"] == "l0" and h["replicable"]]

    wide = metrics.pivot_table(index=["series_id", "family"], columns="model",
                               values="mase").reset_index()
    groups = [(fam, favored, wide[wide["family"] == fam]) for fam, favored in claims]
    m_testable = sum(1 for _, _, grp in groups if len(grp) >= sc.min_series)
    min_attainable_p_holm = (m_testable / sc.n_boot) if m_testable and sc.n_boot else None

    tests, pvals = [], {}
    for fam, favored, grp in groups:
        entry = {"family": fam, "dev_favored": favored,
                 "dev_ratio": dev_ratio.get(fam)}
        sign = 1.0 if favored == name_a else -1.0
        diff = sign * (grp[name_b] - grp[name_a]).to_numpy()
        entry["mde_private"] = mde_paired_bootstrap(
            diff, alpha=cfg.confirm.alpha, n_boot=sc.n_boot,
            seed=cfg.run.seed + 900_000 + len(tests),
            min_attainable_p_holm=min_attainable_p_holm, min_n=sc.min_series)
        if len(grp) < sc.min_series:
            entry.update({"status": "untestable",
                          "reason": f"only {len(grp)} private series"})
            tests.append(entry)
            continue
        res = paired_bootstrap(diff, sc.n_boot, cfg.run.seed + 200 + len(tests), sc.ci)
        entry.update({"status": "tested", **res})
        tests.append(entry)
        pvals[fam] = res["p"]
    adjusted = holm(pvals) if pvals else {}
    for entry in tests:
        if entry["status"] == "tested":
            entry["p_holm"] = adjusted[entry["family"]]
            entry["confirmed"] = bool(entry["p_holm"] < cfg.confirm.alpha
                                      and entry["lo"] > 0)
        else:
            entry["confirmed"] = False
    overall = paired_bootstrap((wide[name_b] - wide[name_a]).to_numpy(),
                               sc.n_boot, cfg.run.seed + 299, sc.ci)
    return {"tests": tests, "overall": overall,
            "overall_direction": f"positive favors {name_a}"}


def _clean_and_corrupted_fingerprint(cfg: PipelineConfig, adapter, contexts: np.ndarray,
                                     corrupted: dict, names: list) -> np.ndarray:
    """Per-series, per-layer relative activation deltas on private contexts.

    The same quantity `l3_perturbation._sensitivity` computes on dev, minus
    its one dependency this stage cannot satisfy: that function reads clean
    activations from the extraction store, which exists only for the dev
    corpus. Here the clean pass is simply run, so nothing else about the
    measurement changes.
    """
    from ..extraction.hooks import ActivationCatcher as _Catcher

    # The dev run screened this model at its own configured capture stride;
    # matching it here keeps the two fingerprints the same shape, which the
    # depth-profile comparison requires.
    layers = adapter.all_layer_names()[:: max(1, adapter.cfg.capture_layer_stride)]
    pool = pooling_matrix(adapter.token_time_spans(), cfg.data.context_len,
                          cfg.alignment.window)

    def _pass(ctx: np.ndarray) -> dict:
        acc = {layer: [] for layer in layers}
        with _Catcher(adapter.module, layers) as catcher:
            for s, e in batch_slices(len(ctx), adapter.cfg.batch_size):
                with torch.no_grad(), torch.autocast(device_type=adapter.device.type,
                                                     dtype=adapter.dtype,
                                                     enabled=adapter.device.type == "cuda"):
                    adapter.forward(adapter.prepare(ctx[s:e]))
                got = catcher.collect()
                for layer in layers:
                    acc[layer].append(
                        align(adapter.postprocess_tokens(layer, got[layer]).float(),
                              pool).cpu().numpy())
        return {layer: np.concatenate(v) for layer, v in acc.items()}

    clean = _pass(contexts)
    per_series = np.zeros((len(contexts), len(layers), len(names)), dtype=np.float32)
    for ci, cname in enumerate(names):
        corr = _pass(corrupted[cname])
        for li, layer in enumerate(layers):
            num = np.linalg.norm(corr[layer] - clean[layer], axis=-1)
            den = np.linalg.norm(clean[layer], axis=-1) + 1e-6
            per_series[:, li, ci] = (num / den).mean(axis=1)
    return per_series, layers


def _replicate_registered_l3(cfg: PipelineConfig, hub, private: BenchmarkData,
                             registry: dict) -> dict:
    """Re-run the corruption battery on private series and re-test each
    registered per-corruption fingerprint agreement.

    Exactly the dev computation on different data -- the same corruptions at
    the same configured strengths, the same relative-activation-delta
    fingerprint, the same `align_on_axis` depth matching and the same paired
    cluster bootstrap. A dev rho is called replicated when it falls inside
    the private CI, the same rule `_replicate_registered_cka` uses, so the
    two replications are read the same way.
    """
    from .depth_axis import depth_axis_for_run
    from .l3_perturbation import (CORRUPTIONS, _agreement_with_ci)

    hyps = [h for h in registry["hypotheses"] if h["stage"] == "l3"]
    if not hyps:
        return {"status": "skipped", "reason": "no registered L3 hypothesis"}
    meta_path = cfg.run_dir() / "l3" / "meta.json"
    if not meta_path.exists():
        return {"status": "skipped", "reason": "no dev L3 artifact to replicate"}
    dev = load_json(meta_path)
    names = [n for n in dev["corruptions"] if n in CORRUPTIONS]
    a, b = cfg.comparison_pair()
    if b is None:
        return {"status": "skipped", "reason": "solo run: agreement needs two models"}

    cap = min(cfg.l3.max_series, private.n)
    rows = np.arange(cap)
    contexts = private.contexts()[rows]
    corrupted = {n: CORRUPTIONS[n](contexts.copy(),
                                   np.random.default_rng(cfg.run.seed + 300 + i),
                                   **cfg.l3.corruptions.get(n, {}))
                 for i, n in enumerate(names)}

    per_series, depths = {}, {}
    for mcfg in (a, b):
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        ps, layers = _clean_and_corrupted_fingerprint(cfg, adapter, contexts,
                                                      corrupted, names)
        per_series[mcfg.name] = ps
        depths[mcfg.name] = depth_axis_for_run(cfg.alignment.depth_axis, None,
                                               mcfg.name, layers,
                                               adapter=adapter).coords
        if not cfg.run.keep_models_loaded:
            hub.release(mcfg.name)

    got = _agreement_with_ci(cfg, per_series[a.name], per_series[b.name], names,
                             depths[a.name], depths[b.name])
    dev_per = ((dev.get("agreement") or {}).get("per_corruption") or {})
    tests = []
    for h in hyps:
        cname = h.get("corruption")
        if cname not in got["per_corruption"] or cname not in dev_per:
            continue
        priv = got["per_corruption"][cname]
        dev_rho = dev_per[cname]["value"]
        if priv.get("lo") is None or priv.get("hi") is None or dev_rho is None:
            tests.append({"corruption": cname, "dev_rho": dev_rho, "private": priv,
                          "replicates": None,
                          "reason": "a correlation is not defined (constant or "
                                    "non-finite depth profile), so it is neither "
                                    "replicated nor refuted"})
            continue
        tests.append({"corruption": cname, "dev_rho": dev_rho, "private": priv,
                      "replicates": bool(priv["lo"] <= dev_rho <= priv["hi"])})
    return {"status": "tested", "model_a": a.name, "model_b": b.name,
            "n_private_series": int(cap), "overall": got["overall"], "tests": tests}


def _replicate_registered_cka(cfg: PipelineConfig, hub, private: BenchmarkData,
                              registry: dict) -> dict:
    """Recompute the registered dev peak-pair CKA on private contexts with a cluster-bootstrap CI."""
    l1_hyp = next((h for h in registry["hypotheses"]
                   if h["stage"] == "l1" and h["replicable"]), None)
    if l1_hyp is None:
        return {"status": "skipped", "reason": "no registered, replicable L1 hypothesis"}
    l1_path = cfg.run_dir() / "l1" / "meta.json"
    best = load_json(l1_path)["best_pair"]
    a, b = cfg.comparison_pair()
    acts = {}
    for mcfg, layer in ((a, best["layer_a"]), (b, best["layer_b"])):
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        pool = pooling_matrix(adapter.token_time_spans(), cfg.data.context_len,
                              cfg.alignment.window)
        chunks = []
        with ActivationCatcher(adapter.module, [layer]) as catcher:
            for s, e in batch_slices(private.n, adapter.cfg.batch_size):
                with torch.no_grad(), torch.autocast(device_type=adapter.device.type,
                                                     dtype=adapter.dtype,
                                                     enabled=adapter.device.type == "cuda"):
                    adapter.forward(adapter.prepare(private.contexts()[s:e]))
                hidden = adapter.postprocess_tokens(layer, catcher.collect()[layer])
                chunks.append(align(hidden.float(), pool).cpu())
        acts[mcfg.name] = torch.cat(chunks)
        if not cfg.run.keep_models_loaded:
            hub.release(mcfg.name)

    xa, xb = acts[a.name], acts[b.name]
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    xa, xb = xa.to(device), xb.to(device)

    def stat(idx: np.ndarray) -> float:
        sel = torch.from_numpy(idx).to(device)
        return linear_cka(xa[sel].reshape(-1, xa.shape[-1]),
                          xb[sel].reshape(-1, xb.shape[-1]))

    ci = bootstrap_ci(stat, xa.shape[0],
                      min(cfg.stats.n_boot, cfg.stats.n_boot_heavy),
                      cfg.run.seed + 210, cfg.stats.ci)
    return {"status": "tested", "layer_a": best["layer_a"], "layer_b": best["layer_b"],
            "dev_cka": best["cka"], "private": ci,
            "replicates": bool(ci["lo"] <= best["cka"] <= ci["hi"])}


# ---------------------------------------------------------------------------
# ROADMAP.md sec 37.10 P7 -- concept_transfer replication on private strata.
# ---------------------------------------------------------------------------

_KNOB_FAMILY_LEDGER_NOTE = (
    "empty by design (sec 37.10 P7 decision item 3): sec 37.9's P6a found no "
    "(concept, knob) response surviving BH on dev (0/98 go/no-go, 0/151 "
    "registered retry), so no dev claim exists to freeze and no private "
    "counterfactual path is built.")


def _capture_private_window_acts(cfg: PipelineConfig, hub, private: BenchmarkData,
                                 model: str, layer: str) -> torch.Tensor:
    """Window-level `[n_private, n_windows, dim]` activations at one target,
    captured on PRIVATE contexts with autocast ON (`cfg.run.dtype`/CUDA) --
    the store's own numeric regime, exactly `_replicate_registered_cka`'s
    capture loop, stopped one step short of its own final series-level
    reduction (concept scoring needs the SAE encoded FIRST, mean-pooled
    over windows SECOND -- `extraction/store.py::write_sae_batch`'s own
    order, mirrored here rather than pooling raw activations before
    encoding, sec 6.2.1 Stage 3d)."""
    adapter = hub.get(model)
    adapter.ensure_loaded()
    pool = pooling_matrix(adapter.token_time_spans(), cfg.data.context_len,
                          cfg.alignment.window)
    chunks = []
    with ActivationCatcher(adapter.module, [layer]) as catcher:
        for s, e in batch_slices(private.n, adapter.cfg.batch_size):
            with torch.no_grad(), torch.autocast(device_type=adapter.device.type,
                                                 dtype=adapter.dtype,
                                                 enabled=adapter.device.type == "cuda"):
                adapter.forward(adapter.prepare(private.contexts()[s:e]))
            hidden = adapter.postprocess_tokens(layer, catcher.collect()[layer])
            chunks.append(align(hidden.float(), pool).cpu())
    acts = torch.cat(chunks)
    if not cfg.run.keep_models_loaded:
        hub.release(model)
    return acts


def _encode_series_pooled(run_dir, model: str, layer: str, acts: torch.Tensor,
                          device: torch.device) -> np.ndarray:
    """Encode window-level activations with the SAVED (never retrained) SAE
    checkpoint and mean-pool over windows -- `sae/train.py::
    encode_and_persist_features`'s own two-step order, replicated on
    private data rather than reading `store.load(..., space="sae")`, which
    only ever holds DEV features. `float64` output: `sae/transfer.py::
    auc_from_ranks` refuses anything narrower (float16 rank quantization,
    CLAUDE.md sec 8)."""
    from ..sae.train import load_sae_checkpoint, sanitize

    ckpt_path = run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}.pt"
    sae = load_sae_checkpoint(str(ckpt_path)).to(device)
    sae.eval()
    b, w, d = acts.shape
    flat = acts.reshape(-1, d).to(device)
    with torch.no_grad():
        feats = sae.encode(flat).cpu().numpy().astype(np.float64).reshape(b, w, -1)
    return feats.mean(axis=1)


def _transfer_targets(registry: dict) -> set:
    """Every target a registered `concept_transfer` claim touches, so the one
    shared private capture (`_capture_private_targets`) also serves it."""
    out = set()
    for h in registry["hypotheses"]:
        if h["stage"] == "concept_transfer":
            out |= {h["src_target"], h["dst_target"]}
    return out


def _replicate_registered_concepts(cfg: PipelineConfig, hub, private: BenchmarkData,
                                   registry: dict, acts_cache: dict | None = None) -> dict:
    """Re-test every registered `concept_transfer` claim on private strata,
    and record the (empty) `concept_knob` family in the same ledger.

    Design choice, stated because it deliberately differs from
    `_replicate_registered_cka`'s fixed-pair convention: unlike a peak-CKA
    pair (chosen by maximizing over every layer x layer combination across
    the whole architecture, a search this function does NOT repeat -- it
    reuses `best_pair` verbatim), a transfer test's "forward leg" is BY
    DEFINITION a search over the destination model's WHOLE dictionary
    (`sae/transfer.py`'s own docstring: "best feature in the destination's
    whole dictionary by AUC on S"), and that search is already corrected by
    `transfer_one`'s own max-over-features null. Re-running `transfer_one`
    verbatim on private data -- letting it re-derive the best destination
    feature via argmax rather than freezing dev's `dst_feature` -- therefore
    replicates the SAME test definition on new data, exactly as
    `_replicate_registered_l3` reruns the whole corruption battery fresh and
    `_test_registered_hypotheses` reruns a fresh paired bootstrap: what is
    frozen is WHICH claim to look at (`src_features`, `dst_target`, the
    concept id), never a value that claim's own defined procedure computes.
    The dev `dst_feature` is still recorded on each test row (`dev_dst_
    feature`) beside the private `feature` so a reader can see whether the
    same feature won again.

    Two families, each its own ledger row (`CLAUDE.md` sec 6.5): the
    `concept_transfer` family (Holm across every registered claim's
    intersection-union p, `p_combined = max(p, rev_p)`, since a claim
    requires BOTH legs to hold -- the standard combination for an AND
    claim, Berger & Hsu 1996) and the `concept_knob` family, always empty
    here (sec 37.9's P6a NO-GO; no private counterfactual path is built).

    ROADMAP.md sec 37.10 P7b -- a registered entry's `mode` (defaulting to
    `"search"` for every pre-P7b claim) selects which claim gets tested here:
    `"search"` runs the paragraph above unchanged; `"frozen"` calls
    `transfer_one_fixed_feature` with the FROZEN `dst_feature` instead of
    `transfer_one`, so the private forward leg scores exactly that one
    feature against its OWN single-feature null (no argmax, no max-over-
    features null) -- the sharper claim "this SPECIFIC dev feature pair
    selects the same series", as opposed to "the destination dictionary has
    SOME feature that does". The combination rule (`max(p, rev_p)`, Holm
    across the family) and the two-family ledger shape are IDENTICAL for
    both modes.
    """
    transfer_hyps = [h for h in registry["hypotheses"] if h["stage"] == "concept_transfer"]
    ids = [h["id"] for h in transfer_hyps]
    if len(set(ids)) != len(ids):
        raise ValueError("concept_transfer claims have duplicate ids; Holm is keyed by "
                         "id, so the family would silently shrink. Re-register with a "
                         "registry built by the current hypotheses.py")
    knob_hyps = [h for h in registry["hypotheses"] if h["stage"] == "concept_knob"]

    if knob_hyps:  # pragma: no cover -- defensive; P7 never registers this family
        knob_block = {"status": "unsupported", "n_registered": len(knob_hyps), "tests": [],
                     "reason": "concept_knob claims were registered, but P7 does not build "
                              "the private counterfactual path (sec 37.9's P6a NO-GO); "
                              "these claims cannot be replicated"}
    else:
        knob_block = {"status": "empty", "n_registered": 0, "tests": [],
                     "reason": _KNOB_FAMILY_LEDGER_NOTE}
    ledger = [{"family": "concept_knob", "m": len(knob_hyps), "n_null": None,
              "min_attainable_p_holm": None, "satisfiable": None,
              "note": _KNOB_FAMILY_LEDGER_NOTE}]

    if not transfer_hyps:
        return {"status": "skipped", "reason": "no registered concept_transfer hypotheses",
               "transfer": {"status": "skipped", "tests": []}, "knob": knob_block,
               "ledger": ledger}

    from ..sae.transfer import _by_stratum, _seed, concept_scores, matched_draws, \
        series_strata, top_series, transfer_one, transfer_one_fixed_feature

    n_null = int(getattr(cfg.confirm, "concept_transfer_n_null", 2000) or 2000)
    base_seed = cfg.run.seed + 700_000
    run_dir = cfg.run_dir()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    targets = sorted({h["src_target"] for h in transfer_hyps}
                     | {h["dst_target"] for h in transfer_hyps})
    pooled: dict = {}
    capture_errors: dict = {}
    for target in targets:
        model, layer = target.split("/", 1)
        try:
            acts = (acts_cache[target] if acts_cache and target in acts_cache
                    else _capture_private_window_acts(cfg, hub, private, model, layer))
            pooled[target] = _encode_series_pooled(run_dir, model, layer, acts, device)
        except Exception as exc:  # capture/encode failure -> every dependent claim
            capture_errors[target] = f"{type(exc).__name__}: {exc}"  # "not replicable", loudly
            log.warning("confirm concept replication: capturing/encoding private "
                       "activations at %s failed (%s); every registered claim touching "
                       "this target is recorded as not replicable", target,
                       capture_errors[target])

    strata = series_strata(private.meta)
    by_stratum = _by_stratum(strata)

    tests, pvals = [], {}
    for h in transfer_hyps:
        mode = h.get("mode", "search")
        entry = {"id": h["id"], "concept": h["concept"], "src_target": h["src_target"],
                 "dst_target": h["dst_target"], "dst_model": h["dst_model"],
                 "dev_auc": h["dev_auc"], "dev_auc_margin": h["dev_auc_margin"],
                 "dev_dst_feature": h["dst_feature"]}
        if mode == "frozen":
            # Additive-only (`CLAUDE.md` sec 7 invariant 13): a "search"-mode
            # entry (the common case, and every existing caller/test) gets
            # none of these keys, so its record stays byte-identical.
            entry["mode"] = "frozen"
            entry["dev_p"] = h.get("dev_p")
        broken = capture_errors.get(h["src_target"]) or capture_errors.get(h["dst_target"])
        if broken:
            entry.update({"status": "not_replicable",
                         "reason": f"private activation capture/encode failed: {broken}"})
            tests.append(entry)
            continue
        try:
            src_pooled, dst_pooled = pooled[h["src_target"]], pooled[h["dst_target"]]
            score_src = concept_scores(src_pooled, h["src_features"])
            k = min(int(h.get("k_top_series") or 20), max(1, src_pooled.shape[0] - 1))
            S = top_series(score_src, k)
            dst_ranks = rankdata(dst_pooled, axis=0)
            fwd_seed = _seed("confirm_concept", h["id"], base=base_seed)
            fwd_draws = matched_draws(S, strata, by_stratum, n_null,
                                      np.random.default_rng(fwd_seed))
            rev_seed = _seed("confirm_concept", h["id"], "rev", base=base_seed)
            if mode == "frozen":
                # ROADMAP.md sec 37.10 P7b -- no search: the dev-frozen
                # `dst_feature` is scored against a SINGLE-FEATURE
                # stratum-matched null, never a max over the destination
                # dictionary (`test_frozen_null_is_not_max_over_features`).
                result = transfer_one_fixed_feature(
                    score_src, dst_ranks, S, fwd_draws, strata, by_stratum,
                    feature=int(h["dst_feature"]), k=k, n_draws=n_null, seed=rev_seed,
                    p_method="exact", fwd_seed=fwd_seed)
            else:
                result = transfer_one(score_src, dst_ranks, S, fwd_draws, strata, by_stratum,
                                      k=k, n_draws=n_null, seed=rev_seed, p_method="exact",
                                      fwd_seed=fwd_seed)
        except ValueError as exc:
            entry.update({"status": "not_replicable",
                         "reason": f"private strata could not build a matched null: {exc}"})
            tests.append(entry)
            continue
        p_combined = max(result["p"], result["rev_p"])
        entry.update({
            "status": "tested", "k_used": k, "n_null": n_null,
            "private_auc": result["auc"], "private_null_p95": result["null_p95"],
            "private_feature": result["feature"], "private_p": result["p"],
            "private_rev_auc": result["rev_auc"], "private_rev_null_p95": result["rev_null_p95"],
            "private_rev_p": result["rev_p"], "private_reciprocal": result["reciprocal"],
            "p_combined": p_combined,
        })
        tests.append(entry)
        pvals[h["id"]] = p_combined

    m = sum(1 for t in tests if t["status"] == "tested")
    min_attainable_p_holm = (m / (n_null + 1)) if m else None
    satisfiable = bool(m and min_attainable_p_holm is not None
                       and min_attainable_p_holm <= cfg.confirm.alpha)
    adjusted = holm(pvals) if pvals else {}
    n_confirmed = 0
    for entry in tests:
        if entry["status"] != "tested":
            entry["confirmed"] = False
            entry["verdict"] = "not replicable"
            continue
        entry["p_holm"] = adjusted[entry["id"]]
        entry["confirmed"] = bool(entry["p_holm"] < cfg.confirm.alpha
                                  and entry["private_reciprocal"])
        entry["verdict"] = "confirmed" if entry["confirmed"] else "not confirmed"
        n_confirmed += int(entry["confirmed"])

    ledger.insert(0, {"family": "concept_transfer", "m": m, "n_null": n_null,
                      "alpha": cfg.confirm.alpha,
                      "min_attainable_p_holm": min_attainable_p_holm,
                      "satisfiable": satisfiable})
    transfer_block = {"status": "tested", "n_registered": len(transfer_hyps),
                      "n_tested": m, "n_confirmed": n_confirmed,
                      "p_combination": "max(p, rev_p) per claim (intersection-union), "
                                      "then Holm across claims",
                      "n_null": n_null, "tests": tests}
    return {"status": "tested", "transfer": transfer_block, "knob": knob_block,
           "ledger": ledger}


# ---------------------------------------------------------------------------
# ROADMAP.md sec 38.2 (K2) -- replication of the CAUSAL concept claims.
#
# Everything is frozen from the registry (dev SAE checkpoints, dev feature
# ids, dev channel and sign); nothing is retrained or re-searched. One private
# activation capture per model covers every target any claim needs. The
# battery is the dev battery (`sae/ablation_run.py::run_ablation_target`,
# `sae/shared_input_agreement.py::run_shared_input_agreement`) called with
# the private corpus as `data`, so the ablation null, the reach probe and the
# seeding (`cfg.run.seed`-derived, the SAME seed for every compared
# `predict()` pair, `CLAUDE.md` sec 11.50) come from the code dev used.
# Three states, never two: `confirmed`, `not confirmed`, and `not testable`
# (reach probe failed on private, the feature fires on no private series, a
# side is not scorable) -- a withheld claim is never counted as a failure
# (`CLAUDE.md` sec 11.37).
# ---------------------------------------------------------------------------

_NOT_TESTABLE = "not testable"


def _device_for(cfg: PipelineConfig) -> torch.device:
    from ..utils import resolve_device
    return resolve_device(cfg.run.device)


def _private_ground_truth_path(cfg: PipelineConfig) -> str:
    """The corpus whose sealed manifest supplies PRIVATE ground-truth periods.
    A non-sealed confirm source has no manifest; the empty string makes the
    lookup fail loudly into `periods = NaN` (seasonal channel unavailable)
    instead of falling back to the DEV table, whose ids are not the private
    series'."""
    return cfg.confirm.path if cfg.confirm.source == "sealed" else ""


def _concept_claims(registry: dict, stage: str) -> list:
    return [h for h in registry["hypotheses"] if h["stage"] == stage]


def _needed_targets(registry: dict) -> list:
    """Every `model/layer` any K2 claim touches, sorted."""
    out = set()
    for h in _concept_claims(registry, "concept_causal"):
        out.add(h["target"])
    for h in _concept_claims(registry, "concept_atlas"):
        out |= {f"{m['model']}/{m['layer']}" for m in h["members"]}
    for h in _concept_claims(registry, "shared_input_agreement"):
        out |= {h["src_target"], h["dst_target"]}
    for h in _concept_claims(registry, "concept_structure"):
        out |= {f"{r['model']}/{r['layer']}" for r in h.get("pool", [])}
        out |= set(h.get("candidates", {}))
    for h in _concept_claims(registry, "concept_atlas_centroid"):
        out |= {f"{m['model']}/{m['layer']}" for m in h["members"]}
    for h in _concept_claims(registry, "family_presence"):
        out |= set(h["targets"])
    return sorted(out)


def check_concept_claims_before_opening(cfg: PipelineConfig, registry: dict) -> list:
    """Refuse, BEFORE the private split is opened, if any registered K2 claim
    family cannot reach `confirm.alpha` after Holm (its smallest attainable
    adjusted p, `m / (n_null + 1)`, already exceeds alpha) or if a registered
    causal claim was frozen under a different ablation null than the one this
    config would run. Returns the (satisfiable) family ledger rows.

    An unsatisfiable family would spend the one private look on tests whose
    non-results are arithmetic, not evidence (`CLAUDE.md` sec 6.5); the
    matching `--doctor` row reads the same function
    (`analysis/hypotheses.py::claim_family_budget`).
    """
    from .hypotheses import claim_family_budget

    problems = []
    rows = [r for r in claim_family_budget(cfg, registry) if r["m"]]
    for r in rows:
        if not r["satisfiable"]:
            problems.append(
                f"family '{r['family']}': {r['m']} claim(s) at n_null={r['n_null']} cannot "
                f"reach alpha={r['alpha']} after Holm (smallest attainable adjusted p "
                f"{r['min_attainable_p_holm']:.4f}); raise the family's null size or "
                f"register fewer claims")
    mode = str(getattr(cfg.sae, "ablation_null", "mean_magnitude") or "mean_magnitude")
    frozen = {h.get("ablation_null") for h in _concept_claims(registry, "concept_causal")}
    frozen |= {h["ablation_null"] for h in _concept_claims(registry, "shared_input_agreement")
               if "ablation_null" in h}
    if frozen and frozen != {mode}:
        problems.append(
            f"registered causal and agreement claims were frozen under ablation null "
            f"{sorted(frozen)} but "
            f"sae.ablation_null is {mode!r}; the private battery must use the null dev used")
    if problems:
        raise RuntimeError(
            "confirm refuses to open the private split: " + "; ".join(problems)
            + " (ROADMAP.md sec 38.2.3). Nothing was read from the private corpus.")
    return rows


def _capture_private_window_acts_multi(cfg: PipelineConfig, hub, private: BenchmarkData,
                                       model: str, layers: list) -> dict:
    """`{layer: [n_private, n_windows, dim]}` window activations of ONE model
    at several layers from a SINGLE forward sweep, in the same numeric regime
    as `_capture_private_window_acts` (autocast ON, same pooling)."""
    adapter = hub.get(model)
    adapter.ensure_loaded()
    pool = pooling_matrix(adapter.token_time_spans(), cfg.data.context_len,
                          cfg.alignment.window)
    chunks = {layer: [] for layer in layers}
    with ActivationCatcher(adapter.module, list(layers)) as catcher:
        for s, e in batch_slices(private.n, adapter.cfg.batch_size):
            with torch.no_grad(), torch.autocast(device_type=adapter.device.type,
                                                 dtype=adapter.dtype,
                                                 enabled=adapter.device.type == "cuda"):
                adapter.forward(adapter.prepare(private.contexts()[s:e]))
            got = catcher.collect()
            for layer in layers:
                hidden = adapter.postprocess_tokens(layer, got[layer])
                chunks[layer].append(align(hidden.float(), pool).cpu())
    if not cfg.run.keep_models_loaded:
        hub.release(model)
    return {layer: torch.cat(v) for layer, v in chunks.items()}


def _capture_private_targets(cfg: PipelineConfig, hub, private: BenchmarkData,
                             targets: list) -> tuple:
    """`(acts, errors)`: `acts[target]` window activations for every target
    and `errors[target]` the failure text for one whose capture raised.
    Grouped by model so each model is swept once for all of its layers -- the
    single private cache every K2 claim type reads."""
    by_model: dict = {}
    for t in targets:
        model, layer = t.split("/", 1)
        by_model.setdefault(model, []).append(layer)
    acts, errors = {}, {}
    for model, layers in sorted(by_model.items()):
        try:
            got = _capture_private_window_acts_multi(cfg, hub, private, model, layers)
            acts.update({f"{model}/{layer}": a for layer, a in got.items()})
        except Exception as exc:  # noqa: BLE001 -- every dependent claim becomes not testable, loudly
            for layer in layers:
                errors[f"{model}/{layer}"] = f"{type(exc).__name__}: {exc}"
            log.warning("confirm concept claims: private capture for %s failed (%s); every "
                        "claim touching it is recorded as not testable", model, exc)
    return acts, errors


def _frozen_features(run_dir, model: str, layer: str, acts: torch.Tensor,
                     device: torch.device) -> dict:
    """Encode private window activations with the FROZEN dev checkpoint
    (`ablation_run.checkpoint_path`, never retrained): `series` is the
    series-level feature matrix the ablation battery selects rows with (the
    dev store's own order of operations: mean-pool windows, THEN encode),
    `pooled` is encode-then-pool in float64 (what transfer and agreement
    score), `alive` the frozen dictionary's alive atoms on private windows."""
    from ..sae.ablation_run import checkpoint_path
    from ..sae.response import alive_feature_mask
    from ..sae.train import load_sae_checkpoint

    sae = load_sae_checkpoint(str(checkpoint_path(run_dir, model, layer))).to(device)
    sae.eval()
    b, w, d = acts.shape
    with torch.no_grad():
        series = sae.encode(acts.mean(dim=1).float().to(device)).cpu().numpy()
        flat = sae.encode(acts.reshape(-1, d).float().to(device)).cpu().numpy()
    pooled = flat.astype(np.float64).reshape(b, w, -1).mean(axis=1)
    alive = alive_feature_mask(sae, acts.reshape(-1, d).float().numpy(), device)
    return {"series": series, "pooled": pooled, "alive": alive}


def _tail_p(obs: float, null_vals, tail: str = "right") -> float:
    """`(1 + #{null >= obs}) / (n + 1)` (right) or `(1 + #{null <= obs}) /
    (n + 1)` (left): floored at `1/(n+1)`, never 0 (`CLAUDE.md` sec 6.5)."""
    v = np.asarray(list(null_vals), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    hits = int(np.sum(v >= obs)) if tail == "right" else int(np.sum(v <= obs))
    return (1 + hits) / (v.size + 1)


def _cfg_with_null(cfg, null_mode: str | None):
    """`cfg` with `sae.ablation_null` set to `null_mode` (a shallow copy; the
    run's own config is never mutated), or `cfg` itself when it already is."""
    from ..sae.ablation_run import cfg_null_mode

    if not null_mode or null_mode == cfg_null_mode(cfg):
        return cfg
    import copy
    out = copy.copy(cfg)
    out.sae = dataclasses.replace(cfg.sae, ablation_null=null_mode)
    return out


def _private_battery(cfg, hub, private, run_dir, model: str, layer: str, feats: dict,
                     candidates: list, k: int, n_null: int, keep_null_draws: bool,
                     signed_null: bool = False, gt_path: str | None = None,
                     null_mode: str | None = None) -> dict:
    """One private battery run at one target with frozen candidates.
    Returns `{"withheld": bool, "reason": str, "by_feature": {f: record}}`.

    ROADMAP.md sec 41 (V3-B) -- three optional arguments, every default the
    legacy behaviour: `signed_null` keeps the signed null draws, `gt_path`
    overrides the ground-truth table's corpus (`""` for a corpus with none, so
    the seasonal channel is unavailable rather than read from dev's table), and
    `null_mode` runs the battery under that ablation null instead of the
    config's."""
    from ..sae.ablation_run import run_ablation_target

    res = run_ablation_target(
        _cfg_with_null(cfg, null_mode), run_dir, hub, private, None, _device_for(cfg), model,
        layer, top_k_series=int(k), n_null_directions=int(n_null),
        max_series=int(cfg.concepts.max_series), keep_forecasts=0,
        ground_truth_path=(_private_ground_truth_path(cfg) if gt_path is None else gt_path),
        candidates=[{"feature": int(f)} for f in candidates],
        activations=feats["series"], keep_null_draws=keep_null_draws,
        keep_signed_null_draws=signed_null)
    if res.get("skipped") or res.get("withheld"):
        reason = res.get("reason") or (res.get("reach") or {}).get("reason") or "withheld"
        return {"withheld": True, "reason": str(reason), "by_feature": {}}
    return {"withheld": False, "reason": "",
            "by_feature": {int(c["feature"]): c for c in res.get("candidates") or []}}


def _holm_confirm(tests: list, pkey: str, alpha: float, extra_ok=lambda t: True) -> tuple:
    """Holm over the `tested` entries' `pkey`; sets `p_holm`, `confirmed`,
    `verdict` on every entry. `not testable` entries stay out of the family
    and are never `not confirmed`. Returns `(m, n_confirmed)`."""
    pv = {t["id"]: t[pkey] for t in tests
          if t["status"] == "tested" and t.get(pkey) is not None and np.isfinite(t[pkey])}
    adj = holm(pv) if pv else {}
    n_conf = 0
    for t in tests:
        if t["status"] != "tested" or t["id"] not in adj:
            t["confirmed"], t["verdict"] = False, _NOT_TESTABLE
            continue
        t["p_holm"] = adj[t["id"]]
        t["confirmed"] = bool(t["p_holm"] < alpha and extra_ok(t))
        t["verdict"] = "confirmed" if t["confirmed"] else "not confirmed"
        n_conf += int(t["confirmed"])
    return len(pv), n_conf


def causal_claim_reading(t: dict, m: int, alpha: float, n_max: int) -> str:
    """Plain-language power reading of one tested causal-concept claim, from
    fields stored in its test entry (`confirmed`, `n_null`, `dev_effect`,
    `mde`) plus the family's Holm size `m`, `alpha` and the adaptive ceiling
    `n_max`. The claim's own MDE is computed from its own null draws, so a
    claim that was not at the p floor (never redrawn, `n_null < n_max`) sees
    `m/(n_null+1) > alpha` and reports `unsatisfiable_correction` even though
    the procedure can reach alpha at `n_max`; that case reads as "not
    detected", and only `m/(n_max+1) > alpha` reads as untestable. A
    confirmed claim never receives a non-confirmation sentence."""
    mde = t.get("mde") or {}
    mde_v = mde.get("mde")
    dev = t.get("dev_effect")
    if t.get("confirmed"):
        if mde_v is not None and dev is not None and dev >= mde_v:
            return "confirmed with power"
        return "confirmed (dev-sized effect below this test's MDE)"
    if mde_v is None:
        reason = mde.get("reason")
        n_used = int(t.get("n_null") or 0)
        if (reason == "unsatisfiable_correction" and n_max > n_used
                and m / (n_max + 1) <= alpha):
            return (f"tested at {n_used} null draws and not at the p floor, so the adaptive "
                    f"redraw to {n_max} could not rescue it: a non-confirmation reads as not "
                    f"detected (power at this effect size not computed)")
        return (f"power not computable ({reason}): read this claim as untestable, "
                f"not absent")
    if dev is not None and dev >= mde_v:
        return ("a dev-sized effect was above this test's MDE: a non-confirmation reads as "
                "absent")
    return ("a dev-sized effect is BELOW this test's MDE: a non-confirmation reads as "
            "underpowered, not absent")


def _confirm_concept_causal(cfg, hub, private, run_dir, registry, feats: dict,
                            errors: dict, batteries: dict) -> dict:
    """Replicate every `concept_causal` claim: on the feature's PRIVATE
    top-k firing series (k as dev), ablate it from the frozen SAE's own
    reconstruction and score its registered channel against the row-matched
    random-direction null. The statistic is the mean |effect| on the
    channel; the p is the exact permutation p against `causal_n_null`
    direction draws, and a claim that sits on the p floor is redrawn with
    `causal_max_null` draws (the adaptive tail p, sec 37.6 P3). The sign must
    also match the dev sign. Confirmed at Holm `alpha` over the tested
    claims of this family."""
    from .power import mde_ablation_effect

    claims = _concept_claims(registry, "concept_causal")
    alpha = float(cfg.confirm.alpha)
    n0, n_max = int(cfg.confirm.causal_n_null), int(cfg.confirm.causal_max_null)
    tests = []
    for h in claims:
        entry = {"id": h["id"], "model": h["model"], "layer": h["layer"],
                 "atlas_concept": h.get("atlas_concept"),
                 "target": h["target"], "feature": h["feature"], "channel": h["channel"],
                 "sign": h["sign"], "dev_effect": h["dev_effect"],
                 "dev_null_p95": h["dev_null_p95"],
                 "dev_effect_over_null_p95": h["dev_effect_over_null_p95"],
                 "ablation_null": h["ablation_null"]}
        b = batteries.get(h["target"])
        if h["target"] in errors:
            entry.update({"status": _NOT_TESTABLE,
                          "reason": f"private capture failed: {errors[h['target']]}"})
        elif b is None or b["withheld"]:
            entry.update({"status": _NOT_TESTABLE, "reason": (
                f"private reach probe withheld this target: {(b or {}).get('reason', '')}")})
        else:
            rec = b["by_feature"].get(int(h["feature"]))
            ch = ((rec or {}).get("channels") or {}).get(h["channel"]) or {}
            if not rec or not rec.get("scorable"):
                entry.update({"status": _NOT_TESTABLE,
                              "reason": "the feature fires on no private series"})
            elif not ch.get("available") or "null_draw_means" not in ch:
                entry.update({"status": _NOT_TESTABLE, "reason": (
                    f"channel {h['channel']!r} unavailable on this feature's private rows: "
                    f"{ch.get('reason', '')}")})
            else:
                obs, draws = float(ch["effect"]), list(ch["null_draw_means"])
                p, hit_floor = _tail_p(obs, draws), not any(d >= obs for d in draws)
                method, n_used, ch_p = "exact", len(draws), ch
                if hit_floor and n_max > n0:
                    big = batteries.get(("adaptive", h["target"], int(h["feature"])))
                    bch = ((big or {}).get("by_feature", {}).get(int(h["feature"])) or {}
                           ).get("channels", {}).get(h["channel"]) or {}
                    if bch.get("available") and "null_draw_means" in bch:
                        obs, draws = float(bch["effect"]), list(bch["null_draw_means"])
                        p, method, n_used, ch_p = _tail_p(obs, draws), "adaptive", len(draws), bch
                signed = float(ch["signed_effect"])
                entry.update({
                    "status": "tested", "private_effect": obs,
                    "private_signed_effect": signed, "private_null_p95": ch.get("null_p95"),
                    "private_n_top_series": rec.get("n_top_series"),
                    "sign_matches_dev": bool(np.sign(signed) == h["sign"]),
                    "p": p, "p_method": method, "n_null": n_used,
                    "clears_null_p95": bool(ch.get("clears_null")),
                    "_power_rows": ch_p.get("row_abs_effects"), "_power_null": draws})
        tests.append(entry)

    m, n_conf = _holm_confirm(tests, "p", alpha, extra_ok=lambda t: t["sign_matches_dev"])
    for t in tests:
        rows, null = t.pop("_power_rows", None), t.pop("_power_null", None)
        if t["status"] != "tested":
            continue
        mde = mde_ablation_effect(rows, null, alpha=alpha, m=max(m, 1),
                                  null_p95=t["private_null_p95"],
                                  seed=cfg.run.seed + 810_000)
        t["mde"] = mde
        t["non_replication_reading"] = causal_claim_reading(t, m, alpha, n_max)
    return {"status": "tested", "n_registered": len(claims), "n_tested": m,
            "n_confirmed": n_conf,
            "n_not_testable": sum(1 for t in tests if t["status"] != "tested"),
            "n_max_null": n_max, "n_initial_null": n0,
            "p_method": "exact permutation p on mean |effect| vs random-direction null "
                        f"({n0} draws); adaptive redraw to {n_max} at the floor",
            "p_combination": "Holm across the tested claims of this family; sign must match dev",
            "tests": tests}


def _cos_matrix_stats(unit: np.ndarray, min_cosine: float) -> tuple:
    iu = np.triu_indices(unit.shape[0], k=1)
    pair = (unit @ unit.T)[iu]
    return float(np.mean(pair >= min_cosine)), pair


def _unit_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1)
    return X / np.where(n == 0, 1.0, n)[:, None]


def _atlas_member_vector(batteries: dict):
    """`f(model, layer, feature) -> private dev-convention 9-vector or None`
    over one `{target: battery}` dict (a withheld target, an unfired feature
    and an unscorable one are all `None`)."""
    from ..sae.concepts import ablation_vector

    def _vec(model, layer, feature):
        b = batteries.get(f"{model}/{layer}")
        if b is None or b["withheld"]:
            return None
        rec = b["by_feature"].get(int(feature))
        return ablation_vector(dict(rec)) if rec and rec.get("scorable") else None
    return _vec


def _private_causal_pool(batteries: dict) -> dict:
    """`{model: unit rows}` of the private causal pool: every scorable frozen
    feature that clears >= 1 channel on private data, the population the atlas
    claims' random member sets are drawn from (row-matched per model)."""
    from ..sae.concepts import ablation_vector

    pool: dict = {}
    for tgt, b in batteries.items():
        if not isinstance(tgt, str) or b["withheld"]:
            continue
        model = tgt.split("/", 1)[0]
        for f, rec in b["by_feature"].items():
            if rec.get("scorable") and rec.get("n_channels_clearing"):
                v = ablation_vector(dict(rec))
                if v is not None and np.linalg.norm(v) > 0:
                    pool.setdefault(model, []).append(v)
    return {m: _unit_rows(np.asarray(v)) for m, v in pool.items()}


def _confirm_atlas(cfg, registry: dict, errors: dict, batteries: dict) -> dict:
    """Replicate every `concept_atlas` claim from the private battery vectors
    of the FROZEN member features (`sae/concepts.py::ablation_vector`, the
    dev vector). Statistics: the fraction of member pairs whose private
    cosine is >= the dev `min_cosine`, and the cosine of the private member
    centroid with the DEV centroid. Confirmed if the centroid cosine >=
    `min_cosine`, at least half of the pairs hold, and both statistics beat
    their null (random same-composition member sets drawn from the private
    causal pool, row-matched per model) at Holm alpha (combined p = the max
    of the two, an intersection-union rule)."""
    claims = _concept_claims(registry, "concept_atlas")
    alpha, n_null = float(cfg.confirm.alpha), int(cfg.confirm.atlas_n_null)
    _vec = _atlas_member_vector(batteries)
    pool = _private_causal_pool(batteries)

    tests = []
    for h in claims:
        entry = {"id": h["id"], "concept": h["concept"], "n_members": len(h["members"]),
                 "models": h["models"], "min_cosine": h["min_cosine"]}
        vecs, kept = [], []
        for m in h["members"]:
            v = _vec(m["model"], m["layer"], m["feature"])
            if v is not None and np.linalg.norm(v) > 0:
                vecs.append(v)
                kept.append(m)
        entry["n_testable_members"] = len(vecs)
        if len(vecs) < max(2, int(h.get("min_members", 3))):
            entry.update({"status": _NOT_TESTABLE, "reason": (
                f"only {len(vecs)} of {len(h['members'])} members are testable on private "
                f"data (reach withheld, unfired, or unscorable)")})
            tests.append(entry)
            continue
        comp: dict = {}
        for m in kept:
            comp[m["model"]] = comp.get(m["model"], 0) + 1
        if any(pool.get(mo) is None or len(pool[mo]) < c for mo, c in comp.items()):
            entry.update({"status": _NOT_TESTABLE, "reason": (
                "the private causal pool of a member's model is smaller than the member "
                "count, so no row-matched null can be drawn")})
            tests.append(entry)
            continue
        dev_centroid = np.asarray(h["dev_centroid"], dtype=np.float64)
        unit = _unit_rows(np.asarray(vecs))
        frac, _pair = _cos_matrix_stats(unit, h["min_cosine"])
        cen = unit.mean(axis=0)
        cen_cos = float(cen @ dev_centroid / (np.linalg.norm(cen) or 1.0))
        rng = np.random.default_rng(cfg.run.seed + 820_000 + int(h["concept"]))
        null_frac, null_cen = [], []
        for _ in range(n_null):
            sets = [pool[mo][rng.choice(len(pool[mo]), size=c, replace=False)]
                    for mo, c in sorted(comp.items())]
            u = np.concatenate(sets)
            f_, _ = _cos_matrix_stats(u, h["min_cosine"])
            c_ = u.mean(axis=0)
            null_frac.append(f_)
            null_cen.append(float(c_ @ dev_centroid / (np.linalg.norm(c_) or 1.0)))
        p_frac, p_cen = _tail_p(frac, null_frac), _tail_p(cen_cos, null_cen)
        entry.update({
            "status": "tested", "private_pair_fraction": frac,
            "private_centroid_cosine": cen_cos, "n_null": n_null,
            "null_pair_fraction_p95": float(np.quantile(null_frac, 0.95)),
            "null_centroid_cosine_p95": float(np.quantile(null_cen, 0.95)),
            "p_pair_fraction": p_frac, "p_centroid": p_cen, "p": max(p_frac, p_cen),
            "rule_holds": bool(cen_cos >= h["min_cosine"] and frac >= 0.5)})
        tests.append(entry)
    m, n_conf = _holm_confirm(tests, "p", alpha, extra_ok=lambda t: t["rule_holds"])
    return {"status": "tested", "n_registered": len(claims), "n_tested": m,
            "n_confirmed": n_conf,
            "n_not_testable": sum(1 for t in tests if t["status"] != "tested"),
            "p_combination": "max(p_pair_fraction, p_centroid) per claim, then Holm; "
                            "confirmed also needs centroid cosine >= min_cosine and >= half "
                            "of member pairs >= min_cosine",
            "n_null": n_null, "tests": tests}


def _confirm_atlas_centroid(cfg, registry: dict, errors: dict, batteries_by_null: dict) -> dict:
    """Replicate every `concept_atlas_centroid` claim (ROADMAP.md sec 41, V3-B):
    the cosine of the private member centroid to the DEV centroid, with NO
    pair-fraction leg. The null is `_confirm_atlas`'s own, unchanged: random
    same-composition member sets drawn from the private causal pool (per-model
    row-matched), seeded identically, so a concept's `p_centroid` here equals
    its legacy `p_centroid`. Confirmed if the centroid cosine >= the frozen
    `min_cosine` and the null p is below alpha at Holm across THIS family.

    `batteries_by_null` is `{null_mode: {target: battery}}`: each claim is
    scored on the battery run under its own registered `null_mode`. A claim
    whose null has no battery here is `not testable` with that stated, never
    scored against another null's vectors. A concept needs at least
    `min_members` (frozen, default 5) testable members, so the claim stays a
    claim about a centroid of that many features."""
    claims = _concept_claims(registry, "concept_atlas_centroid")
    alpha, n_null = float(cfg.confirm.alpha), int(cfg.confirm.atlas_n_null)
    tests = []
    for h in claims:
        entry = {"id": h["id"], "concept": h["concept"], "n_members": len(h["members"]),
                 "models": h["models"], "min_cosine": h["min_cosine"],
                 "null_mode": h["null_mode"], "legacy_claim_id": h.get("legacy_claim_id")}
        batteries = batteries_by_null.get(h["null_mode"])
        if batteries is None:
            entry.update({"status": _NOT_TESTABLE, "reason": (
                f"no private battery was run under the registered null {h['null_mode']!r}")})
            tests.append(entry)
            continue
        _vec, pool = _atlas_member_vector(batteries), _private_causal_pool(batteries)
        vecs, kept = [], []
        for m in h["members"]:
            v = _vec(m["model"], m["layer"], m["feature"])
            if v is not None and np.linalg.norm(v) > 0:
                vecs.append(v)
                kept.append(m)
        entry["n_testable_members"] = len(vecs)
        if len(vecs) < max(2, int(h["min_members"])):
            entry.update({"status": _NOT_TESTABLE, "reason": (
                f"only {len(vecs)} of {len(h['members'])} members are testable on private "
                f"data (reach withheld, unfired, or unscorable); the claim needs "
                f"{h['min_members']}")})
            tests.append(entry)
            continue
        comp: dict = {}
        for m in kept:
            comp[m["model"]] = comp.get(m["model"], 0) + 1
        if any(pool.get(mo) is None or len(pool[mo]) < c for mo, c in comp.items()):
            entry.update({"status": _NOT_TESTABLE, "reason": (
                "the private causal pool of a member's model is smaller than the member "
                "count, so no row-matched null can be drawn")})
            tests.append(entry)
            continue
        dev_centroid = np.asarray(h["dev_centroid"], dtype=np.float64)
        unit = _unit_rows(np.asarray(vecs))
        cen = unit.mean(axis=0)
        cen_cos = float(cen @ dev_centroid / (np.linalg.norm(cen) or 1.0))
        rng = np.random.default_rng(cfg.run.seed + 820_000 + int(h["concept"]))
        null_cen = []
        for _ in range(n_null):
            u = np.concatenate([pool[mo][rng.choice(len(pool[mo]), size=c, replace=False)]
                                for mo, c in sorted(comp.items())])
            c_ = u.mean(axis=0)
            null_cen.append(float(c_ @ dev_centroid / (np.linalg.norm(c_) or 1.0)))
        entry.update({
            "status": "tested", "private_centroid_cosine": cen_cos, "n_null": n_null,
            "null_centroid_cosine_p95": float(np.quantile(null_cen, 0.95)),
            "p": _tail_p(cen_cos, null_cen), "rule_holds": bool(cen_cos >= h["min_cosine"])})
        tests.append(entry)
    m, n_conf = _holm_confirm(tests, "p", alpha, extra_ok=lambda t: t["rule_holds"])
    return {"status": "tested", "n_registered": len(claims), "n_tested": m,
            "n_confirmed": n_conf,
            "n_not_testable": sum(1 for t in tests if t["status"] != "tested"),
            "p_combination": "p_centroid per claim vs the same-composition random-member-set "
                            "null, then Holm across this family; confirmed also needs the "
                            "centroid cosine >= min_cosine (no pair-fraction leg)",
            "n_null": n_null, "tests": tests}


def _confirm_family_presence(cfg, registry: dict, errors: dict, batteries_by_group: dict,
                             tag: str = "private") -> dict:
    """Replicate every `family_presence` claim (ROADMAP.md sec 41, V3-B).

    A claim's frozen features are re-measured by the private battery under the
    claim's own null (`batteries_by_group[(null_mode, k, n_null_directions)]`);
    each scorable feature contributes its observed vector and its own signed
    random-direction draws (`family_claims.observed_and_null_vectors`). The
    statistic is how many features have cosine >= the family threshold to the
    FROZEN dev centroid; the null count is drawn from each feature's own
    random-direction hit probability (`family_claims.presence_statistic`; the
    module docstring has the reasoning). Three states: a target whose capture
    failed or whose reach probe was withheld, and a claim with no scorable
    feature, are `not testable`, never `not confirmed`. Confirmed at Holm
    `alpha` across the tested claims of this family."""
    from .family_claims import observed_and_null_vectors, presence_statistic
    from ..sae.transfer import _seed

    claims = _concept_claims(registry, "family_presence")
    alpha = float(cfg.confirm.alpha)
    n_null = int(cfg.confirm.family_presence_n_null)
    tests = []
    for h in claims:
        entry = {"id": h["id"], "family_id": h["family_id"], "family_title": h["family_title"],
                 "model": h["model"], "null_mode": h["null_mode"], "min_cosine": h["min_cosine"],
                 "dev_count": h["dev_count"], "dev_basis": h["dev_basis"],
                 "n_features_frozen": h["n_features"]}
        group = batteries_by_group.get((h["null_mode"], int(h["k_top_series"]),
                                        int(h["n_null_directions"])))
        centroid = np.asarray(h["dev_centroid"], dtype=np.float64)
        obs_cos, null_cos, reasons = [], [], []
        for tgt, feats_ in h["targets"].items():
            b = (group or {}).get(tgt)
            if tgt in errors:
                reasons.append(f"{tgt}: private capture failed ({errors[tgt]})")
            elif group is None or b is None:
                reasons.append(f"{tgt}: no battery was run under null {h['null_mode']!r}")
            elif b["withheld"]:
                reasons.append(f"{tgt}: reach probe withheld ({b['reason']})")
            else:
                for f in feats_:
                    vn = observed_and_null_vectors(b["by_feature"].get(int(f)))
                    if vn is None:
                        continue
                    obs_cos.append(float(vn["obs"] @ centroid))
                    null_cos.append(vn["null"] @ centroid)
        entry["n_features_testable"] = len(obs_cos)
        if reasons:
            entry["unavailable_targets"] = reasons
        if not obs_cos:
            entry.update({"status": _NOT_TESTABLE, "reason": (
                "no frozen feature of this model is scorable on the held-out corpus "
                + ("(" + "; ".join(reasons) + ")" if reasons else
                   "(none fires there, or none has signed null draws)"))})
            tests.append(entry)
            continue
        rng = np.random.default_rng(_seed("family_presence", h["id"], tag,
                                          base=cfg.run.seed + 830_000))
        stat = presence_statistic(np.asarray(obs_cos), null_cos, h["min_cosine"], n_null, rng)
        entry.update({
            "status": "tested", "observed_count": stat["observed_count"],
            "expected_null_count": stat["expected_null_count"],
            "null_count_mean": stat["null_count_mean"], "null_count_p95": stat["null_count_p95"],
            "n_null": stat["n_null"], "n_null_directions": int(h["n_null_directions"]),
            "p": stat["p"], "observed_cosines": [float(c) for c in obs_cos]})
        tests.append(entry)
    m, n_conf = _holm_confirm(tests, "p", alpha,
                              extra_ok=lambda t: t["observed_count"] >= 1)
    return {"status": "tested", "n_registered": len(claims), "n_tested": m,
            "n_confirmed": n_conf,
            "n_not_testable": sum(1 for t in tests if t["status"] != "tested"),
            "n_null": n_null,
            "p_method": ("plus-one p of the count of features with cosine >= threshold to the "
                         "frozen family centroid, against each feature's own signed "
                         f"random-direction null ({n_null} Monte-Carlo replicates)"),
            "p_combination": "Holm across the tested claims of this family",
            "tests": tests}


def _agreement_key(t: dict) -> tuple:
    return (t["src_target"], t["dst_target"], int(t["concept"]), int(t["dst_feature"]))


def _confirm_agreement(cfg, hub, private, run_dir, registry: dict, feats: dict,
                       errors: dict) -> dict:
    """Replicate every registered `shared_input_agreement` claim by re-running
    the dev measurement (`run_shared_input_agreement`) on PRIVATE data with
    the frozen source/destination sets and the private top-k as `U`. The
    floors are re-drawn at `confirm.agreement_n_null`, and each claim's p is
    formed from the raw floor draws: a `same causal effect` claim needs both
    statistics to beat BOTH floors (p = the max of four right-tail p's, and
    the private verdict must again be `same causal effect`); an `acts
    differently` claim needs one statistic BELOW both floors' p05 (the lower
    tail: p = the min over statistics of the max of the two left-tail p's).
    Failing to beat a p95 is never evidence of disagreement (`CLAUDE.md` sec
    8). A claim frozen with `requires_defined_firing`
    (`confirm.register_requires_defined_firing` at registration) applies the
    L5 defined-firing rule to its private test: only a statistic that
    `firing_defined` marks defined can fire (and enter p), and a claim with
    no defined statistic is `not testable`. A side that is not scorable on private data makes the claim `not
    testable`. Sampled models seed every compared predict pair identically
    inside `shared_input_agreement` (baseline, ablation and null share one
    seed per side)."""
    from ..sae import shared_input_agreement as sia

    claims = _concept_claims(registry, "shared_input_agreement")
    alpha = float(cfg.confirm.alpha)
    units = [{
        "concept": h["concept"], "src_target": h["src_target"], "src_model": h["src_model"],
        "src_features": h["src_features"], "dst_target": h["dst_target"],
        "dst_model": h["dst_model"], "dst_feature": h["dst_feature"],
        "dst_features": h["dst_features"], "dst_set_kind": h["dst_set_kind"],
        "k_top_series": h["k_top_series"], "test_auc": None, "transfer_margin": None}
        for h in claims
        if h["src_target"] in feats and h["dst_target"] in feats]
    sia.reset_caches()
    try:
        doc = sia.run_shared_input_agreement(
            cfg, run_dir, hub, None, private, _device_for(cfg), {}, {}, units=units,
            private_pooled={t: f["pooled"] for t, f in feats.items()},
            private_alive={t: f["alive"] for t, f in feats.items()},
            ground_truth_path=_private_ground_truth_path(cfg),
            n_null=int(cfg.confirm.agreement_n_null), base_seed=cfg.run.seed + 800_000,
            keep_floor_values=True, write=False) if units else {"tests": []}
    finally:
        sia.reset_caches()
    by_key = {_agreement_key(t): t for t in doc["tests"]}

    tests = []
    for h in claims:
        entry = {"id": h["id"], "concept": h["concept"], "src_target": h["src_target"],
                 "dst_target": h["dst_target"], "dst_feature": h["dst_feature"],
                 "dev_verdict": h["dev_verdict"]}
        t = by_key.get(_agreement_key(h))
        missing = [x for x in (h["src_target"], h["dst_target"]) if x not in feats]
        if missing:
            entry.update({"status": _NOT_TESTABLE, "reason": (
                f"private capture failed for {missing[0]}: {errors.get(missing[0], 'unknown')}")})
        elif t is None or t["verdict"] == "not scorable":
            entry.update({"status": _NOT_TESTABLE, "reason": (
                (t or {}).get("reason", "no private measurement was produced"))})
        else:
            entry.update({"status": "tested", "private_verdict": t["verdict"],
                          "n_shared_series": t.get("n_shared_series")})
            si, sii = t.get("statistic_i") or {}, t.get("statistic_ii") or {}
            stats = [s for s in (si, sii) if s.get("observed") is not None]
            entry["private_statistics"] = {
                k: {f: s.get(f) for f in ("observed", "floor_p95_src", "floor_p95_dst",
                                          "floor_p05_src", "floor_p05_dst", "clears",
                                          "below_floor", "n_floor_src", "n_floor_dst")}
                for k, s in (("statistic_i", si), ("statistic_ii", sii))}
            if h.get("requires_defined_firing"):
                defined = sia.record_firing_defined(t)
                entry["firing_defined"] = defined
            if h["dev_verdict"] == "same causal effect":
                ps = [_tail_p(s["observed"], s[f"floor_values_{side}"], "right")
                      for s in (si, sii) for side in ("src", "dst")
                      if s.get("observed") is not None]
                entry["p"] = max(ps) if len(stats) == 2 and ps else float("nan")
                entry["rule"] = "both statistics beat both floors' p95 (verdict same causal effect)"
                entry["rule_holds"] = bool(t["verdict"] == "same causal effect")
            elif h.get("requires_defined_firing"):
                live = [s for s, k in ((si, "i"), (sii, "ii"))
                        if s.get("observed") is not None and defined[k]]
                if not live:
                    entry.update({"status": _NOT_TESTABLE, "reason": (
                        "firing statistic undefined on private data under the defined-firing "
                        "rule: (i) needs `level` to clear on both sides, (ii) needs both sides "
                        "to clear a shape channel")})
                else:
                    ps = [max(_tail_p(s["observed"], s["floor_values_src"], "left"),
                              _tail_p(s["observed"], s["floor_values_dst"], "left"))
                          for s in live]
                    entry["p"] = min(ps)
                    entry["rule"] = ("at least one DEFINED statistic below BOTH floors' p05 "
                                     "(lower tail); (i) is defined only when `level` clears on "
                                     "both sides, (ii) only when both sides clear a shape "
                                     "channel; not clearing a p95 is not disagreement")
                    entry["rule_holds"] = bool(any(s.get("below_floor") for s in live))
            else:
                ps = [max(_tail_p(s["observed"], s["floor_values_src"], "left"),
                          _tail_p(s["observed"], s["floor_values_dst"], "left"))
                      for s in stats]
                entry["p"] = min(ps) if ps else float("nan")
                entry["rule"] = ("at least one statistic below BOTH floors' p05 (lower tail); "
                                 "not clearing a p95 is not disagreement")
                entry["rule_holds"] = bool(si.get("below_floor") or sii.get("below_floor"))
        tests.append(entry)
    m, n_conf = _holm_confirm(tests, "p", alpha, extra_ok=lambda t: t["rule_holds"])
    return {"status": "tested", "n_registered": len(claims), "n_tested": m,
            "n_confirmed": n_conf,
            "n_not_testable": sum(1 for t in tests if t["status"] != "tested"),
            "n_null": int(cfg.confirm.agreement_n_null),
            "p_combination": "empirical tail p from the raw floor draws, Holm across claims; "
                            "the dev rule must also hold on private data",
            "tests": tests}


def _confirm_structure(cfg, registry: dict, errors: dict, batteries: dict) -> dict:
    """Replicate the registered directional aggregates.

    (a) `no_concept_in_all_models` (rule-based, no p): the atlas recomputed
    (`sae/concept_atlas.py::cluster_atlas`, dev `min_cosine`/`min_members`)
    on the PRIVATE battery vectors of the dev causal pool has no concept with
    members in every model. Not testable if any dev-pool model has no
    testable private row: an absent model would make the claim vacuously
    true (`CLAUDE.md` sec 8, "widening the evidence can make an existential
    check vacuous").

    (b) `majority_prominent_features_causally_null`: the share of frozen dev
    candidates that clear no channel on private data; confirmed if the
    one-sided 95% lower bound of that share, from a bootstrap stratified by
    target, is above 0.5. The resampling unit here is the FEATURE, because the
    claim's denominator is features and a series resample would need the
    battery re-run per draw; this is stated in the output, not hidden."""
    from ..sae.concept_atlas import cluster_atlas
    from ..sae.concepts import ablation_vector

    claims = _concept_claims(registry, "concept_structure")
    alpha = float(cfg.confirm.alpha)
    tests = []
    for h in claims:
        entry = {"id": h["id"], "statistic": h["statistic"], "p_valued": h["p_valued"]}
        if h["id"].endswith("no_concept_in_all_models"):
            X, models, pool_models = [], [], sorted(h["dev_models"])
            for r in h["pool"]:
                b = batteries.get(f"{r['model']}/{r['layer']}")
                rec = (b or {}).get("by_feature", {}).get(int(r["feature"])) \
                    if b and not b["withheld"] else None
                v = ablation_vector(dict(rec)) if rec and rec.get("scorable") else None
                if v is not None and np.linalg.norm(v) > 0:
                    X.append(v)
                    models.append(r["model"])
            missing = [m for m in pool_models if m not in set(models)]
            if missing:
                entry.update({"status": _NOT_TESTABLE, "reason": (
                    f"model(s) {missing} have no testable private row, so 'no concept in all "
                    f"models' would hold vacuously")})
            else:
                labels = cluster_atlas(np.asarray(X), h["min_cosine"], h["min_members"])
                per = {}
                for lab, mo in zip(labels, models):
                    if lab >= 0:
                        per.setdefault(int(lab), set()).add(mo)
                mx = max((len(v) for v in per.values()), default=0)
                entry.update({"status": "tested", "private_n_concepts": len(per),
                              "private_max_models_per_concept": mx,
                              "n_models_in_pool": len(pool_models),
                              "dev_max_models_per_concept": h["dev_max_models_per_concept"],
                              "rule_holds": bool(mx < len(pool_models)),
                              "p": None, "rule": "confirmed if the private atlas also has "
                                                 "no concept with members in every model"})
        else:
            per_target = {}
            for tgt, feats_ in h["candidates"].items():
                b = batteries.get(tgt)
                if b is None or b["withheld"]:
                    continue
                flags = [not rec.get("n_channels_clearing")
                         for f in feats_
                         for rec in [b["by_feature"].get(int(f))]
                         if rec and rec.get("scorable")]
                if flags:
                    per_target[tgt] = np.asarray(flags, dtype=float)
            n_tested = sum(len(v) for v in per_target.values())
            if not n_tested:
                entry.update({"status": _NOT_TESTABLE,
                              "reason": "no frozen candidate fired on a reachable private target"})
            else:
                rate = float(np.concatenate(list(per_target.values())).mean())
                rng = np.random.default_rng(cfg.run.seed + 830_000)
                B = int(cfg.confirm.structure_n_boot)
                boots = np.empty(B)
                for i in range(B):
                    num = den = 0
                    for v in per_target.values():
                        s = v[rng.integers(0, len(v), len(v))]
                        num += s.sum()
                        den += len(s)
                    boots[i] = num / den
                lo = float(np.quantile(boots, 0.05))
                entry.update({
                    "status": "tested", "private_n_features": n_tested, "private_rate": rate,
                    "dev_rate": h["dev_rate"], "lower_95_one_sided": lo, "n_boot": B,
                    "resampling_unit": "feature, stratified by target",
                    "p": (1 + int(np.sum(boots <= 0.5))) / (B + 1),
                    "rule_holds": bool(lo > 0.5),
                    "rule": "confirmed if the one-sided 95% lower bound of the private "
                            "causally-null share is above 0.5"})
        tests.append(entry)
    pvalued = [t for t in tests if t["status"] == "tested" and t.get("p") is not None]
    _holm_confirm(pvalued, "p", alpha, extra_ok=lambda t: t["rule_holds"])
    for t in tests:
        if t["status"] == "tested" and t.get("p") is None:
            t["confirmed"] = bool(t["rule_holds"])
            t["verdict"] = "confirmed" if t["confirmed"] else "not confirmed"
        elif t["status"] != "tested":
            t["confirmed"], t["verdict"] = False, _NOT_TESTABLE
    return {"status": "tested", "n_registered": len(claims),
            "n_tested": sum(1 for t in tests if t["status"] == "tested"),
            "n_confirmed": sum(1 for t in tests if t.get("confirmed")),
            "n_not_testable": sum(1 for t in tests if t["status"] != "tested"),
            "exploratory_not_registered": registry.get("concept_claim_candidates", {}).get(
                "concept_structure", {}).get("exploratory_not_registered"),
            "tests": tests}


def _encode_frozen_targets(run_dir, acts: dict, errors: dict, device) -> tuple:
    """`(feats, errors)`: each captured target encoded with its FROZEN dev
    checkpoint (`_frozen_features`); a target whose encode raises joins
    `errors` (a copy of the capture errors) and every claim touching it is
    not testable."""
    feats, feat_errors = {}, dict(errors)
    for tgt, a in acts.items():
        model, layer = tgt.split("/", 1)
        try:
            feats[tgt] = _frozen_features(run_dir, model, layer, a, device)
        except Exception as exc:  # noqa: BLE001
            feat_errors[tgt] = f"{type(exc).__name__}: {exc}"
            log.warning("confirm concept claims: encoding held-out activations at %s with the "
                        "frozen checkpoint failed (%s); dependent claims are not testable",
                        tgt, exc)
    return feats, feat_errors


def _atlas_centroid_batteries(cfg, hub, data, run_dir, claims: list, feats: dict,
                              gt_path: str | None = None) -> dict:
    """`{null_mode: {target: battery}}` for the `concept_atlas_centroid`
    claims: every frozen member re-measured at dev's own `n_null_directions`
    under the claim's registered null."""
    by_mode: dict = {}
    for h in claims:
        for m in h["members"]:
            by_mode.setdefault(h["null_mode"], {}).setdefault(
                f"{m['model']}/{m['layer']}", set()).add(int(m["feature"]))
    out: dict = {}
    for mode, targets in sorted(by_mode.items()):
        out[mode] = {}
        for tgt, fs in sorted(targets.items()):
            if tgt not in feats:
                continue
            model, layer = tgt.split("/", 1)
            out[mode][tgt] = _private_battery(
                cfg, hub, data, run_dir, model, layer, feats[tgt], sorted(fs),
                int(cfg.concepts.top_k_series), int(cfg.concepts.n_null_directions), False,
                gt_path=gt_path, null_mode=mode)
    return out


def _family_presence_batteries(cfg, hub, data, run_dir, claims: list, feats: dict,
                               gt_path: str | None = None) -> dict:
    """`{(null_mode, k, n_null_directions): {target: battery}}` for the
    `family_presence` claims: the union of the claims' frozen features per
    target, with the signed null draws kept."""
    groups: dict = {}
    for h in claims:
        key = (h["null_mode"], int(h["k_top_series"]), int(h["n_null_directions"]))
        for tgt, fs in h["targets"].items():
            groups.setdefault(key, {}).setdefault(tgt, set()).update(int(f) for f in fs)
    out: dict = {}
    for key, targets in sorted(groups.items()):
        mode, k, nd = key
        out[key] = {}
        for tgt, fs in sorted(targets.items()):
            if tgt not in feats:
                continue
            model, layer = tgt.split("/", 1)
            out[key][tgt] = _private_battery(
                cfg, hub, data, run_dir, model, layer, feats[tgt], sorted(fs), k, nd, True,
                signed_null=True, gt_path=gt_path, null_mode=mode)
    return out


def _replicate_causal_concept_claims(cfg: PipelineConfig, hub, private: BenchmarkData,
                                     registry: dict, replication: dict,
                                     acts: dict, errors: dict) -> dict:
    """Run the four K2 claim types on the shared private cache `acts` and add
    `causal` / `atlas` / `agreement` / `structure` blocks (and their ledger
    rows) to `replication`. Existing keys are never rewritten (invariant 13);
    a run that registered no K2 claim never reaches here."""
    from .hypotheses import claim_family_budget

    run_dir = cfg.run_dir()
    device = _device_for(cfg)
    causal = _concept_claims(registry, "concept_causal")
    atlas = _concept_claims(registry, "concept_atlas")
    struct = _concept_claims(registry, "concept_structure")
    k_default = int(cfg.concepts.top_k_series)

    feats, feat_errors = _encode_frozen_targets(run_dir, acts, errors, device)

    # Pass 1: every frozen dev candidate an atlas / structure claim needs, at
    # dev's own `n_null_directions`, so each private vector is on dev's scale.
    p1_features: dict = {}
    for h in atlas:
        for m in h["members"]:
            p1_features.setdefault(f"{m['model']}/{m['layer']}", set()).add(int(m["feature"]))
    for h in struct:
        for r in h.get("pool", []):
            p1_features.setdefault(f"{r['model']}/{r['layer']}", set()).add(int(r["feature"]))
        for tgt, fs in h.get("candidates", {}).items():
            p1_features.setdefault(tgt, set()).update(int(f) for f in fs)
    batteries: dict = {}
    for tgt, fs in sorted(p1_features.items()):
        if tgt not in feats:
            continue
        model, layer = tgt.split("/", 1)
        batteries[tgt] = _private_battery(cfg, hub, private, run_dir, model, layer, feats[tgt],
                                          sorted(fs), k_default,
                                          int(cfg.concepts.n_null_directions), False)
    atlas_batteries = dict(batteries)

    # Pass 2: the causal claims, with the many-draw null and the per-row
    # deltas kept; a claim on the p floor is redrawn (adaptive) at n_max.
    n0, n_max = int(cfg.confirm.causal_n_null), int(cfg.confirm.causal_max_null)
    by_target: dict = {}
    for h in causal:
        by_target.setdefault(h["target"], {}).setdefault(int(h["k_top_series"]), set()).add(
            int(h["feature"]))
    causal_batteries: dict = {}
    for tgt, ks in sorted(by_target.items()):
        if tgt not in feats:
            continue
        model, layer = tgt.split("/", 1)
        merged = {"withheld": False, "reason": "", "by_feature": {}}
        for k, fs in sorted(ks.items()):
            b = _private_battery(cfg, hub, private, run_dir, model, layer, feats[tgt],
                                 sorted(fs), k, n0, True)
            merged["withheld"] = merged["withheld"] or b["withheld"]
            merged["reason"] = merged["reason"] or b["reason"]
            merged["by_feature"].update(b["by_feature"])
        causal_batteries[tgt] = merged
    if n_max > n0:
        for h in causal:
            b = causal_batteries.get(h["target"])
            rec = (b or {}).get("by_feature", {}).get(int(h["feature"]))
            ch = ((rec or {}).get("channels") or {}).get(h["channel"]) or {}
            draws = ch.get("null_draw_means")
            if b and not b["withheld"] and draws and not any(
                    d >= ch["effect"] for d in draws):
                model, layer = h["target"].split("/", 1)
                causal_batteries[("adaptive", h["target"], int(h["feature"]))] = \
                    _private_battery(cfg, hub, private, run_dir, model, layer,
                                     feats[h["target"]], [int(h["feature"])],
                                     int(h["k_top_series"]), n_max, True)

    out = dict(replication)
    if causal:
        out["causal"] = _confirm_concept_causal(cfg, hub, private, run_dir, registry, feats,
                                                feat_errors, causal_batteries)
    if atlas:
        out["atlas"] = _confirm_atlas(cfg, registry, feat_errors, atlas_batteries)
    agreement = _concept_claims(registry, "shared_input_agreement")
    if agreement:
        out["agreement"] = _confirm_agreement(cfg, hub, private, run_dir, registry, feats,
                                              feat_errors)
    if struct:
        out["structure"] = _confirm_structure(cfg, registry, feat_errors, atlas_batteries)
    centroid = _concept_claims(registry, "concept_atlas_centroid")
    if centroid:
        out["atlas_centroid"] = _confirm_atlas_centroid(
            cfg, registry, feat_errors,
            _atlas_centroid_batteries(cfg, hub, private, run_dir, centroid, feats))
    family = _concept_claims(registry, "family_presence")
    if family:
        out["family_presence"] = _confirm_family_presence(
            cfg, registry, feat_errors,
            _family_presence_batteries(cfg, hub, private, run_dir, family, feats))

    out["ledger"] = _k2_ledger(cfg, registry, out)
    if out.get("status") == "skipped" and any(k in out for k in _K2_TESTED.values()):
        out["status"] = "tested"
        out.pop("reason", None)
    return out


_K2_TESTED = {"concept_causal": "causal", "concept_atlas": "atlas",
              "shared_input_agreement": "agreement", "concept_structure": "structure",
              "concept_atlas_centroid": "atlas_centroid", "family_presence": "family_presence"}


def _k2_ledger(cfg: PipelineConfig, registry: dict, out: dict) -> list:
    """`out`'s ledger plus one row per non-empty K2 claim family. Families
    `claim_family_budget` reports that this pass does not test (the
    `reliability_u1` row) are left to their own replication step, which adds
    its own row."""
    from .hypotheses import claim_family_budget
    ledger = list(out.get("ledger") or [])
    for row in claim_family_budget(cfg, registry):
        if not row["m"] or row["family"] not in _K2_TESTED:
            continue
        blk = out.get(_K2_TESTED[row["family"]]) or {}
        ledger.append({**row, "n_tested": blk.get("n_tested"),
                       "n_not_testable": blk.get("n_not_testable")})
    return ledger


# ---------------------------------------------------------------------------
# ROADMAP.md sec 38.4 (K4) -- replication of the U1 reliability claims.
#
# The claim is about a PROCEDURE ("internals add a positive gain over the free
# baseline"), so confirm refits it on private series with everything the
# registry froze: SAE checkpoints (sha256-pinned, never retrained), family
# membership, baseline definition, ridge grid, folds, repeats, seed, strata
# and bootstrap size. It does not score a frozen dev model: that would add
# domain shift to the question. Three states, never two: `confirmed`,
# `not confirmed`, `not testable` (an input this private run cannot supply).
# ---------------------------------------------------------------------------

_RELIABILITY_EVIDENCE = "predictive (behavioral); not causal"


def _reliability_layers(h: dict) -> list:
    """Every `model/layer` target a registered `reliability_u1` claim reads on
    private data: the SAE layers of its families and its crystallization layer."""
    groups = h["spec"]["groups"]
    layers = list((groups.get("sae_families") or {}).get("layers", []))
    if "crystallization_norm" in groups:
        layers.append(groups["crystallization_norm"]["layer"])
    return sorted({f"{h['model']}/{layer}" for layer in layers})


def _frozen_last_window_features(run_dir, model: str, layer: str, acts: torch.Tensor,
                                 device: torch.device) -> np.ndarray:
    """Last-context-window SAE activations `[n, dict_size]` (float64) of private
    window activations, encoded with the FROZEN dev checkpoint
    (`ablation_run.checkpoint_path`), the window the dev K4 feature read from
    the store's `space="sae"` window level. Nothing is trained here."""
    from ..sae.ablation_run import checkpoint_path
    from ..sae.train import load_sae_checkpoint

    sae = load_sae_checkpoint(str(checkpoint_path(run_dir, model, layer))).to(device)
    sae.eval()
    with torch.no_grad():
        return sae.encode(acts[:, -1, :].float().to(device)).cpu().numpy().astype(np.float64)


def _reliability_internal_columns(run_dir, h: dict, acts: dict, n: int,
                                  device: torch.device) -> tuple:
    """`(matrix [n, q], "")` of one claim's internal feature columns in the
    frozen order (SAE family sums, then the crystallization norm), built from
    window activations `acts[model/layer]` with the FROZEN checkpoints; or
    `(None, reason)` when the family ids do not reproduce. The same builder
    works on any split's activations, which is how a test checks it against
    the dev store route."""
    from . import reliability_from_internals as rfi

    model, groups = h["model"], h["spec"]["groups"]
    blocks = []
    if "sae_families" in groups:
        g = groups["sae_families"]
        rows = [{"model": model, "layer": mem["layer"], "feature": mem["feature"],
                 "family": fam["family"]} for fam in g["families"] for mem in fam["members"]]
        last = {layer: _frozen_last_window_features(run_dir, model, layer,
                                                    acts[f"{model}/{layer}"], device)
                for layer in g["layers"]}
        X, fam_ids = rfi.family_columns(last, rows, model, n)
        if fam_ids != [f["family"] for f in g["families"]]:
            return None, "private family columns differ from the frozen family ids"
        blocks.append(X)
    if "crystallization_norm" in groups:
        layer = groups["crystallization_norm"]["layer"]
        blocks.append(rfi.crystallization_norm_feature(
            acts[f"{model}/{layer}"][:, -1, :].float().numpy()))
    return np.concatenate(blocks, axis=1), ""


def _private_l0(cfg: PipelineConfig, hub, private: BenchmarkData, model: str) -> dict:
    """One model's private forecasts and L0 table: `{"point", "quants", "scored"}`,
    with the same `_score` arguments dev's L0 stage used."""
    adapter = hub.get(model)
    adapter.ensure_loaded()
    contexts = private.contexts()
    point, quants = _predict_all(adapter, contexts, private.horizon, cfg.l0.quantiles)
    scored = _score(model, point, quants, contexts, private.targets(), cfg.l0.quantiles,
                    private.meta, cfg.l0.scale, cfg.l0.min_scale_frac)
    return {"point": point, "quants": quants, "scored": scored}


def _reliability_not_testable(entry: dict, reason: str) -> dict:
    entry.update({"status": _NOT_TESTABLE, "reason": reason})
    return entry


def _score_reliability_claim(cfg: PipelineConfig, hub, private: BenchmarkData, run_dir,
                             h: dict, acts: dict, errors: dict, l0_cache: dict) -> dict:
    """Refit one registered U1 claim on private series and return its row.

    Checks, before any private number is computed, that every pinned
    artifact still has its registered sha256 and that the frozen seed
    reproduces. Then: private L0 (`mase`, `mase_reliable`) and the free
    baseline from the model's own forecasts, the frozen SAE family sums and
    crystallization norm from the shared private activation cache, and
    `reliability_from_internals.u1_gain` with the frozen spec. A claim whose
    inputs this private run cannot supply is `not testable` with the reason."""
    from . import reliability_from_internals as rfi
    from .stats import _mase_scale

    spec, model = h["spec"], h["model"]
    entry = {"id": h["id"], "model": model, "task": h["task"], "dev": h["dev"],
             "spec_n_boot": spec["n_boot"]}
    if h["task"] not in ("log_mase_spearman",):
        return _reliability_not_testable(entry, f"task {h['task']!r} is not refit by confirm")
    changed = []
    for rel, registered in {h["artifact"]: h["artifact_sha256"], **(h.get("artifacts") or {})}.items():
        path = Path(run_dir) / rel
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != registered:
            changed.append(rel)
    if changed:
        return _reliability_not_testable(
            entry, f"frozen artifact(s) changed since registration: {sorted(changed)}")
    if rfi.model_seed_for(model, int(spec["seed"])) != int(spec["model_seed"]):
        return _reliability_not_testable(entry, "the frozen model seed does not reproduce")
    targets = _reliability_layers(h)
    bad = {t: errors[t] for t in targets if t in errors}
    if bad:
        return _reliability_not_testable(entry, f"private capture failed: {bad}")
    missing = [t for t in targets if t not in acts]
    if missing:
        return _reliability_not_testable(entry, f"no private activations at {missing}")
    try:
        if model not in l0_cache:
            l0_cache[model] = _private_l0(cfg, hub, private, model)
    except Exception as exc:
        return _reliability_not_testable(
            entry, f"private L0 unavailable: {type(exc).__name__}: {exc}")
    l0 = l0_cache[model]
    scored = l0["scored"]
    contexts = private.contexts()
    scale = _mase_scale(contexts, cfg.l0.scale)
    c22, c22_names = rfi.context_catch22(contexts)
    base, base_names, _ = rfi.baseline_features(l0["point"], l0["quants"], contexts, scale,
                                                c22, c22_names)
    if base_names != spec["baseline_features"]:
        return _reliability_not_testable(entry, "the baseline feature list differs from the "
                                                "frozen one")
    internals, reason = _reliability_internal_columns(run_dir, h, acts, private.n,
                                                      _device_for(cfg))
    if reason:
        return _reliability_not_testable(entry, reason)
    mase = scored["mase"].to_numpy(dtype=np.float64)
    keep = scored["mase_reliable"].to_numpy(dtype=bool) & np.isfinite(mase)
    idx = np.flatnonzero(keep)
    if len(idx) < max(int(cfg.stats.min_series), 2 * int(spec["n_folds"])):
        return _reliability_not_testable(
            entry, f"only {len(idx)} reliable private series with a finite MASE")
    res = rfi.u1_gain(np.log(np.maximum(mase[idx], 1e-6)), base[idx],
                      internals[idx],
                      private.meta["family"].astype(str).to_numpy()[idx],
                      n_folds=int(spec["n_folds"]), n_repeats=int(spec["n_repeats"]),
                      n_boot=int(spec["n_boot"]), model_seed=int(spec["model_seed"]),
                      alphas=spec["ridge_alphas"])
    entry.update({"status": "tested", "n": int(len(idx)),
                  "n_excluded": int(private.n - len(idx)), **res})
    return entry


def _confirm_reliability_u1(cfg, hub, private, run_dir, registry: dict, acts: dict,
                            errors: dict) -> dict:
    """Refit every `reliability_u1` claim on private series, Holm over the family.

    Confirmed iff the claim's private series-bootstrap gain CI lower bound is
    > 0 AND its two-sided bootstrap p is below `confirm.alpha` after Holm
    across the TESTED claims of this family (a bootstrap p is floored at
    `1/n_boot`, so the family's smallest attainable adjusted p is `m/n_boot`,
    the ledger row `claim_family_budget` declares and `confirm` checks before
    opening the split). `not testable` claims stay out of the family and are
    never `not confirmed`. Evidence class: predictive (behavioral), not causal."""
    claims = _concept_claims(registry, RELIABILITY_STAGE)
    ids = [h["id"] for h in claims]
    if len(set(ids)) != len(ids):
        raise ValueError("reliability_u1 claims have duplicate ids; Holm is keyed by id, so "
                         "the family would silently shrink")
    alpha = float(cfg.confirm.alpha)
    l0_cache: dict = {}
    tests = [_score_reliability_claim(cfg, hub, private, run_dir, h, acts, errors, l0_cache)
             for h in claims]
    m, n_conf = _holm_confirm(tests, "p", alpha,
                              extra_ok=lambda t: t["lo"] > 0 and t["gain"] > 0)
    return {"status": "tested", "family": RELIABILITY_STAGE, "n_registered": len(claims),
            "n_tested": m, "n_confirmed": n_conf,
            "n_not_testable": sum(1 for t in tests if t["status"] != "tested"),
            "evidence_class": _RELIABILITY_EVIDENCE,
            "p_combination": "two-sided series-bootstrap p of the gain, Holm across the tested "
                             "claims of this family; the gain CI lower bound must also be > 0",
            "caveat": "A feature that predicts error is not a cause of error; this is predictive "
                      "(behavioral) evidence only.",
            "tests": tests}


def _replicate_reliability_u1(cfg: PipelineConfig, hub, private: BenchmarkData, registry: dict,
                              replication: dict, acts: dict, errors: dict) -> dict:
    """Add the `reliability_u1` block (and its ledger row) to `replication`.

    Captures, in one sweep per model, only the targets the shared private
    cache `acts` does not already hold. Existing keys are never rewritten
    (invariant 13); a run that registered no U1 claim never reaches here."""
    from .hypotheses import claim_family_budget

    claims = _concept_claims(registry, RELIABILITY_STAGE)
    need = sorted({t for h in claims for t in _reliability_layers(h)} - set(acts) - set(errors))
    acts, errors = dict(acts), dict(errors)
    if need:
        more, more_err = _capture_private_targets(cfg, hub, private, need)
        acts.update(more)
        errors.update(more_err)
    block = _confirm_reliability_u1(cfg, hub, private, cfg.run_dir(), registry, acts, errors)
    out = dict(replication)
    out["reliability_u1"] = block
    ledger = list(out.get("ledger") or [])
    for row in claim_family_budget(cfg, registry):
        if row["family"] == RELIABILITY_STAGE and row["m"]:
            ledger.append({**row, "n_tested": block["n_tested"],
                           "n_not_testable": block["n_not_testable"]})
    out["ledger"] = ledger
    if out.get("status") == "skipped":
        out["status"] = "tested"
        out.pop("reason", None)
    return out
