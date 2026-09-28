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

Discipline matters more than machinery here: run this once, at the end.
Repeated peeking consumes the private benchmark (regenerate a fresh epoch
via tsfm_benchmark if that happens). The stage refuses to overwrite existing
confirmation artifacts unless explicitly forced, and by default refuses
unverifiable private corpora.
"""

from __future__ import annotations

import dataclasses
import hashlib

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

# Corpus identity (ROADMAP.md sec 34 item B5): a manifest-derived field
# degrades to this reason, never a bare `None`, when the confirm.source is
# `smoke` or an unverified jsonl -- both skip `load_sealed` entirely, so
# there is no manifest.json to read (CLAUDE.md sec 2.5, sec 6.7's own
# "smoke-mode confirm is mechanically real but distributionally identical to
# dev" warning, which this is what makes visible per-run rather than only
# documented).
_NO_MANIFEST_REASON = "no sealed manifest (smoke source, or an unverified jsonl load)"


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
    if repeated:
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
    a, b = cfg.comparison_pair()
    metrics = _private_behavioral(cfg, hub, private, out_dir)
    hypotheses = _test_registered_hypotheses(cfg, metrics, registry, a.name, b.name)
    replication = _replicate_registered_cka(cfg, hub, private, registry)
    l3_replication = _replicate_registered_l3(cfg, hub, private, registry)
    concept_replication = _replicate_registered_concepts(cfg, hub, private, registry)

    confirmed = sum(1 for h in hypotheses["tests"] if h["confirmed"])
    n_replicable = sum(1 for h in registry["hypotheses"] if h["replicable"])
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


def _replicate_registered_concepts(cfg: PipelineConfig, hub, private: BenchmarkData,
                                   registry: dict) -> dict:
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
            acts = _capture_private_window_acts(cfg, hub, private, model, layer)
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
