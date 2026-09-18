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
