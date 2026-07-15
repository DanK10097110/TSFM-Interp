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

import numpy as np
import pandas as pd
import torch

from ..config import PipelineConfig
from ..data import BenchmarkData, load_benchmark
from ..extraction.alignment import align, pooling_matrix
from ..extraction.hooks import ActivationCatcher
from ..utils import batch_slices, load_json, log, save_json
from .l0_behavioral import _predict_all, _score
from .l1_geometry import linear_cka
from .stats import bootstrap_ci, holm, paired_bootstrap


def run_confirm(cfg: PipelineConfig, hub) -> None:
    """Load the private corpus, re-test dev hypotheses, spot-check the CKA peak."""
    out_dir = cfg.run_dir() / "confirm"
    if (out_dir / "confirmation.json").exists():
        raise RuntimeError(
            "confirmation artifacts already exist; the private benchmark is meant to be "
            "consumed once. Rerun with --force confirm only if you understand that this "
            "constitutes a second look (and consider a fresh private epoch).")
    log.warning("CONFIRM: running the one-shot private-benchmark confirmation; "
                "avoid re-running against the same private epoch")
    out_dir.mkdir(parents=True, exist_ok=True)

    private = _load_private(cfg)
    a, b = cfg.comparison_pair()
    metrics = _private_behavioral(cfg, hub, private, out_dir)
    hypotheses = _test_hypotheses(cfg, metrics, a.name, b.name)
    replication = _replicate_cka(cfg, hub, private)

    confirmed = sum(1 for h in hypotheses["tests"] if h["confirmed"])
    save_json(out_dir / "confirmation.json", {
        "model_a": a.name, "model_b": b.name,
        "n_private_series": private.n,
        "alpha": cfg.confirm.alpha,
        **hypotheses,
        "cka_replication": replication,
    })
    log.info("confirm complete: %d/%d dev hypotheses confirmed on private data",
             confirmed, len(hypotheses["tests"]))


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


def _test_hypotheses(cfg: PipelineConfig, metrics: pd.DataFrame,
                     name_a: str, name_b: str) -> dict:
    """Re-test dev family strengths on private series, Holm-corrected."""
    sc = cfg.stats
    dev = load_json(cfg.run_dir() / "l0" / "summary.json")
    dev_strengths = dev.get("strengths", {})
    dev_ratio = dev.get("mase_ratio", {})
    claims = [(fam, model) for model, fams in dev_strengths.items() for fam in fams]

    wide = metrics.pivot_table(index=["series_id", "family"], columns="model",
                               values="mase").reset_index()
    tests, pvals = [], {}
    for fam, favored in claims:
        grp = wide[wide["family"] == fam]
        entry = {"family": fam, "dev_favored": favored,
                 "dev_ratio": dev_ratio.get(fam)}
        if len(grp) < sc.min_series:
            entry.update({"status": "untestable",
                          "reason": f"only {len(grp)} private series"})
            tests.append(entry)
            continue
        sign = 1.0 if favored == name_a else -1.0
        diff = sign * (grp[name_b] - grp[name_a]).to_numpy()
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


def _replicate_cka(cfg: PipelineConfig, hub, private: BenchmarkData) -> dict:
    """Recompute the dev peak-pair CKA on private contexts with a cluster-bootstrap CI."""
    l1_path = cfg.run_dir() / "l1" / "meta.json"
    if not l1_path.exists():
        return {"status": "skipped", "reason": "no L1 artifacts on dev"}
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
