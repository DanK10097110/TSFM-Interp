"""L0 — behavioral profiling.

Answers the purely behavioral questions ("is one model better on
high-frequency data?") before any internals are touched, and generates the
family-level hypotheses the mechanistic levels then try to explain. Metrics
are computed per series and aggregated per family: MASE against the naive
one-step scale of each context, sMAPE, and mean pinball loss over the
configured quantiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.store import ActivationStore
from ..utils import batch_slices, log, save_json
from .stats import holm, mase as _mase, mean_ci, paired_bootstrap


def run_l0(cfg: PipelineConfig, hub, data: BenchmarkData, store: ActivationStore) -> None:
    """Forecast with every model, score against targets, persist metric tables."""
    out_dir = cfg.run_dir() / "l0"
    out_dir.mkdir(parents=True, exist_ok=True)
    contexts, targets = data.contexts(), data.targets()
    frames = []
    for mcfg in cfg.models:
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        point, quants = _predict_all(adapter, contexts, data.horizon, cfg.l0.quantiles)
        store.write_predictions(mcfg.name, point, quants)
        frames.append(_score(mcfg.name, point, quants, contexts, targets,
                             cfg.l0.quantiles, data.meta))
        if not cfg.run.keep_models_loaded:
            hub.release(mcfg.name)
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_parquet(out_dir / "metrics.parquet")
    save_json(out_dir / "summary.json", _summarize(metrics, cfg))
    log.info("L0 complete: %d model-series scores", len(metrics))


def _predict_all(adapter, contexts: np.ndarray, horizon: int, quantiles: list):
    """Batched forecasting over the full benchmark."""
    points, quants = [], []
    for s, e in batch_slices(len(contexts), adapter.cfg.batch_size):
        out = adapter.predict(contexts[s:e], horizon, quantiles)
        points.append(out["point"])
        quants.append(out["quantiles"])
    return np.concatenate(points), np.concatenate(quants)


def _score(model: str, point: np.ndarray, quants: np.ndarray, contexts: np.ndarray,
           targets: np.ndarray, quantiles: list, meta: pd.DataFrame) -> pd.DataFrame:
    """Per-series metric table for one model."""
    scale = np.abs(np.diff(contexts, axis=1)).mean(axis=1) + 1e-8
    err = targets - point
    smape = (2 * np.abs(err) / (np.abs(targets) + np.abs(point) + 1e-8)).mean(axis=1)
    q = np.asarray(quantiles, dtype=np.float32)[None, None, :]
    diff = targets[:, :, None] - quants
    pinball = np.maximum(q * diff, (q - 1) * diff).mean(axis=(1, 2)) / scale
    return pd.DataFrame({"model": model, "series_id": meta["series_id"],
                         "family": meta["family"], "mase": _mase(point, targets, contexts),
                         "smape": smape, "pinball": pinball})


def _summarize(metrics: pd.DataFrame, cfg: PipelineConfig) -> dict:
    """Family aggregates with CIs, plus Holm-corrected paired tests of the model pair.

    A family "strength" is claimed only when the paired per-series MASE
    difference has a Holm-adjusted bootstrap p below alpha and a CI
    excluding zero; the raw ratio is kept as an effect size.
    """
    sc = cfg.stats
    per_family = (metrics.groupby(["model", "family"])[["mase", "smape", "pinball"]]
                  .mean().reset_index())
    if sc.enabled:
        cis = [mean_ci(grp["mase"].to_numpy(), sc.n_boot, cfg.run.seed + i, sc.ci)
               for i, (_, grp) in enumerate(metrics.groupby(["model", "family"]))]
        per_family["mase_lo"] = [c["lo"] for c in cis]
        per_family["mase_hi"] = [c["hi"] for c in cis]
    summary = {"per_family": per_family.to_dict(orient="records"),
               "overall": metrics.groupby("model")[["mase", "smape", "pinball"]]
               .mean().reset_index().to_dict(orient="records")}
    a, b = cfg.comparison_pair()
    wide = metrics.pivot_table(index=["series_id", "family"], columns="model",
                               values="mase").reset_index()
    if a.name not in wide.columns or b.name not in wide.columns:
        return summary
    pivot = per_family.pivot(index="family", columns="model", values="mase")
    ratio = (pivot[a.name] / pivot[b.name]).sort_values()
    summary["mase_ratio"] = {str(k): float(v) for k, v in ratio.items()}

    if not sc.enabled:
        summary["strengths"] = {a.name: [f for f, r in ratio.items() if r < 0.9],
                                b.name: [f for f, r in ratio.items() if r > 1.1]}
        return summary

    tests, pvals = {}, {}
    for i, (fam, grp) in enumerate(wide.groupby("family")):
        if len(grp) < sc.min_series:
            continue
        diff = (grp[b.name] - grp[a.name]).to_numpy()
        res = paired_bootstrap(diff, sc.n_boot, cfg.run.seed + 40 + i, sc.ci)
        if res is not None:
            tests[str(fam)], pvals[str(fam)] = res, res["p"]
    adjusted = holm(pvals) if pvals else {}
    family_tests = []
    for fam, res in tests.items():
        p_holm = adjusted[fam]
        favored = "none"
        if p_holm < sc.alpha and res["lo"] > 0:
            favored = a.name
        elif p_holm < sc.alpha and res["hi"] < 0:
            favored = b.name
        family_tests.append({"family": fam, "ratio": summary["mase_ratio"].get(fam),
                             **res, "p_holm": p_holm, "favored": favored})
    summary["family_tests"] = sorted(family_tests, key=lambda t: t["p_holm"])
    summary["overall_test"] = paired_bootstrap(
        (wide[b.name] - wide[a.name]).to_numpy(), sc.n_boot, cfg.run.seed + 90, sc.ci)
    summary["strengths"] = {
        a.name: [t["family"] for t in family_tests if t["favored"] == a.name],
        b.name: [t["family"] for t in family_tests if t["favored"] == b.name],
    }
    summary["alpha"] = sc.alpha
    return summary
