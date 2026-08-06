"""Exemplar case studies: a handful of concrete series told end to end.

Aggregate statistics answer "which model is better on this family"; exemplars
answer "what does that difference actually look like". For each family
(ranked by the size of the cross-model MASE gap) a few representative series
are selected from the L0 metric table — the series with the largest gap and
series at interior quantiles of the gap distribution — and for each one the
pipeline records the raw context and target, both models' forecasts, the
per-layer skip-lens forecast quality (where in depth the forecast forms on
this particular series), and a window-pooled attention map (where the model
looks while forming it). Everything is stored as plain arrays so the report
can render the case studies without touching any model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.alignment import pooling_matrix
from ..extraction.store import ActivationStore
from ..utils import log, save_json
from .lens import skip_lens_forecasts


def run_exemplars(cfg: PipelineConfig, hub, store: ActivationStore,
                  data: BenchmarkData, device: torch.device) -> None:
    """Select exemplar series and record forecasts, lens curves, attention maps."""
    out_dir = cfg.run_dir() / "exemplars"
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = _select_exemplars(cfg, data)
    rows = picks["row"].to_numpy()
    contexts, targets = data.contexts()[rows], data.targets()[rows]
    scale = np.abs(np.diff(contexts, axis=1)).mean(axis=1) + 1e-8

    arrays = {"contexts": contexts.astype(np.float32),
              "targets": targets.astype(np.float32)}
    meta = {"exemplars": picks.drop(columns=["row"]).assign(row=rows).to_dict("records"),
            "models": {}}
    fam_first = picks.groupby("family", sort=False).head(1).index.to_numpy()

    for mcfg in cfg.models:
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        layers = store.layers(mcfg.name)[:: max(1, cfg.lens.layer_stride)]
        seed = cfg.run.seed + 11
        lens, final = skip_lens_forecasts(adapter, layers, contexts, data.horizon,
                                          cfg.l0.quantiles, seed)
        arrays[f"forecast_{mcfg.name}"] = final.astype(np.float32)
        arrays[f"lens_mase_{mcfg.name}"] = _lens_mase(lens, targets, scale)
        model_meta = {"layers": layers,
                      "final_mase": _mase(final, targets, scale).tolist()}
        attn = _exemplar_attention(cfg, adapter, contexts, fam_first)
        if attn is None:
            model_meta["attention"] = {"status": "unsupported"}
        else:
            arrays[f"attn_map_{mcfg.name}"] = attn["maps"]
            model_meta["attention"] = attn["meta"]
        meta["models"][mcfg.name] = model_meta
        if not cfg.run.keep_models_loaded:
            hub.release(mcfg.name)

    np.savez(out_dir / "exemplars.npz", **arrays)
    save_json(out_dir / "exemplars.json", meta)
    log.info("exemplars complete: %d series across %d families",
             len(picks), picks["family"].nunique())


def _select_exemplars(cfg: PipelineConfig, data: BenchmarkData) -> pd.DataFrame:
    """Pick per-family series spanning the cross-model MASE gap distribution.

    Families are ranked by mean absolute gap between the comparison pair and
    capped at `max_families`. Within a family the series at the largest
    absolute gap is always taken; remaining slots go to series nearest evenly
    spaced interior quantiles of the gap, so `per_family=2` yields the extreme
    and the median case.
    """
    a, b = cfg.comparison_pair()
    metrics = pd.read_parquet(cfg.run_dir() / "l0" / "metrics.parquet")
    wide = metrics.pivot_table(index=["series_id", "family"], columns="model",
                               values="mase").reset_index()
    if a.name not in wide.columns or b.name not in wide.columns:
        raise RuntimeError("exemplars require L0 metrics for the comparison pair")
    wide["gap"] = wide[a.name] - wide[b.name]

    row_of = {sid: i for i, sid in enumerate(data.meta["series_id"])}
    fam_rank = (wide.groupby("family")["gap"].apply(lambda g: g.abs().mean())
                .sort_values(ascending=False).index[: cfg.exemplars.max_families])
    picked = []
    for fam in fam_rank:
        grp = wide[wide["family"] == fam].sort_values("gap")
        take = [grp["gap"].abs().idxmax()]
        interior = grp.drop(index=take)
        for q in np.linspace(0.5, 0.25, max(0, cfg.exemplars.per_family - 1)):
            if interior.empty:
                break
            idx = (interior["gap"] - interior["gap"].quantile(q)).abs().idxmin()
            take.append(idx)
            interior = interior.drop(index=idx)
        sub = grp.loc[take, ["series_id", "family", a.name, b.name, "gap"]]
        picked.append(sub)
    picks = pd.concat(picked, ignore_index=True)
    picks["row"] = picks["series_id"].map(row_of)
    if picks["row"].isna().any():
        raise RuntimeError("exemplar series_id not found in benchmark metadata")
    picks["row"] = picks["row"].astype(int)
    # Carried through for the report's exemplar cards (sec 15 A9) -- `None`
    # for series whose tier can't express an archetype, matching L0's own
    # per_archetype fallback-to-family convention rather than inventing one.
    picks["archetype"] = data.meta["archetype"].to_numpy()[picks["row"].to_numpy()]
    return picks


def _exemplar_attention(cfg: PipelineConfig, adapter, contexts: np.ndarray,
                        fam_first: np.ndarray):
    """Window-pooled, head-averaged last-layer attention map per family exemplar.

    Token-level patterns are pooled on both axes with the same overlap
    matrix used for activation alignment, giving a [W, W] map in window units
    that is comparable across tokenizers. Returns None when the adapter
    exposes no patterns.
    """
    probe = adapter.prepare(contexts[:1])
    pats = adapter.attention_patterns(probe)
    if pats is None:
        return None
    pool = pooling_matrix(adapter.token_time_spans(), cfg.data.context_len,
                          cfg.alignment.window)
    last = list(pats)[-1]
    maps = []
    for row in fam_first:
        with torch.no_grad():
            att = adapter.attention_patterns(adapter.prepare(contexts[row : row + 1]))
        head_avg = att[last].float().mean(dim=(0, 1)).cpu()
        maps.append((pool @ head_avg @ pool.T).cpu().numpy())
    return {"maps": np.stack(maps).astype(np.float32),
            "meta": {"layer": last, "rows": fam_first.tolist()}}


def _lens_mase(lens: np.ndarray, targets: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Per-layer, per-series MASE of skip-lens forecasts, shape [n_layers, n_series]."""
    return np.stack([_mase(lens[li], targets, scale) for li in range(lens.shape[0])])


def _mase(point: np.ndarray, targets: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Per-series MASE matching the L0 definition."""
    return (np.abs(targets - point).mean(axis=1) / scale).astype(np.float32)
