"""Forecast lens — the logit-lens analog for forecasting models.

Answers "where in depth does each model's forecast crystallize?", the
missing bridge between the behavioral level (L0) and every
representation-level analysis. Two complementary lenses per layer:

1. Skip lens: layer-l token states are patched in as the final block's
   output and the model's own head (for Chronos, the full decoder) produces
   a forecast from them. This is the direct logit-lens translation — it uses
   the model's real output pathway and needs no access to head internals,
   only the existing `token_patch` primitive. Like classic logit lens it can
   be miscalibrated at early layers, which is exactly what the second lens
   corrects for.
2. Tuned lens: a held-out ridge probe from each layer's stored window states
   to (a) the model's own final forecast and (b) the true targets. High R^2
   against the final forecast at layer l means the forecast is already
   linearly readable there even if the raw head cannot decode it.

Both models get both lenses, so "TimesFM front-loads then compresses" versus
"Chronos accumulates" becomes a testable statement about where forecast
quality appears in depth, not just where family information does. The skip
lens inherits L3 patching's constraints: one batch per call (contexts capped
at the model batch size) and a forecasting path that calls the backbone once
per prediction.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.extract import capture_raw_tokens
from ..extraction.hooks import token_patch
from ..extraction.store import ActivationStore
from ..utils import batch_slices, capped_take, log, relative_depths, sample_rows, save_json
from .l2_stitching import ridge_r2
from .stats import mean_ci


def run_lens(cfg: PipelineConfig, hub, store: ActivationStore, data: BenchmarkData,
             device: torch.device) -> None:
    """Compute skip-lens and tuned-lens depth curves for every configured model."""
    out_dir = cfg.run_dir() / "lens"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, arrays = {}, {}
    for mcfg in cfg.models:
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        layers = store.layers(mcfg.name)[:: max(1, cfg.lens.layer_stride)]
        result = _model_lens(cfg, adapter, store, data, layers, device)
        meta[mcfg.name] = result["meta"]
        for key, arr in result["arrays"].items():
            arrays[f"{key}_{mcfg.name}"] = arr
        if not cfg.run.keep_models_loaded:
            hub.release(mcfg.name)
    np.savez(out_dir / "curves.npz", **arrays)
    save_json(out_dir / "lens.json", meta)
    for model, m in meta.items():
        log.info("lens %s: final MASE %.3f, crystallization depth %s",
                 model, m["final_mase"], m["crystallization_depth"])


def skip_lens_forecasts(adapter, layers: list, contexts: np.ndarray, horizon: int,
                        quantiles: list, seed: int) -> tuple:
    """Per-layer forecasts with layer-l token states patched into the final block.

    Returns (lens [n_layers, B, horizon], final [B, horizon]). All layers of
    a residual architecture share the final block's hidden size; a model that
    violates that assumption fails loudly at the patch. Shared by the lens
    stage and the exemplar case studies.
    """
    final_block = adapter.final_block_name()
    clean = capture_raw_tokens(adapter, contexts, layers)
    torch.manual_seed(seed)
    final = adapter.predict(contexts, horizon, quantiles)["point"]
    lens = np.zeros((len(layers),) + final.shape, dtype=np.float32)
    for li, layer in enumerate(tqdm(layers, desc=f"lens {adapter.name}", leave=False)):
        with token_patch(adapter.module, final_block, adapter.token_slice, clean[layer]):
            torch.manual_seed(seed)
            lens[li] = adapter.predict(contexts, horizon, quantiles)["point"]
    return lens, final


def _model_lens(cfg: PipelineConfig, adapter, store: ActivationStore,
                data: BenchmarkData, layers: list, device) -> dict:
    """Skip- and tuned-lens curves plus crystallization depth for one model.

    Uses a fixed seed (not per-model-advancing rng state) so every configured
    model is scored on the *same* stratified row sample -- `CLAUDE.md` sec 5.3
    found lens numbers moving for an unchanged checkpoint purely because two
    configs' different `max_series` caused `rng.choice` to draw disjoint
    samples; sampling deterministically in `(n, k, seed, strata)` and sharing
    the seed across models removes that confound (`ROADMAP.md` sec 15 A4).
    """
    cap = capped_take(cfg.lens.max_series, n_available=data.n, batch_size=adapter.cfg.batch_size)
    take = cap["n_realized"]
    rows = sample_rows(data.n, take, cfg.run.seed + 8, strata=data.meta["family"].to_numpy())
    contexts, targets = data.contexts()[rows], data.targets()[rows]
    scale = np.abs(np.diff(contexts, axis=1)).mean(axis=1) + 1e-8

    lens_fc, final_fc = skip_lens_forecasts(adapter, layers, contexts,
                                            data.horizon, cfg.l0.quantiles,
                                            cfg.run.seed + 81)
    per_series = np.abs(lens_fc - targets[None]).mean(axis=2) / scale[None]
    agreement = (np.abs(lens_fc - final_fc[None]).mean(axis=2) / scale[None]).mean(axis=1)
    final_mase = float((np.abs(final_fc - targets).mean(axis=1) / scale).mean())
    mase_curve = per_series.mean(axis=1)
    mase_ci = [mean_ci(per_series[li], cfg.stats.n_boot, cfg.run.seed + 82 + li,
                       cfg.stats.ci) for li in range(len(layers))] \
        if cfg.stats.enabled else [{"value": float(v)} for v in mase_curve]

    depths = relative_depths(len(layers))
    threshold = final_mase * (1.0 + cfg.lens.crystallization_tol)
    crystal_idx = next((i for i, v in enumerate(mase_curve) if v <= threshold), None)
    crystallization = float(depths[crystal_idx]) if crystal_idx is not None else None

    meta = {"layers": layers, "rel_depth": depths.tolist(),
            "final_mase": final_mase, "mase_ci": mase_ci,
            "crystallization_depth": crystallization,
            "crystallization_tol": cfg.lens.crystallization_tol,
            "n_series_skip": int(take), "n_requested_skip": cap["n_requested"],
            "limited_by_skip": cap["limited_by"]}
    arrays = {"skip_mase": mase_curve.astype(np.float32),
              "skip_agreement": agreement.astype(np.float32)}

    if cfg.lens.tuned:
        r2_model, r2_true, n_tuned = _tuned_lens(cfg, adapter, store, data,
                                                 layers, device)
        meta["n_series_tuned"] = n_tuned
        arrays["tuned_r2_model"] = r2_model
        arrays["tuned_r2_true"] = r2_true
    return {"meta": meta, "arrays": arrays}


def _tuned_lens(cfg: PipelineConfig, adapter, store: ActivationStore,
                data: BenchmarkData, layers: list, device) -> tuple:
    """Held-out ridge R^2 per layer against the model's forecast and the targets.

    Features are the series-mean window state concatenated with the final
    window's state, which keeps the probe dimension bounded while retaining
    the recency that forecasts depend on. Splits are by series, matching the
    L2 leakage discipline, and the lambda grid is shared with L2's ridge.
    """
    n = min(data.n, cfg.lens.tuned_max_series)
    rows = sample_rows(data.n, n, cfg.run.seed + 83, strata=data.meta["family"].to_numpy())
    rng = np.random.default_rng(cfg.run.seed + 83)
    n_val = max(2, int(n * cfg.lens.val_frac))
    perm = rng.permutation(n)
    val, train = perm[:n_val], perm[n_val:]

    if store.has_predictions(adapter.name):
        y_model = store.load_predictions(adapter.name)["point"][rows]
    else:
        log.info("lens %s: no stored predictions (L0 not run); forecasting the "
                 "tuned-lens subset directly", adapter.name)
        y_model = predict_rows(adapter, data.contexts()[rows], data.horizon,
                                cfg.l0.quantiles, cfg.run.seed + 84)
    y_true = data.targets()[rows]
    ytr_m = torch.from_numpy(y_model[train].astype(np.float32)).to(device)
    yva_m = torch.from_numpy(y_model[val].astype(np.float32)).to(device)
    ytr_t = torch.from_numpy(y_true[train].astype(np.float32)).to(device)
    yva_t = torch.from_numpy(y_true[val].astype(np.float32)).to(device)

    r2_model, r2_true = np.zeros(len(layers), np.float32), np.zeros(len(layers), np.float32)
    for li, layer in enumerate(layers):
        win = store.load(adapter.name, layer, level="window", rows=rows).astype(np.float32)
        feats = np.concatenate([win.mean(axis=1), win[:, -1]], axis=1)
        x = torch.from_numpy(feats).to(device)
        r2_model[li] = ridge_r2(x[train], ytr_m, x[val], yva_m, cfg.lens.lambdas)
        r2_true[li] = ridge_r2(x[train], ytr_t, x[val], yva_t, cfg.lens.lambdas)
    return r2_model, r2_true, int(n)


def predict_rows(adapter, contexts: np.ndarray, horizon: int, quantiles: list,
                  seed: int) -> np.ndarray:
    """Batched seeded point forecasts for a row subset."""
    out = []
    for s, e in batch_slices(len(contexts), adapter.cfg.batch_size):
        torch.manual_seed(seed + s)
        out.append(adapter.predict(contexts[s:e], horizon, quantiles)["point"])
    return np.concatenate(out)
