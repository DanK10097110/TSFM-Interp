"""Forecast lens — the logit-lens analog for forecasting models.

Answers "where in depth does each model's forecast crystallize?", the
missing bridge between the behavioral level (L0) and every
representation-level analysis. Two complementary lenses per layer:

1. Skip lens: layer-l token states are patched in as the final block's
   output and the model's own head produces a forecast from them -- "head"
   meaning whatever that architecture actually uses, which for an
   encoder-decoder is its entire decoder. This is the direct logit-lens translation — it uses
   the model's real output pathway and needs no access to head internals,
   only the existing `token_patch` primitive. Like classic logit lens it can
   be miscalibrated at early layers, which is exactly what the second lens
   corrects for.
2. Tuned lens: a held-out ridge probe from each layer's stored window states
   to (a) the model's own final forecast and (b) the true targets. High R^2
   against the final forecast at layer l means the forecast is already
   linearly readable there even if the raw head cannot decode it.

Every model in a run gets both lenses, so a claim of the form "this one
front-loads then compresses, that one accumulates" becomes a testable
statement about where forecast quality appears in depth, not just where
family information does. The shapes are named from the measured curves; no
architecture is described here in advance, since this docstring is rendered
verbatim into the report's methods appendix and would otherwise assert
something about models the run may not contain. The skip
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
from ..utils import batch_slices, capped_take, log, sample_rows, save_json
from .depth_axis import depth_axis_for_run
from .l2_stitching import ridge_r2
from .stats import mean_ci


def run_lens(cfg: PipelineConfig, hub, store: ActivationStore, data: BenchmarkData,
             device: torch.device) -> None:
    """Compute skip-lens and tuned-lens depth curves for every configured model."""
    out_dir = cfg.run_dir() / "lens"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, arrays, convergence = {}, {}, {}
    for mcfg in cfg.models:
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        layers = store.layers(mcfg.name)[:: max(1, cfg.lens.layer_stride)]
        result = _model_lens(cfg, adapter, store, data, layers, device)
        meta[mcfg.name] = result["meta"]
        for key, arr in result["arrays"].items():
            arrays[f"{key}_{mcfg.name}"] = arr
        convergence[mcfg.name] = result["convergence"]
        if not cfg.run.keep_models_loaded:
            hub.release(mcfg.name)
    np.savez(out_dir / "curves.npz", **arrays)
    save_json(out_dir / "lens.json", meta)
    _write_convergence(out_dir, convergence)
    for model, m in meta.items():
        if not m.get("skip_lens_available", True):
            log.info("lens %s: skip lens unavailable (%s); tuned lens only",
                     model, m["skip_lens_unavailable_reason"])
            continue
        log.info("lens %s: final MASE %.3f, crystallization depth %s",
                 model, m["final_mase"], m["crystallization_depth"])


CONVERGENCE_RULE = (
    "first captured layer at which mean_h |lens_forecast - final_forecast| / scale "
    "<= tol, scale = mean |diff(context)| (the L0 MASE scale), tol = "
    "lens.crystallization_tol; label-free (no targets enter it); a series that never "
    "gets within tol at any captured layer is recorded as not converged")


def convergence_depths(lens_fc: np.ndarray, final_fc: np.ndarray, scale: np.ndarray,
                       tol: float, depths: np.ndarray) -> dict:
    """Per-series convergence depth of the skip lens onto the model's own forecast.

    `lens_fc` is `[n_layers, B, horizon]`, `final_fc` `[B, horizon]`, `scale`
    `[B]`. The distance of layer l to the final forecast is
    `mean_h |lens_l - final| / scale`; a series converges at the first layer
    whose distance is `<= tol` (`CONVERGENCE_RULE`). Unlike the mean-curve
    `crystallization_depths`, which compares lens MASE against the final MASE
    and so reads the targets, this rule uses only forecasts, so the value is
    available at inference and can serve as a predictor of failure. Returns
    `layer_index` (`-1` when never converged), `rel_depth` (`nan` when never
    converged) and the boolean `converged`.
    """
    dist = np.abs(lens_fc - final_fc[None]).mean(axis=2) / scale[None]
    ok = dist <= tol
    converged = ok.any(axis=0)
    first = np.where(converged, ok.argmax(axis=0), -1)
    rel = np.where(converged, np.asarray(depths, dtype=np.float64)[np.maximum(first, 0)],
                   np.nan)
    return {"layer_index": first.astype(np.int64), "rel_depth": rel,
            "converged": converged}


def _write_convergence(out_dir, convergence: dict) -> None:
    """Persist per-series convergence depths beside, never inside, the legacy artifacts.

    `curves.npz` and `lens.json` keep exactly their legacy keys
    (`CLAUDE.md` invariant 13); the per-series depths live in
    `convergence.npz` (`series_id_/layer_index_/rel_depth_/converged_<model>`)
    with `convergence.json` carrying the rule, the tolerance and, for a model
    whose skip lens is unavailable, the stated reason.
    """
    arrays, meta = {}, {}
    for model, rec in convergence.items():
        meta[model] = {k: v for k, v in rec.items() if k != "arrays"}
        for key, arr in rec.get("arrays", {}).items():
            arrays[f"{key}_{model}"] = arr
    np.savez(out_dir / "convergence.npz", **arrays)
    save_json(out_dir / "convergence.json", meta)


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


def crystallization_depths(mase_curve: np.ndarray, final_mase: np.ndarray,
                           tol: float, depths: np.ndarray) -> list:
    """First relative depth whose skip-lens MASE lands within `tol` of final.

    `mase_curve` is `[n_layers]` or `[n_layers, horizon]`; `final_mase` is a
    scalar or `[horizon]` to match. Returns a plain float (or `None` if the
    curve never crosses) for the 1-D case, and a per-horizon-step list for
    the 2-D case (`ROADMAP.md` sec 16 E12) -- the same crossing rule applied
    independently at each horizon step, reusing the exact skip-lens forecasts
    already computed for the whole-horizon-averaged number, no new forward
    passes. Pulled out as its own function so both shapes share one
    tested implementation instead of the 2-D case reimplementing the 1-D
    logic in a loop.
    """
    curve = np.atleast_2d(mase_curve.T).T if mase_curve.ndim == 1 else mase_curve
    final = np.atleast_1d(final_mase)
    threshold = final * (1.0 + tol)
    out = []
    for h in range(curve.shape[1]):
        idx = next((i for i in range(curve.shape[0]) if curve[i, h] <= threshold[h]), None)
        out.append(float(depths[idx]) if idx is not None else None)
    return out[0] if mase_curve.ndim == 1 else out


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
    skip_ok = adapter.forecast_reads_patched_positions()
    if not skip_ok:
        log.warning(
            "lens %s: skip lens withheld -- this model's forecast head reads positions "
            "that a FINAL-BLOCK patch of the token_slice span cannot reach, so every "
            "layer would return the same forecast and the flat curve would render as a "
            "result. Widening the patch to the whole sequence is not a fix either: it "
            "does vary by layer, but non-monotonically, so it does not answer how much "
            "of the forecast is formed by depth l -- and norm-matching the patched "
            "positions to the final block does not repair that (measured: <0.0002 MASE "
            "change at any block). The tuned lens is unaffected and still runs.",
            adapter.name)

    cap = capped_take(cfg.lens.max_series, n_available=data.n, batch_size=adapter.cfg.batch_size)
    take = cap["n_realized"]
    rows = sample_rows(data.n, take, cfg.run.seed + 8, strata=data.meta["family"].to_numpy())
    contexts, targets = data.contexts()[rows], data.targets()[rows]
    scale = np.abs(np.diff(contexts, axis=1)).mean(axis=1) + 1e-8

    if not skip_ok:
        meta = {"layers": layers,
                "skip_lens_available": False,
                "skip_lens_unavailable_reason":
                    "the forecast head reads positions that a final-block patch of this "
                    "adapter's token_slice span cannot reach, so every layer would "
                    "return an identical forecast",
                "n_series_skip": 0, "n_requested_skip": cap["n_requested"]}
        da0 = depth_axis_for_run(cfg.alignment.depth_axis, store, adapter.name,
                                 layers, adapter=adapter)
        meta["rel_depth"] = da0.coords.tolist()
        meta["depth_axis"] = da0.axis
        meta["depth_axis_degraded_from"] = da0.fallback_from
        arrays = {}
        if cfg.lens.tuned:
            r2_model, r2_true, n_tuned = _tuned_lens(cfg, adapter, store, data,
                                                     layers, device)
            meta["n_series_tuned"] = n_tuned
            arrays["tuned_r2_model"] = r2_model
            arrays["tuned_r2_true"] = r2_true
        return {"meta": meta, "arrays": arrays,
                "convergence": {"available": False, "rule": CONVERGENCE_RULE,
                                "reason": meta["skip_lens_unavailable_reason"]}}

    lens_fc, final_fc = skip_lens_forecasts(adapter, layers, contexts,
                                            data.horizon, cfg.l0.quantiles,
                                            cfg.run.seed + 81)
    per_series = np.abs(lens_fc - targets[None]).mean(axis=2) / scale[None]
    per_series_h = np.abs(lens_fc - targets[None]) / scale[None, :, None]  # [layers, B, horizon]
    agreement = (np.abs(lens_fc - final_fc[None]).mean(axis=2) / scale[None]).mean(axis=1)
    final_mase = float((np.abs(final_fc - targets).mean(axis=1) / scale).mean())
    final_mase_h = (np.abs(final_fc - targets) / scale[:, None]).mean(axis=0)  # [horizon]
    mase_curve = per_series.mean(axis=1)
    mase_ci = [mean_ci(per_series[li], cfg.stats.n_boot, cfg.run.seed + 82 + li,
                       cfg.stats.ci) for li in range(len(layers))] \
        if cfg.stats.enabled else [{"value": float(v)} for v in mase_curve]

    da = depth_axis_for_run(cfg.alignment.depth_axis, store, adapter.name, layers, adapter=adapter)
    depths = da.coords
    crystallization = crystallization_depths(mase_curve, final_mase,
                                              cfg.lens.crystallization_tol, depths)

    meta = {"layers": layers, "rel_depth": depths.tolist(),
            "skip_lens_available": True,
            "depth_axis": da.axis, "depth_axis_degraded_from": da.fallback_from,
            "final_mase": final_mase, "mase_ci": mase_ci,
            "crystallization_depth": crystallization,
            "crystallization_tol": cfg.lens.crystallization_tol,
            "n_series_skip": int(take), "n_requested_skip": cap["n_requested"],
            "limited_by_skip": cap["limited_by"]}
    arrays = {"skip_mase": mase_curve.astype(np.float32),
              "skip_agreement": agreement.astype(np.float32)}

    if cfg.lens.horizon_resolved:
        mase_curve_h = per_series_h.mean(axis=1)  # [n_layers, horizon]
        meta["crystallization_depth_by_horizon"] = crystallization_depths(
            mase_curve_h, final_mase_h, cfg.lens.crystallization_tol, depths)
        arrays["skip_mase_by_horizon"] = mase_curve_h.astype(np.float32)

    if cfg.lens.tuned:
        r2_model, r2_true, n_tuned = _tuned_lens(cfg, adapter, store, data,
                                                 layers, device)
        meta["n_series_tuned"] = n_tuned
        arrays["tuned_r2_model"] = r2_model
        arrays["tuned_r2_true"] = r2_true
    conv_rows, conv = _convergence_for_rows(
        cfg, adapter, data, layers, depths, rows, lens_fc, final_fc, scale)
    return {"meta": meta, "arrays": arrays,
            "convergence": {
                "available": True, "rule": CONVERGENCE_RULE,
                "tol": cfg.lens.crystallization_tol, "layers": layers,
                "rel_depth_axis": depths.tolist(), "n_series": int(len(conv_rows)),
                "depth_max_series": cfg.lens.depth_max_series,
                "n_converged": int(conv["converged"].sum()),
                "arrays": {
                    "series_id": data.meta["series_id"].to_numpy()[conv_rows].astype(str),
                    "rows": conv_rows.astype(np.int64),
                    "layer_index": conv["layer_index"], "rel_depth": conv["rel_depth"],
                    "converged": conv["converged"]}}}


def _convergence_for_rows(cfg: PipelineConfig, adapter, data: BenchmarkData, layers: list,
                          depths: np.ndarray, rows: np.ndarray, lens_fc: np.ndarray,
                          final_fc: np.ndarray, scale: np.ndarray) -> tuple:
    """Per-series convergence depths for the legacy lens rows, or a larger sample.

    With `lens.depth_max_series` at its default of 0 (or no larger than the
    legacy sample) the depths come from the forecasts `_model_lens` already
    holds: no extra forward pass. Otherwise a separate stratified sample is
    forecast in chunks of the model's batch size, since `token_patch` needs
    one batch per call; the legacy curves are untouched either way.
    """
    want = int(cfg.lens.depth_max_series)
    if want <= len(rows):
        return rows, convergence_depths(lens_fc, final_fc, scale,
                                        cfg.lens.crystallization_tol, depths)
    big = sample_rows(data.n, want, cfg.run.seed + 86, strata=data.meta["family"].to_numpy())
    contexts = data.contexts()[big]
    bscale = np.abs(np.diff(contexts, axis=1)).mean(axis=1) + 1e-8
    parts = []
    for s, e in batch_slices(len(big), adapter.cfg.batch_size):
        lf, ff = skip_lens_forecasts(adapter, layers, contexts[s:e], data.horizon,
                                     cfg.l0.quantiles, cfg.run.seed + 81)
        parts.append(convergence_depths(lf, ff, bscale[s:e], cfg.lens.crystallization_tol,
                                        depths))
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return big, merged


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
