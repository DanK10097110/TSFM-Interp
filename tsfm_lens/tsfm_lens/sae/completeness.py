"""Forecast COMPLETENESS (ROADMAP.md sec 42 task C): how much of the forecast
change is carried by a small set of SAE features, and how much by the SAE's
error term.

The per-feature battery answers "does this atom move the forecast more than a
random direction". It does not answer the question a reader asks next: is the
forecast mostly *in* the features at all? Reconstruction fidelity is the usual
proxy and does not predict causal clarity (FINDINGS MN-34), so this module
measures the coverage directly, on the forecast.

Per series, with the SAE's own reconstruction `R` of the clean activation `A`
at the target layer, three patched forecasts are compared (all through
`extraction/hooks.py::token_patch`, clean capture with autocast OFF, exactly the
patch path of `sae/response.py::feature_ablation_fingerprints`):

  error term    forecast(patch R)  vs  forecast(no patch)
                what is lost by replacing the activation with its
                reconstruction, i.e. by deleting the error `A - R`;
  top-N set     forecast(patch R with the N top-ranked atoms zeroed)  vs
                forecast(patch R)
                the same removal the battery makes, with the error term kept;
  random set    the same, for N alive atoms drawn to match the set's mean
                activation decile (`shared_input_agreement.matched_null_sets`).

Each Delta is the mean over the horizon of |forecast difference|, divided by the
series' context scale (the context's standard deviation, so series of different
magnitude contribute comparably; the battery's own `level` channel is unscaled
and is not reused). The share explained is the ratio of means,
`mean(Delta_features) / (mean(Delta_features) + mean(Delta_error))`, with a
series-bootstrap 95% interval (the series is the resampling unit); the median of the per-series shares
is recorded beside it because the ratio of means is dominated by series with a large Delta. The two
Deltas have different baselines (recon vs clean), so the share is a descriptive
coverage ratio, not a decomposition of one total, and the artifact says so.

The ranking is an input. `fallback_ranking` (mean activation x decoder norm) is
used only when the caller passes none, and the artifact names which one it used
in `ranking_source`; it is a prominence ranking, not an effect ranking
(prominence != importance), so a low share under it is not evidence against an
attribution ranking.

Controls (CLAUDE.md sec 8): the reach probe (self-patch 0.0, cross-layer
nonzero) gates the target; on the first chunk the patch path is also checked
directly (patching the layer's own clean tokens must reproduce the unpatched
forecast exactly, and predicting twice with no patch must be identical); every
forecast is seeded with the same seed so sampled models compare like with like.
A failed control withholds the target with its reason. Evidence class:
descriptive (within-model interventions, no causal claim about a feature).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..extraction.extract import capture_raw_tokens
from ..extraction.hooks import token_patch
from ..utils import load_json, log, sample_rows, save_json
from .eval import _token_level_replacement
from .response import _feature_ablated_replacement, reach_probe
from .shared_input_agreement import matched_null_sets

DEFAULT_SET_SIZES = (1, 5, 20)
FALLBACK_RANKING = "fallback_mean_activation_x_decoder_norm"
ATTRIBUTION_RANKING = "attribution_horizon_l1"
DELTA_DEFINITION = ("mean over the horizon of |forecast difference| divided by the series' "
                    "context standard deviation")


def battery_ranking(run_dir, model: str, layer: str) -> dict | None:
    """The attribution ranking the ablation battery recorded for one target, or `None`.

    Reads `top_forecast_horizon_l1` (the ranking that passed its known-answer re-gate,
    ROADMAP.md sec 42.3) from the target's `*_ablation.json` `attribution` block, in score
    order. `None` when the artifact is absent, withheld, has no attribution block (a
    default config, or a model whose forecast is not differentiable and whose block says
    `unavailable`), or an empty ranking; the caller then uses the named fallback, and the
    completeness artifact names it as such.
    """
    from .ablation_run import ablation_path
    path = ablation_path(run_dir, model, layer)
    if not path.exists():
        return None
    block = load_json(path).get("attribution") or {}
    ranked = block.get("top_forecast_horizon_l1") if block.get("status") == "ok" else None
    if not ranked:
        return None
    return {"source": ATTRIBUTION_RANKING, "features": [int(e["feature"]) for e in ranked]}


def completeness_path(run_dir) -> Path:
    return Path(run_dir) / "sae" / "completeness.json"


def context_scale(contexts: np.ndarray) -> np.ndarray:
    """Per-series context standard deviation; a constant context falls back to its
    mean absolute value, then to 1.0, so the divisor is never zero."""
    c = np.asarray(contexts, dtype=np.float64)
    scale = c.std(axis=-1)
    alt = np.abs(c).mean(axis=-1)
    scale = np.where(scale > 0, scale, alt)
    return np.where(scale > 0, scale, 1.0)


def fallback_ranking(pooled: np.ndarray, decoder_norm: np.ndarray,
                     alive_mask: np.ndarray) -> list:
    """Alive atoms by `mean series-level activation x decoder norm`, descending, ties
    by atom index. A prominence ranking, named as a fallback in the artifact."""
    score = np.asarray(pooled, dtype=np.float64).mean(axis=0) * np.asarray(decoder_norm,
                                                                           dtype=np.float64)
    alive = np.flatnonzero(np.asarray(alive_mask, dtype=bool))
    order = alive[np.lexsort((alive, -score[alive]))]
    return [int(i) for i in order]


def scaled_delta(patched: np.ndarray, baseline: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """`[B]` mean over the horizon of |patched - baseline| / scale; NaN where either
    forecast is non-finite on that row (never a silent zero)."""
    p = np.asarray(patched, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    d = np.abs(p - b).mean(axis=-1) / scale
    bad = ~(np.isfinite(p).all(axis=-1) & np.isfinite(b).all(axis=-1))
    return np.where(bad, np.nan, d)


def _share(mf: np.ndarray, me: np.ndarray) -> np.ndarray:
    tot = mf + me
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, mf / tot, np.nan)


def _ci(x: np.ndarray) -> list:
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if x.size else [None, None]


def summarize_set_size(df: np.ndarray, de: np.ndarray, dr: np.ndarray, draw_means: np.ndarray,
                       n_boot: int, seed: int) -> dict:
    """The per-target numbers for one set size. `df`, `de` `[S]` per-series Deltas
    of the top-N set and of the error term; `dr` `[S]` the per-series mean over the
    random draws; `draw_means` `[n_draws]` the series-mean Delta of each draw.
    Series with any non-finite value are dropped jointly (and counted). Bootstrap
    resamples series, with the same indices for every statistic."""
    keep = np.isfinite(df) & np.isfinite(de) & np.isfinite(dr)
    n_dropped = int((~keep).sum())
    f, e, r = df[keep], de[keep], dr[keep]
    n = int(keep.sum())
    if n == 0:
        return {"n_series": 0, "n_nonfinite_dropped": n_dropped, "measured": False,
                "reason": "no series with a finite forecast under every arm"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    bf, be, br = f[idx].mean(axis=1), e[idx].mean(axis=1), r[idx].mean(axis=1)
    mf, me, mr = float(f.mean()), float(e.mean()), float(r.mean())
    share = _share(np.array(mf), np.array(me))
    per_series = _share(f, e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(per_series)
        med_boot = np.nanmedian(per_series[idx], axis=1)
    dm = np.asarray(draw_means, dtype=np.float64)
    dm = dm[np.isfinite(dm)]
    return {
        "measured": True, "n_series": n, "n_nonfinite_dropped": n_dropped,
        "delta_features": mf, "delta_features_ci": _ci(bf),
        "delta_error": me, "delta_error_ci": _ci(be),
        "share_explained": None if not np.isfinite(share) else float(share),
        "share_explained_ci": _ci(_share(bf, be)),
        "share_explained_median_series": float(med) if np.isfinite(med) else None,
        "share_explained_median_series_ci": _ci(med_boot),
        "delta_random": mr, "delta_random_ci": _ci(br),
        "excess_over_random": mf - mr, "excess_over_random_ci": _ci(bf - br),
        "frac_series_features_exceed_random": float((f > r).mean()),
        "n_random_draws": int(dm.size),
        "random_draw_means": {"min": float(dm.min()), "median": float(np.median(dm)),
                              "max": float(dm.max())} if dm.size else None,
        "p_vs_random_draws": float((1 + (dm >= mf).sum()) / (dm.size + 1)) if dm.size else None,
    }


def _predict_seeded(adapter, contexts, horizon, quantiles, seed):
    torch.manual_seed(seed)
    return adapter.predict(contexts, horizon, quantiles)["point"]


def completeness_target(cfg, adapter, layer: str, sae, data, device, *, pooled: np.ndarray,
                        alive_mask: np.ndarray, ranking: list | None = None,
                        ranking_source: str | None = None, set_sizes=DEFAULT_SET_SIZES,
                        n_random: int = 16, max_series: int = 64, n_boot: int = 1000,
                        seed_offset: int = 420) -> dict:
    """The completeness record of one (model, layer) target.

    `pooled` is `[n_series, dict_size]` series-level SAE features in `data`'s own
    order (used only for the fallback ranking and the activation matching).
    `ranking` is an ordered list of atom indices, best first; `None` means the
    named fallback. Returns `{"withheld": True, ...}` with a reason when the reach
    probe or a patch-path control fails.
    """
    pooled = np.asarray(pooled, dtype=np.float64)
    if pooled.ndim != 2 or pooled.shape[0] != data.n:
        raise ValueError(f"completeness_target: `pooled` must be series-level "
                         f"[n_series={data.n}, dict_size], got {pooled.shape}")
    alive_mask = np.asarray(alive_mask, dtype=bool)
    if ranking is None:
        ranking = fallback_ranking(pooled, sae.W_dec.detach().norm(dim=1).cpu().numpy(), alive_mask)
        ranking_source = FALLBACK_RANKING
    ranking = [int(i) for i in ranking]
    if len(set(ranking)) != len(ranking):
        raise ValueError("completeness_target: `ranking` repeats an atom")
    set_sizes = sorted({int(n) for n in set_sizes})
    base = {"ranking_source": ranking_source or "caller", "ranking_head": ranking[:max(set_sizes)],
            "delta_definition": DELTA_DEFINITION, "set_sizes": set_sizes}

    reach = reach_probe(cfg, adapter, layer, data, device, max_series=16,
                        seed_offset=seed_offset - 10)
    if not reach["reachable"]:
        log.warning("completeness: %s/%s withheld -- %s", adapter.name, layer, reach["reason"])
        return {**base, "reach": reach, "withheld": True, "reason": reach["reason"]}
    adapter.ensure_loaded()

    cap = min(int(max_series), data.n)
    rows = sample_rows(data.n, cap, cfg.run.seed + seed_offset, strata=data.families)
    batch = max(1, int(adapter.cfg.batch_size))
    seed = cfg.run.seed + seed_offset
    horizon, quantiles = cfg.data.horizon, cfg.l0.quantiles
    all_contexts = data.contexts()

    ctx = SimpleNamespace(alive_mask=alive_mask, pooled=pooled)
    draws, diags = {}, {}
    for k, n in enumerate(set_sizes):
        top = ranking[:n]
        if len(top) < n:
            diags[n] = {"note": f"ranking has {len(top)} atoms, fewer than {n}",
                        "effective_set_size": len(top)}
        if not top:
            draws[n] = []
            continue
        draws[n], diag = matched_null_sets(ctx, top, rows, n_random, seed + 1 + k)
        diags[n] = {**diags.get(n, {}), **diag, "effective_set_size": len(top)}

    S = len(rows)
    d_err = np.full(S, np.nan)
    d_feat = {n: np.full(S, np.nan) for n in set_sizes}
    d_rand = {n: np.full((len(draws[n]), S), np.nan) for n in set_sizes}
    controls = {"self_patch_max_abs": None, "repeat_max_abs": None}
    for start in range(0, S, batch):
        sl = slice(start, min(start + batch, S))
        contexts = all_contexts[rows[sl]]
        scale = context_scale(contexts)
        clean = capture_raw_tokens(adapter, contexts, [layer])[layer]

        def _patched(replacement):
            with token_patch(adapter.module, layer, adapter.token_slice, replacement):
                return _predict_seeded(adapter, contexts, horizon, quantiles, seed)

        unpatched = _predict_seeded(adapter, contexts, horizon, quantiles, seed)
        if start == 0:
            again = _predict_seeded(adapter, contexts, horizon, quantiles, seed)
            self_patch = _patched(clean)
            controls["repeat_max_abs"] = float(np.nanmax(np.abs(again - unpatched)))
            controls["self_patch_max_abs"] = float(np.nanmax(np.abs(self_patch - unpatched)))
            if controls["repeat_max_abs"] != 0.0 or controls["self_patch_max_abs"] != 0.0:
                why = ("two seeded predictions with no patch differ" if controls["repeat_max_abs"]
                       else "patching the layer's own clean tokens does not reproduce the "
                            "unpatched forecast")
                log.warning("completeness: %s/%s withheld -- %s", adapter.name, layer, why)
                return {**base, "reach": reach, "controls": controls, "withheld": True,
                        "reason": why}
        recon = _token_level_replacement(clean, sae, device)
        baseline = _patched(recon)
        d_err[sl] = scaled_delta(baseline, unpatched, scale)
        for n in set_sizes:
            top = ranking[:n]
            if top:
                d_feat[n][sl] = scaled_delta(
                    _patched(_feature_ablated_replacement(clean, sae, device, top)),
                    baseline, scale)
            for j, members in enumerate(draws[n]):
                d_rand[n][j, sl] = scaled_delta(
                    _patched(_feature_ablated_replacement(clean, sae, device, members)),
                    baseline, scale)

    per_n = {}
    for k, n in enumerate(set_sizes):
        dr = d_rand[n]
        if dr.shape[0]:
            rand_series = np.nanmean(dr, axis=0)
            draw_means = np.nanmean(dr, axis=1)
        else:
            rand_series = np.full(S, np.nan)
            draw_means = np.array([])
        rec = summarize_set_size(d_feat[n], d_err, rand_series, draw_means, n_boot, seed + 100 + k)
        rec["matching"] = diags.get(n, {})
        per_n[str(n)] = rec
    log.info("completeness: %s/%s: %d series, ranking %s", adapter.name, layer, S,
             base["ranking_source"])
    return {**base, "reach": reach, "controls": controls, "withheld": False,
            "n_series_requested": int(cap), "series_rows": [int(r) for r in rows],
            "n_boot": int(n_boot), "n_random": int(n_random),
            "per_set_size": per_n,
            "per_series": {"delta_error": [None if not np.isfinite(v) else float(v) for v in d_err],
                           "delta_features": {str(n): [None if not np.isfinite(v) else float(v)
                                                       for v in d_feat[n]] for n in set_sizes}}}


def run_completeness(cfg, run_dir, hub, store, data, device, targets: list | None = None,
                     rankings: dict | None = None, out_path=None) -> dict:
    """`completeness_target` over every trained target of `run_dir`'s `sae` stage
    (or `targets`), writing `sae/completeness.json` under `cfg.run_dir()` (or
    `out_path`, which a read-only study of a finished run must pass).

    `rankings` maps `"model/layer"` to `{"source": str, "features": [atom, ...]}`;
    a target without an entry uses the battery's recorded attribution ranking
    (`battery_ranking`) and, failing that, the named fallback. Reads the persisted
    series-level features when present, else encodes them from the store.
    """
    from .ablation_run import checkpoint_path
    from .ground_truth import encode_series_level
    from .response import alive_feature_mask
    from .train import load_all_windows, load_sae_checkpoint

    run_dir = Path(run_dir)
    c = cfg.concepts
    meta = load_json(run_dir / "sae" / "meta.json")
    if targets is None:
        targets = [tuple(k.split("/", 1)) for k in meta]
    sizes = tuple(c.completeness_set_sizes or DEFAULT_SET_SIZES)
    out = {"schema_version": 1, "evidence_class": "descriptive",
           "delta_definition": DELTA_DEFINITION, "set_sizes": list(sizes), "targets": {}}
    for model, layer in targets:
        key = f"{model}/{layer}"
        ckpt = checkpoint_path(run_dir, model, layer)
        if not ckpt.exists():
            out["targets"][key] = {"skipped": True, "reason": f"no trained SAE checkpoint at {ckpt}"}
            continue
        sae = load_sae_checkpoint(str(ckpt)).to(device)
        if (meta.get(key) or {}).get("features_persisted"):
            pooled = np.asarray(store.load(model, layer, level="series", space="sae"),
                                dtype=np.float64)
        else:
            pooled = encode_series_level(sae, store, model, layer, np.arange(data.n), device)
        alive = alive_feature_mask(sae, load_all_windows(store, model, layer), device)
        given = (rankings or {}).get(key) or battery_ranking(run_dir, model, layer)
        out["targets"][key] = {"model": model, "layer": layer, **completeness_target(
            cfg, hub.get(model), layer, sae, data, device, pooled=pooled, alive_mask=alive,
            ranking=given["features"] if given else None,
            ranking_source=given["source"] if given else None, set_sizes=sizes,
            n_random=c.completeness_n_random, max_series=c.completeness_max_series,
            n_boot=c.completeness_n_boot)}
        if not cfg.run.keep_models_loaded:
            hub.release(model)
    save_json(out_path or completeness_path(cfg.run_dir()), out)
    return out
