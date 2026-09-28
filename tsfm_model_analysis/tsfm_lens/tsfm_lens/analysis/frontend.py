"""Input front-end diagnostics -- the `frontend` pipeline stage (`ROADMAP.md`
sec 16 E17).

Every later stage in this repo asks what a model does WITH its input once
it's inside a layer. This stage asks the prior question: what does the
model do TO its input before any layer runs at all? Four diagnostics, all
cheap (tokenizer calls and a handful of `predict()` calls -- no activation
store, no `extract` dependency, so `--stages frontend` is a valid
standalone run against a checkpoint with nothing else built, exactly like
`budget`):

1. **Quantization resolution / dynamic range** -- for a re-quantizing
   tokenizer (Chronos-T5's `MeanScaleUniformBins`), how coarse is one
   quantization step relative to a series' own amplitude, and how much of
   the series gets clamped into the tokenizer's extreme (saturating) bin.
   Pure stats in `analysis/quantization_resolution.py`. `not_applicable`
   for continuous-embedding architectures (TimesFM/Sundial/Chronos-Bolt/
   Chronos-2), which have no bin geometry for this probe to measure.
2. **Scale-equivariance** -- does `predict(k * context) / k` match
   `predict(context)`? Pure stats in `analysis/scale_equivariance.py`.
3. **Context-truncation from the back** -- what happens to forecast
   quality when the most RECENT context is what's missing (a data-
   staleness/reporting-lag scenario), as opposed to the already-built
   `phase_sensitivity.py` (sub-patch-width front shifts) and
   `context_scaling.py` (E20, drops OLD history, keeps recent). Pure stats
   in `analysis/context_truncation.py`.
4. **NaN / missing-timestep handling** -- does `predict()` error, silently
   propagate NaN into the forecast, or genuinely handle a missing value in
   the context? Pure stats in `analysis/nan_handling.py`.

Each diagnostic degrades independently and loudly (`CLAUDE.md` sec 2.5): a
model lacking the capability a diagnostic needs gets an explicit
`not_applicable`/`unmeasurable` record with a stated reason, never a
fabricated zero or a silently missing key. A model whose `predict()` raises
for a specific probe input (an extreme scale factor, a very short truncated
context) has that single point logged and skipped rather than killing the
whole stage.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..models.base import CapabilityUnavailable
from ..utils import batch_slices, log, sample_rows, save_json
from .context_truncation import context_truncation_stats
from .context_scaling import context_length_values
from .nan_handling import nan_handling_stats
from .quantization_resolution import quantization_resolution_stats
from .scale_equivariance import scale_equivariance_stats
from .stats import _mase_scale
from .stats import mase as _mase


_PREDICT_SEED = 0


def _predict_batched(adapter, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
    """`adapter.predict(contexts, ...)`, chunked by the model's own configured
    `batch_size` rather than passed as one call over every requested series.

    `ChronosAdapter.predict()` (and any adapter whose `predict()` samples
    multiple trajectories internally) has no batch-size ceiling of its own --
    it hands the WHOLE input batch straight to the underlying library, which
    then further multiplies it by `num_samples` before running the model.
    `analysis/l0_behavioral.py` already knows this and chunks every
    `predict()` call via `batch_slices(len(contexts), adapter.cfg.batch_size)`
    (`CLAUDE.md`'s own "VRAM knobs" list names `models[*].batch_size` as
    memory-critical for exactly this reason); this stage's diagnostics call
    `predict()` directly and had not been doing the same, which is a real
    bug, not a hypothetical one -- see the frontend-stage trap entry this
    session added to the report. Every diagnostic that calls `predict()`
    over a multi-series batch must call it through here.

    Each batch is reseeded with the same `_PREDICT_SEED` (CLAUDE.md sec 8,
    "sampled models need seeded predict()"): every diagnostic here compares
    two `predict()` calls on the same rows (original vs rescaled, full vs
    truncated context), and for a sampled model (Sundial, Chronos-T5,
    Lag-Llama) an unseeded pair adds sampling noise to the residual --
    measured at 0.357 context-sd units for Sundial and 1.335 for Lag-Llama
    with no rescale at all (spec S1). Deterministic models are unaffected.
    """
    points, quants = [], []
    for s, e in batch_slices(len(contexts), max(1, adapter.cfg.batch_size)):
        torch.manual_seed(_PREDICT_SEED)
        out = adapter.predict(contexts[s:e], horizon, quantiles)
        points.append(out["point"])
        if "quantiles" in out and out["quantiles"] is not None:
            quants.append(out["quantiles"])
    result = {"point": np.concatenate(points, axis=0)}
    if quants:
        result["quantiles"] = np.concatenate(quants, axis=0)
    return result


def _quantization_profile(adapter, contexts: np.ndarray):
    """`(profile, reason)` -- `profile` is `None` (with a stated `reason`)
    for any adapter this probe cannot measure; otherwise a dict carrying the
    live tokenizer object and its bin-center geometry.

    Gated first on `adapter.token_ids()` (the existing, reused signal for
    "does this adapter re-quantize at all" -- `CLAUDE.md` sec 15 A20), then
    on the checkpoint's own tokenizer metadata (`pipeline.tokenizer.centers`,
    the bin CENTERS a `MeanScaleUniformBins`-style tokenizer builds via
    `linspace(low_limit, high_limit, n_bins)` -- read directly, never
    hardcoded, so this keeps working if a checkpoint's bin count changes).
    """
    try:
        prepared = adapter.prepare(contexts)
    except CapabilityUnavailable as exc:
        return None, f"tier {adapter.capability_tier()} (black box): {exc}"
    ids = adapter.token_ids(prepared)
    if ids is None:
        return None, ("adapter.token_ids() is None -- no re-quantizing tokenizer for "
                      "this probe to measure (continuous patch-MLP embedding)")
    tokenizer = getattr(getattr(adapter, "pipeline", None), "tokenizer", None)
    centers = getattr(tokenizer, "centers", None)
    if centers is None or centers.numel() < 2:
        return None, ("adapter.token_ids() reports a re-quantizing tokenizer, but its "
                      "bin-center geometry could not be introspected "
                      "(no pipeline.tokenizer.centers)")
    if not callable(getattr(tokenizer, "context_input_transform", None)):
        return None, ("tokenizer exposes bin centers but no context_input_transform to "
                      "read its per-series scale from")
    return {"tokenizer": tokenizer, "centers": centers.detach().cpu().numpy()}, None


def _run_quantization_resolution(adapter, contexts: np.ndarray, n_boot: int, seed: int) -> dict:
    profile, reason = _quantization_profile(adapter, contexts)
    if profile is None:
        log.info("frontend: quantization-resolution not applicable to '%s' (%s)",
                 adapter.name, reason)
        return {"status": "not_applicable", "reason": reason}
    centers = np.sort(profile["centers"])
    bin_width_scaled = float(np.mean(np.diff(centers)))
    bound = float(np.max(np.abs(centers)))
    tensor = torch.from_numpy(np.ascontiguousarray(contexts)).float()
    _, _, scale = profile["tokenizer"].context_input_transform(tensor)
    scale = scale.detach().cpu().numpy()
    stats = quantization_resolution_stats(contexts, scale, bin_width_scaled, bound,
                                          n_boot=n_boot, seed=seed)
    stats["status"] = "measured"
    return stats


def _release_cuda_after_exception(exc: Exception) -> None:
    """Explicitly drop the caught exception and empty the CUDA cache.

    A bare `except Exception as e:` binds `e` in the enclosing scope for the
    rest of the function, and `e.__traceback__` keeps every frame between the
    raise site and here alive -- including whatever CUDA tensors were live
    inside `predict()`'s own stack frames at the moment it OOM'd. Across a
    sweep that calls `predict()` many times (scale-equivariance's factors,
    context-truncation's lengths), that single retained traceback is enough
    to turn one real OOM into a cascade: memory measured at consecutive
    failures in this stage's own live-checkpoint run climbed 11GiB -> 14GiB
    -> 21GiB across three `predict()` calls that, released promptly, would
    each have needed only their own peak. `del exc` breaks the reference
    before the next iteration allocates; `empty_cache()` returns whatever
    that freed to the allocator immediately rather than waiting for it to be
    reused. A no-op on CPU-only adapters (`torch.cuda.is_available()` gates
    it), so this costs nothing on the mock-adapter test suite.
    """
    del exc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_scale_equivariance(adapter, contexts: np.ndarray, horizon: int, factors: list,
                            n_boot: int, seed: int) -> dict:
    point_orig = _predict_batched(adapter, contexts, horizon, [0.5])["point"]
    context_scale = np.asarray(_mase_scale(contexts, "mean_abs_diff"), dtype=np.float64)
    out = {}
    for i, factor in enumerate(factors):
        try:
            raw = _predict_batched(adapter, contexts * factor, horizon, [0.5])["point"]
        except Exception as e:  # noqa: BLE001 -- an extreme scale factor is exactly
            # the kind of input this probe deliberately stresses; a model that
            # cannot even run at 1000x/0.001x is itself the finding, recorded
            # rather than crashing the whole stage (CLAUDE.md sec 2.5).
            log.warning("frontend: '%s' predict() failed at scale factor %g: %s",
                       adapter.name, factor, e)
            out[str(factor)] = {"status": "error", "error_type": type(e).__name__,
                                "error_msg": str(e)[:300]}
            _release_cuda_after_exception(e)
            continue
        point_unscaled = raw / factor
        stats = scale_equivariance_stats(point_orig, point_unscaled, context_scale,
                                         n_boot=n_boot, seed=seed + i)
        stats["status"] = "measured"
        stats["factor"] = factor
        out[str(factor)] = stats
    return out


def _run_context_truncation(adapter, contexts: np.ndarray, targets: np.ndarray, context_len: int,
                            horizon: int, window: int, n_points: int, min_frac: float,
                            n_boot: int, seed: int) -> dict:
    min_context_len = max(window, int(round(context_len * min_frac)))
    avail_lens = context_length_values(context_len, min_context_len=min_context_len,
                                       window=window, n_points=n_points)
    mase_by_len = {}
    for i, avail_len in enumerate(avail_lens):
        lag = context_len - avail_len
        available_context = contexts[:, :avail_len]
        try:
            point_full = _predict_batched(adapter, available_context, horizon + lag, [0.5])["point"]
        except Exception as e:  # noqa: BLE001 -- a model that cannot forecast far
            # enough to bridge a large staleness gap is a real, reportable
            # limitation, not a bug in this probe (CLAUDE.md sec 2.5).
            log.warning("frontend: '%s' predict() failed at available_context_len=%d "
                       "(lag=%d): %s", adapter.name, avail_len, lag, e)
            _release_cuda_after_exception(e)
            continue
        # Some adapters silently return FEWER than the requested `horizon + lag`
        # steps instead of raising (e.g. a fixed-output-width forecast head) --
        # caught here explicitly rather than letting a short slice silently
        # broadcast-fail deeper in `_mase` with a confusing shape error
        # (CLAUDE.md sec 2.5: a capability shortfall must be a named skip, not
        # an unrelated-looking crash).
        if point_full.shape[1] < lag + horizon:
            log.warning("frontend: '%s' predict() returned only %d steps at "
                       "available_context_len=%d (lag=%d), needed %d -- this adapter "
                       "cannot bridge a staleness gap this large; skipping this point",
                       adapter.name, point_full.shape[1], avail_len, lag, lag + horizon)
            continue
        point_used = point_full[:, lag:lag + horizon]
        mase_by_len[avail_len] = _mase(point_used, targets, available_context)
    if len(mase_by_len) < 3:
        reason = (f"only {len(mase_by_len)} of {len(avail_lens)} available-context "
                 f"lengths produced a usable forecast for '{adapter.name}' -- need at "
                 f"least 3 to characterize a degradation shape")
        log.warning("frontend: context-truncation skipped for '%s' (%s)", adapter.name, reason)
        return {"status": "insufficient_points", "reason": reason,
               "n_successful": len(mase_by_len)}
    stats = context_truncation_stats(mase_by_len, n_boot=n_boot, seed=seed)
    stats["status"] = "measured"
    stats["context_len"] = int(context_len)
    return stats


def _nan_scenarios(context_len: int, nan_frac: float) -> dict:
    k = max(1, int(round(nan_frac * context_len)))
    mid = context_len // 2
    return {
        "front": slice(0, k),
        "middle": slice(max(0, mid - k // 2), max(0, mid - k // 2) + k),
        "back": slice(context_len - k, context_len),
    }


def _run_nan_handling(adapter, contexts: np.ndarray, horizon: int, context_len: int,
                      nan_frac: float) -> dict:
    scenarios = _nan_scenarios(context_len, nan_frac)
    results = []
    for position, sl in scenarios.items():
        corrupted = contexts.copy()
        corrupted[:, sl] = np.nan
        try:
            point = _predict_batched(adapter, corrupted, horizon, [0.5])["point"]
        except Exception as e:  # noqa: BLE001 -- an adapter erroring on NaN input is
            # exactly one of the three possible verdicts this probe reports,
            # not a bug to suppress (CLAUDE.md sec 2.5).
            results.append({"position": position, "raised": True,
                            "error_type": type(e).__name__, "error_msg": str(e)[:300],
                            "output_has_nonfinite": None})
            _release_cuda_after_exception(e)
            continue
        results.append({"position": position, "raised": False, "error_type": None,
                        "output_has_nonfinite": bool(not np.all(np.isfinite(point)))})
    stats = nan_handling_stats(results)
    stats["status"] = "measured"
    return stats


def run_frontend(cfg, hub, data, device) -> dict:
    """Pipeline stage: one front-end-diagnostics record per configured model.

    Needs a loaded model and nothing else -- no activation store, no prior
    analysis -- so like `budget` it can sit anywhere in the DAG and run
    standalone via `--stages frontend`.
    """
    out_dir = cfg.run_dir() / "frontend"
    fc = cfg.frontend
    rows = sample_rows(data.n, fc.max_series, cfg.run.seed + 900, strata=data.families)
    all_contexts = np.asarray(data.contexts()[rows], dtype=np.float32)
    all_targets = np.asarray(data.targets()[rows], dtype=np.float32)
    context_len = int(cfg.data.context_len)
    horizon = int(cfg.data.horizon)
    window = int(cfg.alignment.window)
    n_boot = int(cfg.stats.n_boot)

    records = {}
    for model_cfg in cfg.models:
        adapter = hub.get(model_cfg.name)
        adapter.ensure_loaded()
        record: dict = {}

        if fc.quantization_resolution:
            try:
                record["quantization_resolution"] = _run_quantization_resolution(
                    adapter, all_contexts, n_boot, cfg.run.seed)
            except Exception as e:
                log.warning("frontend: quantization-resolution failed for '%s': %s",
                           model_cfg.name, e)
                record["quantization_resolution"] = {"status": "error", "error": str(e)}
                _release_cuda_after_exception(e)

        if fc.scale_equivariance:
            try:
                record["scale_equivariance"] = _run_scale_equivariance(
                    adapter, all_contexts, horizon, fc.scale_factors, n_boot, cfg.run.seed + 901)
            except Exception as e:
                log.warning("frontend: scale-equivariance failed for '%s': %s",
                           model_cfg.name, e)
                # NOTE: this is a FLAT {status, error} dict, deliberately not
                # nested per-factor -- `_run_scale_equivariance` failed before
                # producing ANY per-factor result (e.g. its baseline predict()
                # call itself raised), so there is no per-factor structure to
                # preserve. Every consumer of this field (the summary log line
                # below, `report.py::_sec_frontend`) must handle both this flat
                # shape and the normal {factor_str: {...}} shape explicitly --
                # do not assume `.items()` always yields per-factor dicts.
                record["scale_equivariance"] = {"status": "error", "error": str(e)}
                _release_cuda_after_exception(e)

        if fc.context_truncation:
            try:
                record["context_truncation"] = _run_context_truncation(
                    adapter, all_contexts, all_targets, context_len, horizon, window,
                    fc.context_truncation_n_points, fc.context_truncation_min_frac,
                    n_boot, cfg.run.seed + 902)
            except Exception as e:
                log.warning("frontend: context-truncation failed for '%s': %s",
                           model_cfg.name, e)
                record["context_truncation"] = {"status": "error", "error": str(e)}
                _release_cuda_after_exception(e)

        if fc.nan_handling:
            try:
                nan_rows = all_contexts[:min(fc.nan_series, all_contexts.shape[0])]
                record["nan_handling"] = _run_nan_handling(
                    adapter, nan_rows, horizon, context_len, fc.nan_frac)
            except Exception as e:
                log.warning("frontend: NaN-handling failed for '%s': %s", model_cfg.name, e)
                record["nan_handling"] = {"status": "error", "error": str(e)}
                _release_cuda_after_exception(e)

        records[model_cfg.name] = record
        se_field = record.get("scale_equivariance")
        if isinstance(se_field, dict) and all(isinstance(v, dict) for v in se_field.values()):
            # Normal shape: {factor_str: {status: ..., ...}, ...}.
            se_summary = {k: v.get("status") for k, v in se_field.items()}
        elif isinstance(se_field, dict):
            # Flat top-level-failure shape (see the comment at the write site
            # above): {"status": "error", "error": "..."}. Summarize as a
            # single status rather than iterating -- iterating this shape's
            # `.items()` yields bare strings, not dicts, which is exactly the
            # AttributeError this branch exists to avoid.
            se_summary = se_field.get("status")
        else:
            se_summary = None
        log.info("frontend: '%s' quantization_resolution=%s scale_equivariance=%s "
                "context_truncation=%s nan_handling=%s", model_cfg.name,
                record.get("quantization_resolution", {}).get("status"),
                se_summary,
                record.get("context_truncation", {}).get("status"),
                record.get("nan_handling", {}).get("status"))
        if not cfg.run.keep_models_loaded:
            hub.release(model_cfg.name)

    payload = {"models": records, "n_series": int(all_contexts.shape[0]),
              "context_len": context_len, "horizon": horizon}
    save_json(out_dir / "frontend.json", payload)
    return payload
