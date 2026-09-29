"""Attention analysis with a time-series twist, plus head-level causality.

Three measurements, each degrading gracefully per model when an adapter
cannot support it:

1. Attention patterns as a function of temporal lag. Per (layer, head), the
   mean attention a query pays to the token `lag` positions behind it,
   aggregated per benchmark family. From these profiles come head taxonomy
   scores (self, previous-token, local, future mass) and a periodicity
   score: excess attention mass at multiples of each family's dominant
   period — the induction-head analog for time series ("attend to the token
   following a similar past motif" becomes "attend at seasonal lags").
2. Head and MLP mean-ablation. Each attention head's slice of the output
   projection input (and each block's MLP output) is fixed to its mean over
   the benchmark subset, and the resulting ΔMASE is scored overall and per
   family. This is the component-level causal map: which heads and blocks a
   family's forecast quality actually depends on.
3. First-step decoder cross-attention (encoder-decoder models only): where
   the decoder reads the context when it starts forecasting, a direct if
   partial view into the otherwise-invisible decoder stage.

Pattern analysis works in token space; for patch tokenizers, sub-patch
periods are invisible by construction and lag units are converted to time
steps via the median token span for cross-model comparability.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.hooks import (ActivationCatcher, InputCatcher,
                                input_slice_ablate, output_mean_ablate)
from ..extraction.store import ActivationStore
from ..utils import batch_slices, log, sample_rows, save_json
from .lens import predict_rows
from .stats import dominant_period as _dominant_period


def _safe(fn, *args):
    """Run an optional per-model analysis step without letting it take down
    every other model's results.

    Adapters vary wildly in what they expose (hookable heads, exposed
    attention weights, an encoder to cross-attend from); returning `None`
    from the adapter side means "this architecture genuinely doesn't
    support this," which is expected and already handled as "unsupported."
    An *exception*, though, means something broke (a bug in a new or edited
    adapter, a version mismatch) — that should surface as a clear per-model
    error in the report, not crash the whole attention stage for every
    other configured model.
    """
    try:
        return fn(*args), None
    except Exception as exc:
        return None, str(exc)


def run_attention(cfg: PipelineConfig, hub, store: ActivationStore,
                  data: BenchmarkData, device: torch.device) -> None:
    """Pattern, ablation, and cross-attention analyses for every configured model."""
    out_dir = cfg.run_dir() / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, arrays, pattern_results = {}, {}, {}
    for mcfg in cfg.models:
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        model_meta = {}
        if cfg.attention.patterns:
            pat, err = _safe(_pattern_analysis, cfg, adapter, data)
            if err is not None:
                log.warning("attention %s: pattern analysis failed (%s)", adapter.name, err)
                model_meta["patterns"] = {"status": "error", "reason": err}
            elif pat is None:
                model_meta["patterns"] = {"status": "unsupported"}
            else:
                model_meta["patterns"] = pat["meta"]
                arrays[f"lag_profile_{mcfg.name}"] = pat["profile"]
                pattern_results[mcfg.name] = pat
                if pat.get("cross") is not None:
                    arrays[f"cross_profile_{mcfg.name}"] = pat["cross"]
        if cfg.attention.ablation:
            abl, err = _safe(_ablation_analysis, cfg, adapter, store, data)
            if err is not None:
                log.warning("attention %s: ablation analysis failed (%s)", adapter.name, err)
                model_meta["ablation"] = {"status": "error", "reason": err}
            elif abl is None:
                model_meta["ablation"] = {"status": "unsupported"}
            else:
                model_meta["ablation"] = abl["meta"]
                for key in ("head_delta", "head_delta_family", "mlp_delta",
                            "mlp_delta_family"):
                    if key in abl:
                        arrays[f"{key}_{mcfg.name}"] = abl[key]
        meta[mcfg.name] = model_meta
        if not cfg.run.keep_models_loaded:
            hub.release(mcfg.name)
    _add_matched_resolution(cfg, meta, arrays, pattern_results)
    np.savez(out_dir / "arrays.npz", **arrays)
    save_json(out_dir / "meta.json", meta)
    log.info("attention complete: %s", {m: sorted(v) for m, v in meta.items()})


def _add_matched_resolution(cfg: PipelineConfig, meta: dict, arrays: dict,
                            pattern_results: dict) -> None:
    """Add the token-resolution-matched taxonomy alongside the native one.

    ROADMAP.md sec 18 F5. Runs as a post-pass rather than inside the
    per-model loop for two reasons: the common bin width is the *coarsest*
    model's token width, which is not known until every model has been seen,
    and doing it here needs no model resident, so nothing about the stage's
    VRAM profile changes.

    Native scores are left exactly as they were -- this adds keys, it does
    not replace any -- so every previously recorded attention number stays
    reproducible (`CLAUDE.md` sec 2.1). `attention.resolution_mode` records
    which of the two a cross-model claim in the report may cite.
    """
    if not pattern_results:
        return
    widths = {m: p["token_width"] for m, p in pattern_results.items()}
    bin_width = max(widths.values())
    coarsest = max(widths, key=widths.get)
    for name, pat in pattern_results.items():
        matched = matched_head_scores(cfg, pat, bin_width)
        block = meta[name]["patterns"]
        block["head_scores_matched"] = matched["head_scores"]
        block["resolution"] = {
            "mode": cfg.attention.resolution_mode,
            "bin_width_steps": float(bin_width),
            "coarsest_model": coarsest,
            "finest_resolvable_lag_steps": float(widths[name]),
            "is_identity": bool(abs(widths[name] - bin_width) < 1e-9),
        }
        arrays[f"lag_profile_matched_{name}"] = matched["profile"]
    log.info("attention: matched lag resolution = %.1f steps (coarsest model '%s')",
             bin_width, coarsest)


def _pattern_analysis(cfg: PipelineConfig, adapter, data: BenchmarkData):
    """Family-conditioned lag profiles, head scores, and cross-attention."""
    probe = adapter.prepare(data.contexts()[:1])
    if not adapter.attention_patterns(probe):
        log.info("attention %s: patterns unsupported by adapter", adapter.name)
        return None

    spans = adapter.token_time_spans()
    token_width = float(np.median(spans[:, 1] - spans[:, 0]))
    n_tokens = spans.shape[0]
    max_lag = int(min(cfg.attention.max_lag_tokens, n_tokens - 1))

    rng = np.random.default_rng(cfg.run.seed + 9)
    families = sorted(set(data.families))
    per_fam = max(1, cfg.attention.max_series // max(1, len(families)))

    fam_profiles, fam_extra, fam_counts, fam_periods = {}, {}, {}, {}
    layer_names = None
    for fam in families:
        fam_rows = np.flatnonzero(data.families == fam)
        rows = np.sort(rng.choice(fam_rows, size=min(len(fam_rows), per_fam),
                                  replace=False))
        contexts = data.contexts()[rows]
        fam_periods[fam] = float(np.median([_dominant_period(c) for c in contexts]))
        prof_sum, extra_sum, count = None, None, 0
        for s, e in batch_slices(len(rows), cfg.attention.batch_series):
            with torch.no_grad():
                pats = adapter.attention_patterns(adapter.prepare(contexts[s:e]))
            layer_names = layer_names or list(pats)
            prof, extra = _batch_profiles(pats, max_lag)
            prof_sum = prof * (e - s) if prof_sum is None else prof_sum + prof * (e - s)
            extra_sum = extra * (e - s) if extra_sum is None else extra_sum + extra * (e - s)
            count += e - s
        fam_profiles[fam], fam_extra[fam], fam_counts[fam] = \
            prof_sum / count, extra_sum / count, count

    total = sum(fam_counts.values())
    profile = sum(fam_profiles[f] * fam_counts[f] for f in families) / total
    extra = sum(fam_extra[f] * fam_counts[f] for f in families) / total
    heads = _head_scores(cfg, profile, extra, fam_profiles, fam_periods,
                         token_width, layer_names)

    cross, cross_meta = _cross_attention(cfg, adapter, data, rng, token_width)
    meta = {"layers": layer_names, "token_width": token_width,
            "max_lag_tokens": max_lag,
            "family_periods_steps": fam_periods,
            "n_series": int(total), "head_scores": heads,
            "cross_attention": cross_meta}
    # The per-family pieces are returned so F5's matched-resolution pass can
    # rerun the taxonomy on a rebinned axis without a second forward pass;
    # they are in-memory only and never serialized (they are large and the
    # aggregate profile is what a reader wants).
    return {"meta": meta, "profile": profile.astype(np.float32), "cross": cross,
            "fam_profiles": fam_profiles, "fam_counts": fam_counts,
            "fam_periods": fam_periods, "extra": extra,
            "token_width": token_width, "layer_names": layer_names}


def _batch_profiles(pats: dict, max_lag: int):
    """Lag profiles [L, H, max_lag+1] and (self, prev, local, future) extras [L, H, 4]."""
    profs, extras = [], []
    for att in pats.values():
        att = att.float()
        t = att.shape[-1]
        prof = torch.zeros(att.shape[1], max_lag + 1, device=att.device)
        for k in range(min(max_lag, t - 1) + 1):
            prof[:, k] = torch.diagonal(att, offset=-k, dim1=-2, dim2=-1).mean(dim=(0, 2))
        future = torch.triu(att, diagonal=1).sum(dim=-1).mean(dim=(0, 2))
        local = prof[:, : min(3, prof.shape[1])].sum(dim=1)
        extras.append(torch.stack([prof[:, 0], prof[:, 1] if max_lag >= 1 else
                                   torch.zeros_like(prof[:, 0]), local, future], dim=1))
        profs.append(prof)
    return (torch.stack(profs).cpu().numpy(),
            torch.stack(extras).cpu().numpy())


def _head_scores(cfg: PipelineConfig, profile: np.ndarray, extra: np.ndarray,
                 fam_profiles: dict, fam_periods: dict, token_width: float,
                 layer_names: list) -> dict:
    """Per-head taxonomy scores and the top periodicity heads across families."""
    n_layers, n_heads, n_lags = profile.shape
    period_scores = np.full((n_layers, n_heads), np.nan, dtype=np.float32)
    period_fam = np.zeros((n_layers, n_heads), dtype=object)
    for fam, prof in fam_profiles.items():
        q = int(round(fam_periods[fam] / token_width))
        if q < 2 or q >= n_lags:
            continue
        score = _periodicity(prof, q)
        better = np.isnan(period_scores) | (score > period_scores)
        period_scores[better] = score[better]
        period_fam[better] = fam
    ranked = []
    if not np.all(np.isnan(period_scores)):
        flat = np.argsort(np.nan_to_num(period_scores, nan=-1).ravel())[::-1]
        for idx in flat[: cfg.attention.top_k]:
            li, hi = np.unravel_index(idx, period_scores.shape)
            if np.isnan(period_scores[li, hi]):
                continue
            ranked.append({"layer": layer_names[li], "head": int(hi),
                           "score": float(period_scores[li, hi]),
                           "family": str(period_fam[li, hi])})
    return {"self_mass": extra[:, :, 0].tolist(), "prev_mass": extra[:, :, 1].tolist(),
            "local_mass": extra[:, :, 2].tolist(), "future_mass": extra[:, :, 3].tolist(),
            "periodicity": np.nan_to_num(period_scores, nan=0.0).tolist(),
            "top_periodicity_heads": ranked}


def rebin_lag_profile(profile: np.ndarray, token_width: float,
                      bin_width: float) -> np.ndarray:
    """Move a token-lag profile onto a coarser *physical-time* lag axis.

    ROADMAP.md sec 18 F5. A lag index means a different number of timesteps
    for each model (TimesFM ~32, Chronos-T5 ~1), so comparing lag profiles
    index-for-index compares unlike to unlike -- a "sharper seasonal
    attention" finding would be partly a statement about patch size
    (`CLAUDE.md` sec 12 item 5). Here token lag `k` is placed in bin
    `floor(k * token_width / bin_width)`, i.e. by the physical lag it
    actually represents.

    Mass is **summed**, not averaged: attention mass is additive over
    positions, and every downstream statistic normalizes by the profile's
    own total, so summing is what leaves that normalization meaningful.
    For the coarsest model (`bin_width == token_width`) this is the
    identity, which is the invariant the tests pin.
    """
    n_lags = profile.shape[-1]
    idx = np.floor(np.arange(n_lags) * token_width / bin_width).astype(int)
    n_bins = int(idx[-1]) + 1 if n_lags else 0
    out = np.zeros(profile.shape[:-1] + (n_bins,), dtype=profile.dtype)
    np.add.at(out.reshape(-1, n_bins).T, idx, profile.reshape(-1, n_lags).T)
    return out


def _matched_extras(profile: np.ndarray, native_extra: np.ndarray) -> np.ndarray:
    """Self/prev/local masses recomputed on the matched axis; future carried over.

    `future_mass` is the share of attention pointing forward in time, which
    no rebinning of the backward-lag axis can change -- so it is carried
    across unchanged rather than recomputed, and says so here so a reader
    doesn't mistake the carry-over for an oversight.
    """
    n_bins = profile.shape[-1]
    prev = profile[..., 1] if n_bins > 1 else np.zeros_like(profile[..., 0])
    local = profile[..., : min(3, n_bins)].sum(axis=-1)
    return np.stack([profile[..., 0], prev, local, native_extra[..., 3]], axis=-1)


def matched_head_scores(cfg: PipelineConfig, pat: dict, bin_width: float) -> dict:
    """Recompute the head taxonomy with every model's lag axis binned alike.

    Takes the per-family profiles the native pass already produced, so this
    costs zero extra forward passes -- the rebinning is arithmetic on arrays
    that are already in memory.
    """
    tw = pat["token_width"]
    fam_binned = {f: rebin_lag_profile(p, tw, bin_width)
                  for f, p in pat["fam_profiles"].items()}
    total = sum(pat["fam_counts"].values())
    profile = sum(fam_binned[f] * pat["fam_counts"][f] for f in fam_binned) / total
    extra = _matched_extras(profile, pat["extra"])
    scores = _head_scores(cfg, profile, extra, fam_binned, pat["fam_periods"],
                          bin_width, pat["layer_names"])
    scores["bin_width_steps"] = float(bin_width)
    scores["n_bins"] = int(profile.shape[-1])
    # A family whose seasonal period is under two bins has no periodicity that
    # survives matching -- `_head_scores` skips it (q < 2) and would otherwise
    # just produce a shorter ranking with no trace of why. Naming it here keeps
    # the degradation loud rather than silent (`CLAUDE.md` sec 2.5): an empty
    # matched ranking with every family listed means the coarser model cannot
    # resolve this corpus's seasonality at all, which is a finding, not a gap.
    scores["unresolvable_families"] = sorted(
        f for f, period in pat["fam_periods"].items()
        if int(round(period / bin_width)) < 2)
    return {"head_scores": scores, "profile": profile.astype(np.float32)}


def _periodicity(prof: np.ndarray, q: int) -> np.ndarray:
    """Excess normalized attention mass within ±1 lag of multiples of period q."""
    n_lags = prof.shape[-1]
    norm = prof / (prof.sum(axis=-1, keepdims=True) + 1e-12)
    mask = np.zeros(n_lags, dtype=bool)
    k = q
    while k < n_lags:
        mask[max(0, k - 1): min(n_lags, k + 2)] = True
        k += q
    baseline = mask.mean()
    return norm[..., mask].sum(axis=-1) - baseline


def _cross_attention(cfg: PipelineConfig, adapter, data: BenchmarkData, rng,
                     token_width: float):
    """Mean first-step decoder cross-attention profile over context recency."""
    probe = adapter.prepare(data.contexts()[:1])
    with torch.no_grad():
        sample = adapter.cross_attention_patterns(probe)
    if sample is None:
        return None, {"status": "unsupported"}
    take = min(data.n, cfg.attention.max_series)
    rows = sample_rows(data.n, take, cfg.run.seed + 9, strata=data.meta["family"].to_numpy())
    contexts = data.contexts()[rows]
    acc, count = None, 0
    for s, e in batch_slices(take, cfg.attention.batch_series):
        with torch.no_grad():
            cross = adapter.cross_attention_patterns(adapter.prepare(contexts[s:e]))
        batch_mean = cross.float().mean(dim=1).flip(-1).cpu().numpy()
        acc = batch_mean * (e - s) if acc is None else acc + batch_mean * (e - s)
        count += e - s
    profile = acc / count
    head_mean = profile.mean(axis=1)
    peaks = [float(np.argmax(head_mean[layer]) * token_width)
             for layer in range(head_mean.shape[0])]
    return profile.astype(np.float32), {
        "status": "computed", "n_series": int(count), "token_width": token_width,
        "peak_lag_steps_per_layer": peaks,
        "note": ("first decoding step only; deeper decoder behavior remains a "
                 "documented blind spot")}


def _ablation_setup(cfg: PipelineConfig, data: BenchmarkData):
    """Rows/contexts/scale/seed shared by mean-ablation ΔMASE (this module) and
    H8's single-head seasonal-power scoring (`seasonality_circuit.py`).

    Factored out so the two harnesses provably sample the same rows at the
    same seed rather than trusting two independent copies to agree
    (`CLAUDE.md` sec 11.24) -- the seed/row-sampling constants below are the
    entire contract a reproduction check like H8 Stage 1's depends on.
    """
    acfg = cfg.attention
    take = min(data.n, acfg.ablation_max_series)
    rows = sample_rows(data.n, take, cfg.run.seed + 10, strata=data.meta["family"].to_numpy())
    contexts, targets = data.contexts()[rows], data.targets()[rows]
    scale = np.abs(np.diff(contexts, axis=1)).mean(axis=1) + 1e-8
    families = data.families[rows]
    fam_list = sorted(set(families))
    seed = cfg.run.seed + 11
    return rows, contexts, targets, scale, families, fam_list, seed


def _ablation_analysis(cfg: PipelineConfig, adapter, store: ActivationStore,
                       data: BenchmarkData):
    """Head and MLP mean-ablation ΔMASE, overall and per family."""
    info = adapter.attention_info()
    mlp = adapter.mlp_info()
    if info is None and mlp is None:
        log.info("attention %s: ablation unsupported by adapter", adapter.name)
        return None
    acfg = cfg.attention
    rows, contexts, targets, scale, families, fam_list, seed = \
        _ablation_setup(cfg, data)
    take = len(rows)

    if store.has_predictions(adapter.name):
        f_clean = store.load_predictions(adapter.name)["point"][rows]
    else:
        f_clean = predict_rows(adapter, contexts, data.horizon, cfg.l0.quantiles, seed)
    mase_clean = np.abs(f_clean - targets).mean(axis=1) / scale

    def delta(forecast: np.ndarray) -> np.ndarray:
        return np.abs(forecast - targets).mean(axis=1) / scale - mase_clean

    out, meta = {}, {"n_series": int(take), "families": fam_list}
    if info is not None:
        blocks = info[:: max(1, acfg.head_layer_stride)]
        head_d, head_f = _ablate_heads(cfg, adapter, blocks, contexts, data.horizon,
                                       seed, delta, families, fam_list)
        out["head_delta"], out["head_delta_family"] = head_d, head_f
        meta["head_blocks"] = [b["block"] for b in blocks]
        meta["n_heads"] = int(blocks[0]["n_heads"])
        meta["top_heads"] = _top_entries(head_d, [b["block"] for b in blocks],
                                         acfg.top_k)
    if mlp is not None:
        names = list(mlp)[:: max(1, acfg.head_layer_stride)]
        mlp_d, mlp_f = _ablate_mlps(cfg, adapter, {n: mlp[n] for n in names},
                                    contexts, data.horizon, seed, delta,
                                    families, fam_list)
        out["mlp_delta"], out["mlp_delta_family"] = mlp_d, mlp_f
        meta["mlp_blocks"] = names
    return {"meta": meta, **out}


def _ablate_heads(cfg: PipelineConfig, adapter, blocks: list, contexts: np.ndarray,
                  horizon: int, seed: int, delta, families: np.ndarray,
                  fam_list: list, on_forecast=None):
    """Mean-ablate every (block, head) via its o_proj input slice and score ΔMASE.

    `on_forecast(block_index, head_index, forecast)`, if given, is called
    with each head's raw per-position forecast array before `delta` reduces
    it to a ΔMASE scalar -- ROADMAP.md sec 20 H8 Stage 1 uses this to score a
    *second* metric (`seasonality_circuit.seasonal_power`) from the exact
    same forward pass, guaranteeing its `head_delta` output is bit-for-bit
    the array this function has always returned rather than a second,
    independently-recomputed copy (`CLAUDE.md` sec 11.24). Default `None` is
    a no-op -- this function's return value is unchanged for every existing
    caller.
    """
    means = _oproj_means(adapter, [b["o_proj"] for b in blocks], contexts)
    n_heads = blocks[0]["n_heads"]
    head_d = np.zeros((len(blocks), n_heads), dtype=np.float32)
    head_f = np.zeros((len(blocks), n_heads, len(fam_list)), dtype=np.float32)
    for bi, block in enumerate(tqdm(blocks, desc=f"head ablation {adapter.name}")):
        dh = block["head_dim"]
        for h in range(n_heads):
            sl = slice(h * dh, (h + 1) * dh)
            with input_slice_ablate(adapter.module, block["o_proj"], sl,
                                    means[block["o_proj"]][sl]):
                f = predict_rows(adapter, contexts, horizon, cfg.l0.quantiles, seed)
            if on_forecast is not None:
                on_forecast(bi, h, f)
            d = delta(f)
            head_d[bi, h] = d.mean()
            for fi, fam in enumerate(fam_list):
                head_f[bi, h, fi] = d[families == fam].mean()
    return head_d, head_f


def _ablate_mlps(cfg: PipelineConfig, adapter, mlp: dict, contexts: np.ndarray,
                 horizon: int, seed: int, delta, families: np.ndarray,
                 fam_list: list):
    """Mean-ablate each block's MLP output and score ΔMASE."""
    means = _output_means(adapter, list(mlp.values()), contexts)
    mlp_d = np.zeros(len(mlp), dtype=np.float32)
    mlp_f = np.zeros((len(mlp), len(fam_list)), dtype=np.float32)
    for bi, name in enumerate(tqdm(mlp.values(), desc=f"mlp ablation {adapter.name}",
                                   total=len(mlp))):
        with output_mean_ablate(adapter.module, name, means[name]):
            f = predict_rows(adapter, contexts, horizon, cfg.l0.quantiles, seed)
        d = delta(f)
        mlp_d[bi] = d.mean()
        for fi, fam in enumerate(fam_list):
            mlp_f[bi, fi] = d[families == fam].mean()
    return mlp_d, mlp_f


def _oproj_means(adapter, names: list, contexts: np.ndarray) -> dict:
    """Mean o_proj input vector per module over batch and positions."""
    sums, counts = {n: None for n in names}, {n: 0 for n in names}
    with InputCatcher(adapter.module, names) as catcher:
        for s, e in batch_slices(len(contexts), adapter.cfg.batch_size):
            with torch.no_grad():
                adapter.forward(adapter.prepare(contexts[s:e]))
            for name, x in catcher.collect().items():
                flat = x.float().reshape(-1, x.shape[-1])
                sums[name] = flat.sum(0) if sums[name] is None else sums[name] + flat.sum(0)
                counts[name] += flat.shape[0]
    return {n: (sums[n] / counts[n]).cpu() for n in names}


def _output_means(adapter, names: list, contexts: np.ndarray) -> dict:
    """Mean output vector per module over batch and positions."""
    sums, counts = {n: None for n in names}, {n: 0 for n in names}
    with ActivationCatcher(adapter.module, names) as catcher:
        for s, e in batch_slices(len(contexts), adapter.cfg.batch_size):
            with torch.no_grad():
                adapter.forward(adapter.prepare(contexts[s:e]))
            for name, x in catcher.collect().items():
                flat = x.float().reshape(-1, x.shape[-1])
                sums[name] = flat.sum(0) if sums[name] is None else sums[name] + flat.sum(0)
                counts[name] += flat.shape[0]
    return {n: (sums[n] / counts[n]).cpu() for n in names}


def _top_entries(matrix: np.ndarray, block_names: list, k: int) -> list:
    """The k most damaging (block, head) entries by ΔMASE."""
    flat = np.argsort(matrix.ravel())[::-1][:k]
    out = []
    for idx in flat:
        bi, hi = np.unravel_index(idx, matrix.shape)
        out.append({"layer": block_names[bi], "head": int(hi),
                    "delta_mase": float(matrix[bi, hi])})
    return out
