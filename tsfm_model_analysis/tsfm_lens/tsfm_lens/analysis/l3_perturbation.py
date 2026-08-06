"""L3 — perturbation and causal analysis.

A battery of parameterized input corruptions (each targeting one structural
property: seasonality, trend, noise floor, frequency content, level, spikes,
local phase, missing data) produces matched clean/corrupted pairs. Three
measurements follow:

1. Sensitivity fingerprints: per-layer relative activation change under each
   corruption, giving each model a [layers x corruptions] signature of where
   in depth it encodes which property.
2. Cross-model fingerprint agreement: fingerprints interpolated onto a
   shared relative-depth axis and rank-correlated per corruption, so models
   of different depth are compared fairly.
3. Within-model activation patching: cached clean token states are written
   back into a corrupted forward one window at a time, and forecast
   restoration measures where (in depth) and where (in time) the corrupted
   property is causally carried. Patching is within-model by construction;
   comparing the resulting restoration-by-depth curves across models is the
   cross-model claim.

Patching deliberately never overwrites a layer's *entire* token state (every
position at once): in a purely sequential residual stack that operation is
mathematically guaranteed to reach 100% restoration at every single layer,
since the patched state completely determines everything computed downstream
and nothing about the corruption survives past the patch point. That holds
regardless of model or corruption, so it would test nothing. Patching one
alignment window's tokens at a time (`_window_restoration`) leaves every
other position's corruption in place, so the resulting restoration is
genuinely graded by depth and by which part of the context was fixed; the
per-layer curve reported to users is the mean of those per-window
restorations, not a separate whole-context patch.

Sensitivity reuses the extraction store for clean activations, so each
corruption costs one forward pass per model.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.alignment import align, pooling_matrix
from ..extraction.extract import capture_raw_tokens
from ..extraction.hooks import ActivationCatcher, token_patch
from ..extraction.store import ActivationStore
from ..utils import batch_slices, capped_take, log, relative_depths, sample_rows, save_json
from .stats import mean_ci


def corrupt_noise(v: np.ndarray, rng, snr_db: float = 6.0) -> np.ndarray:
    """Add white noise at a fixed SNR relative to each series' power."""
    power = v.var(axis=1, keepdims=True) + 1e-8
    noise_std = np.sqrt(power / (10 ** (snr_db / 10.0)))
    return v + rng.normal(0, 1, v.shape).astype(np.float32) * noise_std


def corrupt_detrend(v: np.ndarray, rng) -> np.ndarray:
    """Remove the per-series linear trend while keeping the level."""
    t = np.arange(v.shape[1], dtype=np.float32)
    tc = t - t.mean()
    slope = ((v - v.mean(axis=1, keepdims=True)) * tc).sum(axis=1) / (tc ** 2).sum()
    return v - slope[:, None] * tc[None, :]


def corrupt_deseasonalize(v: np.ndarray, rng, top_k: int = 2) -> np.ndarray:
    """Notch out each series' top-k spectral peaks (and neighbors), excluding DC."""
    spec = np.fft.rfft(v, axis=1)
    mag = np.abs(spec)
    mag[:, 0] = 0.0
    order = np.argsort(mag, axis=1)[:, ::-1][:, :top_k]
    mask = np.zeros(mag.shape, dtype=bool)
    rows = np.arange(v.shape[0])[:, None]
    for off in (-1, 0, 1):
        mask[rows, np.clip(order + off, 0, mag.shape[1] - 1)] = True
    mask[:, 0] = False
    spec[mask] = 0.0
    return np.fft.irfft(spec, n=v.shape[1], axis=1).astype(np.float32)


def corrupt_frequency_shift(v: np.ndarray, rng, factor: float = 2.0) -> np.ndarray:
    """Resample the time axis so all frequencies scale by `factor`."""
    t = v.shape[1]
    idx = np.clip(np.arange(t, dtype=np.float32) * factor, 0, t - 1)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, t - 1)
    frac = (idx - i0).astype(np.float32)
    return v[:, i0] * (1 - frac)[None, :] + v[:, i1] * frac[None, :]


def corrupt_level_shift(v: np.ndarray, rng, position_frac: float = 0.6,
                        scale: float = 3.0) -> np.ndarray:
    """Inject a step change of +-scale std at a fixed relative position."""
    pos = int(v.shape[1] * position_frac)
    sign = rng.choice([-1.0, 1.0], size=(v.shape[0], 1)).astype(np.float32)
    out = v.copy()
    out[:, pos:] += sign * scale * (v.std(axis=1, keepdims=True) + 1e-6)
    return out


def corrupt_spike(v: np.ndarray, rng, count: int = 3, scale: float = 6.0) -> np.ndarray:
    """Add `count` isolated spikes of +-scale std at random positions."""
    out = v.copy()
    amp = scale * (v.std(axis=1) + 1e-6)
    n = v.shape[0]
    for _ in range(count):
        cols = rng.integers(0, v.shape[1], size=n)
        signs = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        out[np.arange(n), cols] += signs * amp
    return out


def corrupt_smooth(v: np.ndarray, rng, kernel: int = 9) -> np.ndarray:
    """Moving-average filtering that suppresses high-frequency content."""
    return uniform_filter1d(v, size=kernel, axis=1, mode="nearest")


def corrupt_warp(v: np.ndarray, rng, strength: float = 0.15, n_knots: int = 6) -> np.ndarray:
    """Local nonlinear time warp: resample through a smooth random monotone map.

    Unlike `frequency_shift` (a uniform global rescale of the whole series),
    this stretches and compresses different parts of each series by
    different amounts, corrupting local phase/alignment while leaving global
    frequency content roughly unchanged on average — a distinct structural
    property from anything else in this battery.
    """
    n, t = v.shape
    knot_x = np.linspace(0, t - 1, n_knots)
    jitter = rng.normal(0, strength * t / n_knots, size=(n, n_knots))
    knot_y = np.sort(np.clip(knot_x[None, :] + jitter, 0, t - 1), axis=1)
    idx = np.stack([np.interp(np.arange(t), knot_x, knot_y[i]) for i in range(n)])
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, t - 1)
    frac = (idx - i0).astype(np.float32)
    rows = np.arange(n)[:, None]
    return (v[rows, i0] * (1 - frac) + v[rows, i1] * frac).astype(np.float32)


def corrupt_dropout(v: np.ndarray, rng, frac: float = 0.15, n_blocks: int = 3) -> np.ndarray:
    """Blank out `n_blocks` random contiguous spans with the series' own mean.

    Simulates missing or intermittent data (sensor dropout, reporting gaps)
    rather than any continuous-signal distortion, giving the battery a
    "data is simply absent" corruption alongside the shape/spectrum ones.
    """
    out = v.copy()
    n, t = v.shape
    block_len = max(1, int(t * frac / max(1, n_blocks)))
    mean = v.mean(axis=1, keepdims=True)
    rows = np.arange(n)[:, None]
    for _ in range(n_blocks):
        starts = rng.integers(0, max(1, t - block_len + 1), size=n)
        cols = starts[:, None] + np.arange(block_len)[None, :]
        out[rows, cols] = mean
    return out


CORRUPTIONS = {
    "noise": corrupt_noise,
    "detrend": corrupt_detrend,
    "deseasonalize": corrupt_deseasonalize,
    "frequency_shift": corrupt_frequency_shift,
    "level_shift": corrupt_level_shift,
    "spike": corrupt_spike,
    "smooth": corrupt_smooth,
    "warp": corrupt_warp,
    "dropout": corrupt_dropout,
}


# Corruptions with a single scalar knob empirically confirmed monotone
# increasing in perturbation energy (checked directly against a real batch
# of series before trusting it, sec 15 A12, `CLAUDE.md` §2.4): (param, lo,
# hi, is_integer). `noise` is handled separately via a closed-form solve
# (its knob, `snr_db`, is *decreasing* in energy). `detrend` (no free
# parameter), `deseasonalize` (integer `top_k`, not a continuous magnitude
# knob), and `frequency_shift` (`factor` -- empirically non-monotone in
# energy past ~1.5x, confirmed by direct sweep, not assumed) are left
# uncalibrated: reported, never adjusted.
_CALIBRATABLE_RANGES = {
    "level_shift": ("scale", 0.1, 20.0, False),
    "spike": ("scale", 0.1, 30.0, False),
    "smooth": ("kernel", 3, 101, True),
    "warp": ("strength", 0.01, 1.0, False),
    "dropout": ("frac", 0.01, 0.9, False),
}


def _perturbation_energy(clean: np.ndarray, corrupted: np.ndarray) -> float:
    """Mean squared per-timestep perturbation -- the "energy" budget calibration targets."""
    return float(((corrupted - clean) ** 2).mean())


def _perturbation_footprint(clean: np.ndarray, corrupted: np.ndarray, rel_tol: float = 1e-3) -> float:
    """Fraction of timesteps actually changed, relative to the input's own scale (sec 15 A12).

    Reported for every corruption regardless of `calibrate` mode -- this is
    what stops a sparse-by-construction corruption (`spike`, `dropout`) from
    being misread as "the model is robust" just because its aggregate
    behavioral-sensitivity bar is short.
    """
    scale = np.abs(clean).std() + 1e-8
    touched = np.abs(corrupted - clean) > rel_tol * scale
    return float(touched.mean())


def _bisect_param(name: str, base_kwargs: dict, param: str, lo: float, hi: float, is_int: bool,
                  contexts: np.ndarray, target: float, seed: int, n_iter: int = 30):
    """Bisection over one corruption's magnitude parameter to hit a target energy.

    Assumes (confirmed empirically per corruption, see `_CALIBRATABLE_RANGES`)
    that energy is monotone increasing in `param` over `[lo, hi]`. Clamps to
    the range's boundary (rather than extrapolating or raising) when the
    target is out of reach within it, and returns the realized value/energy
    either way so a clamp is a recorded fact, not a silent miss.
    """
    fn = CORRUPTIONS[name]

    def energy_at(val: float) -> float:
        v = int(round(val))
        if is_int and v % 2 == 0:
            v += 1
        kwargs = {**base_kwargs, param: v if is_int else float(val)}
        c = fn(contexts.copy(), np.random.default_rng(seed), **kwargs)
        return _perturbation_energy(contexts, c)

    e_lo, e_hi = energy_at(lo), energy_at(hi)
    if target <= e_lo:
        return lo, e_lo
    if target >= e_hi:
        return hi, e_hi
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        if energy_at(mid) < target:
            lo = mid
        else:
            hi = mid
    val = (lo + hi) / 2.0
    return val, energy_at(val)


def calibrate_corruptions(names: list, configs: dict, contexts: np.ndarray, seed: int) -> tuple:
    """Solve each calibratable corruption's magnitude parameter to a common energy budget.

    Pure numpy, no model in the loop (sec 15 A12 fix item 2) -- the budget
    is the *median* of the battery's own natural (pre-calibration) energies,
    so calibration is anchored to this battery's own existing scale rather
    than an arbitrary external constant. Returns `(adjusted_configs,
    calibration_meta)`; every corruption appears in `calibration_meta`
    (`calibrated: bool`), never dropped.
    """
    natural_energy, natural_footprint = {}, {}
    for i, name in enumerate(names):
        c = CORRUPTIONS[name](contexts.copy(), np.random.default_rng(seed + i), **configs[name])
        natural_energy[name] = _perturbation_energy(contexts, c)
        natural_footprint[name] = _perturbation_footprint(contexts, c)
    target = float(np.median(list(natural_energy.values())))

    adjusted = {name: dict(configs[name]) for name in names}
    meta = {}
    for i, name in enumerate(names):
        calibrated_param = None
        if name == "noise":
            power = float(contexts.var(axis=1).mean() + 1e-8)
            adjusted[name]["snr_db"] = float(10.0 * np.log10(power / max(target, 1e-8)))
            calibrated_param, calibrated_value = "snr_db", adjusted[name]["snr_db"]
        elif name in _CALIBRATABLE_RANGES:
            param, lo, hi, is_int = _CALIBRATABLE_RANGES[name]
            val, _ = _bisect_param(name, configs[name], param, lo, hi, is_int,
                                   contexts, target, seed + i)
            if is_int:
                val = int(round(val))
                if val % 2 == 0:
                    val += 1
            adjusted[name][param] = val
            calibrated_param, calibrated_value = param, val
        c = CORRUPTIONS[name](contexts.copy(), np.random.default_rng(seed + i), **adjusted[name])
        meta[name] = {
            "calibrated": calibrated_param is not None,
            "calibrated_param": calibrated_param,
            "calibrated_value": calibrated_value if calibrated_param else None,
            "target_energy": target,
            "realized_energy": _perturbation_energy(contexts, c),
            "footprint": _perturbation_footprint(contexts, c),
            "natural_energy": natural_energy[name],
            "natural_footprint": natural_footprint[name],
        }
    return adjusted, meta


def corruption_footprint_meta(names: list, configs: dict, contexts: np.ndarray, seed: int) -> dict:
    """Footprint/energy for every corruption without changing any parameter (`calibrate: none`)."""
    meta = {}
    for i, name in enumerate(names):
        c = CORRUPTIONS[name](contexts.copy(), np.random.default_rng(seed + i), **configs[name])
        meta[name] = {"calibrated": False, "footprint": _perturbation_footprint(contexts, c),
                     "energy": _perturbation_energy(contexts, c)}
    return meta


def run_l3(cfg: PipelineConfig, hub, store: ActivationStore, data: BenchmarkData,
           device: torch.device) -> None:
    """Sensitivity fingerprints, cross-model agreement, and activation patching."""
    out_dir = cfg.run_dir() / "l3"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(cfg.l3.corruptions.keys())
    unknown = [n for n in names if n not in CORRUPTIONS]
    if unknown:
        raise ValueError(f"unknown corruptions {unknown}; available: {sorted(CORRUPTIONS)}")

    rows = sample_rows(data.n, cfg.l3.max_series, cfg.run.seed + 3,
                      strata=data.meta["family"].to_numpy())
    contexts = data.contexts()[rows]
    scale = np.abs(np.diff(contexts, axis=1)).mean(axis=1) + 1e-8

    if cfg.l3.calibrate == "none":
        corruption_configs = cfg.l3.corruptions
        calibration_meta = corruption_footprint_meta(names, corruption_configs, contexts,
                                                      cfg.run.seed + 500)
    elif cfg.l3.calibrate == "input_energy":
        corruption_configs, calibration_meta = calibrate_corruptions(
            names, cfg.l3.corruptions, contexts, cfg.run.seed + 500)
        log.info("l3: calibrated corruptions to a common energy budget of %.4g -- %s",
                 calibration_meta[names[0]]["target_energy"],
                 {n: calibration_meta[n].get("calibrated_value") for n in names
                  if calibration_meta[n]["calibrated"]})
    else:
        raise ValueError(f"unknown l3.calibrate {cfg.l3.calibrate!r} "
                         f"(expected 'none' or 'input_energy')")
    corrupted = {c: CORRUPTIONS[c](contexts.copy(), np.random.default_rng(cfg.run.seed + 100 + i),
                                   **corruption_configs[c])
                 for i, c in enumerate(names)}

    a, b = cfg.comparison_pair()
    per_series, beh_series, layer_lists, patching = {}, {}, {}, {}
    for mcfg in (a, b):
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        ps, beh, layers = _sensitivity(cfg, adapter, store, rows, contexts,
                                       corrupted, names, scale)
        per_series[mcfg.name], beh_series[mcfg.name] = ps, beh
        layer_lists[mcfg.name] = layers
        if cfg.l3.patching.enabled:
            patching[mcfg.name] = _patching(
                cfg, adapter, layers, rows, contexts, corrupted, data.horizon,
                targets=data.targets()[rows],
                series_ids=data.meta["series_id"].to_numpy()[rows],
                families=data.meta["family"].to_numpy()[rows])
        if not cfg.run.keep_models_loaded:
            hub.release(mcfg.name)

    fingerprints = {k: v.mean(axis=0) for k, v in per_series.items()}
    agreement = _agreement_with_ci(cfg, per_series[a.name], per_series[b.name], names)
    behavior_ci = {
        model: {names[c]: mean_ci(bs[:, c], cfg.stats.n_boot, cfg.run.seed + 30 + c,
                                  cfg.stats.ci)
                for c in range(len(names))}
        for model, bs in beh_series.items()
    } if cfg.stats.enabled else {}
    np.savez(out_dir / "sensitivity.npz",
             **{f"fingerprint_{k}": v for k, v in fingerprints.items()},
             **{f"per_series_{k}": v for k, v in per_series.items()},
             **{f"behavior_{k}": v.mean(axis=0) for k, v in beh_series.items()})
    save_json(out_dir / "meta.json", {
        "model_a": a.name, "model_b": b.name, "corruptions": names,
        "layers": layer_lists, "agreement": agreement, "behavior_ci": behavior_ci,
        "n_series": int(len(rows)), "calibrate": cfg.l3.calibrate,
        "calibration": calibration_meta,
    })
    if patching:
        win_arrays = {f"restoration_windows_{k}": v["restoration_windows"]
                      for k, v in patching.items() if "restoration_windows" in v}
        verbose_arrays, verbose_meta = {}, {}
        for model, v in patching.items():
            verbose_meta[model] = {}
            for cname, entry in v.get("verbose", {}).items():
                prefix = f"verbose_{model}_{cname}_"
                verbose_arrays[prefix + "grid"] = entry["restoration_grid"]
                verbose_arrays[prefix + "context"] = entry["context"]
                if entry["target"] is not None:
                    verbose_arrays[prefix + "target"] = entry["target"]
                verbose_arrays[prefix + "clean"] = entry["forecast_clean"]
                verbose_arrays[prefix + "corrupted"] = entry["forecast_corrupted"]
                verbose_arrays[prefix + "patched"] = entry["forecast_patched"]
                verbose_meta[model][cname] = {"layer": entry["layer"], "window": entry["window"],
                                              "series_ids": entry["series_ids"],
                                              "families": entry["families"]}
        np.savez(out_dir / "patching.npz",
                 **{f"restoration_{k}": v["restoration"] for k, v in patching.items()},
                 **win_arrays, **verbose_arrays)
        save_json(out_dir / "patching.json", {
            k: {"layers": v["layers"], "corruptions": v["corruptions"],
                "rel_depth": relative_depths(len(v["layers"])).tolist(),
                "windows": v.get("windows", []),
                "window_size": cfg.alignment.window,
                "whole_context_patch": v.get("whole_context_patch", False),
                "verbose": verbose_meta.get(k, {}),
                "n_requested": v.get("n_requested"), "n_realized": v.get("n_realized"),
                "limited_by": v.get("limited_by")}
            for k, v in patching.items()})
    log.info("L3 complete: most divergent corruption = %s", agreement["most_divergent"])


def _sensitivity(cfg: PipelineConfig, adapter, store: ActivationStore, rows: np.ndarray,
                 contexts: np.ndarray, corrupted: dict, names: list, scale: np.ndarray):
    """Per-series, per-layer relative activation deltas and forecast deltas for one model.

    Per-series resolution is kept so downstream agreement statistics can
    bootstrap over series, the exchangeable unit.
    """
    layers = store.layers(adapter.name)
    pool = pooling_matrix(adapter.token_time_spans(), cfg.data.context_len,
                          cfg.alignment.window)
    clean_bank = {layer: store.load(adapter.name, layer, "window", rows) for layer in layers}
    if store.has_predictions(adapter.name):
        f_clean = store.load_predictions(adapter.name)["point"][rows]
    else:
        f_clean = _predict_batched(adapter, contexts, cfg.data.horizon,
                                   cfg.l0.quantiles, cfg.run.seed)

    per_series = np.zeros((len(rows), len(layers), len(names)), dtype=np.float32)
    beh_series = np.zeros((len(rows), len(names)), dtype=np.float32)
    for ci, cname in enumerate(tqdm(names, desc=f"L3 sensitivity {adapter.name}")):
        ctx_c = corrupted[cname]
        with ActivationCatcher(adapter.module, layers) as catcher:
            for s, e in batch_slices(len(rows), adapter.cfg.batch_size):
                with torch.no_grad(), torch.autocast(device_type=adapter.device.type,
                                                     dtype=adapter.dtype,
                                                     enabled=adapter.device.type == "cuda"):
                    adapter.forward(adapter.prepare(ctx_c[s:e]))
                acts = catcher.collect()
                for li, layer in enumerate(layers):
                    corr = align(adapter.postprocess_tokens(layer, acts[layer]).float(),
                                 pool).cpu().numpy()
                    clean = clean_bank[layer][s:e].astype(np.float32)
                    num = np.linalg.norm(corr - clean, axis=-1)
                    den = np.linalg.norm(clean, axis=-1) + 1e-6
                    per_series[s:e, li, ci] = (num / den).mean(axis=1)
        f_corr = _predict_batched(adapter, ctx_c, cfg.data.horizon,
                                  cfg.l0.quantiles, cfg.run.seed)
        beh_series[:, ci] = np.abs(f_corr - f_clean).mean(axis=1) / scale
    return per_series, beh_series, layers


def _patching(cfg: PipelineConfig, adapter, layers: list, rows: np.ndarray,
              contexts: np.ndarray, corrupted: dict, horizon: int,
              targets: np.ndarray = None, series_ids: np.ndarray = None,
              families: np.ndarray = None) -> dict:
    """Layer-by-window clean-into-corrupted patching, aggregated to a depth curve.

    Each (layer, window) cell patches only the tokens belonging to that one
    alignment window, leaving every other position's corruption in place —
    the reported per-layer "restoration" curve is the mean of those
    per-window values. This is deliberate: patching a layer's *entire* token
    state (every position at once) would completely determine everything the
    model computes afterward, so a whole-context patch is restored to 100%
    at every single layer regardless of model, corruption, or depth — a
    guaranteed ceiling effect, not a finding. Falls back to whole-context
    patching only if `per_window` is disabled, in which case the resulting
    curve should be read as a wiring sanity-check, not a depth signal.
    """
    pcfg = cfg.l3.patching
    cap = capped_take(pcfg.max_series, n_available=len(rows), batch_size=adapter.cfg.batch_size)
    take = cap["n_realized"]
    # `contexts`/`targets`/`series_ids`/`families` arrive already row-sampled
    # (by `run_l3`) but still in sorted-index order; a further `[:take]` head
    # slice here would re-introduce the family-order bias `sample_rows` exists
    # to avoid, one level removed (ROADMAP.md sec 15 A4) -- so this second cap
    # is stratified too, applied consistently to every per-series array below.
    sel = (sample_rows(len(rows), take, cfg.run.seed + 77,
                       strata=families if families is not None else None)
          if take < len(rows) else np.arange(len(rows)))
    ctx_clean = contexts[sel]
    targets = targets[sel] if targets is not None else None
    series_ids = series_ids[sel] if series_ids is not None else None
    families = families[sel] if families is not None else None
    layers_p = layers[:: max(1, pcfg.layer_stride)]
    corr_names = [c for c in pcfg.corruptions if c in corrupted]
    clean_tokens = capture_raw_tokens(adapter, ctx_clean, layers_p)
    seed = cfg.run.seed + 7
    f_clean = _predict_once(adapter, ctx_clean, horizon, cfg.l0.quantiles, seed)

    n_windows = cfg.data.context_len // cfg.alignment.window
    win_of_token = _token_windows(adapter.token_time_spans(), cfg.alignment.window,
                                  n_windows)
    windows = list(range(0, n_windows, max(1, pcfg.window_stride))) \
        if pcfg.per_window else []
    n_verbose = min(cfg.report.verbose_series, take) if cfg.report.verbose else 0

    restoration = np.zeros((len(corr_names), len(layers_p)), dtype=np.float32)
    rest_win = np.zeros((len(corr_names), len(layers_p), len(windows)), dtype=np.float32)
    verbose_grid = (np.zeros((len(corr_names), len(layers_p), len(windows), n_verbose),
                             dtype=np.float32)
                    if n_verbose and windows else None)
    verbose = {}
    for ci, cname in enumerate(tqdm(corr_names, desc=f"L3 patching {adapter.name}")):
        ctx_corr = corrupted[cname][sel]
        f_corr = _predict_once(adapter, ctx_corr, horizon, cfg.l0.quantiles, seed)
        damage = np.abs(f_corr - f_clean).mean() + 1e-8
        for li, layer in enumerate(layers_p):
            if windows:
                vals = []
                for wi, w in enumerate(windows):
                    tok_idx = np.flatnonzero(win_of_token == w)
                    if not len(tok_idx):
                        continue
                    v, v_series, _ = _window_restoration(
                        adapter, layer, tok_idx, clean_tokens[layer], ctx_corr,
                        horizon, cfg.l0.quantiles, seed, f_clean, damage)
                    rest_win[ci, li, wi] = v
                    vals.append(v)
                    if verbose_grid is not None:
                        verbose_grid[ci, li, wi] = v_series[:n_verbose]
                restoration[ci, li] = float(np.mean(vals)) if vals else float("nan")
            else:
                with token_patch(adapter.module, layer, adapter.token_slice,
                                 clean_tokens[layer]):
                    f_patch = _predict_once(adapter, ctx_corr, horizon, cfg.l0.quantiles, seed)
                restoration[ci, li] = float(1.0 - np.abs(f_patch - f_clean).mean() / damage)
        if verbose_grid is not None:
            verbose[cname] = _verbose_case(
                adapter, layers_p, windows, win_of_token, clean_tokens, ctx_corr,
                horizon, cfg.l0.quantiles, seed, f_clean, f_corr, damage, rest_win[ci],
                verbose_grid[ci], n_verbose, ctx_clean, targets, series_ids, families)
    out = {"restoration": restoration, "layers": layers_p, "corruptions": corr_names,
           "whole_context_patch": not bool(windows),
           "n_requested": cap["n_requested"], "n_realized": cap["n_realized"],
           "limited_by": cap["limited_by"]}
    if windows:
        out["restoration_windows"] = rest_win
        out["windows"] = windows
    if verbose:
        out["verbose"] = verbose
    return out


def _verbose_case(adapter, layers_p: list, windows: list, win_of_token: np.ndarray,
                  clean_tokens: dict, ctx_corr: np.ndarray, horizon: int, quantiles: list,
                  seed: int, f_clean: np.ndarray, f_corr: np.ndarray, damage: float,
                  rest_win_ci: np.ndarray, verbose_grid_ci: np.ndarray, n_verbose: int,
                  ctx_clean: np.ndarray, targets, series_ids, families) -> dict:
    """A concrete before/after for a handful of series at one corruption.

    Reuses the corpus-wide restoration grid to pick the single (layer, window)
    cell that restored the most on average across the whole sampled batch, then
    replays *only* the patch at that one cell for the first `n_verbose` series
    to recover their actual patched forecast (everything else needed — clean
    and corrupted forecasts, and this cell's full per-series restoration grid
    from `verbose_grid_ci` — is already in hand at no extra cost). This is a
    representative "best patch" example, not each series' own individually
    best cell; the report states that distinction explicitly.
    """
    best_li, best_wi = np.unravel_index(int(np.argmax(rest_win_ci)), rest_win_ci.shape)
    best_layer, best_window = layers_p[best_li], windows[best_wi]
    tok_idx = np.flatnonzero(win_of_token == best_window)
    ex = slice(0, n_verbose)
    _, _, f_patch_ex = _window_restoration(
        adapter, best_layer, tok_idx, clean_tokens[best_layer][ex], ctx_corr[ex],
        horizon, quantiles, seed, f_clean[ex], damage)
    return {
        "layer": best_layer, "window": int(best_window),
        "restoration_grid": verbose_grid_ci.copy(),
        "series_ids": [str(s) for s in series_ids[:n_verbose]] if series_ids is not None else [],
        "families": [str(f) for f in families[:n_verbose]] if families is not None else [],
        "context": ctx_clean[:n_verbose].astype(np.float32),
        "target": targets[:n_verbose].astype(np.float32) if targets is not None else None,
        "forecast_clean": f_clean[:n_verbose].astype(np.float32),
        "forecast_corrupted": f_corr[:n_verbose].astype(np.float32),
        "forecast_patched": f_patch_ex.astype(np.float32),
    }


def _token_windows(spans: np.ndarray, window: int, n_windows: int) -> np.ndarray:
    """Assign each token to the alignment window holding its span midpoint.

    Midpoint assignment (rather than any-overlap) patches each token exactly
    once even under overlapping strided tokenizations, keeping per-window
    restorations additive rather than double-counted.
    """
    mid = spans.mean(axis=1)
    return np.clip((mid // window).astype(np.int64), 0, n_windows - 1)


def _window_restoration(adapter, layer: str, tok_idx: np.ndarray,
                        clean_layer: torch.Tensor, ctx_corr: np.ndarray,
                        horizon: int, quantiles: list, seed: int,
                        f_clean: np.ndarray, damage: float):
    """Forecast restoration from patching only one window's tokens at one layer.

    Returns the batch-mean restoration (the reported statistic, unchanged from
    before this also returned per-series/patched-forecast detail), plus the
    per-series restoration and the patched forecast itself — both needed only
    by verbose single-series reporting, which reuses this same patched
    forward instead of re-running the model.
    """
    idx = torch.from_numpy(tok_idx)

    def index_fn(live_len: int):
        base = adapter.token_slice(live_len)
        return idx + base.start

    with token_patch(adapter.module, layer, index_fn, clean_layer[:, tok_idx]):
        f_patch = _predict_once(adapter, ctx_corr, horizon, quantiles, seed)
    per_series = 1.0 - np.abs(f_patch - f_clean).mean(axis=1) / damage
    return float(per_series.mean()), per_series.astype(np.float32), f_patch


def _agreement_with_ci(cfg: PipelineConfig, psa: np.ndarray, psb: np.ndarray,
                       names: list) -> dict:
    """Fingerprint agreement with paired cluster-bootstrap CIs.

    The same series resample is applied to both models before recomputing
    fingerprints, since deltas come from identical series and corruptions.
    """
    point = _fingerprint_agreement(psa.mean(axis=0), psb.mean(axis=0), names)
    out = {"overall": {"value": point["overall"]},
           "per_corruption": {c: {"value": v} for c, v in point["per_corruption"].items()},
           "most_divergent": point["most_divergent"]}
    if not cfg.stats.enabled:
        return out
    n_boot = min(cfg.stats.n_boot, cfg.stats.n_boot_heavy)
    rng = np.random.default_rng(cfg.run.seed + 25)
    overall = np.empty(n_boot)
    per = {c: np.empty(n_boot) for c in names}
    for i in range(n_boot):
        idx = rng.integers(0, psa.shape[0], psa.shape[0])
        agr = _fingerprint_agreement(psa[idx].mean(axis=0), psb[idx].mean(axis=0), names)
        overall[i] = agr["overall"]
        for c in names:
            per[c][i] = agr["per_corruption"][c]
    q = [(1 - cfg.stats.ci) / 2, 1 - (1 - cfg.stats.ci) / 2]
    lo, hi = np.quantile(overall, q)
    out["overall"].update({"lo": float(lo), "hi": float(hi)})
    for c in names:
        lo, hi = np.quantile(per[c], q)
        out["per_corruption"][c].update({"lo": float(lo), "hi": float(hi)})
    return out


def _fingerprint_agreement(fp_a: np.ndarray, fp_b: np.ndarray, names: list) -> dict:
    """Rank agreement of depth profiles per corruption on a shared relative-depth grid."""
    from scipy.stats import spearmanr
    grid = np.linspace(0, 1, 33)
    ia = np.stack([np.interp(grid, relative_depths(fp_a.shape[0]), fp_a[:, c])
                   for c in range(fp_a.shape[1])], axis=1)
    ib = np.stack([np.interp(grid, relative_depths(fp_b.shape[0]), fp_b[:, c])
                   for c in range(fp_b.shape[1])], axis=1)
    per = {}
    for ci, cname in enumerate(names):
        rho = spearmanr(ia[:, ci], ib[:, ci]).statistic
        per[cname] = float(rho) if np.isfinite(rho) else 0.0
    overall = spearmanr(ia.ravel(), ib.ravel()).statistic
    return {"per_corruption": per,
            "overall": float(overall) if np.isfinite(overall) else 0.0,
            "most_divergent": min(per, key=per.get)}


def _predict_batched(adapter, contexts: np.ndarray, horizon: int, quantiles: list,
                     seed: int) -> np.ndarray:
    """Batched point forecasts with per-batch seeding so sampling models are comparable."""
    out = []
    for s, e in batch_slices(len(contexts), adapter.cfg.batch_size):
        torch.manual_seed(seed + s)
        out.append(adapter.predict(contexts[s:e], horizon, quantiles)["point"])
    return np.concatenate(out)


def _predict_once(adapter, contexts: np.ndarray, horizon: int, quantiles: list,
                  seed: int) -> np.ndarray:
    """Single-call seeded point forecast, required by patching's one-batch constraint."""
    torch.manual_seed(seed)
    return adapter.predict(contexts, horizon, quantiles)["point"]
