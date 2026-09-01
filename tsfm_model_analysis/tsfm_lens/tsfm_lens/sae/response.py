"""Component A -- the response fingerprint (ROADMAP.md §25.4, §25.9 Stage 2).

Stop naming an SAE feature by what it correlates with; name it by what it
does to the forecast. For each candidate feature, steer it +-k*sigma_f
through the existing `token_patch` seam, read a fixed battery of
forecast-space statistics against the intact token-level reconstruction
baseline, and score every channel against a random-direction null of the
same injection magnitude: "this feature does X" means "more than an
arbitrary perturbation of this size does".

Deliberately follows `sae/eval.py::feature_steering_effects`'s exact
plumbing (full token-level reconstruction baseline, `capture_raw_tokens` +
`token_patch`, `sample_rows(..., strata=families)`, `cfg.run.seed +
seed_offset`) rather than reimplementing it -- the only new pieces are the
richer 8-channel battery (§25.4's table) in place of trend/seasonal alone,
the random-direction null, and candidate selection restricted to alive
atoms and labelled by nomination rule (`CLAUDE.md` §11.34: a probe must
record which rule resolved it).

The reach gate (`analysis/response_reach.py`) is a *separate*, mandatory
precondition -- per §11.42, this module never substitutes a declaration for
a measurement, and the I/O wrapper `feature_response_fingerprints` calls the
reach probe itself and withholds the whole battery when the target is
unreachable, rather than silently computing a table of zeros that would
read as "these features don't matter" (§25.1 (7)).
"""

from __future__ import annotations

import numpy as np
import torch

from ..extraction.extract import capture_raw_tokens
from ..extraction.hooks import token_patch
from ..utils import capped_take, log, sample_rows
from .eval import _token_level_replacement
from ..analysis.response_reach import reach_probe  # re-exported for convenience
from ..analysis.spectral_lens import _magnitude_spectrum
from ..analysis.stats import in_floor_units, mase, mean_ci
from ..analysis.steering import seasonal_band_magnitude, trend_slope

CHANNELS = ("trend", "seasonal", "spectral_centroid", "level", "dispersion",
           "horizon_shape_near", "horizon_shape_far", "mase", "flatness")

# `dispersion` and `horizon_shape` are each split above into the pieces a
# fixed-width channel table needs (`horizon_shape` -> near/far, per §25.4's
# own table); `n_channels` for the chance-arithmetic in §25.11 is the count
# actually scored per feature, i.e. `len(CHANNELS)`.


# ---------------------------------------------------------------------------
# Pure battery statistics (no forward passes; operate on forecast arrays).
# ---------------------------------------------------------------------------

def _level(x: np.ndarray) -> np.ndarray:
    """Mean of the forecast, per series."""
    return x.mean(axis=-1)


def _dispersion(x: np.ndarray, quantiles: np.ndarray | None = None) -> tuple:
    """(sd of the point forecast, mean quantile width or None).

    `quantiles` is `[B, H, Q]` or None. A model with a degenerate
    (zero-spread) quantile band -- deterministic forecasters, or any run
    where quantiles were not captured -- reports `width=None` rather than
    0.0, so a downstream null-comparison never scores a zero-spread band as
    "no effect" (CLAUDE.md sec 11.37: never let a degenerate baseline
    silently win/lose; report `available: False` instead).
    """
    sd = x.std(axis=-1)
    if quantiles is None:
        return sd, None
    width = (quantiles[..., -1] - quantiles[..., 0]).mean(axis=-1)
    if not np.any(width > 1e-9):
        return sd, None
    return sd, width


def _horizon_shape(x: np.ndarray, baseline: np.ndarray) -> tuple:
    """|delta(steered) - delta(baseline)| split into near (first third) and
    far (last third) horizon thirds, mean absolute deviation per series."""
    h = x.shape[-1]
    third = max(1, h // 3)
    d = x - baseline
    near = np.abs(d[..., :third]).mean(axis=-1)
    far = np.abs(d[..., -third:]).mean(axis=-1)
    return near, far


def _flatness(x: np.ndarray, rel_tol: float = 1e-3) -> np.ndarray:
    """Fraction of near-constant consecutive steps, per series -- the
    intermittency-sensitive channel. Scale-relative: a step is "flat" when
    its first difference is small relative to the series' own amplitude."""
    scale = (np.abs(x).max(axis=-1, keepdims=True) + 1e-8)
    d = np.abs(np.diff(x, axis=-1)) / scale
    return (d < rel_tol).mean(axis=-1)


def _spectral_centroid(x: np.ndarray) -> np.ndarray:
    """Magnitude-weighted mean frequency bin of the forecast's own FFT."""
    mag = _magnitude_spectrum(x)
    freqs = np.arange(mag.shape[-1], dtype=np.float64)
    denom = mag.sum(axis=-1) + 1e-12
    return (mag * freqs[None, :]).sum(axis=-1) / denom


def battery_statistics(steered: np.ndarray, baseline: np.ndarray, targets: np.ndarray,
                       contexts: np.ndarray, periods: np.ndarray,
                       steered_quantiles: np.ndarray | None = None,
                       baseline_quantiles: np.ndarray | None = None) -> dict:
    """Per-series signed effect for every channel: `steered` minus `baseline`
    (or, for `mase`/`dispersion`'s width half, the steered value itself
    against `targets`/no baseline, since those are not naturally a
    difference against another forecast). All arrays `[B, horizon]` except
    `contexts` `[B, context_len]`, `periods` `[B]`, and the two `*_quantiles`
    `[B, horizon, Q]` or None.

    Returns `{channel: {"delta": np.ndarray[B] or None, "available": bool,
    "reason": str}}`. `available=False` for a channel this batch cannot
    score at all (no finite periods for `seasonal`, no quantile spread for
    `dispersion`'s width) -- never a silent zero (sec 11.37).
    """
    out = {}

    out["trend"] = {"delta": trend_slope(steered) - trend_slope(baseline),
                    "available": True, "reason": ""}

    have_periods = bool(np.isfinite(periods).any())
    if have_periods:
        sb = seasonal_band_magnitude(steered, periods)
        bb = seasonal_band_magnitude(baseline, periods)
        d = sb - bb
        out["seasonal"] = {"delta": d if np.isfinite(d).any() else None,
                           "available": bool(np.isfinite(d).any()),
                           "reason": "" if np.isfinite(d).any() else
                                     "no series in this batch has a usable period"}
    else:
        out["seasonal"] = {"delta": None, "available": False,
                           "reason": "no series in this batch carries a finite "
                                     "ground-truth seasonal period"}

    out["spectral_centroid"] = {
        "delta": _spectral_centroid(steered) - _spectral_centroid(baseline),
        "available": True, "reason": ""}

    out["level"] = {"delta": _level(steered) - _level(baseline),
                    "available": True, "reason": ""}

    sd_s, width_s = _dispersion(steered, steered_quantiles)
    sd_b, width_b = _dispersion(baseline, baseline_quantiles)
    # sd-of-forecast is always available; quantile width is a *second*,
    # separately-gated sub-statistic under the same channel name, since the
    # table lists them together ("sd of the forecast, and mean quantile
    # width when the model has one").
    disp_delta = sd_s - sd_b
    width_delta = (width_s - width_b) if (width_s is not None and width_b is not None) else None
    out["dispersion"] = {"delta": disp_delta, "available": True, "reason": "",
                         "width_delta": width_delta,
                         "width_available": width_delta is not None,
                         "width_reason": "" if width_delta is not None else
                                         "quantile band has zero spread; not scored"}

    near, far = _horizon_shape(steered, baseline)
    out["horizon_shape_near"] = {"delta": near, "available": True, "reason": ""}
    out["horizon_shape_far"] = {"delta": far, "available": True, "reason": ""}

    mase_steered = mase(steered, targets, contexts)
    mase_baseline = mase(baseline, targets, contexts)
    out["mase"] = {"delta": mase_steered - mase_baseline, "available": True, "reason": ""}

    out["flatness"] = {"delta": _flatness(steered) - _flatness(baseline),
                       "available": True, "reason": ""}

    return out


def summarize_battery(per_series: dict, unit: str = "series", n_boot: int = 500,
                      seed: int = 0) -> dict:
    """Aggregate `battery_statistics`' per-series deltas with a series-level
    bootstrap CI per channel (invariant 2). Unavailable channels pass through
    unaggregated with `available=False`."""
    out = {}
    for i, (channel, rec) in enumerate(per_series.items()):
        if not rec["available"] or rec["delta"] is None:
            out[channel] = {"available": False, "reason": rec["reason"], "ci": None}
            continue
        finite = np.asarray(rec["delta"], dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            out[channel] = {"available": False, "reason": "no finite values", "ci": None}
            continue
        ci = mean_ci(finite, n_boot=n_boot, seed=seed + i, unit=unit)
        out[channel] = {"available": True, "reason": "", "ci": ci, "mean": ci["value"]}
        if channel == "dispersion":
            out[channel]["width_available"] = rec.get("width_available", False)
            if rec.get("width_available"):
                wd = np.asarray(rec["width_delta"], dtype=np.float64)
                wd = wd[np.isfinite(wd)]
                out[channel]["width_ci"] = (mean_ci(wd, n_boot=n_boot, seed=seed + i + 100, unit=unit)
                                            if wd.size else None)
            else:
                out[channel]["width_ci"] = None
                out[channel]["width_reason"] = rec.get("width_reason", "")
    return out


# ---------------------------------------------------------------------------
# Alive-atom mask (mirrors `sae/eval.py::dead_feature_rate`'s "ever fired"
# logic, returning the per-atom mask rather than the scalar rate).
# ---------------------------------------------------------------------------

@torch.no_grad()
def alive_feature_mask(sae, activations: np.ndarray, device, batch_size: int = 8192,
                       threshold: float = 1e-8) -> np.ndarray:
    """`[dict_size]` bool: which atoms fire above `threshold` at least once
    over the given rows. Same "ever fired" definition
    `sae/eval.py::dead_feature_rate` already uses -- kept consistent rather
    than a second definition of "alive"."""
    from ..utils import batch_slices
    x = torch.from_numpy(activations)
    ever_fired = torch.zeros(sae.dict_size, dtype=torch.bool, device=device)
    for s, e in batch_slices(x.shape[0], batch_size):
        features = sae.encode(x[s:e].to(device))
        ever_fired |= (features.abs() > threshold).any(dim=0)
    return ever_fired.cpu().numpy()


# ---------------------------------------------------------------------------
# Candidate selection (ROADMAP.md §25.4's new priority order).
# ---------------------------------------------------------------------------

def select_candidates(alive_mask: np.ndarray, n_per_rule: int,
                      residualized_structural: dict | None = None,
                      activation_variance: np.ndarray | None = None,
                      provenance_matches: dict | None = None,
                      seed: int = 0) -> list:
    """Nominate candidate feature indices, restricted to `alive_mask`, in
    ROADMAP.md §25.4's stated priority order. Every candidate records EVERY
    rule that nominated it (not just the first) -- `CLAUDE.md` sec 11.34:
    a probe that does not record its own resolution is the trap.

    - `residualized_structural`: `{feature_idx: abs_rho}` (from
      `ground_truth.py::best_ground_truth_matches_separated`'s `structural`
      field), already residualized against provenance.
    - `activation_variance`: `[dict_size]` array, the raw activation
      variance of every atom (dead or alive; this function does the
      alive-restriction).
    - `provenance_matches`: `{feature_idx: abs_rho}` (the `provenance`
      field from the same separated-matches result).

    Returns a list of `{"feature": int, "rules": [str, ...]}`, deduplicated
    by feature index, ordered by first-nomination-rule priority. A
    `random` control sample is always included (rule 4), independent of
    whether the other three inputs are given, since it is "the only way to
    say whether the selected features are unusual" (§25.4) and costs
    nothing to include.
    """
    alive_idx = np.flatnonzero(alive_mask)
    nominations: dict[int, list] = {}

    def _nominate(idx: int, rule: str) -> None:
        if idx not in nominations:
            nominations[idx] = []
        if rule not in nominations[idx]:
            nominations[idx].append(rule)

    order: list = []

    # (1) top-n by residualized structural rho, restricted to alive atoms.
    if residualized_structural:
        ranked = sorted(
            ((f, abs(r)) for f, r in residualized_structural.items() if alive_mask[f]),
            key=lambda kv: -kv[1])
        for f, _ in ranked[:n_per_rule]:
            _nominate(f, "structural")
            if f not in order:
                order.append(f)

    # (2) top-n by activation variance among alive atoms.
    if activation_variance is not None:
        var = np.asarray(activation_variance, dtype=np.float64)
        alive_var = [(int(i), float(var[i])) for i in alive_idx]
        alive_var.sort(key=lambda kv: -kv[1])
        for f, _ in alive_var[:n_per_rule]:
            _nominate(f, "variance")
            if f not in order:
                order.append(f)

    # (3) top-n by raw provenance rho, kept and labelled deliberately.
    if provenance_matches:
        ranked = sorted(
            ((f, abs(r)) for f, r in provenance_matches.items() if alive_mask[f]),
            key=lambda kv: -kv[1])
        for f, _ in ranked[:n_per_rule]:
            _nominate(f, "provenance")
            if f not in order:
                order.append(f)

    # (4) random sample of alive atoms, the within-dictionary control.
    rng = np.random.default_rng(seed)
    n_random = min(n_per_rule, alive_idx.size)
    if n_random > 0:
        random_pick = rng.choice(alive_idx, size=n_random, replace=False)
        for f in random_pick:
            f = int(f)
            _nominate(f, "random")
            if f not in order:
                order.append(f)

    if not order:
        log.info("sae response fingerprint: no alive atoms available for any "
                 "candidate-selection rule (alive_mask.sum()=%d)", int(alive_mask.sum()))

    return [{"feature": f, "rules": nominations[f]} for f in order]


# ---------------------------------------------------------------------------
# Steered-forecast production (mirrors `sae/eval.py::_feature_steered_replacement`).
# ---------------------------------------------------------------------------

@torch.no_grad()
def _direction_steered_replacement(clean_tokens: torch.Tensor, sae, device,
                                   direction: torch.Tensor, magnitude: float) -> torch.Tensor:
    """Token-level SAE reconstruction with a fixed unit `direction` in the
    dictionary's FEATURE space added at magnitude `magnitude` to every
    token's encoded activation, before decoding. Used both for a real
    feature's one-hot steering direction and for the random-direction null
    -- the same mechanism, differing only in which direction vector is
    injected, which is exactly what makes the null a fair comparison."""
    b, t, d = clean_tokens.shape
    features = sae.encode(clean_tokens.reshape(-1, d).to(device))
    features = features + magnitude * direction.to(device)
    recon = sae.decode(features)
    return recon.reshape(b, t, d).cpu()


def feature_response_fingerprints(cfg, adapter, layer: str, sae, data, device,
                                  candidates: list, strength_sigma: float = 2.0,
                                  n_null_directions: int = 24,
                                  max_series: int = 32, seed_offset: int = 210,
                                  floor: dict | None = None,
                                  periods_full: np.ndarray | None = None) -> dict:
    """I/O wrapper: reach gate -> full-reconstruction baseline -> per-candidate
    steering -> random-direction null -> per-channel gating, all in floor
    units for `mase` (ROADMAP.md §25.4).

    `candidates` is `[{"feature": int, "rules": [str, ...]}, ...]` from
    `select_candidates`. `floor` is one model's entry from
    `l0/noise_floor.json` (or None if unmeasured) -- passed through to
    `stats.in_floor_units` for the `mase` channel only, per §25.4's stated
    rule. `periods_full` is `[data.n]` -- each of `data`'s series' own
    ground-truth `seasonal_period_dominant` (`np.nan` where absent), exactly
    the array `sae/train.py::run_sae` already builds once via
    `load_ground_truth_table(cfg.data.path).reindex(series_ids)` and passes
    to `feature_steering_effects` -- reused here rather than re-derived, and
    left `None` (all-NaN) when the caller has no ground-truth table (a
    non-benchmark corpus), which correctly disables the `seasonal` channel
    rather than raising.

    Returns a dict with `reach` (the gate's own record), and, only when
    `reach["reachable"]` is True: `n_series`, `candidates` (each with its
    per-channel effect + null p95 + `clears_null` bool), `null` (the K
    random-direction draws' per-channel p95/mean), and the excess-over-
    chance count (§25.11). When unreachable, returns just `{"reach": ...,
    "withheld": True}` -- no battery numbers at all, so a caller cannot
    accidentally render a table of zeros as a finding (§25.1 (7)).
    """
    reach = reach_probe(cfg, adapter, layer, data, device,
                        max_series=min(max_series, 16), seed_offset=seed_offset - 10)
    if not reach["reachable"]:
        log.warning("sae response fingerprint: %s/%s withheld -- %s",
                   adapter.name, layer, reach["reason"])
        return {"reach": reach, "withheld": True}

    adapter.ensure_loaded()
    cap = capped_take(max_series, n_available=data.n, batch_size=adapter.cfg.batch_size)
    take = cap["n_realized"]
    seed = cfg.run.seed + seed_offset
    families = data.families
    rows = sample_rows(data.n, take, seed, strata=families)
    contexts = data.contexts()[rows]
    targets = data.targets()[rows]

    periods = (np.asarray(periods_full, dtype=np.float64)[rows] if periods_full is not None
              else np.full(take, np.nan))

    clean_tokens = capture_raw_tokens(adapter, contexts, [layer])[layer]
    full_recon = _token_level_replacement(clean_tokens, sae, device)
    with token_patch(adapter.module, layer, adapter.token_slice, full_recon):
        torch.manual_seed(seed)
        rec_full = adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)
    baseline_fc = rec_full["point"]
    baseline_q = rec_full.get("quantiles")

    d_in = sae.d_in
    dict_size = sae.dict_size

    def _run_direction(direction: np.ndarray, magnitude: float) -> dict:
        direction_t = torch.as_tensor(direction, dtype=torch.float32)
        replacement = _direction_steered_replacement(clean_tokens, sae, device,
                                                     direction_t, magnitude)
        with token_patch(adapter.module, layer, adapter.token_slice, replacement):
            torch.manual_seed(seed)
            rec = adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)
        stats = battery_statistics(rec["point"], baseline_fc, targets, contexts, periods,
                                   steered_quantiles=rec.get("quantiles"),
                                   baseline_quantiles=baseline_q)
        return stats

    # --- Random-direction null: K unit-normalized directions in the
    # dictionary's own feature space (one-hot on a single dimension for a
    # real feature; here a dense random unit vector, matching every real
    # feature's injection MAGNITUDE exactly, per §25.4's "injected at the
    # SAME per-series magnitude" rule). One direction's magnitude is the
    # mean of the candidate features' own steering magnitudes so the null is
    # comparable to what this target's real features actually receive,
    # rather than an arbitrary fixed constant.
    rng = np.random.default_rng(cfg.run.seed + seed_offset + 1)
    candidate_sigmas = []
    with torch.no_grad():
        for c in candidates:
            f_idx = int(c["feature"])
            feat_vals = sae.encode(clean_tokens.reshape(-1, d_in).to(device))[:, f_idx]
            sigma = float(feat_vals.std().cpu())
            candidate_sigmas.append(sigma if sigma > 1e-8 else 1.0)
    mean_sigma = float(np.mean(candidate_sigmas)) if candidate_sigmas else 1.0
    null_magnitude = strength_sigma * mean_sigma

    null_per_channel: dict = {ch: [] for ch in CHANNELS}
    for k in range(n_null_directions):
        direction = rng.normal(size=dict_size)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        stats = _run_direction(direction, null_magnitude)
        for ch in CHANNELS:
            rec = stats[ch]
            if rec["available"] and rec["delta"] is not None:
                null_per_channel[ch].append(float(np.nanmean(np.abs(rec["delta"]))))
    null_p95 = {ch: (float(np.quantile(v, 0.95)) if v else None) for ch, v in null_per_channel.items()}
    null_mean = {ch: (float(np.mean(v)) if v else None) for ch, v in null_per_channel.items()}

    # --- Candidates: one-hot steering direction per feature, at that
    # feature's own +-strength_sigma*sigma_f magnitude (feature_steering_
    # effects' existing convention). Both signs run; the LARGER absolute
    # effect per channel is what gets compared to the null, since a feature
    # can (and often does) have an asymmetric response.
    candidate_results = []
    n_clearing_cells = 0
    for c in candidates:
        f_idx = int(c["feature"])
        with torch.no_grad():
            feat_vals = sae.encode(clean_tokens.reshape(-1, d_in).to(device))[:, f_idx]
            sigma = float(feat_vals.std().cpu())
        delta_mag = strength_sigma * sigma if sigma > 1e-8 else strength_sigma
        one_hot = np.zeros(dict_size, dtype=np.float64)
        one_hot[f_idx] = 1.0

        signed = {}
        for sign_name, mag in (("up", delta_mag), ("down", -delta_mag)):
            stats = _run_direction(one_hot, mag)
            per_channel = {}
            for ch in CHANNELS:
                rec = stats[ch]
                if not rec["available"] or rec["delta"] is None:
                    per_channel[ch] = {"available": False, "reason": rec["reason"],
                                       "effect": None, "null_p95": null_p95.get(ch),
                                       "clears_null": False}
                    continue
                effect = float(np.nanmean(np.abs(rec["delta"])))
                p95 = null_p95.get(ch)
                clears = bool(p95 is not None and effect > p95)
                entry = {"available": True, "reason": "", "effect": effect,
                        "signed_mean": float(np.nanmean(rec["delta"])),
                        "null_p95": p95, "null_mean": null_mean.get(ch),
                        "clears_null": clears}
                if ch == "mase":
                    entry["floor_units"] = in_floor_units(entry["signed_mean"], floor)
                per_channel[ch] = entry
            signed[sign_name] = {"delta_magnitude": mag, "channels": per_channel}

        # A feature "clears" a channel if either sign clears it; count each
        # (feature, channel) cell at most once for the chance-arithmetic
        # (§25.11), matching "the count of clearing (feature, channel) cells".
        feature_clears = set()
        for sign_name in ("up", "down"):
            for ch, entry in signed[sign_name]["channels"].items():
                if entry.get("clears_null"):
                    feature_clears.add(ch)
        n_clearing_cells += len(feature_clears)

        candidate_results.append({
            "feature": f_idx, "rules": c["rules"], "steering_sigma": sigma,
            "steering_magnitude": delta_mag, "up": signed["up"], "down": signed["down"],
            "clearing_channels": sorted(feature_clears),
        })

    n_features = len(candidates)
    n_channels = len(CHANNELS)
    chance_expected = 0.05 * n_features * n_channels
    any_clears = any(r["clearing_channels"] for r in candidate_results)

    return {
        "reach": reach,
        "withheld": False,
        "layer": layer,
        "n_series": int(take),
        "n_requested": cap["n_requested"],
        "n_realized": cap["n_realized"],
        "limited_by": cap["limited_by"],
        "strength_sigma": strength_sigma,
        "n_null_directions": n_null_directions,
        "null_magnitude": null_magnitude,
        "null_p95": null_p95,
        "null_mean": null_mean,
        "candidates": candidate_results,
        "n_features": n_features,
        "n_channels": n_channels,
        "n_clearing_cells": n_clearing_cells,
        "chance_expected_clearing_cells": chance_expected,
        "excess_over_chance": n_clearing_cells - chance_expected,
        "any_feature_clears_any_channel": any_clears,
        "meets_stage2_exit": bool(
            reach["reachable"] and reach["self_patch_delta"] == 0.0 and
            any_clears and n_clearing_cells > chance_expected),
    }
