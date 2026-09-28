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
                       baseline_quantiles: np.ndarray | None = None,
                       remove_level: bool = False) -> dict:
    """Per-series signed effect for every channel: `steered` minus `baseline`
    (or, for `mase`/`dispersion`'s width half, the steered value itself
    against `targets`/no baseline, since those are not naturally a
    difference against another forecast). All arrays `[B, horizon]` except
    `contexts` `[B, context_len]`, `periods` `[B]`, and the two `*_quantiles`
    `[B, horizon, Q]` or None.

    `remove_level` (ROADMAP.md sec 37.7, P4): when True, `steered` is
    shifted per series by `-(mean_h(steered) - mean_h(baseline))` BEFORE any
    channel is computed, so the shifted forecast has the same horizon-mean
    as `baseline` by construction. Every channel below is then computed on
    the shifted array -- the same code path, not a second implementation --
    which is itself the audit `CLAUDE.md` sec 8/`ROADMAP.md` sec 37.7 item 2
    calls for: a channel already invariant to an additive per-series
    constant must return the SAME value whether `remove_level` is True or
    False, and a channel that is not must differ. Decided from the code
    (confirmed bit-for-bit by `test_already_invariant_channels_identical`):

      invariant   -- `trend` (`trend_slope` subtracts each row's own mean
                     before fitting, so an additive constant cancels
                     exactly), `seasonal` (`seasonal_band_magnitude` reads a
                     NONZERO FFT bin only -- `in_range` requires
                     `raw_bin >= 1` -- and a constant shift only ever moves
                     the bin-0/DC magnitude), `dispersion` (`std()` subtracts
                     its own mean, and the quantile-width sub-statistic never
                     sees `steered` at all).
      not invariant -- `spectral_centroid` (its denominator sums EVERY bin
                     including bin 0, so a level shift changes the whole
                     spectrum's energy and hence the weighted mean
                     frequency), `level` (this literal channel IS the level
                     being removed -- it goes to ~0 by construction),
                     `horizon_shape_near`/`horizon_shape_far` (built from the
                     RAW per-step delta `steered - baseline`, which the shift
                     changes at every step -- this is the intended "shape"
                     signal), `mase` (absolute error against `targets` moves
                     with the forecast's level), `flatness` (its
                     near-constant threshold is relative to `max(|x|)`,
                     which the shift changes).

    `steered_quantiles` is left untouched: the shift is a property of the
    POINT forecast only (P0's `level_share` and this module's `level`
    channel are both point-forecast quantities), so the quantile-width
    sub-statistic is unaffected either way, exactly as the invariance table
    above states.

    Returns `{channel: {"delta": np.ndarray[B] or None, "available": bool,
    "reason": str}}`. `available=False` for a channel this batch cannot
    score at all (no finite periods for `seasonal`, no quantile spread for
    `dispersion`'s width) -- never a silent zero (sec 11.37).
    """
    if remove_level:
        shift = steered.mean(axis=-1) - baseline.mean(axis=-1)
        steered = steered - shift[..., None]

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


def level_share(steered: np.ndarray, baseline: np.ndarray) -> tuple[float | None, str]:
    """ROADMAP.md sec 37.3 P0's definition, restated as a function so P4
    (sec 37.7) scores it identically rather than re-deriving it:
    `mean_s(a_s**2) / mean_s(mean_h(d_s(h)**2))`, where `d_s = steered_s -
    baseline_s` and `a_s = mean_h(d_s)` -- the fraction of a candidate's mean
    squared forecast movement that is explained by its own per-series level
    shift. Both arrays are `[n_rows, horizon]`, already restricted to
    whatever rows the caller wants scored (a candidate's own top-firing
    series in `feature_ablation_fingerprints`).

    Returns `(None, reason)` when the denominator is exactly zero -- every
    row's forecast is bit-identical to baseline across the whole horizon, so
    the ratio is 0/0, not 0 (`CLAUDE.md` sec 11.37: an undefined statistic
    must never be reported as a measured zero). `n_rows == 0` is the same
    undefined case (P0's own 16-of-452 exclusion).
    """
    if steered.size == 0:
        return None, "no top-firing series to score"
    d = np.asarray(steered, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    a = d.mean(axis=-1)
    denom = float(np.mean(np.mean(d ** 2, axis=-1)))
    if not (denom > 0.0):
        return None, ("exactly-zero denominator: this feature's own top-firing series show "
                      "no forecast movement at all on any horizon step, so level share is "
                      "undefined, not zero (CLAUDE.md sec 11.37)")
    return float(np.mean(a ** 2) / denom), ""


def _score_channel_against_null(delta: np.ndarray, null_draws: list) -> dict:
    """One channel's per-candidate effect (`delta`, already row-selected to
    the candidate's own rows) scored against its row-matched null draws
    (`null_draws`, a list of `[n_rows]` ABSOLUTE-value arrays already
    restricted to the same rows) -- factored out of
    `feature_ablation_fingerprints` so the raw and level-removed
    (`remove_level=True`, sec 37.7 P4) scoring paths share one gating
    implementation rather than two copies that could drift apart. Returns
    the same per-channel schema `feature_ablation_fingerprints` has always
    recorded: `available`, `effect`, `signed_effect`, `null_p95`,
    `clears_null`, `null_degenerate`, `null_nonzero_frac`, `margin`,
    `reason`. Key order matches the pre-refactor inline code exactly, so
    JSON output is unaffected (`CLAUDE.md` invariant 13).

    `available=False` (no finite value on this candidate's own rows) is the
    caller's responsibility to detect and short-circuit before calling this
    -- kept out of this helper because the caller needs a DIFFERENT `reason`
    string for that case than the degenerate-null one below.
    """
    effect = float(np.nanmean(np.abs(delta)))
    signed = float(np.nanmean(delta))
    pooled = np.concatenate(null_draws) if null_draws else np.empty(0)
    pooled = pooled[np.isfinite(pooled)]
    p95 = float(np.quantile(pooled, 0.95)) if pooled.size else None
    # A null with no spread is not a threshold -- see
    # `feature_ablation_fingerprints`'s own module-level note (sec 11.37):
    # quantized decoding can leave every null draw at exactly 0, and then any
    # movement at all would "clear" a channel that measured nothing.
    degenerate = p95 is not None and not (p95 > 0.0)
    clears = bool(p95 is not None and not degenerate and effect > p95)
    return {
        "available": True, "effect": effect, "signed_effect": signed,
        "null_p95": p95, "clears_null": clears,
        "null_degenerate": bool(degenerate),
        "null_nonzero_frac": (float(np.mean(pooled > 0.0))
                              if pooled.size else None),
        "margin": (effect - p95) if (p95 is not None and not degenerate)
                  else None,
        "reason": ("every null draw moved this channel by exactly 0, "
                   "so there is no spread to clear -- not scored")
                  if degenerate else ""}


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


# ---------------------------------------------------------------------------
# Ablation, conditioned on the feature actually firing (ROADMAP.md sec 27).
#
# `feature_response_fingerprints` above answers "what does this DIRECTION do",
# by injecting it into a family-stratified representative sample -- the same
# 32 series for every candidate, whether or not that candidate fires on any
# of them. That is the right question for naming a direction and the wrong
# one for the user-facing question this section exists for: "what does this
# FEATURE contribute, where it is actually active". A TopK dictionary leaves
# any single atom inactive on most series, so injecting into a representative
# sample measures a feature mostly outside its own regime, which pushes every
# candidate toward the same answer -- exactly the wrong bias for a statistic
# meant to TELL FEATURES APART.
#
# So this pass differs in both axes and says so in its own artifact:
#   intervention  ablate (zero the atom in the reconstruction)  not inject
#   conditioning  the atom's own top-firing series               not a sample
#
# It does NOT replace the above -- both are kept, because they answer
# different questions and because replacing would silently rewrite every
# number sec 25 already recorded (CLAUDE.md sec 2.1).
# ---------------------------------------------------------------------------

def _feature_ablated_replacement(clean_tokens: torch.Tensor, sae, device,
                                 f_idx) -> torch.Tensor:
    """Token-level SAE reconstruction with one atom, or a SET of atoms
    (ROADMAP.md sec 37.8 P5b), zeroed together.

    The counterfactual is "this feature (or feature set), removed" -- not
    "reversed" -- so the comparison baseline must be the FULL token-level
    reconstruction rather than the raw clean forecast, or the measured
    effect absorbs the whole dictionary's reconstruction error.
    `eval.py::feature_ablation_effects` already establishes that baseline
    convention; this reuses it rather than picking a second one.

    `f_idx` is an `int` (the original, single-feature contract -- unchanged,
    byte-identical: `features[:, f_idx] = 0.0` already broadcasts the same
    way for a bare int) or an iterable of ints (an atlas concept part's full
    membership, sec 37.8 P5b item 3), converted to a sorted list so fancy
    indexing zeros every member at once.
    """
    b, t, d = clean_tokens.shape
    features = sae.encode(clean_tokens.reshape(-1, d).to(device))
    if not isinstance(f_idx, (int, np.integer)):
        f_idx = sorted(int(i) for i in f_idx)
    features[:, f_idx] = 0.0
    recon = sae.decode(features)
    return recon.reshape(b, t, d).cpu()


def top_firing_rows(activations: np.ndarray, f_idx: int, k: int) -> np.ndarray:
    """The `k` series row indices this atom fires hardest on, strongest first.

    `activations` is `[n_series, dict_size]` -- the persisted series-level
    SAE features (`store.load(..., space="sae")`), so selecting a feature's
    own regime costs zero forward passes.

    Rows where the atom is exactly zero are never returned: "the series it
    fires hardest on" cannot include a series it does not fire on, and
    padding the list to `k` with silent zeros would put series carrying no
    signal into the very panel built to show the feature's effect. A caller
    therefore gets FEWER than `k` rows for a rarely-active atom, which is
    the honest answer and is what `n_top_series` records.
    """
    col = np.asarray(activations, dtype=np.float64)[:, int(f_idx)]
    nz = np.flatnonzero(col > 0.0)
    if nz.size == 0:
        return np.empty(0, dtype=int)
    order = nz[np.argsort(-col[nz], kind="stable")]
    return order[:max(0, int(k))].astype(int)


def _series_ids(data) -> np.ndarray:
    """`BenchmarkData` carries its ids in `meta["series_id"]`; a positional
    fallback keeps a stub or a metadata-less jsonl load renderable rather
    than crashing a whole pass over a label."""
    meta = getattr(data, "meta", None)
    if meta is not None and "series_id" in getattr(meta, "columns", []):
        return meta["series_id"].to_numpy()
    ids = getattr(data, "series_ids", None)
    return np.asarray(ids) if ids is not None else np.array(
        [f"row{i}" for i in range(data.n)])


def _pack_chunks(per_candidate_rows: dict, order: list, cap: int) -> list:
    """Group candidates so each group's UNION of top-firing rows fits one
    forward batch, greedily and in the caller's own candidate order.

    The alternative -- one union over every candidate, truncated to the cap --
    is `CLAUDE.md` sec 11.38's head slice: it keeps the lowest-numbered SERIES
    indices, which is unrelated to which candidate needed them, and silently
    leaves most candidates with no rows at all. Chunking costs one baseline
    and one null sweep per chunk and measures every candidate on the series
    it was selected for.

    A single candidate whose own rows exceed the cap is truncated to its
    STRONGEST rows (`top_firing_rows` is already sorted that way), which is a
    principled restriction of its regime rather than an arbitrary one.
    """
    chunks, cur, cur_rows = [], [], set()
    for f_idx in order:
        rows = set(int(r) for r in per_candidate_rows[f_idx][:cap])
        if not rows:
            continue
        if cur and len(cur_rows | rows) > cap:
            chunks.append((cur, np.array(sorted(cur_rows), dtype=int)))
            cur, cur_rows = [], set()
        cur.append(f_idx)
        cur_rows |= rows
    if cur:
        chunks.append((cur, np.array(sorted(cur_rows), dtype=int)))
    return chunks


def feature_ablation_fingerprints(cfg, adapter, layer: str, sae, data, device,
                                  candidates: list, activations: np.ndarray,
                                  top_k_series: int = 8, n_null_directions: int = 16,
                                  max_series: int = 64, seed_offset: int = 260,
                                  floor: dict | None = None,
                                  periods_full: np.ndarray | None = None,
                                  keep_forecasts: int = 3) -> dict:
    """Ablate each candidate on its OWN top-firing series; score the same
    9-channel battery against a ROW-MATCHED random-direction null.

    Same reach gate, same baseline convention and same channels as
    `feature_response_fingerprints` -- the two differ only in intervention
    (ablate vs inject) and conditioning (own regime vs representative
    sample), which is the whole point of keeping both.

    **The null is row-matched, and that is load-bearing.** Candidates are
    scored on different series, so a single pooled null p95 would compare a
    candidate's effect on ITS rows against a null measured on other rows --
    and a candidate whose top-firing series happen to sit in a high-response
    regime would clear it on that alone. So the null's per-series deltas are
    kept per row and the p95 is taken over each candidate's own rows. This is
    `CLAUDE.md` sec 11.33's confound one level in: a threshold means nothing
    against a reference measured under other conditions.

    `activations` is `[n_series, dict_size]` SERIES-LEVEL pooled features
    (`ground_truth.encode_series_level` over every row), used only to choose
    rows -- a window-level matrix has more rows than there are series and
    would index `data` as a different population, so the shape is checked
    rather than assumed.

    `keep_forecasts` per-series with/without pairs are returned per candidate
    for the report's exemplar overlays. The "with" arm is the FULL
    reconstruction, not the raw clean forecast, matching the channel baseline
    so the picture and the number describe the same contrast; the unpatched
    forecast is kept alongside so the SAE's own reconstruction cost stays
    visible rather than being charged to the feature.

    Returns `{"reach": ..., "withheld": True}` and nothing else when the
    target is unreachable (sec 25.1 (7) -- a table of zeros from a patch that
    never lands reads as "these features don't matter").
    """
    acts = np.asarray(activations, dtype=np.float64)
    if acts.ndim != 2 or acts.shape[0] != data.n:
        raise ValueError(
            f"feature_ablation_fingerprints: `activations` must be series-level "
            f"[n_series={data.n}, dict_size], got {acts.shape}. A window-level "
            f"matrix (train.load_all_windows) has n_series*n_windows rows, so "
            f"its row indices name windows, not series, and would select a "
            f"different population from `data`. Use "
            f"ground_truth.encode_series_level(sae, store, model, layer, "
            f"np.arange(data.n), device).")

    reach = reach_probe(cfg, adapter, layer, data, device,
                        max_series=min(max_series, 16), seed_offset=seed_offset - 10)
    if not reach["reachable"]:
        log.warning("sae ablation fingerprint: %s/%s withheld -- %s",
                   adapter.name, layer, reach["reason"])
        return {"reach": reach, "withheld": True}

    adapter.ensure_loaded()

    order = [int(c["feature"]) for c in candidates]
    rules = {int(c["feature"]): c.get("rules", []) for c in candidates}
    per_candidate_rows = {f: top_firing_rows(acts, f, top_k_series) for f in order}
    if not any(r.size for r in per_candidate_rows.values()):
        return {"reach": reach, "withheld": True,
                "reason": "no candidate fires on any series, so there is no "
                          "regime to ablate it in"}

    cap = capped_take(max_series, batch_size=adapter.cfg.batch_size)["n_realized"]
    chunks = _pack_chunks(per_candidate_rows, order, cap)
    log.info("sae ablation: %s/%s: %d candidates over %d forward chunks "
             "(<=%d series each)", adapter.name, layer, len(order), len(chunks), cap)

    all_contexts, all_targets = data.contexts(), data.targets()
    all_ids = _series_ids(data)
    periods_all = (np.asarray(periods_full, dtype=np.float64)
                   if periods_full is not None else np.full(data.n, np.nan))
    d_in, dict_size = sae.d_in, sae.dict_size
    seed = cfg.run.seed + seed_offset
    rng = np.random.default_rng(seed + 1)

    results, n_clearing_cells, n_series_used = [], 0, set()
    n_shape_clearing_cells = 0
    for chunk_feats, rows in chunks:
        row_pos = {int(r): i for i, r in enumerate(rows)}
        n_series_used.update(int(r) for r in rows)
        contexts, targets = all_contexts[rows], all_targets[rows]
        periods, series_ids = periods_all[rows], [str(s) for s in all_ids[rows]]

        clean_tokens = capture_raw_tokens(adapter, contexts, [layer])[layer]
        full_recon = _token_level_replacement(clean_tokens, sae, device)

        def _forward(replacement):
            with token_patch(adapter.module, layer, adapter.token_slice, replacement):
                torch.manual_seed(seed)
                return adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)

        rec_full = _forward(full_recon)
        baseline_fc, baseline_q = rec_full["point"], rec_full.get("quantiles")
        torch.manual_seed(seed)
        unpatched_fc = adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)["point"]

        def _stats(rec, remove_level=False):
            return battery_statistics(rec["point"], baseline_fc, targets, contexts,
                                      periods, steered_quantiles=rec.get("quantiles"),
                                      baseline_quantiles=baseline_q, remove_level=remove_level)

        # Null: remove an arbitrary direction of the same size. The mirror of
        # Component A's injection null (same mechanism, negative magnitude),
        # sized to the mean per-token activation these ablations actually
        # remove -- so "this feature matters" means "more than removing that
        # much of something arbitrary does".
        with torch.no_grad():
            enc_all = sae.encode(clean_tokens.reshape(-1, d_in).to(device))
            removed = [float(enc_all[:, f].abs().mean().cpu()) for f in chunk_feats]
        nonzero = [m for m in removed if m > 0]
        null_magnitude = -float(np.mean(nonzero) if nonzero else 1.0)

        null_rows: dict = {ch: [] for ch in CHANNELS}
        # ROADMAP.md sec 37.7 P4: the level-removed null, computed from the
        # SAME null forward passes above (no extra forward pass) so a shape
        # channel is never scored against a level-carrying null -- exactly
        # the row-matched discipline this function's own docstring states,
        # applied a second time to the level-removed transform
        # (`test_null_gets_same_transform` is the load-bearing regression for
        # this).
        null_rows_shape: dict = {ch: [] for ch in CHANNELS}
        for _ in range(n_null_directions):
            direction = rng.normal(size=dict_size)
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            rec_null = _forward(_direction_steered_replacement(
                clean_tokens, sae, device,
                torch.as_tensor(direction, dtype=torch.float32), null_magnitude))
            stats = _stats(rec_null)
            stats_shape = _stats(rec_null, remove_level=True)
            for ch in CHANNELS:
                r = stats[ch]
                if r["available"] and r["delta"] is not None:
                    null_rows[ch].append(np.abs(np.asarray(r["delta"], dtype=np.float64)))
                rs = stats_shape[ch]
                if rs["available"] and rs["delta"] is not None:
                    null_rows_shape[ch].append(np.abs(np.asarray(rs["delta"], dtype=np.float64)))

        for f_idx in chunk_feats:
            rows_f = [int(r) for r in per_candidate_rows[f_idx] if int(r) in row_pos]
            idx = np.array([row_pos[r] for r in rows_f], dtype=int)
            rec = _forward(_feature_ablated_replacement(clean_tokens, sae, device, f_idx))
            stats = _stats(rec)
            stats_shape = _stats(rec, remove_level=True)

            per_channel = {}
            per_channel_shape = {}
            for ch in CHANNELS:
                s = stats[ch]
                if not s["available"] or s["delta"] is None:
                    per_channel[ch] = {"available": False, "reason": s["reason"],
                                       "effect": None, "signed_effect": None,
                                       "null_p95": None, "clears_null": False,
                                       "margin": None}
                else:
                    delta = np.asarray(s["delta"], dtype=np.float64)[idx]
                    if not np.isfinite(delta).any():
                        # Every one of THIS candidate's rows is non-finite for
                        # this channel (e.g. no ground-truth period among its
                        # own top-firing series). The channel is available
                        # for the batch and unavailable for this feature --
                        # recording a NaN effect instead would put a NaN into
                        # every downstream fingerprint and comparison.
                        per_channel[ch] = {
                            "available": False, "effect": None, "signed_effect": None,
                            "null_p95": None, "clears_null": False, "margin": None,
                            "reason": "no finite value on this feature's own "
                                      "top-firing series"}
                    else:
                        # Signed too, and it is not redundant: `effect` is
                        # what the null p95 (itself unsigned) can legitimately
                        # be compared against, while DIRECTION is what tells
                        # two features that fire on the same series apart --
                        # one ablation raising the level and another lowering
                        # it are opposite causal roles that an unsigned
                        # fingerprint would call identical.
                        draws = [d[idx] for d in null_rows[ch] if d.size == len(rows)]
                        per_channel[ch] = _score_channel_against_null(delta, draws)
                        n_clearing_cells += int(per_channel[ch]["clears_null"])

                s_shape = stats_shape[ch]
                if ch == "level":
                    per_channel_shape[ch] = {
                        "available": False, "effect": None, "signed_effect": None,
                        "null_p95": None, "clears_null": False, "margin": None,
                        "reason": "removed by construction: the level-removed forecast has "
                                  "the baseline's horizon mean, so both effect and null are "
                                  "float rounding (CLAUDE.md sec 11.48)"}
                elif not s_shape["available"] or s_shape["delta"] is None:
                    per_channel_shape[ch] = {"available": False, "reason": s_shape["reason"],
                                             "effect": None, "signed_effect": None,
                                             "null_p95": None, "clears_null": False,
                                             "margin": None}
                else:
                    delta_shape = np.asarray(s_shape["delta"], dtype=np.float64)[idx]
                    if not np.isfinite(delta_shape).any():
                        per_channel_shape[ch] = {
                            "available": False, "effect": None, "signed_effect": None,
                            "null_p95": None, "clears_null": False, "margin": None,
                            "reason": "no finite value on this feature's own "
                                      "top-firing series (level-removed)"}
                    else:
                        draws_shape = [d[idx] for d in null_rows_shape[ch] if d.size == len(rows)]
                        per_channel_shape[ch] = _score_channel_against_null(delta_shape, draws_shape)
                        n_shape_clearing_cells += int(per_channel_shape[ch]["clears_null"])

            level_share_val, level_share_reason = level_share(
                np.asarray(rec["point"], dtype=np.float64)[idx],
                np.asarray(baseline_fc, dtype=np.float64)[idx])

            keep = rows_f[:max(0, int(keep_forecasts))]
            forecasts = [{
                "series_id": series_ids[row_pos[r]],
                "row": r,
                "activation": float(acts[r, f_idx]),
                "context": np.asarray(contexts[row_pos[r]], dtype=np.float64).tolist(),
                "target": np.asarray(targets[row_pos[r]], dtype=np.float64).tolist(),
                "with_feature": np.asarray(baseline_fc[row_pos[r]], dtype=np.float64).tolist(),
                "without_feature": np.asarray(rec["point"][row_pos[r]], dtype=np.float64).tolist(),
                "unpatched": np.asarray(unpatched_fc[row_pos[r]], dtype=np.float64).tolist(),
            } for r in keep]

            mase_rec = per_channel.get("mase", {})
            results.append({
                "feature": f_idx, "rules": rules[f_idx],
                "n_top_series": int(idx.size), "scorable": True,
                "mean_activation_on_top": float(np.mean(acts[rows_f, f_idx])),
                "channels": per_channel,
                "n_channels_clearing": sum(1 for v in per_channel.values()
                                           if v["clears_null"]),
                "mase_effect_floor_units": (
                    in_floor_units(mase_rec["effect"], floor).get("value")
                    if mase_rec.get("effect") is not None and floor else None),
                "forecasts": forecasts,
                # ROADMAP.md sec 37.7 P4 -- additive (CLAUDE.md invariant 13):
                # `level_share` is P0's ratio (sec 37.3) recomputed on this
                # candidate's own top-firing rows; `shape_channels` is the
                # SAME 9-channel battery scored with the level shift removed
                # first (`battery_statistics(..., remove_level=True)`)
                # against its own row-matched null (`null_rows_shape` above).
                "level_share": level_share_val,
                "level_share_reason": level_share_reason,
                "shape_channels": per_channel_shape,
                "n_shape_channels_clearing": sum(1 for v in per_channel_shape.values()
                                                 if v["clears_null"]),
            })

    for f_idx in order:
        if not any(r["feature"] == f_idx for r in results):
            results.append({"feature": f_idx, "rules": rules[f_idx], "n_top_series": 0,
                            "scorable": False,
                            "reason": "this atom fires on no series, so there is "
                                      "no regime to ablate it in"})
    results.sort(key=lambda r: order.index(r["feature"]))

    scorable = [r for r in results if r.get("scorable")]
    n_cells = len(scorable) * len(CHANNELS)
    return {
        "reach": reach, "withheld": False,
        "intervention": "ablate", "conditioning": "top_firing",
        "top_k_series": int(top_k_series), "n_series_union": len(n_series_used),
        "n_chunks": len(chunks), "series_per_chunk_cap": int(cap),
        "n_null_directions": int(n_null_directions),
        "candidates": results,
        "n_clearing_cells": int(n_clearing_cells),
        "chance_expected_cells": 0.05 * n_cells,
        # 🔴 A RATIO, where the Stage 2 artifact's identically-named field a
        # few hundred lines above is a DIFFERENCE (90 clearing cells against
        # 19.35 expected reads 70.65 there and 4.65 here). Two artifacts a
        # reader lays side by side must not use one name for two quantities,
        # so the ratio carries its units in its name and `excess_over_chance`
        # is kept as the difference, matching Stage 2 exactly. Caught by a
        # background agent's own reading of the two artifacts, not by a test
        # -- both numbers are individually correct.
        "clearing_cells_over_chance_ratio": (
            (n_clearing_cells / (0.05 * n_cells)) if n_cells else None),
        "excess_over_chance": n_clearing_cells - (0.05 * n_cells) if n_cells else None,
        "any_feature_clears_any_channel": any(r.get("n_channels_clearing", 0) > 0
                                              for r in scorable),
        # ROADMAP.md sec 37.7 P4 -- additive, mirrors the raw
        # `n_clearing_cells` above for the level-removed battery.
        "n_shape_clearing_cells": int(n_shape_clearing_cells),
    }
