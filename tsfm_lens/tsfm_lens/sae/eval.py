"""SAE evaluation harness (ROADMAP.md §6.2): reconstruction fidelity,
dead-feature rate, forecast-preservation under reconstruction, and
feature-level ablation -- the causal test §7 bullet 3 / §16 E15's second
half specifies. Feature-level ablation depends on `forecast_preservation`'s
own token-granularity check passing reasonably well first: patching in a
reconstruction that already breaks the forecast on its own would make any
single-feature ablation delta uninterpretable (dominated by SAE
reconstruction error, not the ablated feature's causal contribution).
"""

from __future__ import annotations

import numpy as np
import torch

from ..analysis.l3_perturbation import _token_windows
from ..analysis.stats import mase
from ..analysis.steering import seasonal_band_magnitude, trend_slope
from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.alignment import align, pooling_matrix
from ..extraction.extract import capture_raw_tokens
from ..extraction.hooks import token_patch
from ..extraction.store import ActivationStore
from ..utils import batch_slices, capped_take, sample_rows


def seed_spread(values) -> dict:
    """Mean/sd/min/max/range of one metric across SAE training seeds.

    The SAE analog of the behavioral repeat-run noise floor (`ROADMAP.md`
    §15 A13): a single-seed ΔMASE is one draw from a distribution wide
    enough, on measured runs, to contain both a "passes" and a "fails"
    verdict against a small threshold, so the spread is what a later reader
    has to hold the number against. `sd` is the sample (ddof=1) deviation --
    the quantity to compare a delta to -- and `range` is reported beside it
    because at the handful of seeds this is ever run with, the range is the
    more honest summary of what one unreplicated number could have been.

    Non-finite entries (a failed check's NaN) are dropped rather than
    poisoning the summary, and `n` records how many actually contributed.
    """
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return {"n": 0}
    arr = np.asarray(finite, dtype=np.float64)
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max()),
            "range": float(arr.max() - arr.min()),
            "values": [float(v) for v in arr]}


@torch.no_grad()
def reconstruction_fidelity(sae, activations: np.ndarray, device,
                            batch_size: int = 8192) -> float:
    """Fraction of variance explained by the SAE's reconstruction over the given rows."""
    x = torch.from_numpy(activations)
    mean = x.mean(dim=0).to(device)
    total_resid, total_var = 0.0, 0.0
    for s, e in batch_slices(x.shape[0], batch_size):
        batch = x[s:e].to(device)
        recon, _ = sae(batch)
        total_resid += float(((batch - recon) ** 2).sum())
        total_var += float(((batch - mean) ** 2).sum())
    return 1.0 - total_resid / max(total_var, 1e-8)


@torch.no_grad()
def dead_feature_rate(sae, activations: np.ndarray, device, batch_size: int = 8192,
                      threshold: float = 1e-8) -> float:
    """Fraction of dictionary atoms that never fire above `threshold` over the given rows."""
    x = torch.from_numpy(activations)
    ever_fired = torch.zeros(sae.dict_size, dtype=torch.bool, device=device)
    for s, e in batch_slices(x.shape[0], batch_size):
        features = sae.encode(x[s:e].to(device))
        ever_fired |= (features.abs() > threshold).any(dim=0)
    return float((~ever_fired).float().mean())


@torch.no_grad()
def _window_broadcast_replacement(clean_tokens: torch.Tensor, sae, adapter,
                                  cfg: PipelineConfig, device) -> torch.Tensor:
    """The SAE's window-pooled reconstruction, broadcast back to every raw
    token in its window. See `forecast_preservation`'s "window" granularity
    docstring for why this is a confound, not just a design choice."""
    pool = pooling_matrix(adapter.token_time_spans(), cfg.data.context_len, cfg.alignment.window)
    n_windows = cfg.data.context_len // cfg.alignment.window
    win_of_token = _token_windows(adapter.token_time_spans(), cfg.alignment.window, n_windows)

    pooled = align(clean_tokens, pool)  # [B, W, D] -- exactly what the SAE trained on
    b, w, d = pooled.shape
    recon_pooled, _ = sae(pooled.reshape(-1, d).to(device))
    recon_pooled = recon_pooled.reshape(b, w, d).cpu()
    win_idx = torch.from_numpy(win_of_token)
    return recon_pooled[:, win_idx, :]  # broadcast each token to its window's recon


def _token_level_replacement(clean_tokens: torch.Tensor, sae, device) -> torch.Tensor:
    """Encode/decode every raw token independently through `sae`, with no
    window pooling or broadcast at all. See `forecast_preservation`'s "token"
    granularity docstring for the distribution-shift caveat this carries."""
    b, t, d = clean_tokens.shape
    recon, _ = sae(clean_tokens.reshape(-1, d).to(device))
    return recon.reshape(b, t, d).cpu()


def forecast_preservation(cfg: PipelineConfig, adapter, layer: str, sae,
                          store: ActivationStore, data: BenchmarkData, device,
                          granularity: str = "window",
                          allowed_series: np.ndarray | None = None,
                          split: str = "all") -> dict:
    """Patch the SAE's reconstruction of clean activations into a clean forward pass.

    Compares the resulting forecast's MASE against the model's own unpatched
    forecast, both against real targets.

    `granularity="window"` (the original, default behavior, kept for
    continuity with every number already recorded under this key): the SAE
    trains on window-pooled activations (`act/{model}/{layer}`), so its
    reconstruction is computed at that same pooled granularity, then
    broadcast to every raw token belonging to that window (via the same
    per-token-to-window assignment `analysis/l3_perturbation.py` already
    uses for its own patching) before being patched in as one whole-context
    replacement via `token_patch` -- the same intervention primitive L3
    uses. This means any degradation reported here reflects the SAE's
    reconstruction error *and* the window-pooling information loss
    together, not the SAE in isolation; the two are not decomposed in this
    pass.

    ⚠️ That window-pooling loss is **not the same size for every
    architecture, and the difference is large enough to dominate this
    check's result for some models.** When `alignment.window` equals a
    model's own token width (TimesFM's default: one 32-step patch per
    window), the broadcast is exact -- one token per window, nothing to
    average over -- so this check measures close to pure SAE reconstruction
    error. When a model tokenizes at a finer grain than the window
    (Chronos: one token per *timestep*, so a 32-step window covers 32
    distinct tokens), broadcasting one reconstructed vector across all of
    them destroys real within-window variation regardless of SAE quality.
    Confirmed empirically, not just argued (ROADMAP.md §6.2's Findings):
    after fixing this baseline's dead-feature collapse, TimesFM's check
    passed almost exactly (ΔMASE +0.047) while Chronos-T5-Base's did not
    (ΔMASE +2.37) despite a *similar* reconstruction-fidelity improvement
    for both models -- read Chronos's number here as upper-bounded by this
    granularity mismatch, not as a clean SAE-only validity failure.

    `granularity="token"` (ROADMAP.md §16 E15's fix for the confound above):
    every raw token is encoded/decoded through the SAE independently and
    patched in at its own position -- no window pooling, no broadcast, so
    there is no within-window information loss to confound the result for
    any tokenization granularity. This removes one confound but introduces
    another, smaller one worth reading the number against: the SAE was
    *trained* on window-pooled vectors, so applying it to raw per-token
    vectors is an out-of-training-distribution input for models that
    tokenize finer than the window (Chronos) -- a difference in scale/
    variance between an average-of-32-tokens and a single token, not
    something the dictionary ever saw during training. A `mase_delta` that
    improves substantially under "token" relative to "window" for such a
    model is evidence the window-broadcast confound was the dominant
    driver; one that stays bad under both points at the SAE's own
    reconstruction quality instead.

    `allowed_series` restricts the evaluated sample to one side of the
    train/held-out series split (`train.py::split_series`), so a ΔMASE can be
    reported on series the dictionary never saw; `split` is the label carried
    into the artifact ("train", "heldout", or "all" when unrestricted). The
    split is over SERIES, never windows -- invariant 2 -- because this check
    patches whole contexts and every window of a series shares one forecast.
    """
    if granularity not in ("window", "token"):
        raise ValueError(f"forecast_preservation: unknown granularity {granularity!r}")
    adapter.ensure_loaded()
    requested = cfg.sae.forecast_preservation_max_series
    families = data.meta["family"].to_numpy()
    # `allowed_series` restricts the sample to one side of the train/held-out
    # SERIES split (train.py::split_series). Sampling is stratified WITHIN the
    # allowed set rather than drawn corpus-wide and filtered, so a split that
    # happens to under-represent a family still yields a family-balanced
    # sample of what it does contain instead of a skewed remainder.
    pool = (np.arange(data.n) if allowed_series is None
            else np.asarray(allowed_series, dtype=int))
    cap = capped_take(requested, n_available=int(pool.size), batch_size=adapter.cfg.batch_size)
    take = cap["n_realized"]
    # A head slice here silently scored both models on whichever series happen
    # to sit first in a corpus written grouped by task -- stratified instead
    # (ROADMAP.md sec 15 A4); `n_requested`/`n_realized`/`limited_by` are
    # recorded below so this run's exact sample, and *why* it differs from the
    # configured request when it does, is comparable to another's (sec 15 A16).
    rows = pool[sample_rows(int(pool.size), take, cfg.run.seed + 11, strata=families[pool])]
    contexts = data.contexts()[rows]
    targets = data.targets()[rows]
    seed = cfg.run.seed + 11

    torch.manual_seed(seed)
    f_clean = adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)["point"]
    mase_clean = mase(f_clean, targets, contexts)

    clean_tokens = capture_raw_tokens(adapter, contexts, [layer])[layer]  # [B, n_tokens, D] fp32
    if granularity == "window":
        replacement = _window_broadcast_replacement(clean_tokens, sae, adapter, cfg, device)
    else:
        replacement = _token_level_replacement(clean_tokens, sae, device)

    with token_patch(adapter.module, layer, adapter.token_slice, replacement):
        torch.manual_seed(seed)
        f_patch = adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)["point"]
    mase_patch = mase(f_patch, targets, contexts)

    return {
        "granularity": granularity,
        "split": split,
        "n_series": int(take),
        "n_requested": cap["n_requested"],
        "n_realized": cap["n_realized"],
        "limited_by": cap["limited_by"],
        "rows": [int(r) for r in rows],
        "mase_clean": float(np.mean(mase_clean)),
        "mase_reconstructed": float(np.mean(mase_patch)),
        "mase_delta": float(np.mean(mase_patch) - np.mean(mase_clean)),
    }


@torch.no_grad()
def _feature_ablated_replacement(clean_tokens: torch.Tensor, sae, device,
                                 feature_idx: int) -> torch.Tensor:
    """Token-level SAE reconstruction with one dictionary feature zeroed
    post-encode, before decoding. Same seam `_token_level_replacement` uses,
    with one extra step in between -- this is what isolates a single
    feature's own marginal contribution, holding every other feature's
    reconstruction fixed."""
    b, t, d = clean_tokens.shape
    features = sae.encode(clean_tokens.reshape(-1, d).to(device))
    features[:, feature_idx] = 0.0
    recon = sae.decode(features)
    return recon.reshape(b, t, d).cpu()


def feature_ablation_effects(cfg: PipelineConfig, adapter, layer: str, sae,
                             data: BenchmarkData, device,
                             candidate_features: list, seed_offset: int = 14) -> dict:
    """For each feature in `candidate_features`, zero it in the token-level
    SAE reconstruction and measure the forecast impact (ROADMAP.md §7
    bullet 3 / §16 E15's second half) -- the sharpest test of whether a
    learned feature is real causal computational structure or just a
    descriptive correlation with some ground-truth field.

    Deliberately **token** granularity only, never "window": window-
    broadcast would reintroduce exactly the confound `forecast_preservation`
    §16 E15 diagnosed and fixed (destroying real within-window token
    variation for any model tokenizing finer than the alignment window),
    and here it would additionally hide a single feature's own effect
    behind that same information loss.

    The comparison that isolates a feature's own marginal contribution is
    against the **full** token-level reconstruction (every feature intact,
    i.e. `forecast_preservation(granularity="token")`'s own replacement) --
    not against the raw clean forecast, which conflates "this feature
    matters" with "this SAE's reconstruction error, from every feature
    combined, matters." Both deltas are reported (`mase_delta_vs_full_recon`
    is the causal one to read; `mase_delta_vs_clean` is context). Per-family
    breakdown (via `data.families`) lets a feature matched to e.g. "trend
    order" (by `ground_truth.py::best_ground_truth_matches`) be checked for
    whether its causal effect actually concentrates in trend-dominant
    series, or is spread evenly across families the match itself gives no
    reason to expect an effect in -- exactly the asymmetry a purely
    descriptive (non-causal) correlation would fail to produce.

    `candidate_features` is caller-supplied (typically the top-K feature
    indices by `|rho|` from an already-computed `ground_truth_alignment`
    result) rather than every dictionary atom, since each one costs one
    extra full forward pass over `data.n` capped series -- ablating an
    entire dictionary (`dict_size_mult * d_in`, often thousands of atoms)
    unconditionally would be far more expensive than training the SAE
    itself.
    """
    adapter.ensure_loaded()
    requested = cfg.sae.feature_ablation_max_series
    cap = capped_take(requested, n_available=data.n, batch_size=adapter.cfg.batch_size)
    take = cap["n_realized"]
    seed = cfg.run.seed + seed_offset
    families = data.families
    rows = sample_rows(data.n, take, seed, strata=families)
    contexts = data.contexts()[rows]
    targets = data.targets()[rows]
    row_families = families[rows]

    torch.manual_seed(seed)
    f_clean = adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)["point"]
    mase_clean = mase(f_clean, targets, contexts)

    clean_tokens = capture_raw_tokens(adapter, contexts, [layer])[layer]
    full_recon = _token_level_replacement(clean_tokens, sae, device)
    with token_patch(adapter.module, layer, adapter.token_slice, full_recon):
        torch.manual_seed(seed)
        f_full = adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)["point"]
    mase_full = mase(f_full, targets, contexts)

    feature_results = []
    for f_idx in candidate_features:
        replacement = _feature_ablated_replacement(clean_tokens, sae, device, int(f_idx))
        with token_patch(adapter.module, layer, adapter.token_slice, replacement):
            torch.manual_seed(seed)
            f_ablated = adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)["point"]
        mase_ablated = mase(f_ablated, targets, contexts)
        delta_vs_full = mase_ablated - mase_full
        by_family = {}
        for fam in np.unique(row_families):
            mask = row_families == fam
            if mask.sum() < 3:
                continue
            by_family[str(fam)] = float(np.mean(delta_vs_full[mask]))
        feature_results.append({
            "feature": int(f_idx),
            "mase_delta_vs_full_recon": float(np.mean(delta_vs_full)),
            "mase_delta_vs_clean": float(np.mean(mase_ablated - mase_clean)),
            "mase_delta_by_family": by_family,
        })

    return {
        "granularity": "token",
        "n_series": int(take),
        "n_requested": cap["n_requested"],
        "n_realized": cap["n_realized"],
        "limited_by": cap["limited_by"],
        "rows": [int(r) for r in rows],
        "mase_clean": float(np.mean(mase_clean)),
        "mase_full_reconstruction": float(np.mean(mase_full)),
        "features": feature_results,
    }


@torch.no_grad()
def _feature_steered_replacement(clean_tokens: torch.Tensor, sae, device,
                                 feature_idx: int, delta: float) -> torch.Tensor:
    """Token-level SAE reconstruction with one dictionary feature's
    post-encode activation shifted by `delta` (added, not zeroed -- the
    steering-vector intervention `ROADMAP.md` §16 E14 asks for, sibling to
    `_feature_ablated_replacement`'s zeroing). Applied uniformly to every
    token's own encoded value, including tokens where the feature wasn't
    originally active -- this is a "push this direction" intervention, not
    a "condition on already being active" one."""
    b, t, d = clean_tokens.shape
    features = sae.encode(clean_tokens.reshape(-1, d).to(device))
    features[:, feature_idx] = features[:, feature_idx] + delta
    recon = sae.decode(features)
    return recon.reshape(b, t, d).cpu()


def feature_steering_effects(cfg: PipelineConfig, adapter, layer: str, sae,
                             data: BenchmarkData, device, candidate_features: list,
                             periods: np.ndarray | None = None, seed_offset: int = 15,
                             strength_sigma: float = 2.0) -> dict:
    """For each feature in `candidate_features`, add a signed `±strength_sigma
    * (that feature's own clean activation std)` steering delta to the
    token-level SAE reconstruction and measure both the forecast disruption
    and, where the matched ground-truth field supports it
    (`analysis/steering.py::predicted_direction_metric`), whether the
    forecast's own directional metric (trend slope / seasonal-band FFT
    magnitude) moves the way the feature's signed correlation with that
    field predicts. This is the sharper causal test §16 E14 asks for on top
    of `feature_ablation_effects`: ablation shows a feature matters at all;
    this shows whether pushing it moves the forecast in a *specific,
    falsifiable* direction, not just any direction.

    Deliberately **token** granularity only, mirroring
    `feature_ablation_effects`'s own reasoning for the same choice -- a
    window-broadcast reconstruction would blur exactly the fine-grained
    causal signal a single feature's steering effect depends on.

    Both directional deltas are reported (`trend_response` /
    `seasonal_response`), independent of `best_field` -- `feature_steering
    _effects` itself makes no claim about which field a feature matches or
    whether the response is "correct"; that verdict (via
    `steering.py::evaluate_direction_match`) is computed by the caller,
    which already has each candidate's `best_field`/`rho` from
    `ground_truth_alignment` and shouldn't be duplicated here.

    `periods` is each of `data`'s series' own ground-truth
    `seasonal_period_dominant` (`[data.n]`, `np.nan` where absent/real-
    derived) -- optional, since a feature matched to a non-seasonal field
    never needs it; `seasonal_response` is `None` for every row where no
    finite period was supplied.
    """
    adapter.ensure_loaded()
    requested = cfg.sae.feature_steering_max_series
    cap = capped_take(requested, n_available=data.n, batch_size=adapter.cfg.batch_size)
    take = cap["n_realized"]
    seed = cfg.run.seed + seed_offset
    families = data.families
    rows = sample_rows(data.n, take, seed, strata=families)
    contexts = data.contexts()[rows]
    targets = data.targets()[rows]
    row_periods = periods[rows].astype(np.float64) if periods is not None else np.full(take, np.nan)
    have_periods = bool(np.isfinite(row_periods).any())

    torch.manual_seed(seed)
    f_clean = adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)["point"]
    mase_clean = mase(f_clean, targets, contexts)

    clean_tokens = capture_raw_tokens(adapter, contexts, [layer])[layer]
    full_recon = _token_level_replacement(clean_tokens, sae, device)
    with token_patch(adapter.module, layer, adapter.token_slice, full_recon):
        torch.manual_seed(seed)
        f_full = adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)["point"]
    mase_full = mase(f_full, targets, contexts)
    trend_full = trend_slope(f_full)
    seasonal_full = seasonal_band_magnitude(f_full, row_periods) if have_periods else None

    feature_results = []
    for f_idx in candidate_features:
        b, t, d = clean_tokens.shape
        with torch.no_grad():
            feat_vals = sae.encode(clean_tokens.reshape(-1, d).to(device))[:, int(f_idx)]
            sigma = float(feat_vals.std().cpu())
        delta_mag = strength_sigma * sigma if sigma > 1e-8 else strength_sigma

        signed = {}
        for sign_name, delta in (("up", delta_mag), ("down", -delta_mag)):
            replacement = _feature_steered_replacement(clean_tokens, sae, device, int(f_idx), delta)
            with token_patch(adapter.module, layer, adapter.token_slice, replacement):
                torch.manual_seed(seed)
                f_steered = adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)["point"]
            mase_steered = mase(f_steered, targets, contexts)
            trend_response = float(np.nanmean(trend_slope(f_steered) - trend_full))
            seasonal_response = (float(np.nanmean(seasonal_band_magnitude(f_steered, row_periods) - seasonal_full))
                                 if have_periods else None)
            signed[sign_name] = {
                "delta": float(delta),
                "mase_delta_vs_full_recon": float(np.mean(mase_steered - mase_full)),
                "trend_response": trend_response,
                "seasonal_response": seasonal_response,
            }

        feature_results.append({
            "feature": int(f_idx),
            "steering_sigma": sigma,
            "steering_magnitude": delta_mag,
            "up": signed["up"],
            "down": signed["down"],
        })

    return {
        "granularity": "token",
        "n_series": int(take),
        "n_requested": cap["n_requested"],
        "n_realized": cap["n_realized"],
        "limited_by": cap["limited_by"],
        "rows": [int(r) for r in rows],
        "have_periods": have_periods,
        "mase_clean": float(np.mean(mase_clean)),
        "mase_full_reconstruction": float(np.mean(mase_full)),
        "features": feature_results,
    }
