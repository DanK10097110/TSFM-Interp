"""The seasonality circuit (ROADMAP.md sec 20 H8) — Stage 0 primitives.

A minimal-sufficient-set search over attention heads needs a metric that
measures *seasonality specifically*, not behaviour in general: ΔMASE would
let a head set score well by restoring the forecast's overall level while
the seasonal component the `deseasonalize` corruption notched out stays
gone. `seasonal_power` reads a forecast's own rFFT and reports the fraction
of its (non-DC) spectral energy sitting in the band around a series' known
ground-truth dominant period — the exact quantity `corrupt_deseasonalize`
(`l3_perturbation.py`) removes, at the same +/-1-bin neighbourhood width, so
a head set that restores seasonality restores *this* number, not merely the
forecast's level.

`restoration` is L3's own convention (`_window_restoration`'s inline
formula, pulled out here so H8 and L3 provably mean the same thing by
construction rather than by two independently-typed copies): a set's
necessity/sufficiency score is always read as the fraction of the clean-vs-
corrupted gap on `seasonal_power` that a set's ablation destroys or a set's
patch restores, never as a bare power value on its own.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.store import ActivationStore
from ..utils import log

_EPS = 1e-8


def seasonal_power(forecast: np.ndarray, period, eps: float = _EPS) -> np.ndarray:
    """Fraction of `forecast`'s non-DC spectral energy near `period`.

    `forecast` is `[..., horizon]`; `period` is a per-row ground-truth
    dominant seasonal period in timesteps (a scalar or an array broadcast-
    compatible with `forecast.shape[:-1]`), `NaN`/`<=0` where a series
    carries no seasonality ground truth (every real-derived-tier series and
    any synthetic series with zero seasonalities, `CLAUDE.md` sec 4.1) --
    those rows return `NaN` rather than a fabricated value, matching
    `spectral_lens.py`'s existing convention for the same ground-truth
    column. The target FFT bin is `round(horizon / period)` (identical
    convention to `spectral_lens.py::spectral_lens_stats`, so both readings
    of "which bin is this series' seasonality" agree); the summed band is
    that bin plus its immediate neighbours, matching
    `corrupt_deseasonalize`'s own `(-1, 0, 1)` notch neighbourhood exactly --
    this metric measures the fraction of energy in precisely the band that
    corruption removes, no more and no less. The DC bin is excluded from
    both the band and the normalizing total, since DC is trend/level, not
    seasonality, and letting it dominate the denominator would make every
    series' seasonal *fraction* look artificially small regardless of head
    set.
    """
    forecast = np.asarray(forecast, dtype=np.float64)
    horizon = forecast.shape[-1]
    spec = np.fft.rfft(forecast, axis=-1)
    mag2 = np.abs(spec) ** 2
    mag2[..., 0] = 0.0
    n_bins = mag2.shape[-1]

    period_arr = np.broadcast_to(np.asarray(period, dtype=np.float64), forecast.shape[:-1])
    valid = np.isfinite(period_arr) & (period_arr > 0)
    safe_period = np.where(valid, period_arr, np.inf)
    bin_idx = np.clip(np.round(horizon / safe_period).astype(np.int64), 1, n_bins - 1)

    total = mag2.sum(axis=-1) + eps
    band = np.zeros_like(total)
    for off in (-1, 0, 1):
        neighbour = np.clip(bin_idx + off, 1, n_bins - 1)
        band = band + np.take_along_axis(mag2, neighbour[..., None], axis=-1)[..., 0]

    power = band / total
    return np.where(valid, power, np.nan).astype(np.float32)


def restoration(m_patch: np.ndarray, m_clean: np.ndarray, m_corr: np.ndarray,
                eps: float = _EPS) -> np.ndarray:
    """`1 - |m_patch - m_clean| / |m_corr - m_clean|` — L3's own convention.

    Identical in shape to `l3_perturbation.py::_window_restoration`'s inline
    formula and `restoration_by_horizon`'s, pulled out so an H8 number and an
    L3 number are provably the same computation rather than two hand-copies
    that could silently drift (`CLAUDE.md` sec 11.24's lesson, applied
    pre-emptively): 1.0 means the metric fully recovered to its clean value,
    0.0 means it is exactly as damaged as the corrupted-and-unpatched case,
    and values outside `[0, 1]` are legitimate (over- or under-shoot) rather
    than clamped, matching every existing L3 restoration reading. `eps` is
    added to the denominator (L3's own call sites bake an identical `+1e-8`
    into `damage` before calling the forecast-space version) so a series
    whose corruption happened to leave `seasonal_power` exactly unchanged
    doesn't raise a division-by-zero -- `seasonal_power`'s own +/-1-bin band
    can coincide with `corrupt_deseasonalize`'s notch by construction in a
    way forecast MAE rarely does, so this guard is more load-bearing here
    than in L3's original two call sites.
    """
    m_patch = np.asarray(m_patch, dtype=np.float64)
    m_clean = np.asarray(m_clean, dtype=np.float64)
    m_corr = np.asarray(m_corr, dtype=np.float64)
    damage = np.abs(m_corr - m_clean)
    return (1.0 - np.abs(m_patch - m_clean) / (damage + eps)).astype(np.float32)


def score_single_head_effects(cfg: PipelineConfig, adapter, store: ActivationStore,
                              data: BenchmarkData, top_k: int = None) -> Optional[dict]:
    """Stage 1 (ROADMAP.md sec 20 H8): score every attention head one at a
    time on (a) the existing mean-ablation DeltaMASE and (b) a new causal
    effect on `seasonal_power`, from the SAME forward passes.

    Reuses `attention.py::_ablation_setup`/`_ablate_heads` directly (rather
    than re-deriving rows/contexts/seed independently) so `head_delta`
    reproduces `attention/meta.json`'s recorded array bit-for-bit *by
    construction* -- it is not a second, independently-written computation
    that could silently disagree with the first (`CLAUDE.md` sec 11.24's
    same-inputs discipline, applied here instead of discovered the hard way).
    `_ablate_heads`'s `on_forecast` hook records each head's `seasonal_power`
    effect from the identical per-head forecast the ΔMASE score is already
    computed from, so this costs zero extra forward passes over the existing
    ablation stage.

    Per-series periods come from `stats.dominant_period` -- the same
    autocorrelation estimator `attention.py::_pattern_analysis` already uses
    for `top_periodicity_heads`'s own family periods -- so this scores heads
    on the identical basis the periodicity taxonomy does, not a second,
    ground-truth-derived basis (`sae/ground_truth.py::load_ground_truth_table`,
    which additionally requires a sealed corpus) that could disagree with the
    taxonomy for reasons having nothing to do with which heads actually
    matter.

    Returns `None` (logged, degrading per `CLAUDE.md` sec 2.5) when the
    adapter has no `attention_info()` -- ablation-unsupported is an expected,
    not exceptional, per-adapter outcome.
    """
    from . import attention as _attn
    from .lens import predict_rows
    from .stats import dominant_period

    info = adapter.attention_info()
    if info is None:
        log.info("seasonality_circuit %s: single-head scoring unsupported "
                 "(no attention_info)", adapter.name)
        return None

    acfg = cfg.attention
    rows, contexts, targets, scale, families, fam_list, seed = \
        _attn._ablation_setup(cfg, data)
    periods = np.array([dominant_period(c) for c in contexts], dtype=np.float64)

    if store.has_predictions(adapter.name):
        f_clean = store.load_predictions(adapter.name)["point"][rows]
    else:
        f_clean = predict_rows(adapter, contexts, data.horizon, cfg.l0.quantiles, seed)
    mase_clean = np.abs(f_clean - targets).mean(axis=1) / scale
    power_clean = seasonal_power(f_clean, periods)

    def delta(forecast: np.ndarray) -> np.ndarray:
        return np.abs(forecast - targets).mean(axis=1) / scale - mase_clean

    blocks = info[:: max(1, acfg.head_layer_stride)]
    n_heads = blocks[0]["n_heads"]
    block_names = [b["block"] for b in blocks]
    power_loss = np.full((len(blocks), n_heads), np.nan, dtype=np.float32)

    def on_forecast(bi: int, h: int, f: np.ndarray) -> None:
        # Power LOST when this head is ablated (positive = head helps carry
        # seasonality); NaN-safe since `seasonal_power` returns NaN for rows
        # with no estimable period.
        power_loss[bi, h] = np.nanmean(power_clean - seasonal_power(f, periods))

    head_d, head_f = _attn._ablate_heads(cfg, adapter, blocks, contexts, data.horizon,
                                         seed, delta, families, fam_list,
                                         on_forecast=on_forecast)

    k = acfg.top_k if top_k is None else top_k
    top_power_heads = _rank_head_matrix(power_loss, block_names, k)

    return {"head_delta": head_d, "head_delta_family": head_f,
           "head_power_loss": power_loss, "block_names": block_names,
           "top_power_heads": top_power_heads,
           "n_series": int(len(rows)), "families": fam_list}


def _rank_head_matrix(scores: np.ndarray, block_names: list, k: int) -> list:
    """Top-`k` (block, head) entries of a `[n_blocks, n_heads]` score matrix,
    NaN-safe and highest-first -- the same shape of ranking
    `attention.py::_top_entries` produces for ΔMASE, so a caller can treat
    the two outputs uniformly.
    """
    flat = np.nan_to_num(scores, nan=-np.inf).ravel()
    order = np.argsort(flat)[::-1]
    out = []
    for idx in order[:k]:
        bi, hi = np.unravel_index(idx, scores.shape)
        if not np.isfinite(scores[bi, hi]):
            continue
        out.append({"layer": block_names[bi], "head": int(hi),
                    "score": float(scores[bi, hi])})
    return out


def candidate_head_set(top_periodicity_heads: list, top_power_heads: list,
                       cap: int = 12) -> list:
    """Stage 1's candidate set for the Stage 2 minimal-set search: the union
    of the periodicity taxonomy's top heads (`attention.py::_head_scores`'s
    `top_periodicity_heads`) and the top heads by single-head seasonal-power
    ablation effect (`score_single_head_effects`'s `top_power_heads`), each a
    list of `{"layer": ..., "head": ...}` dicts.

    A head in both lists counts once. Ties beyond `cap` are broken by list
    order (periodicity first, matching this item's own stated ordering) --
    not by score, since the two lists are on different, incommensurable
    scales (attention mass vs. a ΔMASE-shaped power loss) and ranking them
    against each other directly would be a fabricated comparison.
    """
    seen, out = set(), []
    for entries in (top_periodicity_heads, top_power_heads):
        for e in entries:
            key = (e["layer"], int(e["head"]))
            if key not in seen:
                seen.add(key)
                out.append({"layer": key[0], "head": key[1]})
    return out[:cap]


def _capture_oproj_inputs(adapter, names: list, contexts: np.ndarray) -> dict:
    """Full per-position o_proj input tensor per module, one forward pass,
    no internal chunking assumed.

    Unlike `attention.py::_oproj_means` (which reduces to a mean and is
    therefore free to batch via `batch_slices`), Stage 2's sufficiency arm
    needs the exact per-position clean value to patch into a corrupted
    forward, so `contexts` must already fit in a single internal forward
    (the same constraint `hooks.input_slice_ablate`'s per-position path
    already enforces and raises on -- CLAUDE.md sec 11.5's class of trap --
    so callers should size `contexts` the same way `attention.ablation_
    max_series` sizes it for that path). Returns `{name: [B, T, D] tensor}`,
    matching `input_slice_ablate`'s per-position `value` contract exactly.
    """
    import torch
    from ..extraction.hooks import InputCatcher
    with InputCatcher(adapter.module, names) as catcher:
        with torch.no_grad():
            adapter.forward(adapter.prepare(contexts))
        return {name: x.detach().clone() for name, x in catcher.collect().items()}


def _head_specs(blocks_by_name: dict, head_set: list, values: dict) -> list:
    """`hooks.multi_slice_ablate`'s `specs` list for a set of (layer, head)
    entries, given a `{o_proj_module_name: value_tensor}` map already sliced
    to that module's full output width (mean-shaped `[D]` for necessity, or
    per-position `[B, T, D]` for sufficiency -- `input_slice_ablate`'s hook
    body handles either identically, per its own docstring)."""
    specs = []
    for e in head_set:
        block = blocks_by_name[e["layer"]]
        dh = block["head_dim"]
        h = int(e["head"])
        sl = slice(h * dh, (h + 1) * dh)
        v = values[block["o_proj"]]
        specs.append((block["o_proj"], sl, v[..., sl] if v.dim() > 1 else v[sl]))
    return specs


def set_ablation_forecast(cfg: PipelineConfig, adapter, blocks_by_name: dict,
                          head_set: list, contexts: np.ndarray, horizon: int,
                          seed: int) -> np.ndarray:
    """Mean-ablate exactly `head_set` jointly (Stage 2's necessity arm): the
    set's combined effect on a CLEAN forward, via `hooks.multi_slice_ablate`
    -- a set's effect is not assumed to be the sum of its members' single-
    head deltas (that assumption is exactly what this search exists to
    test), so every head in the set is intervened on in the same forward
    pass.
    """
    from . import attention as _attn
    from .lens import predict_rows
    from ..extraction.hooks import multi_slice_ablate

    names = sorted({e["layer"] for e in head_set})
    means = _attn._oproj_means(adapter, [blocks_by_name[n]["o_proj"] for n in names],
                               contexts)
    specs = _head_specs(blocks_by_name, head_set, means)
    with multi_slice_ablate(adapter.module, specs):
        return predict_rows(adapter, contexts, horizon, cfg.l0.quantiles, seed)


def set_patch_forecast(cfg: PipelineConfig, adapter, blocks_by_name: dict,
                       head_set: list, clean_contexts: np.ndarray,
                       corrupted_contexts: np.ndarray, horizon: int,
                       seed: int) -> np.ndarray:
    """Patch exactly `head_set`'s clean per-position o_proj-input values into
    a forward over `corrupted_contexts` (Stage 2's sufficiency arm): does
    restoring only this set's contribution recover the forecast's
    seasonality on an otherwise-corrupted input.
    """
    from .lens import predict_rows
    from ..extraction.hooks import multi_slice_ablate

    names = sorted({e["layer"] for e in head_set})
    clean_vals = _capture_oproj_inputs(adapter, [blocks_by_name[n]["o_proj"] for n in names],
                                       clean_contexts)
    specs = _head_specs(blocks_by_name, head_set, clean_vals)
    with multi_slice_ablate(adapter.module, specs):
        return predict_rows(adapter, corrupted_contexts, horizon, cfg.l0.quantiles, seed)


def greedy_minimal_set(cfg: PipelineConfig, adapter, blocks_by_name: dict,
                       candidates: list, clean_contexts: np.ndarray,
                       corrupted_contexts: np.ndarray, periods: np.ndarray,
                       m_clean: np.ndarray, m_corr: np.ndarray, horizon: int,
                       seed: int, tau: float = 0.8) -> dict:
    """Greedy forward selection over `candidates` by SUFFICIENCY restoration
    (ROADMAP.md sec 20 H8 Stage 2): at each step, add whichever remaining
    head raises the growing set's mean restoration the most; stop once
    restoration reaches `tau` or every candidate has been added. Reports the
    whole trace, not just the final set -- where the curve saturates *is*
    the answer (this item's own stated design), and a set that never
    reaches `tau` is a reportable negative rather than a silently-swallowed
    failure (CLAUDE.md sec 2.5).
    """
    selected, remaining, trace = [], list(candidates), []
    while remaining:
        best_cand, best_mean, best_r = None, -np.inf, None
        for cand in remaining:
            trial = selected + [cand]
            f = set_patch_forecast(cfg, adapter, blocks_by_name, trial,
                                   clean_contexts, corrupted_contexts, horizon, seed)
            r = restoration(seasonal_power(f, periods), m_clean, m_corr)
            r_mean = float(np.nanmean(r))
            if r_mean > best_mean:
                best_cand, best_mean, best_r = cand, r_mean, r
        selected.append(best_cand)
        remaining.remove(best_cand)
        trace.append({"added": best_cand, "set_size": len(selected),
                      "restoration": best_mean})
        if best_mean >= tau:
            break
    return {"trace": trace, "selected_set": selected,
           "selected_restoration_per_series": best_r,
           "reached_tau": bool(trace and trace[-1]["restoration"] >= tau)}


def best_prefix_from_trace(trace: list, full_set: list) -> tuple:
    """The trace's global-best-restoration prefix of `full_set`, and its size.

    ROADMAP.md sec 20 H8 Stage 2's resolution of the originally-specified
    but never-implemented online diminishing-returns stopping rule: rather
    than stopping the greedy search mid-run the moment one step's gain is
    small (which a non-monotonic restoration curve -- observed live on
    Chronos-T5-Base, a dip at steps 2-3 before a rise to the trace's best at
    step 5 -- would truncate before reaching the real optimum), let the
    search run to `tau`/exhaustion as `greedy_minimal_set` already does,
    then take the single best point across the WHOLE trace. This can only
    match or beat any online rule's outcome, since it considers every size
    the search actually visited rather than committing early.

    `trace` is `greedy_minimal_set`'s own `"trace"` list (each entry has
    `"set_size"` and `"restoration"`); `full_set` is that same call's
    `"selected_set"`. Returns `([], 0)` for an empty trace -- an empty
    candidate set has no prefix to take, and this is a fact about the input,
    not an error to fabricate a nonempty answer around.
    """
    if not trace:
        return [], 0
    best_size = max(trace, key=lambda t: t["restoration"])["set_size"]
    return full_set[:best_size], best_size


def random_set_null(cfg: PipelineConfig, adapter, blocks_by_name: dict,
                    all_heads: list, set_size: int, n_draws: int,
                    clean_contexts: np.ndarray, corrupted_contexts: np.ndarray,
                    periods: np.ndarray, m_clean: np.ndarray, m_corr: np.ndarray,
                    horizon: int, seed: int, rng_seed: int) -> np.ndarray:
    """`n_draws` random size-`set_size` subsets of `all_heads` -- every
    scanned (layer, head) pair, not just the candidate pool, since the floor
    this item's own acceptance criterion needs is "a set this size drawn
    from anywhere," not "from the periodicity/power-enriched candidate
    pool," which would bias the floor upward and make the real result look
    better than it is. Returns `[n_draws, n_series]` per-series restoration
    so the caller can cluster-bootstrap the gap over series (invariant 2),
    not just compare two scalar means.
    """
    rng = np.random.default_rng(rng_seed)
    draws = []
    for _ in range(n_draws):
        idx = rng.choice(len(all_heads), size=min(set_size, len(all_heads)), replace=False)
        head_set = [all_heads[j] for j in idx]
        f = set_patch_forecast(cfg, adapter, blocks_by_name, head_set,
                               clean_contexts, corrupted_contexts, horizon, seed)
        draws.append(restoration(seasonal_power(f, periods), m_clean, m_corr))
    return np.stack(draws)


def minimal_set_search(cfg: PipelineConfig, adapter, store: ActivationStore,
                       data: BenchmarkData, candidate_set: list,
                       tau: float = 0.8, n_null_draws: int = 5) -> Optional[dict]:
    """Stage 2 (ROADMAP.md sec 20 H8): the greedy minimal sufficient set,
    its necessity check, and the MANDATORY random-set null with a
    pre-registered abort.

    `greedy_minimal_set` runs its trace to `tau` or exhaustion unchanged;
    this function then takes the trace's GLOBAL best-restoration prefix as
    `selected_set`, not the raw (possibly-exhausted, possibly-monotonically-
    worsening-after-its-peak) full trace. This is the resolution of Stage
    2's originally-specified but never-implemented second stopping
    condition ("stop when the best remaining addition improves it by less
    than the null's own spread") -- an online per-step version of that rule
    was rejected after a real run (Chronos-T5-Base, 2026-08-21) showed a
    genuinely non-monotonic restoration trace (a dip at steps 2-3, then a
    rise to the trace's global best at step 5): an online rule would have
    stopped on the early noise and never reached the real optimum. Taking
    the global argmax over the whole trace cannot make that mistake, and it
    lets the null test below (already mandatory) serve double duty as the
    "is this point real or noise" gate, rather than adding a second,
    redundant one. The full, untrimmed trace and set are still returned
    (`full_trace_selected_set`) so nothing is hidden.

    The null is the acceptance criterion, not a footnote (CLAUDE.md sec 18
    F6's precedent applied here): the (now correctly-sized) selected set's
    per-series sufficiency restoration is compared against same-size random
    sets via a paired cluster bootstrap over series (invariant 2), and
    `cleared_null` is `True` only if that gap's CI excludes zero. If it does
    not, this function's own contract is to say so plainly -- no re-tuning
    `tau`, no second search -- Stage 2's own pre-registered abort rule,
    applied here rather than left for a caller to remember.
    """
    from . import attention as _attn
    from .lens import predict_rows
    from .stats import dominant_period
    from .l3_perturbation import corrupt_deseasonalize
    from ..analysis.stats import paired_bootstrap

    info = adapter.attention_info()
    if info is None:
        log.info("seasonality_circuit %s: minimal-set search unsupported "
                 "(no attention_info)", adapter.name)
        return None

    acfg = cfg.attention
    rows, contexts, targets, scale, families, fam_list, seed = \
        _attn._ablation_setup(cfg, data)
    periods = np.array([dominant_period(c) for c in contexts], dtype=np.float64)
    blocks = info[:: max(1, acfg.head_layer_stride)]
    blocks_by_name = {b["block"]: b for b in blocks}
    n_heads = blocks[0]["n_heads"]
    all_heads = [{"layer": b["block"], "head": h} for b in blocks for h in range(n_heads)]

    corrupted_contexts = corrupt_deseasonalize(contexts, np.random.default_rng(seed))

    f_clean = predict_rows(adapter, contexts, data.horizon, cfg.l0.quantiles, seed)
    f_corr = predict_rows(adapter, corrupted_contexts, data.horizon, cfg.l0.quantiles, seed)
    m_clean = seasonal_power(f_clean, periods)
    m_corr = seasonal_power(f_corr, periods)

    greedy = greedy_minimal_set(cfg, adapter, blocks_by_name, candidate_set,
                                contexts, corrupted_contexts, periods,
                                m_clean, m_corr, data.horizon, seed, tau)
    full_trace_selected_set = greedy["selected_set"]

    # See `best_prefix_from_trace`'s docstring and this function's own
    # docstring above for why this replaces an online per-step diminishing-
    # returns rule (ROADMAP.md sec 20 H8 Stage 2, corrected 2026-08-21).
    selected, best_size = best_prefix_from_trace(greedy["trace"], full_trace_selected_set)

    if selected and selected != full_trace_selected_set:
        f_trim = set_patch_forecast(cfg, adapter, blocks_by_name, selected,
                                    contexts, corrupted_contexts, data.horizon, seed)
        selected_restoration_per_series = restoration(
            seasonal_power(f_trim, periods), m_clean, m_corr)
    else:
        selected_restoration_per_series = greedy["selected_restoration_per_series"]

    f_ablate = set_ablation_forecast(cfg, adapter, blocks_by_name, selected,
                                     contexts, data.horizon, seed)
    necessity_r = restoration(seasonal_power(f_ablate, periods), m_clean, m_corr)

    null_draws = random_set_null(cfg, adapter, blocks_by_name, all_heads,
                                 len(selected), n_null_draws, contexts,
                                 corrupted_contexts, periods, m_clean, m_corr,
                                 data.horizon, seed, rng_seed=seed + 999)
    null_floor_per_series = np.nanmean(null_draws, axis=0)

    # Must use the (possibly-trimmed) `selected_restoration_per_series`, not
    # `greedy["selected_restoration_per_series"]` (the full/exhausted set's
    # array) -- the null just above is drawn at `len(selected)`, the trimmed
    # size, so it has to be compared against that same trimmed set's own
    # restoration, not the untrimmed one.
    gap = selected_restoration_per_series - null_floor_per_series
    valid = np.isfinite(gap)
    gap_test = paired_bootstrap(gap[valid]) if valid.sum() >= 3 else None
    cleared_null = bool(gap_test is not None and gap_test["lo"] > 0)

    return {
        "trace": greedy["trace"], "selected_set": selected,
        "full_trace_selected_set": full_trace_selected_set,
        "best_prefix_size": best_size,
        "stopping_rule": "global_argmax_prefix",
        "reached_tau": greedy["reached_tau"],
        "sufficiency_restoration": float(np.nanmean(selected_restoration_per_series)),
        "necessity_restoration": float(np.nanmean(necessity_r)),
        "null_floor_mean": float(np.nanmean(null_floor_per_series)),
        "null_draws_restoration_mean": [float(np.nanmean(d)) for d in null_draws],
        "gap_vs_null": gap_test,
        "cleared_null": cleared_null,
        "n_series": int(len(rows)), "tau": tau, "n_null_draws": n_null_draws,
    }


def periodicity_power_rank_correlation(periodicity_scores: np.ndarray,
                                       periodicity_layers: list,
                                       power_loss: np.ndarray,
                                       power_layers: list) -> dict:
    """Spearman rank correlation between the periodicity taxonomy's per-head
    score (`attention.py::_head_scores`'s `periodicity` array) and this
    stage's per-head seasonal-power ablation effect -- ROADMAP.md sec 20 H8
    Stage 1's "record the rank correlation either way" requirement. A
    disagreement is a finding, not a failure (the taxonomy scores attention
    *mass*, this scores causal *effect* -- CLAUDE.md sec 18 F5's own lesson
    that a plausible-looking ranking can be an artifact of its axis).

    The two harnesses can run at different `capture_layer_stride`/
    `head_layer_stride`, so alignment is over the actual intersection of
    layer-name lists, never an assumed shared shape. Returns `rho: None`
    with a stated reason (degrading per `CLAUDE.md` sec 2.5) when fewer than
    2 shared blocks or fewer than 2 finite paired values exist -- too few
    points for a correlation coefficient to mean anything.
    """
    from scipy.stats import spearmanr

    shared = [b for b in power_layers if b in periodicity_layers]
    if len(shared) < 2:
        return {"rho": None, "n_shared_blocks": len(shared),
               "reason": "fewer than 2 shared blocks between the two layer axes"}
    pidx = [periodicity_layers.index(b) for b in shared]
    aidx = [power_layers.index(b) for b in shared]
    p = np.asarray(periodicity_scores)[pidx].ravel()
    a = np.asarray(power_loss)[aidx].ravel()
    valid = np.isfinite(p) & np.isfinite(a)
    if valid.sum() < 2:
        return {"rho": None, "n_shared_blocks": len(shared),
               "reason": "fewer than 2 finite paired (periodicity, power_loss) values"}
    rho, pval = spearmanr(p[valid], a[valid])
    return {"rho": float(rho), "p_value": float(pval), "n_pairs": int(valid.sum()),
           "n_shared_blocks": len(shared)}
