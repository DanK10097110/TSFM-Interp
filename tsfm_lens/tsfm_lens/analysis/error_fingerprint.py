"""Idiosyncratic-error fingerprinting (`ROADMAP.md` sec 6.3.1, Option C).

The provenance question this reopens -- "were these two checkpoints trained
from a shared lineage?" -- already has one falsified answer on record
(sec 6.3): a representation-similarity method that a same-architecture control
reproduced without any shared training at all. Option C's claim to being
different is structural rather than empirical: two `random_init` twins have
*identical* architecture and make errors that are pure noise, so the control
that killed the earlier method cannot fake this one. That control is run
first, and its number is reported next to every real pair's -- exactly the
"quote the floor beside the value" discipline sec 6.2.1 arrived at the hard
way.

The mechanism, per the item's spec:

1. Per-series error vectors from forecasts this run already made. Nothing
   here loads a model or runs a forward pass -- `store.load_predictions` and
   the persisted targets are the whole input, so a pair costs seconds.
2. Errors are divided by each series' own MASE scale, so a large-amplitude
   series does not dominate the correlation by amplitude alone. This is the
   same denominator L0 scores against, deliberately, so "correlated error"
   here means correlated in the units L0 already reports.
3. **Difficulty is regressed out.** Hard series are hard for everyone, so the
   raw correlation between two competent models' errors is large and means
   nothing. Each model's error is regressed on a series-level difficulty
   basis (`difficulty_features`) and the *residuals* are correlated. This is
   the load-bearing step and the direct analog of L2 reporting only the gain
   over its input-feature baseline.
   The regression is **cross-fitted** over series folds: an in-sample
   residual is shrunk toward zero by whatever the ridge overfit, and that
   shrinkage is shared across models, which would put a spurious common
   component into exactly the quantity being correlated.
   The item's spec named L2's `_baseline_features` (raw window values, rFFT
   magnitudes, per-window statistics) as that basis. **It was measured and
   rejected**: cross-fitted, its out-of-fold R2 is -0.62 -- it predicts
   per-series error worse than the mean does, so it removes no difficulty
   while still producing something shaped like an adjusted residual, and
   under it the pure untrained control scored a magnitude correlation of
   0.979. See `CLAUDE.md` sec 11.36. `adjustment_ok` exists so that failure
   mode cannot recur silently; note it certifies the adjustment is not
   *vacuous*, never that it is sufficient.
4. Both a per-series scalar (mean absolute error, the coarse fingerprint) and
   the **signed per-horizon-step shape** are correlated, since shared
   over-shoot at h=1 followed by shared reversion at h=20 is a far more
   specific signature than one number per series.
5. The series is the resampling unit (invariant 2); CIs are cluster
   bootstraps over series.
6. **Series with an unreliable (near-zero) MASE scale are excluded before
   anything else is computed**, via the same `mase_reliability` mask L0
   already uses for its own aggregate MASE (sec 15 A11) -- found necessary
   by a real run (sec 6.3.1 Option E's lineage-pair experiment), where a
   handful of near-flat-context series turned any nonzero error into a
   multi-million-unit value that swamped both the raw and the
   "difficulty-adjusted" correlation, since ridge regression cannot move an
   outlier that large anywhere near zero. See `error_fingerprint`'s own
   docstring for the full mechanism.

**What this cannot establish.** A positive result is consistent with shared
lineage *and* with two models having been trained on overlapping public
corpora, which is true of nearly every TSFM. Option C does not escape that
confound; it relocates it somewhere measurable. Read a positive as "these
two make the same specific mistakes", which is evidence about training data
and recipe jointly, never about weight inheritance alone.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..extraction.store import ActivationStore, nonfinite_series_mask
from ..utils import log
from .l0_behavioral import _mase_scale
from .stats import bootstrap_ci, mase_reliability


def difficulty_features(contexts: np.ndarray, targets: np.ndarray,
                        scale: np.ndarray) -> tuple:
    """A low-dimensional, model-independent "how hard is this series" basis.

    Deliberately NOT L2's `_baseline_features`, which was the first thing
    tried and is the wrong tool here: it is ~200 per-window raw values and
    rFFT magnitudes designed to *predict a representation*, and against 288
    series it cross-fits to a NEGATIVE out-of-fold R² -- i.e. it removes no
    difficulty at all while looking like it did. What matters for difficulty
    is a handful of series-level quantities, and the single best of them is
    the error a reference forecaster makes, which is by construction
    model-independent and is what "hard series" means operationally.

    Columns: two reference-forecaster errors (last-value naive and
    seasonal-naive at the context's dominant period), then log-scale, log
    target dispersion, lag-1 autocorrelation, spectral centroid, zero
    fraction, and a normalized trend slope. Returned with names so a fit that
    explains nothing can be inspected rather than guessed at.
    """
    n, t = contexts.shape
    h = targets.shape[1]
    eps = 1e-8
    naive = np.abs(targets - contexts[:, -1:]).mean(1) / (scale + eps)

    centered = contexts - contexts.mean(1, keepdims=True)
    spec = np.abs(np.fft.rfft(centered, axis=1))
    spec[:, 0] = 0.0
    freqs = np.arange(spec.shape[1], dtype=np.float64)
    period = np.where(spec.argmax(1) > 0, t / np.maximum(spec.argmax(1), 1), t)
    period = np.clip(period.astype(int), 1, t)
    snaive = np.empty(n)
    for i in range(n):
        idx = t - period[i] + (np.arange(h) % period[i])
        snaive[i] = np.abs(targets[i] - contexts[i, idx]).mean()
    snaive = snaive / (scale + eps)

    cen = centered
    denom = (cen ** 2).sum(1) + eps
    lag1 = (cen[:, 1:] * cen[:, :-1]).sum(1) / denom
    centroid = (spec * freqs).sum(1) / (spec.sum(1) + eps)
    zero_frac = (np.abs(contexts) < eps).mean(1)
    x = np.arange(t, dtype=np.float64)
    xc = x - x.mean()
    slope = (cen * xc).sum(1) / (xc ** 2).sum()
    cols = [naive, snaive, np.log(scale + eps), np.log(targets.std(1) + eps),
            lag1, centroid, zero_frac, slope / (contexts.std(1) + eps)]
    names = ["naive_mase", "seasonal_naive_mase", "log_scale", "log_target_sd",
             "lag1_autocorr", "spectral_centroid", "zero_fraction", "norm_slope"]
    return np.column_stack(cols), names


def _cross_fitted_residuals(features: np.ndarray, y: np.ndarray, n_folds: int,
                            alpha: float, seed: int) -> np.ndarray:
    """Residuals of `y` on `features`, each series predicted out of fold.

    Closed-form ridge on the normal equations, matching `l2_stitching`'s own
    solve rather than importing sklearn for a 3-line fit. `y` may be
    [n_series] or [n_series, k]; the same folds apply to every column so the
    per-step shapes stay comparable across horizon steps.
    """
    n = features.shape[0]
    y2 = y[:, None] if y.ndim == 1 else y
    rng = np.random.default_rng(seed)
    folds = rng.permutation(n) % n_folds
    x = np.concatenate([features, np.ones((n, 1), dtype=np.float64)], axis=1)
    resid = np.empty_like(y2, dtype=np.float64)
    eye = np.eye(x.shape[1])
    eye[-1, -1] = 0.0  # never penalize the intercept
    for f in range(n_folds):
        te = folds == f
        tr = ~te
        if tr.sum() < 2 or not te.any():
            resid[te] = y2[te]
            continue
        xt = x[tr]
        beta = np.linalg.solve(xt.T @ xt + alpha * eye, xt.T @ y2[tr])
        resid[te] = y2[te] - x[te] @ beta
    return resid[:, 0] if y.ndim == 1 else resid


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, 0.0 when either side is constant."""
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return 0.0 if d < 1e-12 else float((a * b).sum() / d)


def _nonfinite_rows_of(store, model: str) -> np.ndarray:
    """`[N]` bool over the store's rows: a non-finite stored point or quantile
    forecast (the same rule as the store's write-time `nonfinite_predictions`
    record)."""
    preds = store.load_predictions(model)
    point = np.asarray(preds["point"])
    return nonfinite_series_mask(point, preds.get("quantiles", point))


def _scaled_errors(store: ActivationStore, model: str, contexts: np.ndarray,
                   targets: np.ndarray, scale_mode: str,
                   row_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """[n_series, horizon] forecast error in each series' own MASE units.

    `row_mask` (over the store's full, unfiltered row order) subsets the
    loaded predictions to match `contexts`/`targets` when the caller has
    already excluded unreliable-scale rows from those two -- the store
    itself always returns predictions for every series in the run.
    """
    point = np.asarray(store.load_predictions(model)["point"], dtype=np.float64)
    if row_mask is not None:
        point = point[row_mask]
    scale = np.asarray(_mase_scale(contexts, scale_mode), dtype=np.float64)
    return (point - targets) / scale[:, None]


def _oof_r2(features: np.ndarray, y: np.ndarray, n_folds: int, alpha: float,
            seed: int) -> float:
    r = _cross_fitted_residuals(features, y, n_folds, alpha, seed)
    return 1.0 - float(np.sum(r ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12))


_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def error_fingerprint(store: ActivationStore, model_a: str, model_b: str,
                      contexts: np.ndarray, targets: np.ndarray,
                      *, scale_mode: str = "mean_abs_diff", n_folds: int = 5,
                      n_boot: int = 2000, seed: int = 0,
                      min_scale_frac: float = 0.05) -> dict:
    """Correlate two models' difficulty-adjusted forecast errors.

    Returns both the raw and the residualized correlations. Reporting the raw
    one is not padding: the gap between them is the size of the difficulty
    confound on this corpus, and a residualized value quoted alone gives a
    reader no way to see how much work the adjustment did.

    The ridge penalty is chosen per model by out-of-fold R² over
    `_ALPHA_GRID`, and that R² is reported. It is the number that says
    whether the adjustment happened at all: a residual correlation computed
    against a basis with R² ≤ 0 has removed nothing, and quoting it as
    "difficulty-adjusted" would be the silent degradation invariant 8
    forbids. `adjustment_ok` is False in that case and callers should treat
    the residual figures as uninterpretable rather than as evidence.

    **Unreliable-scale series are excluded before anything else is computed**
    (`ROADMAP.md` sec 6.3.1 Option E's first live run, found this the hard
    way). `_scaled_errors` divides by the same per-series MASE denominator
    L0 already uses, floored at 1e-8 -- a genuinely near-flat context (this
    module's own §23.3 D2 lineage-pair run hit it on 16 of 288 series at its
    short `context_len=64`) turns any nonzero forecast error into a value in
    the hundreds of thousands to millions once divided by that floor. Two
    lightly-diverged children of one parent checkpoint produce *nearly
    identical* absolute error on those same degenerate series, so the
    resulting scaled values come out bit-identical between the two models --
    a handful of astronomical, model-agnostic values that dominate both the
    raw correlation and, because ridge regression cannot move a
    multi-million-unit outlier's residual anywhere near zero, the
    "difficulty-adjusted" residual correlation too, defeating the entire
    point of the adjustment. L0 already solved exactly this for its own
    aggregate MASE (`mase_reliability`, sec 15 A11, `min_scale_frac`
    default 0.05) -- reused here rather than re-invented (`CLAUDE.md` §2.2),
    with the same default so the exclusion threshold matches what L0's own
    report already applies to this corpus. `n_excluded_unreliable` records
    how many series were dropped so this can never happen silently again.
    """
    scale_full = np.asarray(_mase_scale(contexts, scale_mode), dtype=np.float64)
    reliable = mase_reliability(contexts, targets, scale_mode, min_scale_frac=min_scale_frac)
    n_excluded = int((~reliable).sum())
    nonfinite_rows = np.flatnonzero(_nonfinite_rows_of(store, model_a)
                                    | _nonfinite_rows_of(store, model_b))
    n_nonfinite = int((reliable[nonfinite_rows]).sum())
    if n_nonfinite:
        log.warning("error fingerprint %s vs %s: excluding %d series with a non-finite "
                    "stored forecast (pred/<model> `nonfinite_predictions` record)",
                    model_a, model_b, n_nonfinite)
        reliable = reliable.copy()
        reliable[nonfinite_rows] = False
    if n_excluded:
        log.warning("error fingerprint %s vs %s: excluding %d of %d series with an "
                    "unreliable (near-zero) MASE scale before correlating -- see "
                    "ROADMAP.md sec 6.3.1 Option E's lineage-pair Findings",
                    model_a, model_b, n_excluded, len(contexts))
    contexts, targets, scale = contexts[reliable], targets[reliable], scale_full[reliable]
    ea = _scaled_errors(store, model_a, contexts, targets, scale_mode, row_mask=reliable)
    eb = _scaled_errors(store, model_b, contexts, targets, scale_mode, row_mask=reliable)
    n, horizon = ea.shape

    raw, names = difficulty_features(contexts, targets, scale)
    feats = (raw - raw.mean(0)) / (raw.std(0) + 1e-8)

    mag_a, mag_b = np.abs(ea).mean(1), np.abs(eb).mean(1)
    alpha_a = max(_ALPHA_GRID, key=lambda al: _oof_r2(feats, mag_a, n_folds, al, seed))
    alpha_b = max(_ALPHA_GRID, key=lambda al: _oof_r2(feats, mag_b, n_folds, al, seed))
    r2_a = _oof_r2(feats, mag_a, n_folds, alpha_a, seed)
    r2_b = _oof_r2(feats, mag_b, n_folds, alpha_b, seed)

    ra = _cross_fitted_residuals(feats, mag_a, n_folds, alpha_a, seed)
    rb = _cross_fitted_residuals(feats, mag_b, n_folds, alpha_b, seed)
    sa = _cross_fitted_residuals(feats, ea, n_folds, alpha_a, seed)
    sb = _cross_fitted_residuals(feats, eb, n_folds, alpha_b, seed)

    def _mag(idx):
        return _corr(ra[idx], rb[idx])

    def _shape(idx):
        return _corr(sa[idx].ravel(), sb[idx].ravel())

    all_idx = np.arange(n)
    ok = r2_a > 0.0 and r2_b > 0.0
    out = {
        "model_a": model_a, "model_b": model_b, "n_series": int(n),
        "n_excluded_unreliable": n_excluded,
        **({"n_excluded_nonfinite_forecast": n_nonfinite} if n_nonfinite else {}),
        "min_scale_frac": float(min_scale_frac),
        "horizon": int(horizon), "n_folds": int(n_folds), "scale_mode": scale_mode,
        "difficulty_basis": names,
        "difficulty_oof_r2": {model_a: r2_a, model_b: r2_b},
        "difficulty_alpha": {model_a: float(alpha_a), model_b: float(alpha_b)},
        "adjustment_ok": bool(ok),
        "magnitude": {
            "raw_corr": _corr(mag_a, mag_b),
            "residual_corr": _mag(all_idx),
            "residual_ci": bootstrap_ci(_mag, n, n_boot=n_boot, seed=seed),
        },
        "shape": {
            "raw_corr": _corr(ea.ravel(), eb.ravel()),
            "residual_corr": _shape(all_idx),
            "residual_ci": bootstrap_ci(_shape, n, n_boot=n_boot, seed=seed),
        },
        # Per-horizon-step, un-bootstrapped: the point of this row is the
        # *profile* (does agreement concentrate at h=1 and decay, or is it
        # flat?), which a CI per step would clutter without informing.
        "shape_by_step": [_corr(sa[:, h], sb[:, h]) for h in range(horizon)],
    }
    if not ok:
        log.warning("error fingerprint %s vs %s: the difficulty basis explains nothing "
                    "out of fold (R2 %.3f / %.3f) -- the residual correlations below are "
                    "NOT difficulty-adjusted and must not be read as evidence",
                    model_a, model_b, r2_a, r2_b)
    log.info("error fingerprint %s vs %s: magnitude raw=%.3f resid=%.3f, "
             "shape raw=%.3f resid=%.3f (n=%d, difficulty R2 %.3f/%.3f)",
             model_a, model_b, out["magnitude"]["raw_corr"],
             out["magnitude"]["residual_corr"], out["shape"]["raw_corr"],
             out["shape"]["residual_corr"], n, r2_a, r2_b)
    return out


def run_error_fingerprint(run_dir, *, models: Optional[list] = None,
                          n_boot: int = 2000, seed: int = 0) -> dict:
    """Fingerprint one already-completed run's model pair from its artifacts.

    Reads the run's own `config_resolved.yaml` and its store -- no checkpoint
    is loaded and no forward pass is run -- so it can be pointed at any run
    directory that got as far as L0, including runs built for an entirely
    different purpose. That is what makes the control set cheap: a
    same-family pair, several cross-family pairs and two architecture-matched
    `random_init` nulls already exist on disk from earlier work.

    The corpus is re-derived through the run's recorded config rather than
    read back from the store, because contexts are not persisted there.
    `data.source: smoke` regenerates deterministically from the same seed;
    a sealed corpus is re-read and re-verified.
    """
    from pathlib import Path

    from ..config import load_config
    from ..data import load_benchmark

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config_resolved.yaml")
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    names = models or [m.name for m in cfg.models if store.has_predictions(m.name)]
    if len(names) < 2:
        raise ValueError(f"{run_dir}: need two models with stored predictions, found {names}")
    if len(names) > 2:
        log.warning("error fingerprint: %d models have predictions (%s); fingerprinting "
                    "the first two -- pass --models to choose a different pair",
                    len(names), ", ".join(names))
        names = names[:2]
    data = load_benchmark(cfg.data, cfg.run.seed)
    out = error_fingerprint(store, names[0], names[1],
                            np.asarray(data.contexts(), dtype=np.float64),
                            np.asarray(data.targets(), dtype=np.float64),
                            scale_mode=cfg.l0.scale, n_boot=n_boot, seed=seed,
                            min_scale_frac=cfg.l0.min_scale_frac)
    out["run_dir"] = str(run_dir)
    return out
