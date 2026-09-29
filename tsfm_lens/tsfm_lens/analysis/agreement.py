"""Cross-model forecast agreement as a reliability signal (`ROADMAP.md` sec 20 H4).

Every other stage in this repo asks what the comparison *reveals* about the
models. This one asks what the comparison is *for*: if two models disagreeing
on a series predicts that both are about to be wrong on it, then "run two and
check whether they agree" is an actionable rule a practitioner can apply
without labels, at inference time, on data they have no ground truth for.

The mechanism is a pure reduction over forecasts `l0` already made and stored
-- no checkpoint is loaded and no forward pass is run -- so this costs seconds
against any run directory that reached L0.

**The load-bearing part is the baseline, not the correlation.** A positive
disagreement-predicts-error correlation is nearly guaranteed and nearly
uninformative on its own: hard series are hard, and on a hard series both
models are both wrong *and* likely to differ. The question a practitioner
actually faces is whether running a *second model* buys anything over reading
the *first model's own* quantile width, which is free and needs no second
checkpoint. So every correlation here is reported beside that baseline, and
the verdict is the difference between them with a CI (`H4`'s own acceptance
criterion). If self-reported uncertainty predicts error just as well,
cross-model agreement adds nothing, and this module says so in those words.

Two further disciplines carried from elsewhere in the repo:

- **The series is the resampling unit** (invariant 2). All CIs are cluster
  bootstraps over series, and the disagreement-vs-error relationship is
  summarized per family and per horizon step rather than pooled into one
  number that a family imbalance could drive.
- **Errors and widths are in MASE units**, divided by each series' own scale
  via the same `_mase_scale` L0 scores against. Without that, correlation
  between disagreement and error is dominated by amplitude: a large-amplitude
  series has both a large absolute error and a large absolute spread, in every
  model, for reasons that have nothing to do with reliability.

**What this cannot establish.** A correlation here is a property of *this
corpus and this pair*. It says nothing about whether the rule transfers to a
different data distribution, and a threshold read off the calibration curve is
fitted on the same data it is evaluated on unless a caller holds series out.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..extraction.store import ActivationStore
from ..utils import log
from .stats import _mase_scale, bootstrap_ci


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, reported beside Pearson because the relationship is not linear.

    Disagreement-vs-error is typically monotone with a long right tail (a few
    series where both models fail badly and differ wildly). Pearson on the raw
    values is then largely a statement about those few points; the rank
    version answers the question a practitioner asks -- "does a higher
    disagreement rank mean a higher error rank" -- without letting the tail
    set the number.
    """
    return _pearson(_ranks(x), _ranks(y))


def _ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with **tied values averaged**, which is not optional here.

    The obvious implementation (`argsort` + `arange`) breaks ties by position,
    so a perfectly constant column comes back as `0, 1, 2, ... n-1` in row
    order -- a clean ramp that correlates with whatever else happens to be
    ordered in the corpus. That is not a hypothetical: `GenericHFAdapter`
    exposes only a point head, so its quantile band has width exactly 0 for
    every series, and the naive ranking gave that degenerate baseline a
    Spearman of -0.192 against error, which then produced a confident
    "cross-model agreement BEATS own width" verdict out of a column with no
    information in it at all.
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, returning 0.0 rather than NaN when either side is constant.

    A constant side means the statistic is undefined, not zero -- but a NaN
    propagating into a bootstrap turns the whole CI into NaN and reports
    nothing at all. 0.0 is the value that says "no relationship detected",
    which is the honest reading of a degenerate input here, and the
    degenerate case is visible anyway because a constant column has zero
    spread in the artifact.
    """
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
    return float((xc * yc).sum() / denom) if denom > 0 else 0.0


def pointwise_disagreement(point_a: np.ndarray, point_b: np.ndarray,
                           scale: np.ndarray) -> np.ndarray:
    """Per-series mean absolute gap between two point forecasts, in MASE units."""
    return np.abs(point_a - point_b).mean(axis=1) / scale


def distributional_disagreement(quants_a: np.ndarray, quants_b: np.ndarray,
                                scale: np.ndarray) -> np.ndarray:
    """Per-series mean absolute gap between two quantile *curves*, in MASE units.

    A distinct signal from the pointwise one: two models can agree exactly on
    the median and disagree completely about how uncertain they are. Averaging
    |q_a - q_b| over the shared quantile levels is the discrete form of the
    Wasserstein-1 distance between the two implied distributions, which is why
    it stays in the same units as the pointwise gap and can be compared to it
    directly.
    """
    return np.abs(quants_a - quants_b).mean(axis=(1, 2)) / scale


def quantile_width(quants: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """One model's OWN uncertainty, in MASE units -- the baseline H4 must beat.

    Widest-minus-narrowest quantile at each horizon step, averaged. This is
    the cheap thing a practitioner already has: it needs one model, no second
    checkpoint, and no comparison machinery at all.
    """
    return (quants.max(axis=2) - quants.min(axis=2)).mean(axis=1) / scale


def _mase(point: np.ndarray, targets: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.abs(targets - point).mean(axis=1) / scale


def calibration_curve(signal: np.ndarray, error: np.ndarray, n_bins: int = 10) -> dict:
    """Mean error per signal decile -- what a user reads a threshold off.

    Deliberately equal-*count* bins rather than equal-width: disagreement is
    heavy-tailed, so equal-width bins put almost every series in the first bin
    and a handful in the rest, and the resulting curve describes the tail
    rather than the corpus.
    """
    n = len(signal)
    n_bins = int(min(n_bins, max(2, n // 5)))
    order = np.argsort(signal, kind="mergesort")
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    rows = []
    for i in range(n_bins):
        idx = order[edges[i]:edges[i + 1]]
        if len(idx) == 0:
            continue
        rows.append({"bin": i, "n": int(len(idx)),
                     "signal_lo": float(signal[idx].min()),
                     "signal_hi": float(signal[idx].max()),
                     "mean_signal": float(signal[idx].mean()),
                     "mean_error": float(error[idx].mean())})
    return {"n_bins": len(rows), "bins": rows}


def _corr_block(signal: np.ndarray, error: np.ndarray, *, n_boot: int, seed: int) -> dict:
    def pear(rows):
        return _pearson(signal[rows], error[rows])

    def spear(rows):
        return _spearman(signal[rows], error[rows])

    return {"pearson": bootstrap_ci(pear, len(signal), n_boot=n_boot, seed=seed),
            "spearman": bootstrap_ci(spear, len(signal), n_boot=n_boot, seed=seed + 1)}


def agreement_reliability(store: ActivationStore, model_a: str, model_b: str,
                          contexts: np.ndarray, targets: np.ndarray, *,
                          families: Optional[np.ndarray] = None,
                          scale_mode: str = "mean_abs_diff",
                          n_bins: int = 10, n_boot: int = 2000,
                          seed: int = 0) -> dict:
    """Does cross-model disagreement predict error, and does it beat self-reported width?

    Returns the two disagreement signals, each model's own quantile-width
    baseline, the paired difference between them with a CI, per-family and
    per-horizon-step breakdowns, and a calibration curve.

    The comparison target is **each model's own error**, not the pair's mean
    error, and both are reported. Predicting the mean is the easier task and
    the less useful one: a practitioner running model A wants to know whether
    *A* is about to be wrong.
    """
    pa, pb = store.load_predictions(model_a), store.load_predictions(model_b)
    scale = _mase_scale(contexts, scale_mode)
    err = {model_a: _mase(pa["point"], targets, scale),
           model_b: _mase(pb["point"], targets, scale)}
    err_mean = 0.5 * (err[model_a] + err[model_b])

    signals = {
        "disagreement_point": pointwise_disagreement(pa["point"], pb["point"], scale),
        "disagreement_quantile": distributional_disagreement(pa["quantiles"],
                                                             pb["quantiles"], scale),
        f"own_width_{model_a}": quantile_width(pa["quantiles"], scale),
        f"own_width_{model_b}": quantile_width(pb["quantiles"], scale),
    }

    targets_named = {model_a: err[model_a], model_b: err[model_b], "mean": err_mean}
    corr = {sname: {tname: _corr_block(sig, tgt, n_boot=n_boot, seed=seed)
                    for tname, tgt in targets_named.items()}
            for sname, sig in signals.items()}

    # H4's acceptance criterion: the verdict is this difference, not the raw
    # correlation. Paired on the same bootstrap resample of series, so the two
    # correlations move together and the CI is on their gap rather than on two
    # independently-noisy numbers a reader would have to eyeball.
    verdict = {}
    for m in (model_a, model_b):
        dis, own = signals["disagreement_point"], signals[f"own_width_{m}"]
        tgt = err[m]
        # A model with no quantile head (or one whose adapter fills the band
        # with copies of the point forecast, as `GenericHFAdapter` does) has
        # no baseline to beat. Comparing against a constant column would
        # hand cross-model agreement a free win against nothing, which is
        # the opposite of what H4's acceptance criterion is for -- so the
        # comparison is withheld and the reason recorded.
        if float(own.std()) == 0.0:
            verdict[m] = {
                "own_width_available": False,
                "reason": (f"'{m}' reports a quantile band of width 0 for every series "
                           f"(no quantile head, or an adapter that fills the band with "
                           f"the point forecast), so there is no self-reported "
                           f"uncertainty baseline to compare against"),
                "spearman_disagreement": _corr_block(dis, tgt, n_boot=n_boot,
                                                     seed=seed)["spearman"],
                "beats_own_width": False, "adds_nothing": False,
            }
            continue

        def gap(rows, dis=dis, own=own, tgt=tgt):
            return _spearman(dis[rows], tgt[rows]) - _spearman(own[rows], tgt[rows])

        ci = bootstrap_ci(gap, len(tgt), n_boot=n_boot, seed=seed + 2)
        verdict[m] = {
            "own_width_available": True,
            "spearman_gap_vs_own_width": ci,
            "beats_own_width": bool(ci["lo"] > 0.0),
            "adds_nothing": bool(ci["hi"] <= 0.0),
        }

    by_family = {}
    if families is not None:
        fam = np.asarray(families)
        for f in sorted(set(fam.tolist())):
            rows = np.flatnonzero(fam == f)
            if len(rows) < 8:
                # Below this a series bootstrap reports a CI wider than the
                # range of the statistic; omitting is more honest than
                # rendering a number nothing supports.
                continue
            by_family[str(f)] = {
                "n": int(len(rows)),
                "disagreement_point": _corr_block(signals["disagreement_point"][rows],
                                                  err_mean[rows], n_boot=n_boot, seed=seed),
                f"own_width_{model_a}": _corr_block(signals[f"own_width_{model_a}"][rows],
                                                    err[model_a][rows],
                                                    n_boot=n_boot, seed=seed)}

    h = targets.shape[1]
    step_dis = np.abs(pa["point"] - pb["point"]) / scale[:, None]
    step_err = 0.5 * (np.abs(targets - pa["point"]) + np.abs(targets - pb["point"])) / scale[:, None]
    by_step = [_spearman(step_dis[:, t], step_err[:, t]) for t in range(h)]

    out = {
        "model_a": model_a, "model_b": model_b,
        "n_series": int(len(targets)), "horizon": int(h),
        "scale_mode": scale_mode,
        "signal_means": {k: float(v.mean()) for k, v in signals.items()},
        "signal_sd": {k: float(v.std()) for k, v in signals.items()},
        "mase": {m: float(v.mean()) for m, v in err.items()},
        "correlations": corr,
        "verdict": verdict,
        "by_family": by_family,
        "spearman_by_horizon_step": [float(v) for v in by_step],
        "calibration": {
            "disagreement_point": calibration_curve(signals["disagreement_point"],
                                                    err_mean, n_bins),
            f"own_width_{model_a}": calibration_curve(signals[f"own_width_{model_a}"],
                                                      err[model_a], n_bins),
        },
    }
    for m, v in verdict.items():
        if not v.get("own_width_available", True):
            log.warning("agreement '%s': no own-width baseline -- %s", m, v["reason"])
            continue
        log.info("agreement '%s': disagreement-vs-own-width Spearman gap %.3f "
                 "[%.3f, %.3f] -- %s",
                 m, v["spearman_gap_vs_own_width"]["value"],
                 v["spearman_gap_vs_own_width"]["lo"], v["spearman_gap_vs_own_width"]["hi"],
                 "beats own width" if v["beats_own_width"]
                 else ("adds nothing over own width" if v["adds_nothing"] else "inconclusive"))
    return out


def run_agreement(run_dir, *, models: Optional[list] = None, n_boot: int = 2000,
                  n_bins: int = 10, seed: int = 0) -> dict:
    """Read one completed run's stored forecasts and score the agreement heuristic.

    Same read-only contract as `analysis/error_fingerprint.py`: the run's own
    `config_resolved.yaml` and store are the whole input, contexts are
    re-derived through the recorded data config (they are not persisted), and
    nothing is loaded or re-run.
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
        log.warning("agreement: %d models have predictions (%s); using the first two -- "
                    "pass --models to choose a different pair", len(names), ", ".join(names))
        names = names[:2]
    data = load_benchmark(cfg.data, cfg.run.seed)
    out = agreement_reliability(store, names[0], names[1],
                                np.asarray(data.contexts(), dtype=np.float64),
                                np.asarray(data.targets(), dtype=np.float64),
                                families=data.meta["family"].to_numpy(),
                                scale_mode=cfg.l0.scale, n_bins=n_bins,
                                n_boot=n_boot, seed=seed)
    out["run_dir"] = str(run_dir)
    return out
