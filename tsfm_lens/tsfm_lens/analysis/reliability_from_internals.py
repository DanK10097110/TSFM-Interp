"""Does looking inside a model tell you when to distrust its forecast? (ROADMAP §38.4, K4)

Practitioner question behind U1 and U2. `analysis/agreement.py` (§20 H4)
found that cross-model disagreement predicts error but loses to each model's
own quantile width, a free output-only signal. That sets the bar here: an
internal feature is only worth reporting for what it adds *beyond* a free
baseline, the same gain-over-baseline doctrine L2 follows (invariant 3).

U1, failure prediction. Per (model, series) two targets: log MASE, and a
binary failure (the model's MASE exceeds the seasonal-naive MASE on that
series). Baseline features cost nothing at inference: the model's own mean
relative quantile width (`agreement.quantile_width`), the forecast's
flatness (forecast sd over context sd, the PM-07 collapse signal) and the
catch22 features of the z-scored context. Internal features come in three
groups, each degraded loudly when its input is missing:

* `sae_families`: SAE activations at the last context window, summed over each
  SAE target layer and over the members of each concept family in
  `sae/concept_families.json` (rows carry model, layer, feature, family).
  When that artifact is absent or not `measured`, the fallback is the
  `top_n` most active features per target layer (label-free selection),
  recorded as `sae_top_n`.
* `lens_depth`: per-series lens convergence depth from
  `lens/convergence.npz` (`analysis/lens.py`, label-free by construction),
  plus a never-converged indicator. Needs full coverage of the analysed
  series; otherwise skipped with the coverage stated.
* `crystallization_norm`: log norm of the last-window residual at the layer
  nearest the model's mean-curve crystallization depth (`lens/lens.json`);
  skipped when the model never crystallizes.

Method: ridge (log MASE) and logistic (failure), cross-fitted over series
folds stratified by generator family, refit `n_repeats` times with different
fold assignments and averaged. Every prediction scored is out of fold. The
score is Spearman (log MASE) or AUROC (failure) of baseline+internals minus
baseline alone; the CI is a series cluster bootstrap over the out-of-fold
predictions with those predictions held fixed (like L2's validation-series
CI). Per model, never pooled across models. Per family is a secondary.

U2, per-series routing. The model with the lowest cross-fitted predicted log
MASE is chosen for each series, using baseline features only or
baseline+internals. Realized MASE is compared with the best single model and
the oracle. The best single model is picked on the same rows it is scored
on, which favors that comparator.

Evidence class: predictive (behavioral). A feature that predicts failure is
not a cause of failure; nothing here is causal.

Float16 store reads are cast to float64 before any ranking: scipy's
`rankdata` keeps a float16 input's dtype and quantizes rank sums.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import rankdata

from ..utils import load_json, log
from .agreement import quantile_width
from .stats import bootstrap_ci, bootstrap_ci_diff, mean_ci, paired_bootstrap

EVIDENCE_CLASS = "predictive (behavioral); not causal"
NO_SKILL_SEASONAL = "__seasonal_naive__"
ALPHAS = tuple(np.logspace(-2, 3, 11))


def _seed(*parts, base: int = 0) -> int:
    """Stable sha256 seed (never Python's salted `hash()`)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return (base + int.from_bytes(h[:4], "big")) % (2 ** 32)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Tie-averaged Spearman rho, ranking in float64.

    The cast is the point: scipy's `rankdata` keeps a float16 input's dtype,
    and a float16 rank sum or product is quantized. A constant side returns
    0.0 rather than NaN so a bootstrap cannot turn to NaN on a degenerate
    resample.
    """
    rx = rankdata(np.asarray(x, dtype=np.float64))
    ry = rankdata(np.asarray(y, dtype=np.float64))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUROC of `score` for the positive class, ranking in float64.

    Returns 0.5 when one class is empty in the sample (undefined, reported as
    no signal, so a resample that lacks a class cannot poison a CI).
    """
    lab = np.asarray(label).astype(bool)
    k = int(lab.sum())
    n = lab.size
    if k == 0 or k == n:
        return 0.5
    ranks = rankdata(np.asarray(score, dtype=np.float64))
    return float((ranks[lab].sum() - k * (k + 1) / 2.0) / (k * (n - k)))


def stratified_folds(strata: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold id per series, balanced within every stratum (never a head slice).

    Each stratum's members are permuted and dealt round-robin across folds,
    continuing the deal where the previous stratum stopped, so fold sizes
    stay within one of each other and every family appears in every fold
    whenever it has at least `n_folds` members.
    """
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(strata), dtype=np.int64)
    offset = 0
    for label in np.unique(strata):
        members = np.flatnonzero(strata == label)
        members = members[rng.permutation(len(members))]
        folds[members] = (offset + np.arange(len(members))) % n_folds
        offset += len(members)
    return folds


def _make_ridge(seed: int, alphas=ALPHAS):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=tuple(alphas)))


def _make_logistic(seed: int):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegressionCV(Cs=8, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                             max_iter=2000, scoring="neg_log_loss"))


def cross_fitted_predictions(X: np.ndarray, y: np.ndarray, strata: np.ndarray, kind: str,
                             n_folds: int = 5, n_repeats: int = 3, seed: int = 0,
                             alphas=ALPHAS) -> np.ndarray:
    """Out-of-fold prediction for every series; no series is scored by a fit that saw it.

    `kind` is `"ridge"` (predicts `y`) or `"logistic"` (predicts P(y = 1)).
    Fold assignment is stratified and is redrawn `n_repeats` times; each
    series' prediction is the mean of its out-of-fold predictions across
    repeats, which reduces fold-assignment noise without letting any fit
    see the series it scores. Imputation, standardization and the ridge
    penalty or logistic C are all chosen inside the training fold. `alphas`
    is the ridge grid (default `ALPHAS`); `confirm` passes the grid frozen at
    registration.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    out = np.zeros(n, dtype=np.float64)
    for rep in range(n_repeats):
        folds = stratified_folds(strata, n_folds, _seed("folds", rep, base=seed))
        for f in range(n_folds):
            te, tr = np.flatnonzero(folds == f), np.flatnonzero(folds != f)
            if len(te) == 0:
                continue
            if kind == "ridge":
                model = _make_ridge(seed, alphas).fit(X[tr], y[tr])
                out[te] += model.predict(X[te])
            else:
                model = _make_logistic(_seed("inner", rep, f, base=seed)).fit(X[tr], y[tr].astype(int))
                out[te] += model.predict_proba(X[te])[:, 1]
    return out / n_repeats


def _score_with_ci(fn, pred: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict:
    return bootstrap_ci(lambda idx: fn(pred[idx], y[idx]), len(y), n_boot=n_boot, seed=seed)


def _gain_with_ci(fn, pred_base: np.ndarray, pred_full: np.ndarray, y: np.ndarray,
                  n_boot: int, seed: int, with_p: bool = False) -> dict:
    out = bootstrap_ci_diff(lambda idx: fn(pred_full[idx], y[idx]),
                            lambda idx: fn(pred_base[idx], y[idx]),
                            len(y), n_boot=n_boot, seed=seed, paired=True)
    rec = {"baseline_plus_internals": out["a"], "baseline": out["b"],
           "gain": out["diff"], "lo": out["diff_lo"], "hi": out["diff_hi"],
           "ci_excludes_zero_positive": bool(out["diff_lo"] > 0), "n_boot": n_boot,
           "resample_unit": out["resample_unit"]}
    if with_p:
        rec["p"] = out["p"]
    return rec


def _stack(blocks: list) -> np.ndarray:
    return np.concatenate([np.asarray(b, dtype=np.float64) for b in blocks], axis=1) \
        if blocks else np.zeros((0, 0))


def analyze_model(log_mase: np.ndarray, failure: np.ndarray, baseline: np.ndarray,
                  groups: dict, skipped: dict, strata: np.ndarray, *, n_folds: int = 5,
                  n_repeats: int = 3, n_boot: int = 1000, seed: int = 0,
                  min_family_n: int = 30, min_family_class: int = 5) -> dict:
    """U1 for one model. Pure: arrays in, record out; returns predictions for U2.

    `baseline` is `[n, p]`; `groups` maps a group name to an `[n, q]` array;
    `failure` is a 0/1 float array with NaN where the seasonal-naive
    reference is unavailable (those series drop out of the failure task
    only). `skipped` maps a skipped group to its stated reason and is copied
    into the record. Besides baseline vs baseline+all groups, every group is
    scored alone on top of the baseline, which names the group carrying any
    gain and is where a decoy or pure-noise group is expected to score zero.
    """
    n = len(log_mase)
    fail_ok = np.isfinite(failure)
    sets = {"baseline": baseline}
    if groups:
        sets["baseline+internals"] = np.concatenate(
            [baseline] + [np.asarray(g, dtype=np.float64) for g in groups.values()], axis=1)
        if len(groups) > 1:
            for name, g in groups.items():
                sets[f"baseline+{name}"] = np.concatenate(
                    [baseline, np.asarray(g, dtype=np.float64)], axis=1)
    preds = {"ridge": {}, "logistic": {}}
    for key, X in sets.items():
        preds["ridge"][key] = cross_fitted_predictions(
            X, log_mase, strata, "ridge", n_folds, n_repeats, _seed("ridge", key, base=seed))
        preds["logistic"][key] = np.full(n, np.nan)
        if fail_ok.sum() > 2 * n_folds and 0 < failure[fail_ok].sum() < fail_ok.sum():
            preds["logistic"][key][fail_ok] = cross_fitted_predictions(
                X[fail_ok], failure[fail_ok], strata[fail_ok], "logistic", n_folds, n_repeats,
                _seed("logistic", key, base=seed))
    record = {"n_series": int(n), "n_failure_scorable": int(fail_ok.sum()),
              "failure_rate": float(np.nanmean(failure)) if fail_ok.any() else None,
              "baseline_features": int(baseline.shape[1]),
              "internal_groups": {k: int(np.asarray(v).shape[1]) for k, v in groups.items()},
              "internal_groups_skipped": dict(skipped),
              "u1": {}, "by_group": {}, "by_family": {}}
    if not groups:
        record["u1"]["note"] = ("no internal feature group was available for this model; "
                                "there is no gain to report (see internal_groups_skipped)")
        return {"record": record, "predictions": preds}

    tasks = {"log_mase_spearman": ("ridge", spearman, log_mase, np.ones(n, bool)),
             "failure_auroc": ("logistic", auroc, failure, fail_ok)}
    for task, (kind, fn, y, mask) in tasks.items():
        if not mask.any() or np.isnan(preds[kind]["baseline"][mask]).any():
            record["u1"][task] = {"scorable": False,
                                  "reason": "no failure task: one class only or too few series"}
            continue
        pb, pf = preds[kind]["baseline"][mask], preds[kind]["baseline+internals"][mask]
        yy = y[mask]
        record["u1"][task] = {"scorable": True, "n": int(mask.sum()),
                              "baseline": _score_with_ci(fn, pb, yy, n_boot, _seed(task, "b", base=seed)),
                              "baseline_plus_internals": _score_with_ci(
                                  fn, pf, yy, n_boot, _seed(task, "f", base=seed)),
                              "gain": _gain_with_ci(fn, pb, pf, yy, n_boot, _seed(task, "g", base=seed))}
        for name in groups:
            key = f"baseline+{name}"
            if key not in preds[kind]:
                record["by_group"].setdefault(name, {})[task] = record["u1"][task]["gain"]
                continue
            record["by_group"].setdefault(name, {})[task] = _gain_with_ci(
                fn, pb, preds[kind][key][mask], yy, n_boot, _seed(task, name, base=seed))
        fam = strata[mask]
        for label in np.unique(fam):
            sel = np.flatnonzero(fam == label)
            entry = record["by_family"].setdefault(str(label), {})
            n_pos = int(yy[sel].sum()) if kind == "logistic" else None
            if len(sel) < min_family_n or (
                    kind == "logistic" and (n_pos < min_family_class
                                            or len(sel) - n_pos < min_family_class)):
                entry[task] = {"scorable": False, "n": int(len(sel)),
                               "reason": f"fewer than {min_family_n} series"
                               if len(sel) < min_family_n else
                               f"fewer than {min_family_class} series in one class"}
                continue
            entry[task] = {"scorable": True, "n": int(len(sel)),
                           **_gain_with_ci(fn, pb[sel], pf[sel], yy[sel], n_boot,
                                           _seed(task, "fam", label, base=seed))}
    return {"record": record, "predictions": preds}


U1_TASK_LOG_MASE = "log_mase_spearman"


def u1_gain(log_mase: np.ndarray, baseline: np.ndarray, internals: np.ndarray,
            strata: np.ndarray, *, n_folds: int, n_repeats: int, n_boot: int, model_seed: int,
            alphas=ALPHAS) -> dict:
    """The U1 log-MASE gain of one model, refit from arrays: exactly the
    `analyze_model` procedure for the `log_mase_spearman` task, with the same
    per-key seeds, so a refit on the dev arrays reproduces the dev record.

    `internals` is `[n, q]` (every internal group concatenated in dev order),
    `model_seed` is the per-model seed `run_reliability` derived
    (`_seed("model", name, base=seed)`) and `alphas` the ridge grid. Returns
    the `_gain_with_ci` record plus the two-sided bootstrap `p` (floored at
    `1/n_boot`). Used by `confirm` to refit the frozen procedure on private
    series.
    """
    pb = cross_fitted_predictions(baseline, log_mase, strata, "ridge", n_folds, n_repeats,
                                  _seed("ridge", "baseline", base=model_seed), alphas)
    X = np.concatenate([np.asarray(baseline, dtype=np.float64),
                        np.asarray(internals, dtype=np.float64)], axis=1)
    pf = cross_fitted_predictions(X, log_mase, strata, "ridge", n_folds, n_repeats,
                                  _seed("ridge", "baseline+internals", base=model_seed), alphas)
    return _gain_with_ci(spearman, pb, pf, log_mase, n_boot,
                         _seed(U1_TASK_LOG_MASE, "g", base=model_seed), with_p=True)


def model_seed_for(name: str, seed: int) -> int:
    """The per-model seed `run_reliability` hands `analyze_model`."""
    return _seed("model", name, base=seed)


def baseline_feature_names() -> list:
    """Names of the baseline columns in order, from the same definitions
    `baseline_features` uses (two output-only signals, then catch22)."""
    _, names = context_catch22(np.sin(np.linspace(0.0, 12.0, 64))[None, :])
    return ["log1p_own_width", "log10_forecast_flatness"] + [f"catch22_{n}" for n in names]


def family_columns(last_by_layer: dict, family_rows: list, model: str,
                   n: int) -> tuple:
    """`(X [n, n_families], family_ids)` from in-memory last-window SAE
    activations `{layer: [n, dict_size]}`: the same per-family sum
    `family_activation_features` takes from the store, for a caller (confirm)
    that encodes private activations itself. Only layers present in
    `last_by_layer` contribute; families with no member there give no column."""
    total = {}
    for r in family_rows:
        if r["model"] == model and r["layer"] in last_by_layer:
            col = total.setdefault(int(r["family"]), np.zeros(n, dtype=np.float64))
            col += np.asarray(last_by_layer[r["layer"]], dtype=np.float64)[:, int(r["feature"])]
    fam_ids = sorted(total)
    if not fam_ids:
        return np.zeros((0, 0)), []
    return np.column_stack([total[f] for f in fam_ids]), fam_ids


def crystallization_norm_feature(last_window_act: np.ndarray) -> np.ndarray:
    """`[n, 1]` log norm of last-window residual states, as `load_run_inputs` builds it."""
    last = np.asarray(last_window_act, dtype=np.float64)
    return np.log(np.linalg.norm(last, axis=1) + 1e-12)[:, None]


def route_policies(mase: dict, pred_base: dict, pred_full: dict, n_boot: int = 1000,
                   seed: int = 0) -> dict:
    """U2: realized MASE of four routing policies on the rows every model shares.

    `mase`, `pred_base`, `pred_full` map model to `[n]` arrays: realized
    MASE, and cross-fitted predicted log MASE from baseline-only and
    baseline+internals features. Routing picks each series' argmin. The
    best single model is the one with the lowest mean MASE on these same
    rows (favoring that comparator). `gap_baseline_minus_internals` is the
    value of interpretability: positive means internals routing had the
    lower realized MASE. CIs resample series.
    """
    names = sorted(mase)
    M = np.stack([np.asarray(mase[m], dtype=np.float64) for m in names])
    n = M.shape[1]

    def realized(pred: dict) -> tuple:
        P = np.stack([np.asarray(pred[m], dtype=np.float64) for m in names])
        pick = P.argmin(axis=0)
        return M[pick, np.arange(n)], pick

    base_r, base_pick = realized(pred_base)
    full_r, full_pick = realized(pred_full)
    best_i = int(M.mean(axis=1).argmin())
    policies = {"baseline_routing": base_r, "baseline_plus_internals_routing": full_r,
                "best_single_model": M[best_i], "oracle": M.min(axis=0)}
    out = {"models": names, "n_series": int(n), "best_single_model": names[best_i],
           "policies": {k: mean_ci(v, n_boot=n_boot, seed=_seed("u2", k, base=seed))
                        for k, v in policies.items()},
           "routing_share": {
               "baseline_routing": {m: float((base_pick == i).mean()) for i, m in enumerate(names)},
               "baseline_plus_internals_routing": {
                   m: float((full_pick == i).mean()) for i, m in enumerate(names)}},
           "per_model_mean_mase": {m: float(M[i].mean()) for i, m in enumerate(names)}}
    out["gap_baseline_minus_internals"] = paired_bootstrap(
        base_r - full_r, n_boot=n_boot, seed=_seed("u2gap", base=seed))
    out["gap_best_single_minus_internals"] = paired_bootstrap(
        policies["best_single_model"] - full_r, n_boot=n_boot, seed=_seed("u2gap2", base=seed))
    return out


def context_catch22(contexts: np.ndarray) -> tuple:
    """`(features [n, 22], names)`: catch22 of each z-scored context, NaN kept as NaN.

    catch22 is defined on z-scored series; a constant context (sd 0) yields
    all-NaN features, which the pipeline's median imputer handles inside each
    training fold.
    """
    import pycatch22
    rows, names = [], None
    for c in np.asarray(contexts, dtype=np.float64):
        sd = c.std()
        z = (c - c.mean()) / sd if sd > 0 else np.zeros_like(c)
        res = pycatch22.catch22_all(z.tolist())
        names = res["names"]
        rows.append(res["values"])
    arr = np.asarray(rows, dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    return arr, list(names)


def baseline_features(point: np.ndarray, quants: np.ndarray, contexts: np.ndarray,
                      scale: np.ndarray, catch22: np.ndarray, catch22_names: list) -> tuple:
    """Free, output-only baseline: `(X, names, notes)`.

    `log1p` of the mean relative quantile width (`agreement.quantile_width`,
    MASE units), `log10(forecast sd / context sd + 1e-3)` (flatness; below
    0.1 is PM-07's "flat") and the catch22 columns. A model with a zero-width
    band contributes a constant column; that is stated in `notes`, not
    dropped silently.
    """
    width = quantile_width(np.asarray(quants, dtype=np.float64), scale)
    ctx_sd = np.asarray(contexts, dtype=np.float64).std(axis=1)
    ratio = np.asarray(point, dtype=np.float64).std(axis=1) / np.maximum(ctx_sd, 1e-12)
    X = np.column_stack([np.log1p(np.maximum(width, 0.0)), np.log10(ratio + 1e-3), catch22])
    notes = {}
    if not np.any(width > 0):
        notes["own_width"] = "quantile band has width 0 for every series; column is constant"
    return X, ["log1p_own_width", "log10_forecast_flatness"] + [f"catch22_{n}" for n in catch22_names], notes


def _last_window(store, model: str, layer: str, start: int, stop: int, space: str) -> np.ndarray:
    """Last-context-window states of rows `[start, stop)` as float64 `[rows, dim]`."""
    block = store.load(model, layer, level="window", rows=np.arange(start, stop), space=space)
    return np.asarray(block[:, -1, :], dtype=np.float64)


def family_activation_features(store, model: str, layers: list, family_rows: list,
                               chunk: int = 128) -> tuple:
    """`(X [n, n_families], family_ids)`: last-window SAE activation summed per family.

    `family_rows` are dicts with model, layer, feature, family. For each
    target layer that has persisted SAE features, the last context window's
    activation is summed over the family's member features and the sums are
    added across layers. Families with no members for this model give no
    column. Reads are chunked over series and cast to float64.
    """
    per_layer = {}
    for r in family_rows:
        if r["model"] == model and r["layer"] in layers:
            per_layer.setdefault(r["layer"], {}).setdefault(int(r["family"]), []).append(int(r["feature"]))
    n = None
    total = {}
    for layer, fams in per_layer.items():
        if not store.has_sae_features(model, layer):
            continue
        n = store.load(model, layer, level="series", space="sae").shape[0]
        for s in range(0, n, chunk):
            block = _last_window(store, model, layer, s, min(s + chunk, n), "sae")
            for fam, idx in fams.items():
                col = total.setdefault(fam, np.zeros(n, dtype=np.float64))
                col[s:s + chunk] += block[:, idx].sum(axis=1)
    fam_ids = sorted(total)
    if not fam_ids:
        return np.zeros((0, 0)), []
    return np.column_stack([total[f] for f in fam_ids]), fam_ids


def top_n_activation_features(store, model: str, layers: list, top_n: int,
                              chunk: int = 128) -> tuple:
    """Fallback when no family artifact exists: last-window activation of the
    `top_n` most active SAE features per target layer, chosen by mean
    last-window activation over the series (label-free selection)."""
    cols, names = [], []
    for layer in layers:
        if not store.has_sae_features(model, layer):
            continue
        n = store.load(model, layer, level="series", space="sae").shape[0]
        last = np.concatenate([_last_window(store, model, layer, s, min(s + chunk, n), "sae")
                               for s in range(0, n, chunk)])
        pick = np.argsort(-last.mean(axis=0), kind="mergesort")[:top_n]
        cols.append(last[:, pick])
        names += [f"{layer}#{int(p)}" for p in pick]
    return (np.concatenate(cols, axis=1) if cols else np.zeros((0, 0))), names


def _family_rows(run_dir: Path) -> Optional[list]:
    path = Path(run_dir) / "sae" / "concept_families.json"
    if not path.exists():
        return None
    art = load_json(path)
    if not art.get("measured", False) or not art.get("rows"):
        return None
    return [r for r in art["rows"] if r.get("family") is not None]


def load_run_inputs(run_dir, data_path: Optional[str] = None, top_n: int = 16) -> dict:
    """Read one finished run into the arrays `run_reliability` reduces. Read-only.

    Contexts are not persisted, so they are re-derived through the run's own
    data config (`data_path` overrides the recorded path, which is relative
    to the working directory of the original run) and their series ids are
    checked against the run's `meta.parquet`: a mismatch fails loudly rather
    than mis-align a feature with a target. The store is opened `mode="r"`.
    """
    import pandas as pd

    from ..config import load_config
    from ..data import load_benchmark
    from ..extraction.store import ActivationStore, load_meta
    from .stats import _mase_scale

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config_resolved.yaml")
    if data_path:
        cfg.data.path = str(data_path)
    data = load_benchmark(cfg.data, cfg.run.seed)
    meta = load_meta(run_dir)
    ids = meta["series_id"].astype(str).to_numpy()
    if not np.array_equal(ids, data.meta["series_id"].astype(str).to_numpy()):
        raise ValueError(f"{run_dir}: re-derived corpus series ids differ from the run's "
                         f"meta.parquet; pass --data-path pointing at the run's corpus")
    contexts = np.asarray(data.contexts(), dtype=np.float64)
    scale = _mase_scale(contexts, cfg.l0.scale)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    metrics = pd.read_parquet(run_dir / "l0" / "metrics.parquet")
    metrics["series_id"] = metrics["series_id"].astype(str)
    idx = {s: i for i, s in enumerate(ids)}

    def column(model: str, col: str) -> np.ndarray:
        sub = metrics[metrics["model"] == model]
        out = np.full(len(ids), np.nan, dtype=np.float64)
        out[[idx[s] for s in sub["series_id"]]] = pd.to_numeric(sub[col], errors="coerce").to_numpy()
        return out

    seasonal = column(NO_SKILL_SEASONAL, "mase")
    reliable = column(next(m.name for m in cfg.models), "mase_reliable")
    catch22, catch22_names = context_catch22(contexts)
    fam_rows = _family_rows(run_dir)
    lens_meta = load_json(run_dir / "lens" / "lens.json") if (run_dir / "lens" / "lens.json").exists() else {}
    conv_meta = (load_json(run_dir / "lens" / "convergence.json")
                 if (run_dir / "lens" / "convergence.json").exists() else None)
    conv = np.load(run_dir / "lens" / "convergence.npz") if conv_meta else None

    models = {}
    for mcfg in cfg.models:
        name = mcfg.name
        pred = store.load_predictions(name)
        base, base_names, notes = baseline_features(
            pred["point"], pred["quantiles"], contexts, scale, catch22, catch22_names)
        groups, skipped = {}, {}
        layers = store.layers(name)
        feats_layers = [l for l in layers if store.has_sae_features(name, l)]
        if not feats_layers:
            skipped["sae_families"] = "no persisted SAE features (sae.persist_features was off)"
        else:
            if fam_rows:
                X, fam_ids = family_activation_features(store, name, feats_layers, fam_rows)
                choice = "concept_families"
            else:
                X, fam_ids = top_n_activation_features(store, name, feats_layers, top_n)
                choice = "top_n"
            if X.shape[1] == 0:
                skipped["sae_families"] = ("concept_families.json has no member features in this "
                                           "model's persisted SAE layers")
            else:
                groups["sae_families" if choice == "concept_families" else "sae_top_n"] = X
                notes["sae_choice"] = {"kind": choice, "columns": [str(f) for f in fam_ids]}
        rec = (conv_meta or {}).get(name)
        if conv is None:
            skipped["lens_depth"] = "lens/convergence.json absent (lens stage predates K4; rerun it)"
        elif not rec or not rec.get("available"):
            skipped["lens_depth"] = (rec or {}).get("reason", "no convergence record for this model")
        else:
            depth = np.full(len(ids), np.nan)
            conv_ids = conv[f"series_id_{name}"].astype(str)
            depth[[idx[s] for s in conv_ids]] = conv[f"rel_depth_{name}"]
            seen = np.zeros(len(ids), bool)
            seen[[idx[s] for s in conv_ids]] = True
            groups["lens_depth"] = np.column_stack(
                [np.where(np.isnan(depth), 1.0, depth), (seen & np.isnan(depth)).astype(float)])
            notes["lens_depth_coverage"] = {"n_with_depth": int(seen.sum()), "n": int(len(ids)),
                                            "n_converged": int(np.isfinite(depth).sum()),
                                            "tol": rec.get("tol")}
            notes["_lens_seen"] = seen
        crys = (lens_meta.get(name) or {}).get("crystallization_depth")
        if crys is None:
            skipped["crystallization_norm"] = (
                "the mean skip-lens curve never reaches the final forecast within tolerance "
                "(or the skip lens is unavailable), so there is no crystallization layer")
        else:
            rel = np.asarray(lens_meta[name]["rel_depth"], dtype=np.float64)
            li = int(np.argmin(np.abs(rel - float(crys))))
            layer = lens_meta[name]["layers"][li]
            n_rows = len(ids)
            last = np.concatenate([_last_window(store, name, layer, s, min(s + 128, n_rows), "act")
                                   for s in range(0, n_rows, 128)])
            groups["crystallization_norm"] = crystallization_norm_feature(last)
            notes["crystallization_layer"] = layer
        models[name] = {"mase": column(name, "mase"), "baseline": base,
                        "baseline_names": base_names, "groups": groups,
                        "skipped": skipped, "notes": notes}
    return {"run": run_dir.name, "series_id": ids, "strata": data.meta["family"].astype(str).to_numpy(),
            "reliable": reliable, "seasonal_mase": seasonal, "models": models,
            "n_boot_default": int(cfg.stats.n_boot)}


def run_reliability(inputs: dict, *, n_folds: int = 5, n_repeats: int = 3, n_boot: int = 1000,
                    seed: int = 0, min_lens_coverage: float = 1.0) -> dict:
    """U1 and U2 over the arrays from `load_run_inputs` (or a test fixture).

    Analysed series: MASE-reliable in the L0 sense and finite for every
    model, so the rows are identical across models and U2 can compare them.
    A lens-depth group is kept only when it covers at least
    `min_lens_coverage` of them; otherwise it is skipped with the coverage
    stated. Each model is fit alone; nothing is pooled across models.
    """
    models = inputs["models"]
    names = sorted(models)
    keep = np.ones(len(inputs["strata"]), bool)
    if "reliable" in inputs and inputs["reliable"] is not None:
        keep &= np.nan_to_num(np.asarray(inputs["reliable"], dtype=np.float64), nan=1.0) > 0
    for m in names:
        keep &= np.isfinite(models[m]["mase"])
    rows = np.flatnonzero(keep)
    strata = np.asarray(inputs["strata"])[rows]
    seasonal = np.asarray(inputs["seasonal_mase"], dtype=np.float64)[rows]
    out = {"schema_version": 1, "evidence_class": EVIDENCE_CLASS,
           "run": inputs.get("run"), "n_series": int(len(rows)),
           "n_excluded": int((~keep).sum()),
           "settings": {"n_folds": n_folds, "n_repeats": n_repeats, "n_boot": n_boot, "seed": seed,
                        "ridge_alphas": [float(a) for a in ALPHAS],
                        "cv_strata": "family", "ci": "series cluster bootstrap over out-of-fold "
                        "predictions (predictions held fixed; refit variance not included)",
                        "min_lens_coverage": min_lens_coverage},
           "caveat": "A feature that predicts failure is not a cause of failure; this is "
                     "predictive (behavioral) evidence only.",
           "models": {}}
    pred_base, pred_full, mase_by = {}, {}, {}
    for m in names:
        rec = models[m]
        mase_m = np.asarray(rec["mase"], dtype=np.float64)[rows]
        log_mase = np.log(np.maximum(mase_m, 1e-6))
        failure = np.where(np.isfinite(seasonal), (mase_m > seasonal).astype(float), np.nan)
        groups, skipped = {}, dict(rec["skipped"])
        for g, arr in rec["groups"].items():
            a = np.asarray(arr, dtype=np.float64)
            if g == "lens_depth":
                seen = rec["notes"].get("_lens_seen")
                cov = float(np.asarray(seen)[rows].mean()) if seen is not None else 1.0
                if cov < min_lens_coverage:
                    skipped[g] = (f"lens convergence depth covers {cov:.3f} of the analysed series, "
                                  f"below the required {min_lens_coverage}")
                    continue
            groups[g] = a[rows]
        res = analyze_model(log_mase, failure, np.asarray(rec["baseline"], dtype=np.float64)[rows],
                            groups, skipped, strata, n_folds=n_folds, n_repeats=n_repeats,
                            n_boot=n_boot, seed=_seed("model", m, base=seed))
        notes = {k: v for k, v in rec["notes"].items() if not k.startswith("_")}
        res["record"]["notes"] = notes
        out["models"][m] = res["record"]
        pred_base[m] = res["predictions"]["ridge"]["baseline"]
        pred_full[m] = res["predictions"]["ridge"].get("baseline+internals",
                                                       res["predictions"]["ridge"]["baseline"])
        mase_by[m] = mase_m
        log.info("reliability %s: %d groups, skipped %s", m, len(groups), list(skipped))
    out["u2"] = route_policies(mase_by, pred_base, pred_full, n_boot=n_boot, seed=seed) \
        if len(names) >= 2 else {"note": "routing needs at least two models"}
    if len(names) >= 2:
        out["u2"]["models_without_internals"] = [m for m in names
                                                 if not out["models"][m]["internal_groups"]]
    return out
