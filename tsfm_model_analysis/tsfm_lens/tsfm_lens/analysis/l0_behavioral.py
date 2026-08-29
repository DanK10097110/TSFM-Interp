"""L0 — behavioral profiling.

Answers the purely behavioral questions ("is one model better on
high-frequency data?") before any internals are touched, and generates the
family-level hypotheses the mechanistic levels then try to explain. Metrics
are computed per series and aggregated per family: MASE against the naive
one-step scale of each context, sMAPE, and mean pinball loss over the
configured quantiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.store import ActivationStore
from ..utils import batch_slices, log, sample_rows, save_json
from .calibration import reliability_from_own_width, summarize_calibration
from .stats import (_mase_scale, holm, mae_over_mad, mase as _mase,
                    mase_pinball_by_horizon, mase_reliability, mean_ci, paired_bootstrap)


def run_l0(cfg: PipelineConfig, hub, data: BenchmarkData, store: ActivationStore) -> None:
    """Forecast with every model, score against targets, persist metric tables."""
    out_dir = cfg.run_dir() / "l0"
    out_dir.mkdir(parents=True, exist_ok=True)
    contexts, targets = data.contexts(), data.targets()
    nf_rows = None
    if cfg.l0.noise_floor_repeats >= 2:
        k = min(data.n, cfg.l0.noise_floor_series)
        nf_rows = sample_rows(data.n, k, cfg.run.seed + 77, strata=data.meta["family"].to_numpy())
    frames, noise_floor, calibration, horizon_resolved, reliability = [], {}, {}, {}, {}
    for mcfg in cfg.models:
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        point, quants = _predict_all(adapter, contexts, data.horizon, cfg.l0.quantiles)
        store.write_predictions(mcfg.name, point, quants)
        frames.append(_score(mcfg.name, point, quants, contexts, targets,
                             cfg.l0.quantiles, data.meta, cfg.l0.scale, cfg.l0.min_scale_frac))
        if cfg.l0.calibration and len(cfg.l0.quantiles) >= 2:
            calibration[mcfg.name] = summarize_calibration(
                quants, cfg.l0.quantiles, targets, data.meta["family"].to_numpy())
            reliability[mcfg.name] = reliability_from_own_width(
                point, quants, targets, _mase_scale(contexts, cfg.l0.scale))
        if cfg.l0.horizon_resolved and data.horizon > 1:
            horizon_resolved[mcfg.name] = _summarize_by_horizon(
                point, quants, targets, contexts, cfg.l0.quantiles, cfg.l0.scale,
                data.meta["family"].to_numpy())
        if nf_rows is not None:
            noise_floor[mcfg.name] = _measure_noise_floor(
                adapter, contexts[nf_rows], targets[nf_rows], data.horizon, cfg.l0.quantiles,
                cfg.l0.scale, cfg.l0.noise_floor_repeats, data.meta["family"].to_numpy()[nf_rows])
        if not cfg.run.keep_models_loaded:
            hub.release(mcfg.name)
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_parquet(out_dir / "metrics.parquet")
    save_json(out_dir / "summary.json", _summarize(metrics, cfg))
    if noise_floor:
        save_json(out_dir / "noise_floor.json", noise_floor)
    if calibration:
        save_json(out_dir / "calibration.json", calibration)
    if reliability:
        save_json(out_dir / "reliability.json", reliability)
    if horizon_resolved:
        save_json(out_dir / "horizon_resolved.json", horizon_resolved)
    log.info("L0 complete: %d model-series scores", len(metrics))


def _summarize_by_horizon(point: np.ndarray, quants: np.ndarray, targets: np.ndarray,
                          contexts: np.ndarray, quantiles: list, scale_mode: str,
                          families: np.ndarray) -> dict:
    """Pooled + per-family MASE/pinball curves over horizon step (sec 16 E12)."""
    by_h = mase_pinball_by_horizon(point, quants, targets, contexts, quantiles, scale_mode)
    mase_h, pinball_h = by_h["mase_by_horizon"], by_h["pinball_by_horizon"]
    out = {"mase_by_horizon": mase_h.mean(axis=0).tolist(),
          "pinball_by_horizon": pinball_h.mean(axis=0).tolist(), "n_series": int(len(mase_h))}
    per_family = {}
    for fam in sorted(set(families.tolist())):
        mask = families == fam
        if mask.sum() == 0:
            continue
        per_family[str(fam)] = {"mase_by_horizon": mase_h[mask].mean(axis=0).tolist(),
                                "pinball_by_horizon": pinball_h[mask].mean(axis=0).tolist(),
                                "n_series": int(mask.sum())}
    out["by_family"] = per_family
    return out


def _measure_noise_floor(adapter, contexts: np.ndarray, targets: np.ndarray, horizon: int,
                         quantiles: list, scale_mode: str, repeats: int,
                         families: np.ndarray) -> dict:
    """How much this model's own MASE varies between repeat `predict()` calls (sec 15 A13).

    Every ΔMASE elsewhere in the repo (head/MLP ablation, SAE forecast-
    preservation, L3 restoration) implicitly compares against zero; this is
    the floor those deltas should actually be read against. A model with no
    sampling in its forecast path (TimesFM, Chronos-Bolt) should show
    `deterministic: true` and a floor of exactly 0 -- a free wiring sanity
    check in the same spirit as TimesFM's exactly-0.0 `future_mass`
    (`CLAUDE.md` §9/§5.4) -- since repeated calls with no RNG dependence
    have nothing to vary between them.
    """
    mases = np.stack([_mase(_predict_all(adapter, contexts, horizon, quantiles)[0],
                            targets, contexts, scale_mode) for _ in range(repeats)])
    deltas = np.abs(np.diff(mases, axis=0))
    per_family = {}
    for fam in sorted(set(families.tolist())):
        mask = families == fam
        fam_deltas = deltas[:, mask]
        if fam_deltas.size:
            per_family[str(fam)] = {"mean": float(fam_deltas.mean()),
                                    "p95": float(np.quantile(fam_deltas, 0.95))}
    return {"repeats": repeats, "n_series": int(contexts.shape[0]),
           "deterministic": bool(np.allclose(deltas, 0.0, atol=1e-6)),
           "mase_abs_delta_mean": float(deltas.mean()),
           "mase_abs_delta_p95": float(np.quantile(deltas, 0.95)),
           "mase_abs_delta_max": float(deltas.max()),
           "per_family": per_family}


def _predict_all(adapter, contexts: np.ndarray, horizon: int, quantiles: list):
    """Batched forecasting over the full benchmark."""
    points, quants = [], []
    for s, e in batch_slices(len(contexts), adapter.cfg.batch_size):
        out = adapter.predict(contexts[s:e], horizon, quantiles)
        points.append(out["point"])
        quants.append(out["quantiles"])
    return np.concatenate(points), np.concatenate(quants)


def _score(model: str, point: np.ndarray, quants: np.ndarray, contexts: np.ndarray,
           targets: np.ndarray, quantiles: list, meta: pd.DataFrame,
           scale_mode: str = "mean_abs_diff", min_scale_frac: float = 0.0) -> pd.DataFrame:
    """Per-series metric table for one model.

    `mase_reliable` (sec 15 A11) flags series whose MASE denominator is too
    small relative to the target's own level to trust -- heavy intermittency
    or a near-flat context both degenerate this way, and because the
    denominator is shared with pinball's normalization, pinball inherits the
    same `scale_mode`. sMAPE audited separately (already floored against the
    both-zero case via `+ 1e-8`; a genuinely wrong nonzero forecast against a
    zero target correctly scores near the metric's own max rather than
    exploding, so no fix needed there -- see this item's Findings).
    `mae_over_mad` is a context-independent companion: a floor effect from
    context degeneracy shows up as MASE and `mae_over_mad` diverging, not as
    a shared, invisible compression.
    """
    scale = _mase_scale(contexts, scale_mode)
    err = targets - point
    smape = (2 * np.abs(err) / (np.abs(targets) + np.abs(point) + 1e-8)).mean(axis=1)
    q = np.asarray(quantiles, dtype=np.float32)[None, None, :]
    diff = targets[:, :, None] - quants
    pinball = np.maximum(q * diff, (q - 1) * diff).mean(axis=(1, 2)) / scale
    return pd.DataFrame({"model": model, "series_id": meta["series_id"],
                         "family": meta["family"], "archetype": meta["archetype"],
                         "generator": meta["generator"],
                         "mase": _mase(point, targets, contexts, scale_mode),
                         "mase_reliable": mase_reliability(contexts, targets, scale_mode,
                                                           min_scale_frac=min_scale_frac),
                         "mae_over_mad": mae_over_mad(point, targets),
                         "smape": smape, "pinball": pinball})


def _archetype_summary(metrics: pd.DataFrame, cfg: PipelineConfig) -> dict:
    """Per-archetype MASE ratio/Holm tests, finer-grained than `per_family` (sec 15 A9).

    `random_parametric` samples all collapse into one `family` label, so a
    claim like "the gap holds on `intermittent_bursts` too, not only the
    smooth archetypes it was first observed on" was previously unanswerable.
    Series whose tier can't express an archetype (real-derived, smoke) fall
    back to their family label -- stated in `fallback_to_family` below, not
    silently dropped or silently blended into archetype-labeled rows.

    MASE-based columns (mean, ratio, tests) use only `mase_reliable` rows
    (sec 15 A11); `smape`/`pinball`/`mae_over_mad` use every row, since they
    don't share MASE's scale degeneracy.
    """
    sc = cfg.stats
    if not metrics["archetype"].notna().any():
        return {"applicable": False,
               "reason": "no series in this corpus carry an archetype label "
                         "(only random_parametric records one); nothing "
                         "finer-grained than the per-family table above to report."}
    metrics = metrics.assign(
        archetype_group=metrics["archetype"].where(metrics["archetype"].notna(), metrics["family"]))
    counts = metrics.groupby("archetype_group").size()
    dropped = sorted(counts[counts < sc.min_series].index.tolist())
    kept = metrics[~metrics["archetype_group"].isin(dropped)]
    reliable = kept[kept["mase_reliable"]]
    per_arch = (kept.groupby(["model", "archetype_group"])[["smape", "pinball", "mae_over_mad"]]
                .mean().reset_index().rename(columns={"archetype_group": "archetype"}))
    mase_means = (reliable.groupby(["model", "archetype_group"])["mase"].mean()
                 .rename("mase").reset_index().rename(columns={"archetype_group": "archetype"}))
    per_arch = per_arch.merge(mase_means, on=["model", "archetype"], how="left")
    excl = (kept.drop_duplicates("series_id").assign(unreliable=lambda d: ~d["mase_reliable"])
           .groupby("archetype_group")["unreliable"].sum().astype(int))
    per_arch["mase_n_excluded"] = per_arch["archetype"].map(excl).fillna(0).astype(int)
    out = {"rows": per_arch.to_dict(orient="records"),
          "fallback_to_family": sorted(metrics.loc[metrics["archetype"].isna(), "family"].unique().tolist()),
          "dropped_min_n": dropped, "min_n": sc.min_series}
    n_groups = kept["archetype_group"].nunique()
    # ROADMAP.md sec 24.3: a solo run has per-archetype METRICS but no paired
    # test to run, and `comparison_pair()` raises before the `applicable`
    # guard below could say so. Answer in the guard's own shape rather than
    # letting the stage die -- every row above this line is a within-model
    # measurement and survives untouched.
    if cfg.run_shape() == "solo":
        out["tests"] = {"applicable": False,
                        "reason": "solo run (1 model): the per-archetype metrics above "
                                  "are within-model and stand, but a paired test needs "
                                  "two models to pair"}
        return out
    a, b = cfg.comparison_pair()
    wide = reliable.pivot_table(index=["series_id", "archetype_group"], columns="model",
                               values="mase").reset_index()
    if n_groups < 2 or a.name not in wide.columns or b.name not in wide.columns:
        out["tests"] = {"applicable": False,
                        "reason": (f"only {n_groups} archetype group has >= {sc.min_series} series"
                                  if n_groups < 2 else "comparison pair not present in metrics")}
        return out
    pivot = per_arch.pivot(index="archetype", columns="model", values="mase")
    ratio = (pivot[a.name] / pivot[b.name]).dropna().sort_values()
    out["mase_ratio"] = {str(k): float(v) for k, v in ratio.items()}
    if not sc.enabled:
        out["tests"] = {"applicable": False, "reason": "stats.enabled is false"}
        return out
    tests, pvals = {}, {}
    for i, (grp_key, grp) in enumerate(wide.groupby("archetype_group")):
        diff = (grp[b.name] - grp[a.name]).to_numpy()
        res = paired_bootstrap(diff, sc.n_boot, cfg.run.seed + 140 + i, sc.ci)
        if res is not None:
            tests[str(grp_key)], pvals[str(grp_key)] = res, res["p"]
    adjusted = holm(pvals) if pvals else {}
    archetype_tests = []
    for grp_key, res in tests.items():
        p_holm = adjusted[grp_key]
        favored = "none"
        if p_holm < sc.alpha and res["lo"] > 0:
            favored = a.name
        elif p_holm < sc.alpha and res["hi"] < 0:
            favored = b.name
        archetype_tests.append({"archetype": grp_key, "ratio": out["mase_ratio"].get(grp_key),
                                **res, "p_holm": p_holm, "favored": favored})
    out["tests"] = sorted(archetype_tests, key=lambda t: t["p_holm"])
    out["strengths"] = {
        a.name: [t["archetype"] for t in archetype_tests if t["favored"] == a.name],
        b.name: [t["archetype"] for t in archetype_tests if t["favored"] == b.name],
    }
    out["alpha"] = sc.alpha
    return out


def _summarize(metrics: pd.DataFrame, cfg: PipelineConfig) -> dict:
    """Family aggregates with CIs, plus Holm-corrected paired tests of the model pair.

    A family "strength" is claimed only when the paired per-series MASE
    difference has a Holm-adjusted bootstrap p below alpha and a CI
    excluding zero; the raw ratio is kept as an effect size.

    MASE-based numbers (means, CIs, ratios, tests) are computed on the
    `mase_reliable` subset only (sec 15 A11); `smape`/`pinball`/
    `mae_over_mad` use every row -- excluded counts are recorded in
    `mase_reliability` (corpus-wide) and each `per_family`/`per_archetype`
    row's `mase_n_excluded`, not silently absorbed into the mean.
    """
    sc = cfg.stats
    reliable = metrics[metrics["mase_reliable"]]
    per_family = (metrics.groupby(["model", "family"])[["smape", "pinball", "mae_over_mad"]]
                  .mean().reset_index())
    mase_means = (reliable.groupby(["model", "family"])["mase"].mean()
                 .rename("mase").reset_index())
    per_family = per_family.merge(mase_means, on=["model", "family"], how="left")
    excl = (metrics.drop_duplicates("series_id").assign(unreliable=lambda d: ~d["mase_reliable"])
           .groupby("family")["unreliable"].sum().astype(int))
    per_family["mase_n_excluded"] = per_family["family"].map(excl).fillna(0).astype(int)
    if sc.enabled:
        cis = []
        for _, row in per_family.iterrows():
            grp = reliable[(reliable["model"] == row["model"]) & (reliable["family"] == row["family"])]
            cis.append(mean_ci(grp["mase"].to_numpy(), sc.n_boot, cfg.run.seed + len(cis), sc.ci)
                      if len(grp) else None)
        per_family["mase_lo"] = [c["lo"] if c else None for c in cis]
        per_family["mase_hi"] = [c["hi"] if c else None for c in cis]
    overall = metrics.groupby("model")[["smape", "pinball", "mae_over_mad"]].mean().reset_index()
    overall_mase = reliable.groupby("model")["mase"].mean().rename("mase").reset_index()
    overall = overall.merge(overall_mase, on="model", how="left")
    summary = {"per_family": per_family.to_dict(orient="records"),
               "overall": overall.to_dict(orient="records"),
               "mase_reliability": {
                   "scale": cfg.l0.scale, "min_scale_frac": cfg.l0.min_scale_frac,
                   "n_total": int(len(metrics)), "n_excluded": int((~metrics["mase_reliable"]).sum()),
               }}
    summary["per_archetype"] = _archetype_summary(metrics, cfg)
    n_families = int(metrics["family"].nunique())
    if n_families < 2:
        # A per-family *comparison* (ratio/Holm test/"strengths") needs >=2
        # groups to compare; with one, it would just restate the pooled
        # "overall" number under a family-shaped table -- disable it with a
        # stated reason instead of rendering a degenerate one-row comparison
        # (`ROADMAP.md` sec 15 A6). `per_family`/`overall` above are plain
        # aggregates, not comparisons, so they stay.
        summary["family_comparisons"] = {
            "applicable": False,
            "reason": f"only {n_families} family present in this corpus "
                      f"({'that family' if n_families else 'no families'} would be "
                      f"compared against itself); per-family strength tests need >=2 "
                      f"families. The Overall metrics above still reflect the full "
                      f"pooled comparison.",
        }
        return summary
    # ROADMAP.md sec 24.3, as above: the per-family table, the calibration
    # block and every quantile metric are within-model and already in
    # `summary`; only the paired strength tests need a second model.
    if cfg.run_shape() == "solo":
        # A DEDICATED key, not `pairwise`. `report.py` iterates `pairwise` as
        # a list of `{"a": ..., "b": ...}` entries, so overloading it with a
        # status dict would iterate its KEYS and raise on `e["a"]` -- the
        # report-key-drift failure `CLAUDE.md` sec 11.6 already records, and
        # the reason `n_families < 2` above uses `family_comparisons` rather
        # than reusing an existing key too.
        summary["comparison"] = {"applicable": False,
                                 "reason": "solo run (1 model): per-family metrics above are "
                                           "within-model and stand; strength tests compare "
                                           "two models and have nothing to pair"}
        return summary
    a, b = cfg.comparison_pair()
    wide = reliable.pivot_table(index=["series_id", "family"], columns="model",
                                values="mase").reset_index()
    if a.name not in wide.columns or b.name not in wide.columns:
        return summary
    pivot = per_family.pivot(index="family", columns="model", values="mase")
    ratio = (pivot[a.name] / pivot[b.name]).dropna().sort_values()
    summary["mase_ratio"] = {str(k): float(v) for k, v in ratio.items()}

    if not sc.enabled:
        summary["strengths"] = {a.name: [f for f, r in ratio.items() if r < 0.9],
                                b.name: [f for f, r in ratio.items() if r > 1.1]}
        return summary

    # Every configured model pair, Holm-corrected across the WHOLE (pair,
    # family) set at once rather than once per pair (`ROADMAP.md` sec 18 F8).
    # With two models this set is exactly the family set, so every adjusted
    # p-value is bit-identical to the single-pair correction that preceded
    # this -- the widening only happens where a third model actually adds
    # comparisons, which is the point (sec 2.1: no recorded number moves).
    present = [m for m in cfg.models if m.name in wide.columns]
    pairs = [(x, y) for i, x in enumerate(present) for y in present[i + 1:]]
    # Pair 0 is always the designated `comparison_pair()`, and it keeps its
    # historical seeds EXACTLY -- `+40+fi` with `fi` advancing past skipped
    # small families just as the old `enumerate` did, and `+90` for the
    # overall test -- so a two-model run reproduces bit-for-bit (sec 2.1).
    # Additional pairs draw from a disjoint high range rather than continuing
    # the low one, because a running counter would both re-seed pair 0 after
    # any skip and eventually collide with `+90`, silently making two tests
    # share one resample stream.
    def _fam_seed(pi: int, fi: int) -> int:
        return cfg.run.seed + (40 + fi if pi == 0 else 100_000 + pi * 1000 + fi)

    def _overall_seed(pi: int) -> int:
        return cfg.run.seed + (90 if pi == 0 else 200_000 + pi)

    raw = {}
    for pi, (x, y) in enumerate(pairs):
        for fi, (fam, grp) in enumerate(wide.groupby("family")):
            if len(grp) < sc.min_series:
                continue
            diff = (grp[y.name] - grp[x.name]).to_numpy()
            res = paired_bootstrap(diff, sc.n_boot, _fam_seed(pi, fi), sc.ci)
            if res is not None:
                raw[(x.name, y.name, str(fam))] = res

    adjusted = holm({"\u241f".join(k): v["p"] for k, v in raw.items()}) if raw else {}

    def _rows(x_name: str, y_name: str) -> list:
        out_rows = []
        for (xn, yn, fam), res in raw.items():
            if (xn, yn) != (x_name, y_name):
                continue
            p_holm = adjusted["\u241f".join((xn, yn, fam))]
            favored = "none"
            if p_holm < sc.alpha and res["lo"] > 0:
                favored = xn
            elif p_holm < sc.alpha and res["hi"] < 0:
                favored = yn
            ratio = None
            if xn in pivot.columns and yn in pivot.columns:
                num, den = pivot[xn].get(fam), pivot[yn].get(fam)
                if num is not None and den not in (None, 0) \
                        and np.isfinite(num) and np.isfinite(den):
                    ratio = float(num / den)
            out_rows.append({"family": fam, "ratio": ratio, **res,
                             "p_holm": p_holm, "favored": favored})
        return sorted(out_rows, key=lambda t: t["p_holm"])

    pairwise = []
    for i, (x, y) in enumerate(pairs):
        rows = _rows(x.name, y.name)
        pairwise.append({
            "a": x.name, "b": y.name, "family_tests": rows,
            "overall_test": paired_bootstrap((wide[y.name] - wide[x.name]).to_numpy(),
                                             sc.n_boot, _overall_seed(i), sc.ci),
            "strengths": {x.name: [t["family"] for t in rows if t["favored"] == x.name],
                          y.name: [t["family"] for t in rows if t["favored"] == y.name]},
        })
    summary["pairwise"] = pairwise
    summary["multiplicity"] = {
        "scope": "l0.family", "method": "holm", "alpha": sc.alpha,
        "n_models": len(present), "n_pairs": len(pairs), "n_tests": len(raw),
        "most_stringent_threshold": (sc.alpha / len(raw)) if raw else None,
        "designated_pair": [a.name, b.name],
        "pairs_examined_by_other_stages": 1,
        # `n_boot` belongs here because bootstrap p-values are FLOORED at
        # 1/n_boot (sec 6.6), so the smallest Holm-adjusted p this family can
        # ever produce is n_tests/n_boot -- independent of the data. Once that
        # exceeds alpha the correction is unsatisfiable: no effect of any size
        # can be significant, and the report must say so rather than render a
        # table of honestly-computed non-results (invariant 8).
        "n_boot": sc.n_boot,
        "min_attainable_p_holm": (len(raw) / sc.n_boot) if raw and sc.n_boot else None,
    }

    # The designated pair keeps its existing top-level keys: every consumer
    # (report, confirm, meta_report) reads these, and a run with two models
    # must produce byte-identical output to before this change.
    mine = next((e for e in pairwise if (e["a"], e["b"]) == (a.name, b.name)), None)
    if mine is not None:
        summary["family_tests"] = mine["family_tests"]
        summary["overall_test"] = mine["overall_test"]
        summary["strengths"] = mine["strengths"]
    summary["alpha"] = sc.alpha
    return summary
