"""Phase 2a (ROADMAP.md §6.1): does a layer's effective dimensionality
predict how interpretable/controllable that layer is?

Pools one record per (run, model, layer) from artifacts every existing run
directory already has -- internals' effective dimensionality, input-CKA,
and family-probe decodability; lens's tuned-lens R² against the model's own
forecast; L3's per-layer causal-sensitivity fingerprint; and (once
`run_sae_layer_sweep.py` has been run against a given run directory) SAE
ground-truth-alignment mean |ρ| as a fourth interpretability proxy -- and
correlates each candidate "worth interpreting" metric against each proxy.
Nothing is re-run; a run missing a stage (or missing the optional SAE
sweep) just contributes fewer records for that metric, matching the rest of
the pipeline's degrade-with-a-log discipline (CLAUDE.md §2.5).

The resampling unit for the bootstrap CI is the (run, model) group, not the
individual layer. Layers within one model are strongly depth-autocorrelated
(effective dimensionality and probe decodability both trend with depth), so
treating every layer as an independent sample would understate how few
truly independent observations exist -- the same reasoning CLAUDE.md §6.6
applies to series and windows, carried over to the next enclosing dependent
unit for this differently-shaped question.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ..utils import load_json, log
from .stats import bootstrap_ci

CANDIDATE_METRICS = ["effective_dim", "input_cka", "l3_entropy"]
# `sae_ground_truth_rho` (ROADMAP.md §6.1's "feed SAE eval results back into
# layer-selection" item): populated only for runs with a `sae_layer_sweep.json`
# artifact (`run_sae_layer_sweep.py`), so it degrades to "no data" like every
# other optional proxy here on a run that hasn't had the sweep run against it.
PROXY_METRICS = ["probe_decodability", "tuned_r2_model", "l3_mean_sensitivity", "sae_ground_truth_rho"]


def _entropy(row: np.ndarray) -> float:
    """Normalized Shannon entropy of a nonnegative row, in [0, 1]."""
    row = np.clip(np.asarray(row, dtype=np.float64), 0, None)
    total = row.sum()
    if total <= 0 or len(row) < 2:
        return 0.0
    p = row[row > 0] / total
    return float(-(p * np.log(p)).sum() / np.log(len(row)))


def _safe_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception as e:
        log.warning(f"layer_selection: could not read {path}: {e}")
        return None


def collect_layer_records(run_dirs: list) -> list[dict]:
    """One dict per (run, model, layer) with whichever candidate/proxy metrics exist."""
    records = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        profile = _safe_json(run_dir / "internals" / "profile.json")
        if not profile:
            log.info(f"layer_selection: {run_dir} has no internals/profile.json; skipped")
            continue

        lens_meta = _safe_json(run_dir / "lens" / "lens.json") or {}
        curves_path = run_dir / "lens" / "curves.npz"
        curves = np.load(curves_path) if curves_path.exists() else None

        l3_meta = _safe_json(run_dir / "l3" / "meta.json") or {}
        l3_layers = l3_meta.get("layers", {})
        l3_arrs_path = run_dir / "l3" / "sensitivity.npz"
        l3_arrs = np.load(l3_arrs_path) if l3_arrs_path.exists() else None

        sae_sweep = _safe_json(run_dir / "sae_layer_sweep.json")
        sae_results = (sae_sweep or {}).get("results", {})

        for model, info in profile.items():
            layers = info["layers"]
            for i, layer in enumerate(layers):
                rec = {
                    "run": run_dir.name, "model": model, "layer": layer,
                    "rel_depth": info["rel_depth"][i],
                    "effective_dim": info["effective_dim"][i],
                    "input_cka": info["input_cka"][i],
                    "probe_decodability": info["probe"][i]["value"],
                }
                if model in lens_meta and lens_meta[model].get("layers") == layers and curves is not None:
                    key = f"tuned_r2_model_{model}"
                    if key in curves:
                        rec["tuned_r2_model"] = float(curves[key][i])

                if l3_layers.get(model) == layers and l3_arrs is not None:
                    key = f"fingerprint_{model}"
                    if key in l3_arrs:
                        fp_row = l3_arrs[key][i]
                        rec["l3_mean_sensitivity"] = float(np.mean(fp_row))
                        rec["l3_entropy"] = _entropy(fp_row)

                sae_entry = sae_results.get(f"{model}/{layer}")
                if sae_entry is not None:
                    gt = sae_entry.get("ground_truth_alignment", {})
                    if "error" not in gt and gt.get("n_features_matched", 0) > 0:
                        rec["sae_ground_truth_rho"] = float(gt["mean_abs_rho_matched"])

                records.append(rec)
    return records


def _group_key(rec: dict) -> tuple:
    return (rec["run"], rec["model"])


def cluster_bootstrap_spearman(records: list[dict], x_key: str, y_key: str,
                               n_boot: int = 2000, seed: int = 0) -> dict | None:
    """Spearman correlation between two per-layer metrics, CI via resampling (run, model) groups.

    Returns None if fewer than 3 groups have both metrics present, or if
    fewer than 3 total records exist -- correlation with less than that is
    not a claim worth reporting.
    """
    usable = [r for r in records if x_key in r and y_key in r]
    if len(usable) < 3:
        return None
    groups: dict[tuple, list[dict]] = {}
    for r in usable:
        groups.setdefault(_group_key(r), []).append(r)
    group_list = list(groups.values())
    if len(group_list) < 3:
        return None

    def stat_fn(idx: np.ndarray) -> float:
        pooled = [rec for gi in idx for rec in group_list[gi]]
        xs = np.array([rec[x_key] for rec in pooled])
        ys = np.array([rec[y_key] for rec in pooled])
        if np.std(xs) == 0 or np.std(ys) == 0:
            return 0.0
        rho, _ = spearmanr(xs, ys)
        return float(rho) if np.isfinite(rho) else 0.0

    ci = bootstrap_ci(stat_fn, n_units=len(group_list), n_boot=n_boot, seed=seed,
                      unit="(run, model) group")
    boots_zero_straddle = ci["lo"] <= 0.0 <= ci["hi"]
    return {**ci, "n_records": len(usable), "n_groups": len(group_list),
            "significant": not boots_zero_straddle}


def _add_depth_residuals(records: list[dict], keys: list) -> list[dict]:
    """Add `{key}_resid` = key's value after removing its linear trend vs rel_depth.

    Every metric here plausibly trends with depth on its own; a candidate
    metric "predicting" a proxy could just mean both trend with depth
    independently, not that the candidate carries information beyond depth.
    The detrending line is fit once on the full pooled sample rather than
    refit inside each bootstrap iteration -- an approximation, acceptable at
    this record count, but worth knowing if extending this to a much larger
    pooled sample later.
    """
    out = [dict(r) for r in records]
    for key in keys:
        usable_idx = [i for i, r in enumerate(out) if key in r and "rel_depth" in r]
        if len(usable_idx) < 3:
            continue
        xs = np.array([out[i]["rel_depth"] for i in usable_idx])
        ys = np.array([out[i][key] for i in usable_idx])
        design = np.vstack([xs, np.ones_like(xs)]).T
        coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
        resid = ys - design @ coef
        for i, r_resid in zip(usable_idx, resid):
            out[i][f"{key}_resid"] = float(r_resid)
    return out


def run_layer_selection_study(run_dirs: list, n_boot: int = 2000, seed: int = 0) -> dict:
    """Every candidate-metric x proxy-metric Spearman correlation, pooled across run_dirs.

    Reports both the raw correlation and a depth-controlled version (§2.2's
    null-control discipline extended to the most obvious confound here:
    every metric plausibly trends with depth, so a raw correlation could
    just mean "both trend with depth" rather than the candidate metric
    carrying information beyond that).
    """
    records = collect_layer_records(run_dirs)
    all_keys = CANDIDATE_METRICS + PROXY_METRICS
    resid_records = _add_depth_residuals(records, all_keys)

    correlations = {}
    depth_trend = {}
    for key in all_keys:
        result = cluster_bootstrap_spearman(records, "rel_depth", key, n_boot=n_boot, seed=seed)
        if result is not None:
            depth_trend[key] = result

    for cand in CANDIDATE_METRICS:
        for proxy in PROXY_METRICS:
            raw = cluster_bootstrap_spearman(records, cand, proxy, n_boot=n_boot, seed=seed)
            if raw is None:
                log.info(f"layer_selection: not enough data for {cand} vs {proxy}; skipped")
                continue
            controlled = cluster_bootstrap_spearman(
                resid_records, f"{cand}_resid", f"{proxy}_resid", n_boot=n_boot, seed=seed)
            correlations[f"{cand}__vs__{proxy}"] = {"raw": raw, "depth_controlled": controlled}

    return {
        "n_records": len(records),
        "runs": sorted({r["run"] for r in records}),
        "models": sorted({r["model"] for r in records}),
        "depth_trend": depth_trend,
        "correlations": correlations,
    }


# Which direction of which candidate metric predicted each proxy, per the
# 2026-08-05 study below (ROADMAP.md §6.1 Findings) -- both directions were
# significant after depth control, on n_groups=8, so read this as the
# current best-available policy, not a settled fact. There is no single
# "worth interpreting" score: effective dimensionality predicts the two
# proxies in *opposite* directions, so `goal` must be given explicitly
# rather than defaulting to one composite (`CLAUDE.md` §2.6's evidence-class
# honesty, applied to a layer-selection policy rather than a report figure).
_POLICIES = {
    "forecast_readability": ("effective_dim", -1),   # low effective_dim -> high tuned-lens R^2
    "family_decodability": ("l3_entropy", +1),        # high L3-fingerprint entropy -> high probe accuracy
}


def recommend_layers(records: list[dict], run: str, model: str, goal: str, top_k: int = 3) -> list[dict]:
    """Rank one model's own captured layers by the metric direction §6.1's study found predictive.

    `goal` must be one of `_POLICIES`'s keys -- there is deliberately no
    goal-less "auto", since effective dimensionality predicts
    forecast-readability and family-decodability in opposite directions
    (see module docstring and ROADMAP.md §6.1). Distinct from
    `analysis/clustering.py`'s own `layer: auto`, which picks the L1
    peak-CKA pair for an unrelated reason (cross-model similarity, not
    single-model interpretability) -- the two "auto" policies answer
    different questions and should not be conflated.

    ⚠️ Validated only as a description of the pooled, depth-controlled,
    cross-(run, model) trend -- **not** as a within-single-model ranking
    rule, and the difference is not hypothetical: applying this policy to
    `runs/medium_run_chronos_base`'s own Chronos-T5-Base layers for
    `"forecast_readability"` picked `encoder.block.0`, whose own
    `tuned_r2_model` (0.433) is the *lowest* of any of its 12 layers -- the
    literal opposite of intent -- because within that one model, effective
    dimensionality and tuned-lens R² both *rise* together over early-to-mid
    depth (the same depth-driven co-trend the pooled study's depth control
    was built to strip out across models, not within one). The same policy
    picked TimesFM's actual best layer correctly in the same run. Read: this
    is a real, checked limitation, not a hedge -- inspect a model's own
    per-layer profile (`collect_layer_records`) before trusting this
    function's pick for that specific model, especially for architectures
    unlike the ones §6.1's study was run on. See ROADMAP.md §6.2's Findings
    for the full comparison.
    """
    if goal not in _POLICIES:
        raise ValueError(f"goal must be one of {list(_POLICIES)}, got {goal!r}")
    metric, direction = _POLICIES[goal]
    subset = [r for r in records if r["run"] == run and r["model"] == model and metric in r]
    ranked = sorted(subset, key=lambda r: direction * r[metric], reverse=True)
    return ranked[:top_k]
