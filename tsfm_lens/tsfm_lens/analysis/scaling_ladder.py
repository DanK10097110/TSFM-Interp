"""Model-family scaling ladder (ROADMAP.md sec 20 H1).

Chronos-T5 ships as tiny -> mini -> small -> base -> large: five checkpoints
sharing an architecture, a training corpus and a tokenizer, differing only in
size. `ROADMAP.md` sec 5.3 used two of those points once, as a size *control*,
and found L1/L2 growing with size while L4 clustering AMI shrank. This module
turns that two-point curiosity into a curve, by reducing N already-built run
directories -- one per ladder rung, each pairing the ladder model against the
same fixed reference model -- into one metric-vs-size table.

It re-runs nothing and loads no checkpoint. `report/meta_report.py::
summarize_run` already extracts the run-level half of what a ladder needs
(L0 MASE, L1 peak CKA, L2 gain, crystallization depth, clustering AMI,
capture coverage, provenance), so this module calls it rather than
reimplementing it, and adds only the per-model artifacts that a cross-run
comparison of *one* model needs and a two-model side-by-side did not:
`internals/profile.json`'s effective dimensionality and probe decodability,
`sae/meta.json`'s dead-feature rate and ground-truth alignment, and
`l0/calibration.json`'s coverage error.

Three decisions are load-bearing and worth stating here rather than in the
code they constrain:

1. **The ladder axis is a measured parameter count**, read from `budget/
   model_budget.json` (sec 18 F2), never parsed out of a checkpoint name.
   "chronos-t5-large" is a claim nobody in this repo checks; `parameters.
   total` was counted from the loaded module. A run whose budget stage never
   ran is *excluded with a reason*, not placed on the axis by its name
   (`CLAUDE.md` sec 11.34 -- a probe path must refuse rather than guess).

2. **Significance is an exact permutation test, not a bootstrap.** A five-rung
   ladder has five points; resampling five numbers estimates nothing. Every
   ordering of n points is enumerable (n! = 120 at n=5), so the two-sided
   p-value for Spearman's rho is exact and cheap. Its own floor is reported
   next to it: at n=5 no metric can score below p = 2/120 = 0.0167, which is
   a property of the ladder's *length*, not of the metric -- the same trap
   `CLAUDE.md` sec 11.35 records for a bounded score whose ceiling sits below
   1.0, caught here before it could be read as a weak result.

3. **"Flat" is a verdict that requires within-run CIs, and says so when it
   cannot be reached.** A metric whose across-ladder range is smaller than its
   own median within-run CI width is flat: size did not move it by more than
   one run's noise. Metrics whose artifacts carry no CI (crystallization
   depth, SAE dead rate, mean effective dimensionality) get `flat: None` with
   a stated reason rather than a comparison against an assumed noise level --
   H1's acceptance criterion asks which metrics are flat, and an unbacked
   "not flat" would be the wrong way to answer it.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from ..report.meta_report import summarize_run, _safe_load_json
from ..utils import log


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average tied ranks (CLAUDE.md sec 11.37 -- position tie-breaks fabricate order)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
    return 0.0 if denom == 0.0 else float((xc * yc).sum() / denom)


def spearman_exact(sizes: list[float], values: list[float], max_exact: int = 8) -> dict:
    """Spearman's rho against ladder position, with an exact permutation p-value.

    Enumerating every ordering is the right test for a ladder: n is small by
    construction, so the null distribution of rho is finite and computable
    rather than estimated. `p_floor` is the smallest p this many rungs can
    produce -- read a p at the floor as "as extreme as this ladder length
    permits", never as "highly significant".
    """
    x = np.asarray(sizes, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    n = len(x)
    if n < 3:
        return {"rho": None, "p": None, "p_floor": None, "n": n,
                "reason": f"a ladder needs at least 3 rungs to have a rank correlation; got {n}"}
    rx, ry = _ranks(x), _ranks(y)
    rho = _pearson(rx, ry)
    if n <= max_exact:
        perms = list(itertools.permutations(range(n)))
        null = np.array([_pearson(rx, ry[list(p)]) for p in perms])
        p = float((np.abs(null) >= abs(rho) - 1e-12).mean())
        p_floor = float(2.0 / math.factorial(n))
        exact = True
    else:
        rng = np.random.default_rng(0)
        null = np.array([_pearson(rx, rng.permutation(ry)) for _ in range(20000)])
        p = float((np.abs(null) >= abs(rho) - 1e-12).mean())
        p_floor = 1.0 / 20000.0
        exact = False
    return {"rho": rho, "p": p, "p_floor": p_floor, "n": n, "exact": exact}


def classify(sizes: list[float], values: list[float],
             ci_widths: list[float] | None = None) -> dict:
    """Monotone / non-monotone / flat, with flatness withheld when unbacked."""
    order = np.argsort(np.asarray(sizes, dtype=np.float64))
    v = np.asarray(values, dtype=np.float64)[order]
    if len(v) < 3:
        # `np.all(np.diff(v) > 0)` is vacuously True on one point, so a
        # degenerate ladder would otherwise report a confident
        # "monotone_increasing" with range 0 -- the same shape of failure
        # CLAUDE.md sec 11.37 records, where an absent measurement produced a
        # confident verdict instead of an abstention.
        return {"shape": "too_few_rungs", "range": float(v.max() - v.min()) if len(v) else 0.0,
                "flat": None,
                "flat_reason": f"a shape needs at least 3 rungs; got {len(v)}"}
    diffs = np.diff(v)
    if np.all(diffs > 0):
        shape = "monotone_increasing"
    elif np.all(diffs < 0):
        shape = "monotone_decreasing"
    elif np.all(diffs == 0):
        shape = "constant"
    else:
        shape = "non_monotone"
    out: dict = {"shape": shape, "range": float(v.max() - v.min())}
    if ci_widths is not None and len(ci_widths) == len(values) and all(
            w is not None and np.isfinite(w) for w in ci_widths):
        median_ci = float(np.median(np.asarray(ci_widths, dtype=np.float64)))
        out["median_ci_width"] = median_ci
        out["flat"] = bool(out["range"] <= median_ci)
    else:
        out["flat"] = None
        out["flat_reason"] = ("no within-run CI on this metric, so an across-ladder "
                              "range cannot be compared against one run's own noise")
    return out


def _ci_width(ci: dict | None) -> float | None:
    if not isinstance(ci, dict):
        return None
    lo, hi = ci.get("lo"), ci.get("hi")
    if lo is None or hi is None:
        return None
    return float(hi) - float(lo)


def rung_metrics(run_dir: str | Path, ladder_model: str) -> dict:
    """Every ladder-comparable number one run directory holds for one model.

    Values are `{"value": float, "ci": {...}|None}` so the caller never has to
    know which artifacts happen to carry an interval. A metric whose stage did
    not run is absent from the returned dict rather than present as `None` --
    the difference matters downstream, where "the stage never ran" and "the
    stage ran and measured zero" must not plot as the same point.
    """
    run_dir = Path(run_dir)
    s = summarize_run(run_dir)
    out: dict = {"run_dir": str(run_dir), "label": s.get("label"),
                 "checkpoint": (s.get("checkpoints") or {}).get(ladder_model),
                 "provenance": s.get("provenance", {}),
                 "capture_coverage": (s.get("capture_coverage") or {}).get(ladder_model)}
    m: dict = {}

    if ladder_model in (s.get("l0_overall") or {}):
        m["mase"] = {"value": float(s["l0_overall"][ladder_model]), "ci": None}
    if s.get("l1_peak_cka") is not None:
        l1 = _safe_load_json(run_dir / "l1" / "meta.json") or {}
        m["l1_peak_cka"] = {"value": float(s["l1_peak_cka"]),
                            "ci": (l1.get("best_pair") or {}).get("ci")}
    for direction, gain in (s.get("l2_best_gain") or {}).items():
        if gain is not None:
            m[f"l2_gain[{direction}]"] = {"value": float(gain), "ci": None}
    if ladder_model in (s.get("crystallization_depth") or {}):
        cd = s["crystallization_depth"][ladder_model]
        if cd is not None:
            m["crystallization_depth"] = {"value": float(cd), "ci": None}
    if s.get("clustering_ami") is not None:
        cl = _safe_load_json(run_dir / "clustering" / "comparison.json") or {}
        m["clustering_ami"] = {"value": float(s["clustering_ami"]),
                               "ci": cl.get("ami")}

    prof = (_safe_load_json(run_dir / "internals" / "profile.json") or {}).get(ladder_model)
    if isinstance(prof, dict):
        eff = prof.get("effective_dim") or []
        if eff:
            m["effective_dim_mean"] = {"value": float(np.mean(eff)), "ci": None}
            m["effective_dim_peak"] = {"value": float(np.max(eff)), "ci": None}
        probe = prof.get("probe") or []
        vals = [p.get("value") for p in probe if isinstance(p, dict) and p.get("value") is not None]
        if vals:
            best = int(np.argmax(vals))
            m["probe_peak"] = {"value": float(vals[best]), "ci": probe[best]}
            chance = prof.get("chance")
            if chance is not None:
                m["probe_peak_over_chance"] = {"value": float(vals[best]) - float(chance),
                                               "ci": None}

    sae = _safe_load_json(run_dir / "sae" / "meta.json") or {}
    targets = [k for k in sae if k.split("/", 1)[0] == ladder_model]
    if len(targets) == 1:
        rec = sae[targets[0]]
        if rec.get("dead_feature_rate") is not None:
            m["sae_dead_rate"] = {"value": float(rec["dead_feature_rate"]), "ci": None}
        if rec.get("reconstruction_fidelity") is not None:
            m["sae_fidelity"] = {"value": float(rec["reconstruction_fidelity"]), "ci": None}
        gt = rec.get("ground_truth_alignment") or {}
        if gt.get("mean_abs_rho_matched") is not None:
            m["sae_gt_alignment"] = {"value": float(gt["mean_abs_rho_matched"]), "ci": None}
    elif len(targets) > 1:
        log.info(f"scaling_ladder: {run_dir} has {len(targets)} SAE targets for "
                 f"{ladder_model}; SAE metrics omitted (a ladder point must be one number)")

    calib = _safe_load_json(run_dir / "l0" / "calibration.json") or {}
    rec = (calib.get("models") or calib).get(ladder_model) if isinstance(calib, dict) else None
    if isinstance(rec, dict) and rec.get("mean_abs_coverage_error") is not None:
        m["calibration_error"] = {"value": float(rec["mean_abs_coverage_error"]), "ci": None}

    out["metrics"] = m
    return out


def ladder_size(run_dir: str | Path, ladder_model: str) -> float | None:
    """Measured parameter count for the ladder model, or None with a logged reason."""
    budget = _safe_load_json(Path(run_dir) / "budget" / "model_budget.json")
    rec = ((budget or {}).get("models") or {}).get(ladder_model)
    total = ((rec or {}).get("parameters") or {}).get("total")
    if total is None:
        log.warning(f"scaling_ladder: {run_dir} has no measured parameter count for "
                    f"{ladder_model} (budget stage not run); this rung is EXCLUDED -- "
                    f"the ladder axis is a measurement, never a checkpoint name")
        return None
    return float(total)


def run_scaling_ladder(run_dirs: list[str | Path], ladder_model: str) -> dict:
    """Reduce N run directories into one metric-vs-size ladder for `ladder_model`."""
    rungs, excluded = [], []
    for rd in run_dirs:
        size = ladder_size(rd, ladder_model)
        if size is None:
            excluded.append({"run_dir": str(rd),
                             "reason": "no measured parameter count (budget stage not run)"})
            continue
        rec = rung_metrics(rd, ladder_model)
        if not rec["metrics"]:
            excluded.append({"run_dir": str(rd), "reason": "no ladder-comparable metrics found"})
            continue
        rec["n_params"] = size
        rungs.append(rec)
    rungs.sort(key=lambda r: r["n_params"])

    metrics: dict = {}
    all_keys = sorted({k for r in rungs for k in r["metrics"]})
    for key in all_keys:
        pts = [(r["n_params"], r["metrics"][key], r["label"]) for r in rungs if key in r["metrics"]]
        sizes = [p[0] for p in pts]
        values = [p[1]["value"] for p in pts]
        widths = [_ci_width(p[1]["ci"]) for p in pts]
        entry = {
            "n_params": sizes,
            "values": values,
            "labels": [p[2] for p in pts],
            "ci": [p[1]["ci"] for p in pts],
            "n_rungs": len(pts),
            "missing_rungs": [r["label"] for r in rungs if key not in r["metrics"]],
        }
        entry.update(spearman_exact(sizes, values))
        entry.update(classify(sizes, values, widths))
        metrics[key] = entry

    coverage = {r["label"]: r["capture_coverage"] for r in rungs}
    return {
        "ladder_model": ladder_model,
        "rungs": [{"label": r["label"], "run_dir": r["run_dir"],
                   "checkpoint": r["checkpoint"], "n_params": r["n_params"]} for r in rungs],
        "excluded": excluded,
        "metrics": metrics,
        "capture_coverage": coverage,
        "coverage_is_constant": (len({v for v in coverage.values() if v is not None}) <= 1),
    }
