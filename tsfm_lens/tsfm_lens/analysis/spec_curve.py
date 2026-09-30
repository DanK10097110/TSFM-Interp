"""Specification curve over the analysis knobs (`ROADMAP.md` §34 item A2).

Every analysis knob in this repo is individually defended, and the defences
are good -- but none is swept, and this repo's own history is evidence that
matters: §9's layer-screen row records selector verdicts *flipping* when the
gold ranking changed (`CLAUDE.md` §11.18/§11.23), and §6.2's L1/L2
untrained-weights null work records a verdict flipping depending on which
layer of the null run it was read against. Knob sensitivity is a demonstrated
property of this pipeline, not a hypothetical this module has to argue for.

This is **not a pipeline stage** -- a standalone reducer over one already-
completed run directory, exactly the shape of `report/meta_report.py` and
`analysis/error_fingerprint.py`. It re-derives the corpus deterministically
from the run's own `config_resolved.yaml` (contexts are not persisted in the
activation store) and reads everything else -- predictions, lens curves,
layer-screen scores, attention head tables, per-block FLOPs -- from artifacts
already on disk. **No model is loaded and no forward pass is run.**

Scope, per A2.1: only knobs that need no re-extraction. Six are swept:

  alignment.depth_axis        index | block | compute | functional
  attention.resolution_mode   native | matched
  l0.scale                    mean_abs_diff | seasonal_naive
  stats.n_boot                configured value | 4x
  corpus_subsample_seed       3 seeds, 80% stratified draw
  layer_screen.method         work_bend | coverage | factor_emergence

Four claim families, one per group of `report/findings.json` claim_ids that
actually depends on one of the six knobs above -- everything else in a run's
findings is recorded as excluded, with the specific reason, rather than
silently left out of the denominator (A2's own stated failure mode: "Silently
sweeping a subset and reporting '11 of 11 robust' is the failure this bullet
exists to prevent"):

  l0_strength_set       which families a model is significantly stronger on
                         (matches the `l0.*` "is significantly stronger on/
                         than" findings). Applicable: l0.scale, stats.n_boot,
                         corpus_subsample_seed.
  lens_crystallization  whether a model's forecast crystallizes within
                         tolerance at all (matches `lens.*`). Applicable:
                         alignment.depth_axis only -- the VALUE of a
                         crystallization depth is axis-dependent by
                         definition (the same layer gets a different
                         coordinate under `index` vs `block`), so the only
                         fair thing to compare across axes is the qualitative
                         state (crystallizes / never crystallizes within the
                         configured tolerance), not the raw number.
  layer_screen_selection which layers a selector method picks (matches
                         `layer_screen.*`). Applicable: layer_screen.method
                         only. Recomputed against the run's own MAIN store
                         (already on disk), not a fresh stride-1 screening
                         extraction -- `CLAUDE.md` §34.0's premise that this
                         knob is "free" turned out to be wrong as literally
                         stated (the dedicated screening store is deleted
                         after selection by default, `layer_screen.keep_store:
                         false`); re-running `coverage`/`factor_emergence`
                         the way the pipeline stage does would need a second
                         extraction pass, which A2.1 rules out of scope. What
                         IS free is re-scoring the SAME three selectors
                         against the main store's already-captured (possibly
                         strided) layers -- exactly what `run_layer_screen`
                         itself falls back to when `require_full_capture:
                         false`. Every cell here is therefore marked
                         `fair_to_all_layers: false`, an honest, stated
                         deviation from the production selection's own
                         fairness guarantee (`ROADMAP.md` §15 A1) -- see this
                         module's own Findings for the corrected premise.
  attention_periodicity  which head is the strongest periodicity head
                         (matches `attention.*`). Applicable:
                         attention.resolution_mode only. Both resolutions are
                         *always* computed and persisted in
                         `attention/meta.json` (`ROADMAP.md` §18 F5), so this
                         one needs no recomputation at all -- just reading
                         two keys that are already on disk.

A2.3: never render the best cell, only the fraction. A claim's `robust_frac`
is computed over its *applicable* cells only -- a `not_applicable` cell is
excluded from the denominator, never counted as robust (A2's own failure-mode
bullet). `evidence_class` for every claim here is `descriptive`: this module
answers "how much does the reported conclusion depend on an analysis choice",
never "is this difference real" -- reading a specification curve as a
significance test is the specific misuse A2 warns against.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import PipelineConfig, load_config
from ..data import BenchmarkData, load_benchmark
from ..extraction.store import ActivationStore, load_meta, nonfinite_series_mask
from ..utils import log, sample_rows, save_json
from .depth_axis import AXES as DEPTH_AXES
from .depth_axis import depth_axis_for_run
from .l0_behavioral import _score, _summarize
from .layer_screen import select_layers
from .lens import crystallization_depths

RESOLUTION_MODE_VALUES = ("native", "matched")
L0_SCALE_VALUES = ("mean_abs_diff", "seasonal_naive")
N_BOOT_MULTIPLIERS = (1, 4)
CORPUS_SUBSAMPLE_SEEDS = (0, 1, 2)
LAYER_SCREEN_METHODS = ("work_bend", "coverage", "factor_emergence")
CORPUS_SUBSAMPLE_FRACTION = 0.8

KNOB_GRID = {
    "alignment.depth_axis": DEPTH_AXES,
    "attention.resolution_mode": RESOLUTION_MODE_VALUES,
    "l0.scale": L0_SCALE_VALUES,
    "stats.n_boot": N_BOOT_MULTIPLIERS,
    "corpus_subsample_seed": CORPUS_SUBSAMPLE_SEEDS,
    "layer_screen.method": LAYER_SCREEN_METHODS,
}


@dataclass
class ClaimCell:
    """One (knob, grid value) cell for one claim."""

    knob: str
    grid_value: object
    status: str  # "ok" | "not_applicable"
    value: object = None
    robust: Optional[bool] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {"knob": self.knob, "grid_value": self.grid_value, "status": self.status,
                "value": self.value, "robust": self.robust, "reason": self.reason}


@dataclass
class ClaimSweep:
    """One headline claim, swept across every knob that can apply to it."""

    claim_id: str
    matched_finding_claim_id: Optional[str]
    family: str
    stage: str
    baseline_value: object
    cells: list = field(default_factory=list)

    def to_dict(self) -> dict:
        applicable = [c for c in self.cells if c.status == "ok"]
        n_robust = sum(1 for c in applicable if c.robust)
        return {
            "claim_id": self.claim_id,
            "matched_finding_claim_id": self.matched_finding_claim_id,
            "family": self.family,
            "stage": self.stage,
            "evidence_class": "descriptive",
            "baseline_value": self.baseline_value,
            "cells": [c.to_dict() for c in self.cells],
            "n_applicable": len(applicable),
            "n_robust": n_robust,
            "robust_frac": (n_robust / len(applicable)) if applicable else None,
        }


def _match_finding_claim_id(findings: list, *, contains: list) -> Optional[str]:
    """First `findings.json` entry whose text contains every string in `contains`.

    Best-effort, by design: A2 is not the source of truth for a claim's
    wording, `report.py` is. A miss returns `None` rather than fabricating an
    id, and the caller records the claim under a synthetic id instead
    (`CLAUDE.md` §2.5 -- degrade loudly, never invent).
    """
    for f in findings:
        text = f.get("text", "")
        if all(s in text for s in contains):
            return f.get("claim_id")
    return None


def _load_findings(run_dir: Path) -> list:
    import json

    p = run_dir / "report" / "findings.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("findings", [])


# ---------------------------------------------------------------------------
# Claim family 1 -- L0 family-strength sets
# ---------------------------------------------------------------------------

def _l0_recompute(cfg: PipelineConfig, store: ActivationStore, data: BenchmarkData,
                  models: list, *, scale: str, n_boot_mult: int,
                  row_seed: Optional[int]) -> dict:
    """`_summarize`'s output, re-scored under one (scale, n_boot, seed) cell.

    Zero forward passes: every model's forecast is already in `store` (L0
    already ran), and `_score` is the exact same per-series metric function
    `run_l0` calls -- reused, not re-implemented (`CLAUDE.md` §2.2). Row
    subsampling (`row_seed`) draws a stratified `CORPUS_SUBSAMPLE_FRACTION`
    of the corpus, matching the family-composition discipline `sample_rows`
    already enforces elsewhere (`ROADMAP.md` §15 A4) rather than a plain
    head slice.
    """
    contexts, targets, meta = data.contexts(), data.targets(), data.meta
    rows = None
    if row_seed is not None:
        k = max(2, round(CORPUS_SUBSAMPLE_FRACTION * data.n))
        rows = sample_rows(data.n, k, row_seed, strata=data.families)
        contexts, targets = contexts[rows], targets[rows]
        meta = meta.iloc[rows].reset_index(drop=True)
    frames = []
    for name in models:
        preds = store.load_predictions(name)
        point, quants = preds["point"], preds["quantiles"]
        if rows is not None:
            point, quants = point[rows], quants[rows]
        frames.append(_score(name, point, quants, contexts, targets, cfg.l0.quantiles,
                             meta, scale, cfg.l0.min_scale_frac))
    metrics = pd.concat(frames, ignore_index=True)
    ids_full = data.meta["series_id"].to_numpy()
    bad_ids = np.unique(np.concatenate([
        ids_full[nonfinite_series_mask(d["point"], d["quantiles"])]
        for d in (store.load_predictions(name) for name in models)]))
    if bad_ids.size:
        log.warning("spec_curve: dropping %d series with a non-finite stored forecast from "
                    "every model's recomputed L0 (same rule as L0's "
                    "l0/nonfinite_forecasts.json)", bad_ids.size)
        metrics = metrics[~metrics["series_id"].isin(bad_ids)].reset_index(drop=True)
    cfg2 = copy.deepcopy(cfg)
    cfg2.l0.scale = scale
    cfg2.stats.n_boot = cfg.stats.n_boot * n_boot_mult
    return _summarize(metrics, cfg2, noise_floor=None)


def _strength_sets(summary: dict) -> dict:
    """{(a, b, model): sorted(families)} for every non-empty strength list.

    Covers both `report.py` call sites that generate an `l0.*` "significantly
    stronger" finding: the designated pair's top-level `summary["strengths"]`
    and every other pair's `summary["pairwise"][i]["strengths"]`.
    """
    out = {}
    designated = tuple((summary.get("multiplicity") or {}).get("designated_pair") or ())
    for model, fams in (summary.get("strengths") or {}).items():
        if fams:
            out[(designated[0] if designated else "?",
                designated[1] if designated else "?", model)] = sorted(fams)
    for entry in summary.get("pairwise", []):
        a, b = entry["a"], entry["b"]
        if (a, b) == designated:
            continue
        for model, fams in (entry.get("strengths") or {}).items():
            if fams:
                out[(a, b, model)] = sorted(fams)
    return out


def sweep_l0_family_strength(run_dir: Path, cfg: PipelineConfig, store: ActivationStore,
                             data: BenchmarkData, findings: list) -> list:
    """One `ClaimSweep` per (pair, model) with a non-empty baseline strength set."""
    models = [m.name for m in cfg.models if store.has_predictions(m.name)]
    if len(models) < 2:
        return []
    baseline_summary = _l0_recompute(cfg, store, data, models, scale=cfg.l0.scale,
                                     n_boot_mult=1, row_seed=None)
    baseline_sets = _strength_sets(baseline_summary)
    if not baseline_sets:
        return []

    cache: dict = {}

    def recompute(scale, n_boot_mult, row_seed):
        key = (scale, n_boot_mult, row_seed)
        if key not in cache:
            cache[key] = _strength_sets(
                _l0_recompute(cfg, store, data, models, scale=scale,
                              n_boot_mult=n_boot_mult, row_seed=row_seed))
        return cache[key]

    out = []
    for (a, b, model), baseline in baseline_sets.items():
        other = b if model == a else a
        finding_id = (_match_finding_claim_id(
            findings, contains=[f"{model} is significantly stronger on: "])
            if (a, b) == tuple((baseline_summary.get("multiplicity") or {})
                               .get("designated_pair") or ()) else
            _match_finding_claim_id(
                findings, contains=[f"{model} is significantly stronger than {other} on: "]))
        claim = ClaimSweep(
            claim_id=f"spec.l0_strength.{a}.{b}.{model}",
            matched_finding_claim_id=finding_id, family="l0_strength_set", stage="l0",
            baseline_value=baseline)
        for scale in L0_SCALE_VALUES:
            sets = recompute(scale, 1, None)
            val = sets.get((a, b, model), [])
            claim.cells.append(ClaimCell("l0.scale", scale, "ok", val, set(val) == set(baseline)))
        for mult in N_BOOT_MULTIPLIERS:
            sets = recompute(cfg.l0.scale, mult, None)
            val = sets.get((a, b, model), [])
            claim.cells.append(ClaimCell("stats.n_boot", f"{mult}x", "ok", val,
                                         set(val) == set(baseline)))
        for seed in CORPUS_SUBSAMPLE_SEEDS:
            sets = recompute(cfg.l0.scale, 1, seed)
            val = sets.get((a, b, model), [])
            claim.cells.append(ClaimCell("corpus_subsample_seed", seed, "ok", val,
                                         set(val) == set(baseline)))
        for knob in ("alignment.depth_axis", "attention.resolution_mode", "layer_screen.method"):
            for gv in KNOB_GRID[knob]:
                claim.cells.append(ClaimCell(
                    knob, gv, "not_applicable", reason=
                    "L0 family-strength verdicts are computed from stored predictions and "
                    "per-series MASE alone; this knob does not enter that computation."))
        out.append(claim)
    return out


# ---------------------------------------------------------------------------
# Claim family 2 -- lens crystallization state
# ---------------------------------------------------------------------------

def sweep_lens_crystallization(run_dir: Path, cfg: PipelineConfig,
                               store: ActivationStore, findings: list) -> list:
    """One `ClaimSweep` per model whose skip lens ran, over `alignment.depth_axis`."""
    import json

    lens_path = run_dir / "lens" / "lens.json"
    curves_path = run_dir / "lens" / "curves.npz"
    if not lens_path.exists() or not curves_path.exists():
        return []
    lens_meta = json.loads(lens_path.read_text(encoding="utf-8"))
    curves = np.load(curves_path)
    budget_path = run_dir / "budget" / "model_budget.json"
    budget_all = (json.loads(budget_path.read_text(encoding="utf-8")).get("models", {})
                 if budget_path.exists() else {})

    out = []
    for model, meta in lens_meta.items():
        if not meta.get("skip_lens_available"):
            continue
        layers = meta["layers"]
        skip_key, tuned_key = f"skip_mase_{model}", f"tuned_r2_model_{model}"
        if skip_key not in curves.files:
            continue
        skip_mase = curves[skip_key]
        final_mase = meta["final_mase"]
        tol = meta["crystallization_tol"]
        baseline_depth = meta.get("crystallization_depth")
        baseline_state = "crystallizes" if baseline_depth is not None else "never"
        finding_id = _match_finding_claim_id(
            findings, contains=[f"{model}: forecast crystallizes"])
        claim = ClaimSweep(claim_id=f"spec.lens_crystallization.{model}",
                           matched_finding_claim_id=finding_id,
                           family="lens_crystallization", stage="lens",
                           baseline_value=baseline_state)
        for axis in DEPTH_AXES:
            functional_values = curves[tuned_key] if axis == "functional" and tuned_key in curves.files else None
            if axis == "functional" and functional_values is None:
                claim.cells.append(ClaimCell(
                    "alignment.depth_axis", axis, "not_applicable", reason=
                    "no tuned-lens R^2 persisted for this model (lens.tuned was off), and the "
                    "functional axis needs a measured per-layer property to align on."))
                continue
            budget = budget_all.get(model) if axis == "compute" else None
            if axis == "compute" and not budget:
                claim.cells.append(ClaimCell(
                    "alignment.depth_axis", axis, "not_applicable", reason=
                    "budget/model_budget.json is missing from this run -- the compute depth "
                    "axis has no per-block FLOPs to normalize by (this is A2's own load-bearing "
                    "negative: a missing budget artifact must not silently fall back to another "
                    "axis and report a robust result)."))
                continue
            da = depth_axis_for_run(axis, store, model, layers, adapter=None,
                                    budget=budget, functional_values=functional_values)
            depth = crystallization_depths(skip_mase, final_mase, tol, da.coords)
            state = "crystallizes" if depth is not None else "never"
            claim.cells.append(ClaimCell(
                "alignment.depth_axis", axis, "ok",
                {"state": state, "rel_depth": depth, "axis_used": da.axis,
                "degraded_from": da.fallback_from},
                state == baseline_state))
        for knob in ("attention.resolution_mode", "l0.scale", "stats.n_boot",
                    "corpus_subsample_seed", "layer_screen.method"):
            for gv in KNOB_GRID[knob]:
                claim.cells.append(ClaimCell(
                    knob, gv, "not_applicable", reason=
                    "crystallization depth is a deterministic skip-lens-MASE threshold "
                    "crossing computed from already-stored per-layer curves; it has no "
                    "bootstrap, scale-mode, subsampling, attention, or layer-selection "
                    "dependency."))
        out.append(claim)
    return out


# ---------------------------------------------------------------------------
# Claim family 3 -- layer_screen selection
# ---------------------------------------------------------------------------

def sweep_layer_screen_selection(run_dir: Path, cfg: PipelineConfig,
                                 store: ActivationStore, findings: list) -> list:
    """One `ClaimSweep` per model, over `layer_screen.method`.

    Recomputed against `store` (the main analysis store), not a fresh
    stride-1 screening extraction -- see this module's docstring and its own
    Findings for why `CLAUDE.md` §34.0's "free" premise needed a correction
    here. `fair_to_all_layers` is always `False` in this sweep's cells.
    """
    import json

    sel_path = run_dir / "layer_screen" / "selection.json"
    if not sel_path.exists():
        return []
    baseline_sel = json.loads(sel_path.read_text(encoding="utf-8"))

    gt, gt_cols, series_ids_full = None, None, None
    if "factor_emergence" in LAYER_SCREEN_METHODS and cfg.data.source == "sealed" and cfg.data.path:
        try:
            from ..sae.ground_truth import load_ground_truth_table
            gt = load_ground_truth_table(cfg.data.path)
            gt_cols = [c for c in gt.columns if c != "generator"]
        except Exception as e:
            log.warning("spec_curve: could not load ground truth from %r (%s); "
                       "factor_emergence cells will be not_applicable", cfg.data.path, e)

    full_meta = load_meta(run_dir)
    families = full_meta["family"].to_numpy()
    seed = cfg.layer_screen.seed if cfg.layer_screen.seed is not None else cfg.run.seed
    rows = sample_rows(len(full_meta), cfg.layer_screen.max_series, seed, strata=families)
    series_ids = full_meta["series_id"].to_numpy()[rows]

    out = []
    for model, base in baseline_sel.items():
        layers = store.layers(model)
        if not layers:
            continue
        baseline_selected = set(base.get("selected", []))
        budget = max(cfg.layer_screen.min_budget,
                    round(cfg.layer_screen.budget_frac * len(layers)))
        budget = min(budget, len(layers))
        finding_id = _match_finding_claim_id(
            findings, contains=[f"{model}: {base.get('method', '')} selected"])
        claim = ClaimSweep(claim_id=f"spec.layer_screen.{model}",
                           matched_finding_claim_id=finding_id,
                           family="layer_screen_selection", stage="layer_screen",
                           baseline_value=sorted(baseline_selected))
        for method in LAYER_SCREEN_METHODS:
            if method == "factor_emergence" and gt is None:
                claim.cells.append(ClaimCell(
                    "layer_screen.method", method, "not_applicable", reason=
                    "factor_emergence needs a sealed corpus with ground truth; "
                    f"this run's data.source is {cfg.data.source!r} or the ground-truth "
                    "table could not be loaded."))
                continue
            kwargs = {"device": None, "seed": seed, "rows": rows,
                     "use_curvature": cfg.layer_screen.use_curvature,
                     "min_gap": cfg.layer_screen.min_gap}
            if method == "factor_emergence":
                kwargs.update(gt=gt, series_ids=series_ids, gt_cols=gt_cols)
            try:
                sel = select_layers(method, store, model, layers, budget, **kwargs)
            except Exception as e:
                claim.cells.append(ClaimCell(
                    "layer_screen.method", method, "not_applicable",
                    reason=f"selector raised on this store: {e}"))
                continue
            selected = set(sel["selected"])
            inter = len(selected & baseline_selected)
            union = len(selected | baseline_selected) or 1
            claim.cells.append(ClaimCell(
                "layer_screen.method", method, "ok",
                {"selected": sorted(selected), "jaccard_vs_baseline": inter / union,
                "fair_to_all_layers": False},
                selected == baseline_selected))
        for knob in ("alignment.depth_axis", "attention.resolution_mode", "l0.scale",
                    "stats.n_boot", "corpus_subsample_seed"):
            for gv in KNOB_GRID[knob]:
                claim.cells.append(ClaimCell(
                    knob, gv, "not_applicable", reason=
                    "layer selection depends only on the configured selector method and the "
                    "store's own activations; it has no depth-axis, attention-resolution, "
                    "L0-scale, bootstrap, or corpus-subsampling dependency."))
        out.append(claim)
    return out


# ---------------------------------------------------------------------------
# Claim family 4 -- attention periodicity head identity
# ---------------------------------------------------------------------------

def sweep_attention_periodicity(run_dir: Path, findings: list) -> list:
    """One `ClaimSweep` per model, over `attention.resolution_mode`.

    Both resolutions are always computed and persisted (`ROADMAP.md` §18 F5),
    so this is a pure read -- no recomputation, no model, no store.
    """
    import json

    p = run_dir / "attention" / "meta.json"
    if not p.exists():
        return []
    meta = json.loads(p.read_text(encoding="utf-8"))

    def _top_head(entry, key):
        heads = ((entry.get("patterns") or {}).get(key) or {}).get("top_periodicity_heads") or []
        if not heads:
            return None
        return (heads[0]["layer"], heads[0]["head"])

    out = []
    for model, entry in meta.items():
        native, matched = _top_head(entry, "head_scores"), _top_head(entry, "head_scores_matched")
        if native is None and matched is None:
            continue
        baseline = {"native": native, "matched": matched}["matched"] \
            if matched is not None else native
        finding_id = _match_finding_claim_id(
            findings, contains=[f"{model}: at the resolution-matched"]) or \
            _match_finding_claim_id(findings, contains=[f"{model}: strongest periodicity head"])
        claim = ClaimSweep(claim_id=f"spec.attention_periodicity.{model}",
                           matched_finding_claim_id=finding_id,
                           family="attention_periodicity", stage="attention",
                           baseline_value=list(baseline) if baseline else None)
        for mode, val in (("native", native), ("matched", matched)):
            if val is None:
                claim.cells.append(ClaimCell(
                    "attention.resolution_mode", mode, "not_applicable",
                    reason="no periodicity heads recorded at this resolution for this model."))
                continue
            claim.cells.append(ClaimCell("attention.resolution_mode", mode, "ok",
                                         list(val), val == baseline))
        for knob in ("alignment.depth_axis", "l0.scale", "stats.n_boot",
                    "corpus_subsample_seed", "layer_screen.method"):
            for gv in KNOB_GRID[knob]:
                claim.cells.append(ClaimCell(
                    knob, gv, "not_applicable", reason=
                    "which head scores highest for excess seasonal mass depends only on the "
                    "attention patterns already captured and the lag-binning resolution; it has "
                    "no depth-axis, L0-scale, bootstrap, corpus-subsampling, or layer-selection "
                    "dependency."))
        out.append(claim)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_spec_curve(run_dir: Path) -> dict:
    """Sweep every applicable knob against one already-completed run's claims.

    Reads `config_resolved.yaml`, re-derives the corpus deterministically
    (`load_benchmark`, no model load), opens the activation store read-only,
    and runs all four claim-family sweeps. Writes `spec_curve/results.json`.
    """
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config_resolved.yaml")
    findings = _load_findings(run_dir)

    claims: list = []
    store_path = run_dir / "activations.zarr"
    if store_path.exists():
        store = ActivationStore(store_path, mode="r")
        data = load_benchmark(cfg.data, cfg.run.seed)
        claims += sweep_l0_family_strength(run_dir, cfg, store, data, findings)
        claims += sweep_lens_crystallization(run_dir, cfg, store, findings)
        claims += sweep_layer_screen_selection(run_dir, cfg, store, findings)
    else:
        log.warning("spec_curve: %s has no activations.zarr; skipping every "
                   "activation-store-dependent claim family", run_dir)
    claims += sweep_attention_periodicity(run_dir, findings)

    claim_dicts = [c.to_dict() for c in claims]
    fully_robust = [c for c in claim_dicts if c["n_applicable"] and c["robust_frac"] == 1.0]
    not_fully_robust = [c for c in claim_dicts if c["n_applicable"] and c["robust_frac"] < 1.0]
    excluded = [c for c in claim_dicts if not c["n_applicable"]]
    out = {
        "run_dir": str(run_dir),
        "grid": {k: list(v) for k, v in KNOB_GRID.items()},
        "grid_mode": "one_knob_at_a_time",
        "claims": claim_dicts,
        "summary": {
            "n_claims": len(claim_dicts),
            "n_fully_robust": len(fully_robust),
            "n_not_fully_robust": len(not_fully_robust),
            "n_excluded_claims": len(excluded),
            "excluded_claim_ids": [c["claim_id"] for c in excluded],
        },
    }
    save_json(run_dir / "spec_curve" / "results.json", out)
    log.info("spec_curve: %d claims swept (%d fully robust, %d not fully robust, %d excluded)",
             len(claim_dicts), len(fully_robust), len(not_fully_robust), len(excluded))
    return out
