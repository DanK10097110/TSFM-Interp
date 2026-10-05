"""ROADMAP.md sec 37 Spec A -- per-concept PROFILES for the cross-model SAE
concept ATLAS (`sae/concept_atlas.py`).

An atlas concept is defined purely by its EFFECT on the forecast: a cluster
of causal features, from possibly several models and layers, whose 9-channel
ablation fingerprints point the same way. That leaves three questions a
reader asks next, none of which the atlas itself answers:

  1. What INPUTS make the concept fire? (correlational -- Spearman against
     the corpus's own generative ground truth, `sae/ground_truth.py`.)
  2. Do the models that share the concept's EFFECT also share its INPUTS?
     (correlational -- a stratum-matched permutation test, reusing
     `sae/transfer.py`'s own machinery, never re-derived.)
  3. Does the model that has the concept forecast BETTER because of it?
     (behavioral for the raw gap; within-model CAUSAL, and only for that one
     model, when the concept's own ablation battery already moved MASE on
     the same series -- never inferred from the gap alone.)

`ROADMAP.md` sec 37's own measured facts motivate all three: effect-space
concepts often fire on different inputs per model (atlas C6), many features
are provenance detectors that must be labelled as such rather than reported
as structural discoveries, and a model can hold a uniquely interpretable
concept without that concept explaining why it wins (TimesFM's C17 fires
cleanly on random walks yet TimesFM is not the best model on that family) --
this module's whole job is to let a reader check that link instead of
assuming it.

EVIDENCE CLASSES, stated once here because every consumer of this artifact
inherits them:
  - input profile: correlational (Spearman, permutation-null corrected).
  - input agreement: correlational (a stratum-matched permutation test).
  - effect profile: causal within-model (the ablation battery already
    established this; this module only reduces it to a part's own mean).
  - the behavioral gap between models on a concept's top-firing series:
    behavioral (an observed MASE difference, no manipulation).
  - the link between that gap and the concept: correlational, UNLESS the
    concept's own ablation battery moves MASE on those same series, which
    makes it a within-model causal statement about that one model only.

Nothing here re-derives clustering, ablation arithmetic or the transfer
machinery: `pooled_features`/`CHANNELS` come from `concept_atlas.py`/
`concepts.py` unchanged, and every permutation test reuses
`transfer.py`'s `series_strata`/`concept_scores`/`top_series`/`_seed`/
`benjamini_hochberg`/`_by_stratum` and `concept_atlas.py`'s own
`_right_tail_p` floor convention. Structural-field residualization against
corpus provenance (tier and real-derived generator dummies, never archetype
or synthetic generators: those are structural recipes, and residualizing on
archetypes measured 48 of 55 parts as provenance-driven on the 4-model run
by gutting every structural rho) reuses `ground_truth.py::residualize_against_provenance`
directly, including its own degenerate-residual gate (a field the
provenance one-hots predict almost perfectly leaves nothing but rounding
noise to correlate against, `ground_truth.py` sec 26 A3) -- reusing that
gate here is deliberate, not incidental: this module's whole
`provenance_driven` question would be silently wrong on exactly the fields
that gate exists for.

Degrade loudly, never silently (`CLAUDE.md` sec 2.5): a part whose target
never had its SAE features persisted, a concept with no `concept_stability.json`
or `atlas_transfer.json` on disk, or a behavioral link with too few series to
bootstrap, all record `"not measured: <reason>"` rather than a fabricated
False/0.

Everything here is deterministic given the run directory and config: every
random draw is seeded via `transfer.py::_seed` off `cfg.sae.transfer_seed`,
never Python's builtin `hash()`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, rankdata, spearmanr

from ..analysis.stats import mean_ci
from ..extraction.store import ActivationStore, load_meta
from ..utils import load_json, log, save_json
from .ablation_run import ablation_path
from .concept_atlas import _right_tail_p, pooled_features
from .concepts import CHANNELS
from .ground_truth import (is_provenance_field, load_ground_truth_table,
                           residualize_against_provenance)
from .transfer import (_by_stratum, _seed, benjamini_hochberg, concept_scores, matched_draws,
                       series_strata, top_series)

__all__ = ["run_concept_profiles"]

SCHEMA_VERSION = 1

_STRUCTURAL_FIELDS = (
    "trend_order", "trend_scale", "n_seasonalities", "seasonal_amplitude_max",
    "seasonal_period_dominant", "ar_order", "ar_coeff_sum", "noise_scale",
    "n_changepoints", "n_anomalies", "has_random_walk", "has_intermittency",
    "has_heteroskedastic",
)
_REAL_DERIVED_GENERATORS = ("mixture", "block_bootstrap", "sequential_par")
_MIN_RESIDUAL_SCALE = 0.01
_PROVENANCE_SHARE_THRESHOLD = 0.8
_ENRICHMENT_Q = 0.05
_AGREEMENT_Q = 0.05
_MASE_CHANNEL_IDX = CHANNELS.index("mase")

EVIDENCE_CLASSES = {
    "input_profile": "correlational",
    "input_agreement": "correlational",
    "effect_profile": "causal within-model",
    "behavioral_gap": "behavioral",
    "behavioral_link": ("correlational, unless the concept's own ablation moves MASE "
                        "on those series -- then a within-model causal statement about "
                        "that model only"),
}


# ---------------------------------------------------------------------------
# Parts: an atlas concept, split by the target(s) its member features live at.
# ---------------------------------------------------------------------------

def _concept_parts(atlas: dict) -> list:
    """One row per (atlas concept, target it has members at), carrying both
    the member feature ids AND their row indices into `atlas["rows"]`/the
    pooled matrix `X` this module recomputes -- unlike
    `sae/stability.py::_atlas_parts`, which only needs the feature ids and
    therefore drops the row index, this module needs both (the row index to
    slice `X` for the effect profile, the feature id to look the member back
    up in its own raw ablation-battery candidate record).
    """
    rows = atlas.get("rows", [])
    out: list = []
    for concept in atlas.get("concepts", []):
        cid = int(concept["concept"])
        by_target: dict = {}
        for idx in concept.get("members", []):
            r = rows[idx]
            key = f"{r['model']}/{r['layer']}"
            by_target.setdefault(key, {"model": r["model"], "layer": r["layer"],
                                       "row_idx": [], "feature_ids": []})
            by_target[key]["row_idx"].append(int(idx))
            by_target[key]["feature_ids"].append(int(r["feature"]))
        for target, rec in sorted(by_target.items()):
            order = np.argsort(rec["feature_ids"])
            out.append({
                "concept": cid, "target": target, "model": rec["model"], "layer": rec["layer"],
                "row_idx": [rec["row_idx"][i] for i in order],
                "feature_ids": [rec["feature_ids"][i] for i in order],
            })
    return out


def _ablation_candidates(run_dir: Path, target: str, cache: dict) -> dict:
    """`{feature_id: candidate}` for one target's raw ablation-battery file,
    cached across parts that share a target (several concepts can each have
    a part at the same target)."""
    if target not in cache:
        model, layer = target.split("/", 1)
        art = load_json(ablation_path(run_dir, model, layer))
        cache[target] = {int(c["feature"]): c for c in (art.get("candidates") or [])}
    return cache[target]


# ---------------------------------------------------------------------------
# Input profile: structural fields, provenance, enrichment.
# ---------------------------------------------------------------------------

def _structural_field_records(s: np.ndarray, joined: pd.DataFrame,
                              provenance_cols: list, seed: int) -> list:
    """One record per `_STRUCTURAL_FIELDS` entry: raw Spearman rho, the
    provenance-residualized rho (`None` with a reason when the field is not
    separable from provenance, `ground_truth.py`'s own degenerate-residual
    gate), the out-of-fold R2 the provenance basis achieves predicting the
    field, and the valid sample size."""
    out = []
    for field in _STRUCTURAL_FIELDS:
        gvals = joined[field].to_numpy(dtype=np.float64)
        valid = ~np.isnan(gvals)
        n = int(valid.sum())
        if n < 2 or np.std(gvals[valid]) == 0 or np.std(s[valid]) == 0:
            out.append({"field": field, "raw_rho": None, "resid_rho": None,
                       "oof_r2": None, "residual_scale": None, "n": n,
                       "reason": "fewer than 2 valid rows, or no variance in the field or score"})
            continue
        raw_rho, _ = spearmanr(s[valid], gvals[valid])
        raw_rho = float(raw_rho) if np.isfinite(raw_rho) else None
        varying = [c for c in provenance_cols
                   if np.nanstd(joined[c].to_numpy(dtype=np.float64)[valid]) > 0]
        if not varying:
            out.append({"field": field, "raw_rho": raw_rho, "resid_rho": raw_rho,
                       "oof_r2": 0.0, "residual_scale": 1.0, "n": n,
                       "reason": "provenance is constant on this field's valid rows; not residualized"})
            continue
        resid, oof_r2 = residualize_against_provenance(
            joined, field, provenance_cols, valid, seed=seed)
        field_scale = float(np.std(gvals[valid]))
        resid_scale = float(np.std(resid) / field_scale) if field_scale > 0 else 0.0
        if resid_scale < _MIN_RESIDUAL_SCALE:
            out.append({"field": field, "raw_rho": raw_rho, "resid_rho": None,
                       "oof_r2": float(oof_r2), "residual_scale": resid_scale, "n": n,
                       "reason": (f"not separable from provenance (oof_r2={oof_r2:.4f}, "
                                 f"residual keeps {resid_scale:.2e} of the field's own scale)")})
            continue
        resid_rho, _ = spearmanr(s[valid], resid)
        resid_rho = float(resid_rho) if np.isfinite(resid_rho) else None
        out.append({"field": field, "raw_rho": raw_rho, "resid_rho": resid_rho,
                   "oof_r2": float(oof_r2), "residual_scale": resid_scale, "n": n, "reason": ""})
    return out


def _p_max_structural(s: np.ndarray, joined: pd.DataFrame, field_records: list,
                      real_max_abs_raw_rho: float, seed: int, n_perm: int) -> float:
    """Search-corrected permutation p for "the largest |raw rho| over the 13
    structural fields is at least this large by chance", floored at
    `1/(n_perm+1)` (`concept_atlas.py::_right_tail_p`'s own convention,
    reused rather than re-derived).

    `s` is permuted only among the rows where ANY structural field is
    defined (the corpus's synthetic tier -- real-derived series carry no
    structural ground truth at all, `ground_truth.py`'s own module
    docstring); every field's own valid subset is a SUBSET of that set, so
    permuting there and re-slicing per field is exact, not an approximation.
    """
    fields = [r["field"] for r in field_records if r["raw_rho"] is not None]
    if not fields:
        return float("nan")
    synthetic_mask = joined[list(_STRUCTURAL_FIELDS)].notna().any(axis=1).to_numpy()
    synth_idx = np.flatnonzero(synthetic_mask)
    if synth_idx.size < 2:
        return float("nan")

    field_valid: dict = {}
    field_rank: dict = {}
    for field in fields:
        gvals = joined[field].to_numpy(dtype=np.float64)
        valid = ~np.isnan(gvals)
        field_valid[field] = valid
        field_rank[field] = rankdata(gvals[valid])

    rng = np.random.default_rng(seed)
    s_perm = s.copy()
    null_max = np.empty(n_perm, dtype=np.float64)
    for p in range(n_perm):
        perm = rng.permutation(synth_idx.size)
        s_perm[synth_idx] = s[synth_idx][perm]
        best = 0.0
        for field in fields:
            valid = field_valid[field]
            sub = s_perm[valid]
            if np.std(sub) == 0:
                continue
            rank_s = rankdata(sub)
            a = rank_s - rank_s.mean()
            b = field_rank[field] - field_rank[field].mean()
            denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
            if denom > 0:
                best = max(best, abs(float(np.sum(a * b) / denom)))
        null_max[p] = best
    return _right_tail_p(real_max_abs_raw_rho, null_max)


def _provenance_profile(s: np.ndarray, joined: pd.DataFrame) -> dict:
    """`tier_synthetic`'s own rho, and the largest |rho| among the
    real-derived `generator_*` dummies -- the two provenance signals
    `provenance_driven` below reads (raw correlation). Synthetic generators
    and archetypes are excluded: they are recipes for structure, so a
    feature tracking them is tracking structure, not corpus origin."""
    out = {"tier_synthetic_rho": None, "max_generator_rho": None, "max_generator_field": None}
    if "tier_synthetic" in joined.columns:
        gvals = joined["tier_synthetic"].to_numpy(dtype=np.float64)
        valid = ~np.isnan(gvals)
        if valid.sum() >= 2 and np.std(gvals[valid]) > 0 and np.std(s[valid]) > 0:
            rho, _ = spearmanr(s[valid], gvals[valid])
            out["tier_synthetic_rho"] = float(rho) if np.isfinite(rho) else None
    best_field, best_rho = None, 0.0
    real_derived_cols = {f"generator_{g}" for g in _REAL_DERIVED_GENERATORS}
    for col in joined.columns:
        if col not in real_derived_cols:
            continue
        gvals = joined[col].to_numpy(dtype=np.float64)
        valid = ~np.isnan(gvals)
        if valid.sum() < 2 or np.std(gvals[valid]) == 0 or np.std(s[valid]) == 0:
            continue
        rho, _ = spearmanr(s[valid], gvals[valid])
        if np.isfinite(rho) and abs(rho) > abs(best_rho):
            best_field, best_rho = col, float(rho)
    out["max_generator_field"] = best_field
    out["max_generator_rho"] = best_rho if best_field is not None else None
    return out


def _enrichment(S: np.ndarray, strata: np.ndarray, q: float = _ENRICHMENT_Q) -> list:
    """Hypergeometric enrichment of every stratum label present in the
    top-`k` set `S`, BH-corrected within this one part (never pooled across
    parts -- `CLAUDE.md` sec 6.5's per-family discipline). Returns only the
    labels that survive, each with its count, fold enrichment over the
    population base rate, and BH-adjusted p (`q`)."""
    n_total = int(strata.size)
    k = int(S.size)
    labels_in_S, counts_in_S = np.unique(strata[S], return_counts=True)
    pvals: dict = {}
    detail: dict = {}
    for label, count in zip(labels_in_S, counts_in_S):
        pop_count = int(np.sum(strata == label))
        p = float(hypergeom.sf(int(count) - 1, n_total, pop_count, k))
        pvals[str(label)] = p
        fold = (count / k) / (pop_count / n_total) if pop_count and n_total else float("nan")
        detail[str(label)] = {"label": str(label), "count": int(count), "k": k,
                              "pop_count": pop_count, "pop_total": n_total,
                              "fold_enrichment": float(fold), "p": p}
    bh = benjamini_hochberg(pvals, q)
    survivors = []
    for label, rec in bh.items():
        if rec["survives"]:
            row = dict(detail[label])
            row["q"] = rec["p_bh"]
            survivors.append(row)
    survivors.sort(key=lambda r: -r["fold_enrichment"])
    return survivors


def _provenance_driven(field_records: list, provenance: dict, S: np.ndarray,
                       generators: np.ndarray) -> tuple:
    """`(bool, components)`. True when `S` is dominated by one real-derived
    generator (>= 80% of the top-k series). On the 4-model run the share is
    bimodal (23 of 55 parts at 0.0, 23 at >= 0.85), so the cut sits in the
    gap. The rho comparison (`provenance_exceeds_structural`) is recorded
    as descriptive only: it set a correlation over every series against a
    residualized one over the synthetic rows, and flagged a random-walk
    concept whose top-k held no real-derived series."""
    resid_rhos = [abs(r["resid_rho"]) for r in field_records if r["resid_rho"] is not None]
    max_resid_structural = max(resid_rhos) if resid_rhos else 0.0
    prov_candidates = [v for v in (provenance.get("tier_synthetic_rho"),
                                   provenance.get("max_generator_rho")) if v is not None]
    max_prov = max((abs(v) for v in prov_candidates), default=0.0)
    rho_component = bool(max_prov > max_resid_structural)

    gens_in_S = generators[S]
    k = int(S.size)
    gen_shares = {g: float(np.mean(gens_in_S == g)) for g in _REAL_DERIVED_GENERATORS if k}
    dominant_gen = max(gen_shares, key=gen_shares.get) if gen_shares else None
    dominant_share = gen_shares.get(dominant_gen, 0.0) if dominant_gen else 0.0
    generator_component = bool(dominant_share >= _PROVENANCE_SHARE_THRESHOLD)

    components = {
        "max_provenance_abs_rho": max_prov, "max_residualized_structural_abs_rho": max_resid_structural,
        "provenance_exceeds_structural": rho_component,
        "dominant_real_derived_generator": dominant_gen, "dominant_generator_share": dominant_share,
        "generator_share_threshold": _PROVENANCE_SHARE_THRESHOLD,
        "generator_dominates": generator_component,
    }
    return bool(generator_component), components


def _top_structural(field_records: list, top_n: int = 3) -> list:
    scored = [r for r in field_records if r["resid_rho"] is not None]
    scored.sort(key=lambda r: -abs(r["resid_rho"]))
    return [{"field": r["field"], "raw_rho": r["raw_rho"], "resid_rho": r["resid_rho"], "n": r["n"]}
           for r in scored[:top_n]]


def _input_profile(s: np.ndarray, S: np.ndarray, joined: pd.DataFrame, provenance_cols: list,
                   strata: np.ndarray, generators: np.ndarray, seed: int, n_perm: int) -> dict:
    field_records = _structural_field_records(s, joined, provenance_cols, seed)
    raw_abs = [abs(r["raw_rho"]) for r in field_records if r["raw_rho"] is not None]
    real_max_abs_raw = max(raw_abs) if raw_abs else 0.0
    p_max_structural = (_p_max_structural(s, joined, field_records, real_max_abs_raw, seed, n_perm)
                        if raw_abs else float("nan"))
    provenance = _provenance_profile(s, joined)
    provenance_driven, prov_components = _provenance_driven(field_records, provenance, S, generators)
    return {
        "fields": field_records,
        "max_abs_raw_rho": real_max_abs_raw,
        "p_max_structural": p_max_structural,
        "provenance": provenance,
        "enrichment": _enrichment(S, strata),
        "provenance_driven": provenance_driven,
        "provenance_driven_components": prov_components,
        "top_structural": _top_structural(field_records),
    }


# ---------------------------------------------------------------------------
# Effect profile and exemplar.
# ---------------------------------------------------------------------------

def _effect_profile(X: np.ndarray, row_idx: list) -> dict:
    members = X[row_idx]
    mean_vec = members.mean(axis=0)
    return {"mean_profile": {ch: float(v) for ch, v in zip(CHANNELS, mean_vec)},
           "n_members": len(row_idx)}


def _exemplar(run_dir: Path, part: dict, X: np.ndarray, meta_by_series: pd.DataFrame,
             ablation_cache: dict) -> dict:
    """The member with the largest ablation-vector norm's first kept
    forecast, its `context` truncated to the last 128 values (`CLAUDE.md`
    sec 2.5 -- everything else in this artifact is a summary statistic;
    keeping a full-length raw series here would be the one array-sized
    exception)."""
    norms = np.linalg.norm(X[part["row_idx"]], axis=1)
    best_local = int(np.argmax(norms))
    feature_id = part["feature_ids"][best_local]
    candidates = _ablation_candidates(run_dir, part["target"], ablation_cache)
    cand = candidates.get(feature_id)
    if cand is None:
        return {"status": "not measured", "reason": "no ablation candidate record for this feature"}
    forecasts = cand.get("forecasts") or []
    if not forecasts:
        return {"status": "not measured",
               "reason": "this candidate's ablation record kept no forecast exemplars"}
    fc = forecasts[0]
    series_id = fc.get("series_id")
    stratum = None
    if series_id is not None and series_id in meta_by_series.index:
        row = meta_by_series.loc[series_id]
        stratum = row.get("archetype") if pd.notna(row.get("archetype")) else row.get("family")
    context = fc.get("context") or []
    return {
        "status": "measured", "feature": int(feature_id), "series_id": series_id,
        "stratum": (str(stratum) if stratum is not None else None),
        "context_last_128": [float(v) for v in context[-128:]],
        "target": [float(v) for v in (fc.get("target") or [])],
        "with_feature": [float(v) for v in (fc.get("with_feature") or [])],
        "without_feature": [float(v) for v in (fc.get("without_feature") or [])],
        "unpatched": [float(v) for v in (fc.get("unpatched") or [])],
    }


# ---------------------------------------------------------------------------
# Behavioral link.
# ---------------------------------------------------------------------------

def _causal_mase_effect(X: np.ndarray, part: dict, ablation_cache: dict, run_dir: Path) -> dict:
    mean_effect = float(X[part["row_idx"], _MASE_CHANNEL_IDX].mean())
    candidates = _ablation_candidates(run_dir, part["target"], ablation_cache)
    any_clears = False
    for fid in part["feature_ids"]:
        cand = candidates.get(fid) or {}
        if ((cand.get("channels") or {}).get("mase") or {}).get("clears_null"):
            any_clears = True
            break
    return {"mean_signed_effect_over_null_p95": mean_effect, "any_member_clears_null": bool(any_clears),
           "n_members": len(part["feature_ids"])}


def _behavioral_link(part_model: str, S_series_ids: np.ndarray, metrics_wide: pd.DataFrame,
                     causal: dict, n_boot: int, seed: int) -> dict:
    """For every OTHER model `B`, `d = log MASE_part_model - log MASE_B` over
    `S`'s series (paired, series is the unit -- `CLAUDE.md` sec 6.5). Sign
    convention (pinned by a test): `d < 0` means the part's model is BETTER
    on `S` than `B`."""
    other_models = [m for m in metrics_wide.columns if m != part_model]
    if not other_models:
        return {"status": "not measured", "reason": "only one model in this run"}
    sub = metrics_wide.reindex(S_series_ids)
    vs_models: dict = {}
    classifications: dict = {}
    for i, model_b in enumerate(other_models):
        pair = sub[[part_model, model_b]].dropna()
        vals_a = pair[part_model].to_numpy(dtype=np.float64)
        vals_b = pair[model_b].to_numpy(dtype=np.float64)
        finite = np.isfinite(vals_a) & np.isfinite(vals_b) & (vals_a > 0) & (vals_b > 0)
        vals_a, vals_b = vals_a[finite], vals_b[finite]
        if vals_a.size < 3:
            vs_models[model_b] = {"status": "not measured",
                                  "reason": f"fewer than 3 series with finite MASE for both models"}
            classifications[model_b] = "not measured"
            continue
        d = np.log(vals_a) - np.log(vals_b)
        ci = mean_ci(d, n_boot=n_boot, seed=_seed("profiles", part_model, model_b, base=seed), unit="series")
        if ci["hi"] < 0:
            cls = "better"
        elif ci["lo"] > 0:
            cls = "worse"
        else:
            cls = "indistinguishable"
        classifications[model_b] = cls
        vs_models[model_b] = {"status": "measured", "mean_log_mase_gap": ci["value"],
                              "ci": [ci["lo"], ci["hi"]], "n_series": int(vals_a.size),
                              "classification": cls}

    measured = {m: c for m, c in classifications.items() if c != "not measured"}
    if not measured:
        return {"status": "not measured", "reason": "no other model had enough paired series",
               "vs_models": vs_models}

    all_better = all(c == "better" for c in measured.values())
    n_worse = sum(1 for c in measured.values() if c == "worse")
    causal_clears = bool(causal["any_member_clears_null"]) and causal["mean_signed_effect_over_null_p95"] > 0
    if all_better and causal_clears:
        why_verdict = "advantage carried by this concept"
    elif all_better:
        why_verdict = "advantage, not traced to this concept"
    elif n_worse > 0:
        why_verdict = f"worse than {n_worse} model(s)"
    else:
        why_verdict = "no advantage on these inputs: indistinguishable"

    return {"status": "measured", "vs_models": vs_models, "causal_mase_effect": causal,
           "why_verdict": why_verdict,
           "why_verdict_detail": {"all_better": bool(all_better), "n_worse": int(n_worse),
                                  "n_models_measured": len(measured),
                                  "causal_mase_clears_and_positive": causal_clears}}


# ---------------------------------------------------------------------------
# Input agreement and sharing class.
# ---------------------------------------------------------------------------

def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else float("nan")

def _permute_within_stratum(vals: np.ndarray, strata: np.ndarray, by_stratum: dict,
                            rng: np.random.Generator) -> np.ndarray:
    out = vals.copy()
    for stratum_vals in by_stratum.values():
        out[stratum_vals] = vals[rng.permutation(stratum_vals)]
    return out


def _pair_agreement(s_a: np.ndarray, s_b: np.ndarray, S_a: np.ndarray, S_b: np.ndarray,
                    strata: np.ndarray, by_stratum: dict, seed: int, n_perm: int) -> dict:
    rho, _ = spearmanr(s_a, s_b)
    rho_undefined = not np.isfinite(rho)
    rho = None if rho_undefined else float(rho)
    jaccard = _jaccard(S_a, S_b)

    p_uncond = p_within_stratum = None
    if not rho_undefined:
        rng_uncond = np.random.default_rng(seed)
        null_uncond = np.empty(n_perm, dtype=np.float64)
        for p in range(n_perm):
            perm_b = s_b[rng_uncond.permutation(s_b.size)]
            r, _ = spearmanr(s_a, perm_b)
            null_uncond[p] = abs(float(r)) if np.isfinite(r) else 0.0
        p_uncond = _right_tail_p(abs(rho), null_uncond)

        rng_within = np.random.default_rng(seed + 1)
        null_within = np.empty(n_perm, dtype=np.float64)
        for p in range(n_perm):
            perm_b = _permute_within_stratum(s_b, strata, by_stratum, rng_within)
            r, _ = spearmanr(s_a, perm_b)
            null_within[p] = abs(float(r)) if np.isfinite(r) else 0.0
        p_within_stratum = _right_tail_p(abs(rho), null_within)

    overlap, expected, p_overlap, p_overlap_within = _top_k_overlap(
        S_a, S_b, s_a.size, strata, by_stratum, seed + 2, n_perm)
    return {"rho": rho, "jaccard_top_k": jaccard, "p_uncond": p_uncond,
           "p_within_stratum": p_within_stratum, "top_k_overlap": overlap,
           "top_k_overlap_expected": expected, "p_overlap": p_overlap,
           "p_overlap_within_stratum": p_overlap_within,
           **({"rho_undefined_reason": "a part's per-series score is constant or "
                                       "non-finite, so the whole-series rank correlation "
                                       "is not defined (top-k overlap is unaffected)"}
              if rho_undefined else {})}


def _top_k_overlap(S_a: np.ndarray, S_b: np.ndarray, n: int, strata: np.ndarray,
                   by_stratum: dict, seed: int, n_perm: int) -> tuple:
    """`(overlap, expected, p, p_within_stratum)` for the two parts' top-k
    series sets. `p` is the exact hypergeometric tail of the overlap under
    two independent random k-subsets; `p_within_stratum` redraws `S_b` with
    its own per-stratum composition (`transfer.matched_draws`), so an
    overlap explained by both parts favouring one archetype does not clear
    it. SAE features are sparse, so the shared top-firing series, not a
    rank correlation over every series, is what "same inputs" means."""
    set_a = set(S_a.tolist())
    overlap = int(len(set_a & set(S_b.tolist())))
    expected = float(len(S_a) * len(S_b) / n) if n else float("nan")
    p = float(hypergeom.sf(overlap - 1, n, len(S_a), len(S_b)))
    draws = matched_draws(S_b, strata, by_stratum, n_perm, np.random.default_rng(seed))
    in_a = np.isin(draws, S_a)
    null = in_a.sum(axis=1)
    p_within = float((1 + np.sum(null >= overlap)) / (1 + n_perm))
    return overlap, expected, p, p_within


def _mark_agreement(cross_model_pairs: list) -> None:
    """Set `agrees`, `p_overlap_bh` and `beyond_stratum` on each cross-model
    pair record in place: BH over the concept's top-k overlap p-values.
    The broad-correlation p (`p_uncond`) never decides agreement."""
    pvals = {i: rec["p_overlap"] for i, rec in enumerate(cross_model_pairs)}
    bh = benjamini_hochberg(pvals, _AGREEMENT_Q)
    for i, rec in enumerate(cross_model_pairs):
        rec["p_overlap_bh"] = bh[i]["p_bh"] if i in bh else None
        rec["agrees"] = bool(bh[i]["survives"]) if i in bh else False
        rec["beyond_stratum"] = bool(rec["agrees"] and rec["p_overlap_within_stratum"] < _AGREEMENT_Q)


def _residualization_cols(columns) -> list:
    """Provenance dummies structural fields are residualized against: tier
    and the real-derived generators. Archetype and synthetic-generator
    dummies are excluded because they are structural recipes: on the
    4-model run, residualizing on `generator_random_parametric` cut the
    random-walk concept's `has_random_walk` rho from 0.833 to 0.309."""
    real_derived = {f"generator_{g}" for g in _REAL_DERIVED_GENERATORS}
    return [c for c in columns if is_provenance_field(c)
            and (c.startswith("tier_") or c in real_derived)]


def _sharing_class(models_with_parts: set, agreeing_pairs: list) -> str:
    if len(models_with_parts) <= 1:
        return "single-model"
    parent = {m: m for m in models_with_parts}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in agreeing_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = {find(m) for m in models_with_parts}
    if not agreeing_pairs:
        return "convergent (same effect, different inputs)"
    if len(roots) == 1:
        return "shared (same effect, same inputs)"
    return "partially shared"


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def run_concept_profiles(run_dir, cfg) -> dict:
    """Per-atlas-concept profiles: input triggers, cross-model input
    agreement, effect profile, an exemplar and the behavioral link to each
    concept's own causal MASE effect. Requires `sae/concept_atlas.json`;
    reads `sae/concept_stability.json`/`sae/atlas_transfer.json` if present.
    Always writes `sae/concept_profiles.json`, including the degenerate
    "no atlas concepts" case, so this function is safe to call directly.
    """
    run_dir = Path(run_dir)
    out_path = run_dir / "sae" / "concept_profiles.json"
    concepts_cfg = cfg.concepts
    sae_cfg = cfg.sae
    k_top = int(getattr(sae_cfg, "transfer_top_k", 20))
    n_perm = int(getattr(concepts_cfg, "profile_n_perm", 1000))
    base_seed = int(getattr(sae_cfg, "transfer_seed", 0))
    n_boot = int(cfg.stats.n_boot)

    atlas_path = run_dir / "sae" / "concept_atlas.json"
    if not atlas_path.exists():
        raise FileNotFoundError(f"run_concept_profiles requires {atlas_path}; the atlas must run first")
    atlas = load_json(atlas_path)

    causal_only = bool(getattr(sae_cfg, "concept_causal_only", True))
    X, pooled_rows = pooled_features(run_dir, causal_only=causal_only)
    atlas_rows = atlas.get("rows") or []
    if len(pooled_rows) != len(atlas_rows) or any(
            pooled_rows[i]["model"] != atlas_rows[i]["model"]
            or pooled_rows[i]["layer"] != atlas_rows[i]["layer"]
            or pooled_rows[i]["feature"] != atlas_rows[i]["feature"]
            for i in range(len(atlas_rows))):
        raise RuntimeError("run_concept_profiles: the pooled ablation matrix no longer matches "
                           "concept_atlas.json's own rows -- the atlas is stale, re-run it first")

    stability_path = run_dir / "sae" / "concept_stability.json"
    stability_by_concept: dict = {}
    stability_status = "not measured: sae/concept_stability.json does not exist"
    if stability_path.exists():
        stability_doc = load_json(stability_path)
        if stability_doc.get("measured"):
            stability_status = "measured"
            for rec in stability_doc.get("concepts", []):
                stability_by_concept[int(rec["concept"])] = rec.get("stability", {}).get("stable")
        else:
            stability_status = f"not measured: {stability_doc.get('reason', 'stability stage did not measure')}"

    atlas_transfer_path = run_dir / "sae" / "atlas_transfer.json"
    transfer_by_concept: dict = {}
    transfer_status = "not measured: sae/atlas_transfer.json does not exist"
    if atlas_transfer_path.exists():
        transfer_doc = load_json(atlas_transfer_path)
        transfer_status = "measured"
        for rec in transfer_doc.get("concept_summary", []):
            transfer_by_concept[int(rec["concept"])] = rec.get("transfers_to_models_fdr", [])

    meta = load_meta(run_dir)
    strata = series_strata(meta)
    by_stratum = _by_stratum(strata)
    generators = meta["generator"].to_numpy()
    series_ids = meta["series_id"].to_numpy()
    ground_truth_status = {"available": True, "reason": ""}
    try:
        gt = load_ground_truth_table(cfg.data.path)
    except Exception as exc:  # noqa: BLE001 -- degrade the structural profile, not the stage
        ground_truth_status = {"available": False,
                               "reason": (f"no ground truth for this corpus ({type(exc).__name__}: "
                                          f"{exc}); every structural field is unscored")}
        log.warning("concept_profiles: %s", ground_truth_status["reason"])
        gt = pd.DataFrame(columns=list(_STRUCTURAL_FIELDS), dtype=np.float64)
    joined = gt.reindex(series_ids)
    provenance_cols = _residualization_cols(joined.columns)

    metrics_path = run_dir / "l0" / "metrics.parquet"
    metrics_wide = None
    if metrics_path.exists():
        metrics = pd.read_parquet(metrics_path)
        metrics_wide = metrics.pivot_table(index="series_id", columns="model", values="mase")

    meta_by_series = meta.set_index("series_id")
    ablation_cache: dict = {}
    pooled_cache: dict = {}
    meta_sae_path = run_dir / "sae" / "meta.json"
    meta_sae = load_json(meta_sae_path) if meta_sae_path.exists() else {}
    store = ActivationStore(run_dir / "activations.zarr", mode="r")

    parts_all = _concept_parts(atlas)
    parts_by_concept: dict = {}
    for p in parts_all:
        parts_by_concept.setdefault(p["concept"], []).append(p)

    concepts_out = []
    n_provenance_driven_parts = 0
    n_parts_total = 0
    sharing_counts: dict = {}
    sharing_counts_stable: dict = {}
    single_model_stable_why: dict = {}

    for concept_rec in sorted(atlas.get("concepts") or [], key=lambda c: int(c["concept"])):
        cid = int(concept_rec["concept"])
        parts = parts_by_concept.get(cid, [])
        part_scores: dict = {}
        part_S: dict = {}
        part_outputs = []
        for part in parts:
            n_parts_total += 1
            pooled = _pooled_for_target(store, meta_sae, part["target"], pooled_cache)
            if pooled is None:
                part_outputs.append({
                    "target": part["target"], "model": part["model"], "layer": part["layer"],
                    "n_members": len(part["feature_ids"]),
                    "input_profile": {"status": "not measured",
                                      "reason": "SAE features were not persisted for this target"},
                    "effect_profile": _effect_profile(X, part["row_idx"]),
                    "exemplar": _exemplar(run_dir, part, X, meta_by_series, ablation_cache),
                    "behavioral_link": {"status": "not measured",
                                        "reason": "no per-series score without persisted SAE features"},
                })
                continue
            # `pooled` is the store's raw float16 array (`CLAUDE.md` sec 6.4);
            # `scipy.stats.rankdata` PRESERVES input dtype, so a float16 `s`
            # squared inside a Spearman-by-hand computation (every
            # permutation test below does exactly that) silently overflows
            # float16's ~65504 ceiling on a run of even a few hundred series.
            # Upcast once, here, at the single point every downstream
            # computation in this module reads `s` from.
            s = concept_scores(pooled, part["feature_ids"]).astype(np.float64)
            S = top_series(s, k_top)
            part_scores[part["target"]] = s
            part_S[part["target"]] = S

            seed_here = _seed("profiles", "input", cid, part["target"], base=base_seed)
            input_profile = _input_profile(s, S, joined, provenance_cols, strata, generators,
                                           seed_here, n_perm)
            if input_profile["provenance_driven"]:
                n_provenance_driven_parts += 1

            effect_profile = _effect_profile(X, part["row_idx"])
            exemplar = _exemplar(run_dir, part, X, meta_by_series, ablation_cache)
            causal = _causal_mase_effect(X, part, ablation_cache, run_dir)
            if metrics_wide is None or part["model"] not in metrics_wide.columns:
                behavioral = {"status": "not measured",
                              "reason": "no l0/metrics.parquet, or this model has no column in it"}
            else:
                S_series_ids = series_ids[S]
                seed_beh = _seed("profiles", "behavioral", cid, part["target"], base=base_seed)
                behavioral = _behavioral_link(part["model"], S_series_ids, metrics_wide, causal,
                                              n_boot, seed_beh)

            part_outputs.append({
                "target": part["target"], "model": part["model"], "layer": part["layer"],
                "n_members": len(part["feature_ids"]), "top_series_ids": series_ids[S].tolist(),
                "input_profile": input_profile, "effect_profile": effect_profile,
                "exemplar": exemplar, "behavioral_link": behavioral,
            })

        models_with_parts = {p["model"] for p in parts}
        cross_model_pairs = []
        within_model_pairs = []
        agreeing_pairs = []
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                pa, pb = parts[i], parts[j]
                if pa["model"] == pb["model"]:
                    within_model_pairs.append({"target_a": pa["target"], "target_b": pb["target"]})
                    continue
                if pa["target"] not in part_scores or pb["target"] not in part_scores:
                    continue
                seed_pair = _seed("profiles", "agree", cid, pa["target"], pb["target"], base=base_seed)
                res = _pair_agreement(part_scores[pa["target"]], part_scores[pb["target"]],
                                      part_S[pa["target"]], part_S[pb["target"]],
                                      strata, by_stratum, seed_pair, n_perm)
                res.update({"model_a": pa["model"], "target_a": pa["target"],
                           "model_b": pb["model"], "target_b": pb["target"]})
                cross_model_pairs.append(res)

        _mark_agreement(cross_model_pairs)
        agreeing_pairs = [(r["model_a"], r["model_b"]) for r in cross_model_pairs if r["agrees"]]

        sharing_class = _sharing_class(models_with_parts, agreeing_pairs)
        if stability_status != "measured" or cid not in stability_by_concept:
            stable_out = "not measured: " + (stability_status if stability_status != "measured"
                                             else "this concept had no scoreable stability parts")
        elif stability_by_concept[cid] is None:
            stable_out = "not measured: fewer than one scoreable part for this concept"
        else:
            stable_out = bool(stability_by_concept[cid])
        transfer_models = (transfer_by_concept.get(cid) if transfer_status == "measured" else "not measured")
        provenance_driven_parts = sum(
            1 for po in part_outputs
            if isinstance(po.get("input_profile"), dict) and po["input_profile"].get("provenance_driven"))

        sharing_counts[sharing_class] = sharing_counts.get(sharing_class, 0) + 1
        if stable_out is True:
            sharing_counts_stable[sharing_class] = sharing_counts_stable.get(sharing_class, 0) + 1
            if sharing_class == "single-model" and len(part_outputs) == 1:
                bl = part_outputs[0].get("behavioral_link") or {}
                verdict = bl.get("why_verdict", bl.get("status", "not measured"))
                model_name = part_outputs[0]["model"]
                single_model_stable_why.setdefault(model_name, {})
                single_model_stable_why[model_name][verdict] = (
                    single_model_stable_why[model_name].get(verdict, 0) + 1)

        concepts_out.append({
            "concept": cid, "name": concept_rec.get("name"), "n_models": concept_rec.get("n_models"),
            "parts": part_outputs, "within_model_pairs": within_model_pairs,
            "cross_model_pairs": cross_model_pairs, "sharing_class": sharing_class,
            "stable": stable_out, "input_transfer_models_fdr": transfer_models,
            "provenance_driven_parts": provenance_driven_parts, "n_parts": len(part_outputs),
        })

    summary = {
        "sharing_class_counts": sharing_counts, "sharing_class_counts_stable_only": sharing_counts_stable,
        "n_provenance_driven_parts": n_provenance_driven_parts, "n_parts": n_parts_total,
        "single_model_stable_why_verdict_counts": single_model_stable_why,
    }
    out = {
        "schema_version": SCHEMA_VERSION,
        "params": {"transfer_top_k": k_top, "profile_n_perm": n_perm, "transfer_seed": base_seed,
                  "n_boot": n_boot, "min_residual_scale": _MIN_RESIDUAL_SCALE,
                  "provenance_share_threshold": _PROVENANCE_SHARE_THRESHOLD,
                  "enrichment_q": _ENRICHMENT_Q, "agreement_q": _AGREEMENT_Q,
                  "agreement_rule": (f"a cross-model part pair agrees when the overlap of their "
                                     f"top-k series is larger than two random k-subsets give "
                                     f"(exact hypergeometric p, BH at q={_AGREEMENT_Q} within the "
                                     f"concept); beyond_stratum records whether the overlap also "
                                     f"beats redraws with the same per-stratum composition. rho "
                                     f"over all series is reported as broad co-variation only: on "
                                     f"the 4-model run rho 0.804 came with zero top-k overlap"),
                  "residualization_basis": ("tier_* and real-derived generator_* dummies only; "
                                            "archetype_* and synthetic generators are structural "
                                            "recipes, so residualizing on them removes the "
                                            "structure being measured")},
        "concepts": concepts_out, "summary": summary, "evidence_classes": EVIDENCE_CLASSES,
        "ground_truth": ground_truth_status,
    }
    save_json(out_path, out)
    log.info("concept profiles: %d concept(s), %d part(s) (%d provenance-driven); "
            "sharing classes %s", len(concepts_out), n_parts_total, n_provenance_driven_parts,
            sharing_counts)
    return out


def _pooled_for_target(store, meta_sae: dict, target: str, cache: dict):
    """Persisted SAE `space="sae"` pooled features for one target, or `None`
    when they were never persisted (`sae/meta.json`'s own
    `features_persisted` flag -- the same check `transfer.py::
    _live_atlas_targets` uses, reused rather than re-derived via a second
    store probe). `cache` is the caller's own dict, shared across every part
    of every concept in one `run_concept_profiles` call, so a target visited
    by two different concepts is loaded from the store once."""
    if target not in cache:
        rec = meta_sae.get(target) or {}
        if not rec.get("features_persisted"):
            cache[target] = None
        else:
            model, layer = target.split("/", 1)
            cache[target] = store.load(model, layer, level="series", space="sae")
    return cache[target]
