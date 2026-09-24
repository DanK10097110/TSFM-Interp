"""Pairwise model similarity across every existing comparison metric, plus a
consensus-and-contrasts reduction over them (ROADMAP.md sec 37, cmp-B spec).

"How similar are models A and B?" has no single answer on this project's own
reference run: representation metrics call one pair by far the closest
(best CKA 0.883, stitching gain ~0.71), behavioral agreement calls a
*different* pair the closest (error agreement 0.949), and one model is a
behavioral outlier despite scoring well on representation metrics with its
closest geometric partner. That disagreement is the finding, not noise to
average away -- this module never pools its metrics into one similarity
score, because doing so would hide exactly the disagreement a reader needs
to see.

This module is a PURE REDUCTION over already-committed stage artifacts (l0,
l1, l2, clustering, and the sae/concept-atlas chain), plus two cheap
computations of its own: a partial Spearman correlation of per-series log
MASE controlling for a model-free difficulty covariate (the seasonal-naive
forecaster's own MASE), and a model-label permutation test over the concept
atlas's cross-model co-membership. It trains nothing, re-derives no stage's
own arithmetic, and reuses `sae/transfer.py`'s seeding (`_seed`, never
Python's builtin `hash()`) and `sae/concept_atlas.py`'s own null-tail
helpers so this module's numbers are computed the SAME way those modules
already compute the same kind of thing.

Every metric entry states its own evidence class, its own reference (a null,
a chance level, or the input-feature baseline), its own uncertainty where
one exists, and a stated reason instead of a fabricated value wherever an
input artifact is missing (CLAUDE.md sec 2.5's "degrade loudly, never
silently" -- a missing artifact is `status: "not measured"` with a reason,
never a 0). No pair metric here depends on which OTHER models are in the
panel (CLAUDE.md sec 18 F8): every metric below is computed from exactly
the two models' own rows/keys, so a pair's number cannot change when a third
model is added to the run.

Two entry points: `build_model_similarity(run_dir, cfg) -> dict` (the pure
reduction) and `write_model_similarity(run_dir, cfg) -> dict` (writes
`report/model_similarity.json`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from ..sae.concept_atlas import _left_tail_p, _right_tail_p
from ..sae.transfer import _seed
from ..utils import load_json, log, save_json
from .l0_behavioral import _seasonal_naive_forecast
from .stats import bootstrap_ci, mase as _mase

__all__ = ["METRIC_DOCS", "build_model_similarity", "write_model_similarity"]

_ATLAS_NULL_N = 1000
_CONSENSUS_N_PERM = 2000
_REPRESENTATION_METRICS = ("cka_best", "stitching_gain", "cluster_ami")

METRIC_ORDER = ["error_agreement", "cka_best", "stitching_gain", "cluster_ami",
               "atlas_co_membership", "input_transfer", "shared_concepts"]

# One doc block per metric: `label`, `family` (the six/seven named buckets a
# contrast is drawn across), `what_it_measures` (one plain sentence),
# `evidence_class`, `higher_means` (every metric here is oriented so higher
# = more similar, which is what makes a single rank table meaningful), and
# `reference_text` (how to read the metric's own reference/null/baseline).
METRIC_DOCS = {
    "error_agreement": {
        "label": "Error agreement", "family": "behavioral",
        "what_it_measures": ("Whether the two models make similar per-series "
                             "forecast errors (Spearman correlation of log MASE "
                             "across series)."),
        "evidence_class": "behavioral", "higher_means": "more similar",
        "reference_text": ("Every pair's raw agreement tends to run high because "
                           "series difficulty dominates MASE for every model; read "
                           "the DIFFERENCES between pairs, and the partial "
                           "correlation controlling for a model-free difficulty "
                           "covariate (detail.partial), rather than the raw value "
                           "alone."),
    },
    "cka_best": {
        "label": "Best-layer CKA", "family": "geometric",
        "what_it_measures": ("The most similar pair of layers' linear "
                             "representational geometry: the max linear CKA over "
                             "every layer-pair between the two models."),
        "evidence_class": "geometric", "higher_means": "more similar",
        "reference_text": ("A max over many layer pairs is biased upward relative "
                           "to any single pre-specified pair; read it against the "
                           "shuffled-series null (reference) and the mean over the "
                           "whole matrix (detail.mean_cka), never the max alone."),
    },
    "stitching_gain": {
        "label": "Stitching gain", "family": "linear",
        "what_it_measures": ("How much of one model's activations a linear probe "
                             "trained on the other model's activations explains, "
                             "beyond an input-feature baseline (both directions, "
                             "averaged)."),
        "evidence_class": "linear-translatable", "higher_means": "more similar",
        "reference_text": ("0 = no better than the input-feature baseline "
                           "(CLAUDE.md invariant 3); a positive gain is "
                           "linear-translatable structure, never causal evidence "
                           "on its own."),
    },
    "cluster_ami": {
        "label": "Cluster agreement (AMI)", "family": "clustering",
        "what_it_measures": ("Whether the two models' activation clusters group "
                             "the same series together (adjusted mutual "
                             "information)."),
        "evidence_class": "descriptive", "higher_means": "more similar",
        "reference_text": "0 = chance agreement (AMI is already chance-adjusted).",
    },
    "atlas_co_membership": {
        "label": "Atlas co-membership", "family": "sae_effect",
        "what_it_measures": ("How often the two models' causally-ablated SAE "
                             "features land in the same cross-model concept "
                             "(shared causal effect profile), versus a "
                             "model-label permutation null."),
        "evidence_class": "causal within-model", "higher_means": "more similar",
        "reference_text": ("1 = exactly the co-membership rate expected under a "
                           "model-label permutation that preserves each model's "
                           "own row count; the two-tailed p-values in detail say "
                           "whether the observed count is an outlier in either "
                           "direction."),
    },
    "input_transfer": {
        "label": "Input transfer", "family": "sae_input",
        "what_it_measures": ("Whether an atlas concept's top-firing series in one "
                             "model are also grouped by the OTHER model's own SAE "
                             "dictionary (shared input selectivity, both "
                             "directions)."),
        "evidence_class": "descriptive", "higher_means": "more similar",
        "reference_text": ("R_rel = 1 means cross-model transfer matches each "
                           "model's own within-model (replicate-SAE) "
                           "reproducibility ceiling; below 1 means cross-model "
                           "transfer is weaker than a model's own reproducibility "
                           "already allows."),
    },
    "shared_concepts": {
        "label": "Shared concepts", "family": "sae_combined",
        "what_it_measures": ("Count of cross-model concepts classed as sharing "
                             "BOTH causal effect and input selectivity between "
                             "the two models."),
        "evidence_class": "descriptive", "higher_means": "more similar",
        "reference_text": ("A raw count, not a rate; read alongside the total "
                           "number of atlas concepts available in this run."),
    },
}


def _pair_key(a: str, b: str) -> str:
    return f"{a}|{b}"


def _not_measured(evidence_class: str, reason: str) -> dict:
    """The one shape every metric uses when its input artifact is missing or
    degenerate -- `value: None`, never a fabricated 0 (CLAUDE.md sec 2.5)."""
    return {"value": None, "ci": None, "reference": None,
           "evidence_class": evidence_class, "status": "not measured",
           "reason": reason, "detail": {}}


# ---------------------------------------------------------------------------
# 1. error_agreement (behavioral) -- l0/metrics.parquet.
# ---------------------------------------------------------------------------

def _seasonal_naive_mase_by_series(cfg) -> tuple:
    """`-> (series_id -> mase, reason)`. The model-free difficulty covariate
    `error_agreement`'s partial correlation controls for: the seasonal-naive
    forecaster's own MASE, scored with the SAME `mase()` function and the
    SAME `scale`/`min_scale_frac` convention `l0` used, via
    `l0_behavioral._seasonal_naive_forecast` (never re-derived). `l0` itself
    drops these reserved rows before writing `metrics.parquet` (`CLAUDE.md`
    sec 11.39's "the reserved namespace never enters a real comparison"), so
    this covariate is not otherwise available and must be recomputed here
    from the corpus directly. Returns `(None, reason)` when the corpus
    cannot be loaded at all -- a stage-input problem, not a per-pair one.
    """
    from ..data import load_benchmark

    try:
        data = load_benchmark(cfg.data, cfg.run.seed)
    except Exception as e:  # noqa: BLE001 -- any load failure degrades loudly, not silently
        return None, f"could not load corpus: {e}"
    contexts, targets = data.contexts(), data.targets()
    sn_point, sn_available = _seasonal_naive_forecast(contexts, data.horizon)
    m = _mase(sn_point, targets, contexts, cfg.l0.scale)
    m = np.where(sn_available, m, np.nan)
    series_id = data.meta["series_id"].to_numpy()
    return dict(zip(series_id, m.tolist())), None


def _partial_error_agreement(common_ids: np.ndarray, log_a: np.ndarray, log_b: np.ndarray,
                             naive_by_id, naive_reason, n_boot: int, seed: int, ci: float) -> dict:
    """Partial Spearman correlation of `log_a`/`log_b` controlling for the
    seasonal-naive log-MASE covariate `Z`, via the standard rank-based
    partial-correlation formula `(r_ab - r_az*r_bz) / sqrt((1-r_az^2)(1-r_bz^2))`.
    Series-cluster bootstrap CI, same discipline as every other CI here.
    """
    if naive_by_id is None:
        return {"status": "not measured", "reason": naive_reason}
    z_raw = np.array([naive_by_id.get(sid, np.nan) for sid in common_ids], dtype=np.float64)
    mask = np.isfinite(z_raw) & (z_raw > 0)
    if int(mask.sum()) < 8:
        return {"status": "not measured",
               "reason": f"only {int(mask.sum())} series have a defined seasonal-naive "
                         f"MASE (need >=8)"}
    z = np.log(z_raw[mask])
    xa, xb = log_a[mask], log_b[mask]

    def _stat(idx: np.ndarray) -> float:
        a_i, b_i, z_i = xa[idx], xb[idx], z[idx]
        r_ab = float(spearmanr(a_i, b_i).statistic)
        r_az = float(spearmanr(a_i, z_i).statistic)
        r_bz = float(spearmanr(b_i, z_i).statistic)
        denom = np.sqrt(max((1.0 - r_az ** 2) * (1.0 - r_bz ** 2), 1e-12))
        return (r_ab - r_az * r_bz) / denom

    ci_out = bootstrap_ci(_stat, int(mask.sum()), n_boot, seed, ci)
    return {"status": "measured", "value": ci_out["value"],
           "ci": [ci_out["lo"], ci_out["hi"]], "n_series": int(mask.sum())}


def _error_agreement(run_dir: Path, cfg, a: str, b: str, naive_by_id, naive_reason,
                     n_boot: int, seed: int) -> dict:
    path = run_dir / "l0" / "metrics.parquet"
    if not path.exists():
        return _not_measured("behavioral", "l0/metrics.parquet not found")
    df = pd.read_parquet(path, columns=["model", "series_id", "mase", "mase_reliable"])
    da = df[(df["model"] == a) & df["mase_reliable"]].set_index("series_id")["mase"]
    db = df[(df["model"] == b) & df["mase_reliable"]].set_index("series_id")["mase"]
    common = da.index.intersection(db.index)
    if len(common) < 8:
        return _not_measured("behavioral",
                             f"only {len(common)} series reliably scored for both "
                             f"models (need >=8)")
    common_ids = common.to_numpy()
    log_a = np.log(np.clip(da.loc[common].to_numpy(dtype=np.float64), 1e-12, None))
    log_b = np.log(np.clip(db.loc[common].to_numpy(dtype=np.float64), 1e-12, None))

    def _stat(idx: np.ndarray) -> float:
        return float(spearmanr(log_a[idx], log_b[idx]).statistic)

    ci_out = bootstrap_ci(_stat, len(common_ids), n_boot, seed, cfg.stats.ci)
    partial = _partial_error_agreement(common_ids, log_a, log_b, naive_by_id, naive_reason,
                                       n_boot, seed + 1, cfg.stats.ci)
    return {"value": ci_out["value"], "ci": [ci_out["lo"], ci_out["hi"]],
           "reference": None, "evidence_class": "behavioral", "status": "measured",
           "reason": None, "detail": {"n_series": len(common_ids), "partial": partial}}


# ---------------------------------------------------------------------------
# 2. cka_best (geometric) -- l1/cka.npz + l1/meta.json.
# ---------------------------------------------------------------------------

def _find_pair_record(meta: dict, a: str, b: str) -> tuple:
    """`-> (record, swapped)`. `record["model_a"] == a` unless `swapped`,
    in which case `record["model_a"] == b` and every `_a`/`_b`-suffixed
    field in it names the OTHER model -- callers must un-swap before
    reading `layers_a`/`layers_b`/`rel_depth_a`/`rel_depth_b`."""
    for rec in meta.get("pairs", []):
        if rec.get("model_a") == a and rec.get("model_b") == b:
            return rec, False
        if rec.get("model_a") == b and rec.get("model_b") == a:
            return rec, True
    return None, False


def _cka_best(run_dir: Path, cfg, a: str, b: str) -> dict:
    cka_path = run_dir / "l1" / "cka.npz"
    meta_path = run_dir / "l1" / "meta.json"
    if not cka_path.exists():
        return _not_measured("geometric", "l1/cka.npz not found")
    key_ab, key_ba = f"cka_window__{a}__{b}", f"cka_window__{b}__{a}"
    with np.load(cka_path) as z:
        if key_ab in z.files:
            mat, reversed_key = np.asarray(z[key_ab]), False
        elif key_ba in z.files:
            mat, reversed_key = np.asarray(z[key_ba]), True
        else:
            return _not_measured("geometric", f"neither {key_ab!r} nor {key_ba!r} "
                                              f"present in l1/cka.npz")
    if mat.size == 0:
        return _not_measured("geometric", f"{key_ab if not reversed_key else key_ba} "
                                         f"is empty")
    flat = int(np.argmax(mat))
    i, j = np.unravel_index(flat, mat.shape)
    # `mat` rows/cols are in (a, b) order unless the reversed key was the
    # only one present, in which case rows are `b`'s layers and cols `a`'s.
    i_a, j_b = (int(j), int(i)) if reversed_key else (int(i), int(j))
    value = float(mat[i, j])
    mean_cka = float(mat.mean())

    layer_a = layer_b = None
    rel_depth_a = rel_depth_b = None
    rel_depth_basis = "index_fallback"
    ci, null_ref = None, None
    ci_reason = "l1/meta.json has no matching pair record"
    if meta_path.exists():
        meta = load_json(meta_path)
        rec, swapped = _find_pair_record(meta, a, b)
        if rec is not None:
            layers_a_field = rec["layers_b"] if swapped else rec["layers_a"]
            layers_b_field = rec["layers_a"] if swapped else rec["layers_b"]
            rel_a_field = rec.get("rel_depth_b") if swapped else rec.get("rel_depth_a")
            rel_b_field = rec.get("rel_depth_a") if swapped else rec.get("rel_depth_b")
            if i_a < len(layers_a_field):
                layer_a = layers_a_field[i_a]
            if j_b < len(layers_b_field):
                layer_b = layers_b_field[j_b]
            if rel_a_field is not None and i_a < len(rel_a_field):
                rel_depth_a = float(rel_a_field[i_a])
                rel_depth_basis = "l1_meta"
            if rel_b_field is not None and j_b < len(rel_b_field):
                rel_depth_b = float(rel_b_field[j_b])
            best_pair = rec.get("best_pair") or {}
            best_layer_a = best_pair.get("layer_b") if swapped else best_pair.get("layer_a")
            best_layer_b = best_pair.get("layer_a") if swapped else best_pair.get("layer_b")
            if best_layer_a == layer_a and best_layer_b == layer_b:
                if best_pair.get("ci"):
                    ci = [best_pair["ci"]["lo"], best_pair["ci"]["hi"]]
                    ci_reason = None
                else:
                    ci_reason = "cfg.stats.enabled was False when l1 ran"
                if best_pair.get("null_ci"):
                    null_ref = best_pair["null_ci"]
            else:
                ci_reason = ("l1/meta.json's recorded best_pair argmax does not match "
                            "this pair's own cka.npz argmax")
    if rel_depth_a is None:
        n_a = mat.shape[1] if reversed_key else mat.shape[0]
        rel_depth_a = i_a / max(n_a - 1, 1)
    if rel_depth_b is None:
        n_b = mat.shape[0] if reversed_key else mat.shape[1]
        rel_depth_b = j_b / max(n_b - 1, 1)

    reference = None
    if null_ref is not None:
        reference = {"kind": "null", "value": null_ref["value"],
                    "text": "linear CKA at this same layer pair after breaking "
                            "series correspondence (shuffled-series null)"}
    return {"value": value, "ci": ci, "reference": reference, "evidence_class": "geometric",
           "status": "measured", "reason": ci_reason,
           "detail": {"layer_a": layer_a, "layer_b": layer_b,
                     "rel_depth_a": rel_depth_a, "rel_depth_b": rel_depth_b,
                     "rel_depth_basis": rel_depth_basis, "mean_cka": mean_cka,
                     "matrix_shape": [int(mat.shape[0]), int(mat.shape[1])]}}


# ---------------------------------------------------------------------------
# 3. stitching_gain (linear-translatable) -- l2/stitching.json.
# ---------------------------------------------------------------------------

def _stitching_gain(run_dir: Path, cfg, a: str, b: str) -> dict:
    path = run_dir / "l2" / "stitching.json"
    if not path.exists():
        return _not_measured("linear-translatable", "l2/stitching.json not found")
    doc = load_json(path)
    directions = doc.get("directions", {})
    detail: dict = {}
    values: list = []
    for key, (src, dst) in ((f"{a}->{b}", (a, b)), (f"{b}->{a}", (b, a))):
        rec = directions.get(key)
        if rec is None or rec.get("best_gain") is None:
            detail[key] = {"status": "not measured",
                          "reason": f"{key!r} missing from l2/stitching.json"}
            continue
        best = rec.get("best") or {}
        gain_ci = best.get("gain_ci")
        detail[key] = {"best_gain": rec["best_gain"],
                      "gain_ci": [gain_ci["lo"], gain_ci["hi"]] if gain_ci else None,
                      "src_layer": best.get("src_layer"), "dst_layer": best.get("dst_layer")}
        values.append(float(rec["best_gain"]))
    if not values:
        return _not_measured("linear-translatable",
                             f"neither direction present for ({a}, {b}) in "
                             f"l2/stitching.json")
    return {"value": float(np.mean(values)), "ci": None,
           "reference": {"kind": "baseline", "value": 0.0,
                        "text": "0 = no better than the input-feature baseline "
                                "(CLAUDE.md invariant 3)"},
           "evidence_class": "linear-translatable", "status": "measured",
           "reason": "CI is reported per direction in detail, not for the "
                    "two-direction mean", "detail": detail}


# ---------------------------------------------------------------------------
# 4. cluster_ami (descriptive) -- clustering/comparison.json.
# ---------------------------------------------------------------------------

def _cluster_ami(run_dir: Path, cfg, a: str, b: str) -> dict:
    path = run_dir / "clustering" / "comparison.json"
    if not path.exists():
        return _not_measured("descriptive", "clustering/comparison.json not found")
    doc = load_json(path)
    rec = next((p for p in doc.get("pairs", [])
               if {p.get("model_a"), p.get("model_b")} == {a, b}), None)
    if rec is None:
        return _not_measured("descriptive", f"no pair record for ({a}, {b}) in "
                                           f"clustering/comparison.json")
    ami = rec.get("ami") or {}
    if ami.get("value") is None:
        return _not_measured("descriptive", "clustering pair record has no ami.value")
    has_ci = "lo" in ami and "hi" in ami
    return {"value": float(ami["value"]), "ci": [ami["lo"], ami["hi"]] if has_ci else None,
           "reference": {"kind": "chance", "value": 0.0,
                        "text": "0 = chance agreement (AMI is already chance-adjusted)"},
           "evidence_class": "descriptive", "status": "measured",
           "reason": None if has_ci else "cfg.stats.enabled was False when clustering ran",
           "detail": {"n_rows_a": len(rec.get("rows_a", [])),
                     "n_cols_b": len(rec.get("cols_b", []))}}


# ---------------------------------------------------------------------------
# 5. atlas_co_membership (causal within-model) -- sae/concept_atlas.json.
# ---------------------------------------------------------------------------

def _atlas_co_membership(run_dir: Path, cfg, a: str, b: str, n_null: int = _ATLAS_NULL_N) -> dict:
    path = run_dir / "sae" / "concept_atlas.json"
    if not path.exists():
        return _not_measured("causal within-model", "sae/concept_atlas.json not found")
    atlas = load_json(path)
    rows = atlas.get("rows", [])
    if not rows:
        return _not_measured("causal within-model",
                             "concept atlas has no pooled causal features")
    models = np.array([r["model"] for r in rows])
    labels = np.array([r["concept"] if r.get("concept") is not None else -1 for r in rows])
    concept_ids = sorted({int(c) for c in labels.tolist() if c >= 0})
    if not concept_ids:
        return _not_measured("causal within-model",
                             "concept atlas assigned zero features to any concept")
    member_idx = {c: np.where(labels == c)[0] for c in concept_ids}

    def _co_count(models_arr: np.ndarray) -> int:
        return sum(1 for c in concept_ids
                  if a in set(models_arr[member_idx[c]].tolist())
                  and b in set(models_arr[member_idx[c]].tolist()))

    observed = _co_count(models)
    seed = _seed("model_similarity_atlas_co_membership", a, b,
                base=int(getattr(getattr(cfg, "run", None), "seed", 0) or 0))
    rng = np.random.default_rng(seed)
    n = len(models)
    null_counts = np.empty(n_null, dtype=np.int64)
    for k in range(n_null):
        null_counts[k] = _co_count(models[rng.permutation(n)])
    expected = float(null_counts.mean())
    p_above = _right_tail_p(observed, null_counts)
    p_below = _left_tail_p(observed, null_counts)

    if expected > 0:
        value, basis = observed / expected, "ratio"
    elif observed > 0:
        value, basis = float("inf"), "expected_zero_observed_positive"
    else:
        value, basis = float("nan"), "both_zero"

    return {"value": value, "ci": None,
           "reference": {"kind": "null", "value": expected,
                        "text": "expected co-membership count under a model-label "
                                "permutation preserving each model's own row count "
                                "(matches sae/concept_atlas.py's cross-model null)"},
           "evidence_class": "causal within-model", "status": "measured",
           "reason": None if basis == "ratio" else f"value basis: {basis}",
           "detail": {"observed": int(observed), "expected": expected,
                     "p_above": p_above, "p_below": p_below, "n_null": n_null,
                     "n_concepts": len(concept_ids), "value_basis": basis}}


# ---------------------------------------------------------------------------
# 6. input_transfer (descriptive) -- sae/atlas_transfer.json + ceiling.
# ---------------------------------------------------------------------------

def _load_ceilings(run_dir: Path) -> dict:
    path = run_dir / "sae" / "concept_stability.json"
    if not path.exists():
        return {}
    doc = load_json(path)
    by_model = ((doc.get("ceiling") or {}).get("by_model")) or {}
    return {m: rec.get("rate") for m, rec in by_model.items()}


def _input_transfer(run_dir: Path, cfg, a: str, b: str) -> dict:
    path = run_dir / "sae" / "atlas_transfer.json"
    if not path.exists():
        return _not_measured("descriptive", "sae/atlas_transfer.json not found")
    doc = load_json(path)
    pair_summary = doc.get("pair_summary", [])
    ceilings = _load_ceilings(run_dir)

    def _rec(src: str, dst: str):
        return next((p for p in pair_summary
                    if p.get("src_model") == src and p.get("dst_model") == dst), None)

    detail: dict = {}
    r_unc_vals, r_rel_vals, any_found = [], [], False
    for key, (src, dst) in ((f"{a}->{b}", (a, b)), (f"{b}->{a}", (b, a))):
        rec = _rec(src, dst)
        if rec is None or not rec.get("n_tests"):
            detail[key] = {"status": "not measured",
                          "reason": "no atlas_transfer tests for this direction"}
            continue
        any_found = True
        n_tests = int(rec["n_tests"])
        r_unc = rec["n_uncorrected_reciprocal"] / n_tests
        r_fdr = rec["n_fdr_reciprocal"] / n_tests
        ceil_src, ceil_dst = ceilings.get(src), ceilings.get(dst)
        r_rel = None
        if ceil_src and ceil_dst and ceil_src > 0 and ceil_dst > 0:
            r_rel = r_unc / np.sqrt(ceil_src * ceil_dst)
        detail[key] = {"n_tests": n_tests, "R_unc": r_unc, "R_fdr": r_fdr, "R_rel": r_rel,
                      "ceiling_src": ceil_src, "ceiling_dst": ceil_dst}
        r_unc_vals.append(r_unc)
        if r_rel is not None:
            r_rel_vals.append(r_rel)
    if not any_found:
        return _not_measured("descriptive",
                             "no atlas_transfer tests in either direction for this pair")
    if len(r_rel_vals) == 2:
        value, basis = float(np.mean(r_rel_vals)), "R_rel"
    else:
        value, basis = float(np.mean(r_unc_vals)), "R_unc"
    detail["value_basis"] = basis
    return {"value": value, "ci": None,
           "reference": {"kind": "ceiling", "value": 1.0,
                        "text": "R_rel = 1 means cross-model transfer matches the "
                                "within-model replicate-SAE reproducibility ceiling"},
           "evidence_class": "descriptive", "status": "measured",
           "reason": f"value = mean of {basis} across the direction(s) tested",
           "detail": detail}


# ---------------------------------------------------------------------------
# 7. shared_concepts (descriptive) -- sae/concept_profiles.json (may not exist).
# ---------------------------------------------------------------------------

_SHARED_CLASSES = {"shared (same effect, same inputs)", "partially shared"}


def _agreeing_component(concept: dict, a: str, b: str) -> bool:
    """True when models `a` and `b` are connected through the concept's
    agreeing cross-model part pairs (`cross_model_pairs[*].agrees`). A
    `partially shared` concept can hold parts in both models whose inputs
    do not agree with each other; that pair does not share it."""
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rec in concept.get("cross_model_pairs") or []:
        if isinstance(rec, dict) and rec.get("agrees"):
            parent[find(rec.get("model_a"))] = find(rec.get("model_b"))
    return a in parent and b in parent and find(a) == find(b)


def _shared_concepts(run_dir: Path, cfg, a: str, b: str) -> dict:
    """Concepts classed shared or partially shared in which `a` and `b` sit
    in one agreeing component, read from `sae/concept_profiles.json`."""
    path = run_dir / "sae" / "concept_profiles.json"
    if not path.exists():
        return _not_measured("descriptive",
                             "sae/concept_profiles.json not found (the concepts "
                             "stage's profiles step has not run)")
    try:
        doc = load_json(path)
        concepts = doc.get("concepts", [])
    except Exception as e:  # noqa: BLE001 -- read defensively, never crash on a foreign schema
        return _not_measured("descriptive",
                             f"could not read sae/concept_profiles.json: {e}")
    matched = []
    for c in concepts:
        if not isinstance(c, dict) or c.get("sharing_class") not in _SHARED_CLASSES:
            continue
        if _agreeing_component(c, a, b):
            matched.append(c.get("concept"))
    return {"value": float(len(matched)), "ci": None, "reference": None,
           "evidence_class": "descriptive", "status": "measured", "reason": None,
           "detail": {"concepts": matched, "n_total_concepts": len(concepts)}}


# ---------------------------------------------------------------------------
# Consensus across metrics.
# ---------------------------------------------------------------------------

def _pair_values(metrics: dict, key: str, pair_keys: list) -> dict:
    out = {}
    for pk in pair_keys:
        rec = metrics[key]["pairs"][pk]
        v = rec.get("value")
        if rec.get("status") == "measured" and v is not None and np.isfinite(v):
            out[pk] = float(v)
    return out


def _ranks_for_metric(values: dict) -> dict:
    """Rank 1 = most similar (highest value); every metric here is oriented
    so higher means more similar (`METRIC_DOCS[*]["higher_means"]`)."""
    keys = list(values)
    if not keys:
        return {}
    ranks = rankdata([-values[k] for k in keys], method="average")
    return {k: float(r) for k, r in zip(keys, ranks)}


def _extreme_pairs(values: dict) -> dict:
    if not values:
        return {"most_similar": None, "least_similar": None}
    most = max(values, key=values.get)
    least = min(values, key=values.get)
    return {"most_similar": most, "most_similar_value": values[most],
           "least_similar": least, "least_similar_value": values[least]}


def _kendall_w(ranks_by_metric: dict, pair_keys: list, n_perm: int, seed: int) -> dict:
    """Kendall's coefficient of concordance over metrics with COMPLETE ranks
    (every pair measured) -- an incomplete metric is excluded entirely
    rather than imputed. Permutation p: shuffle each metric's own rank
    vector independently (permuting which pair each rank belongs to,
    per metric), `n_perm` draws, one-sided (does the real concordance sit
    at least this high by chance)."""
    complete = {m: r for m, r in ranks_by_metric.items() if set(r) == set(pair_keys)}
    m, n = len(complete), len(pair_keys)
    if m < 2 or n < 3:
        return {"w": None, "p": None, "n_metrics": m, "n_pairs": n,
               "metrics_used": sorted(complete),
               "reason": "need >=2 complete-rank metrics and >=3 pairs for Kendall's W"}
    order = sorted(complete)
    R = np.array([[complete[met][pk] for pk in pair_keys] for met in order])

    def _w_stat(mat: np.ndarray) -> float:
        m_, n_ = mat.shape
        rsum = mat.sum(axis=0)
        rbar = m_ * (n_ + 1) / 2.0
        s = float(np.sum((rsum - rbar) ** 2))
        t = 0.0
        for row in mat:
            _, counts = np.unique(row, return_counts=True)
            t += float(np.sum(counts ** 3 - counts))
        denom = m_ ** 2 * (n_ ** 3 - n_) - m_ * t
        return 12.0 * s / denom if denom > 0 else float("nan")

    w_obs = _w_stat(R)
    rng = np.random.default_rng(seed)
    n_ge = 0
    for _ in range(int(n_perm)):
        rp = np.stack([rng.permutation(row) for row in R])
        w_p = _w_stat(rp)
        if np.isfinite(w_p) and w_p >= w_obs:
            n_ge += 1
    p = (1 + n_ge) / (1 + int(n_perm))
    return {"w": float(w_obs), "p": float(p), "n_metrics": m, "n_pairs": n,
           "metrics_used": order, "n_perm": int(n_perm)}


def _rank_summary_per_pair(ranks_by_metric: dict, pair_keys: list, metrics_used: list) -> dict:
    out = {}
    for pk in pair_keys:
        vals = [ranks_by_metric[m][pk] for m in metrics_used if pk in ranks_by_metric.get(m, {})]
        if not vals:
            out[pk] = {"median_rank": None, "rank_range": None, "n_metrics": 0}
        else:
            out[pk] = {"median_rank": float(np.median(vals)),
                      "rank_range": float(max(vals) - min(vals)), "n_metrics": len(vals)}
    return out


def _consensus(metrics: dict, pair_keys: list, base_seed: int) -> dict:
    values_by_metric = {k: _pair_values(metrics, k, pair_keys) for k in METRIC_ORDER}
    ranks_by_metric = {k: _ranks_for_metric(v) for k, v in values_by_metric.items() if v}
    extremes = {k: _extreme_pairs(v) for k, v in values_by_metric.items()}

    w_all = _kendall_w(ranks_by_metric, pair_keys, _CONSENSUS_N_PERM,
                       _seed("model_similarity_kendall_w", "all", base=base_seed))
    rep_ranks = {k: v for k, v in ranks_by_metric.items() if k in _REPRESENTATION_METRICS}
    w_rep = _kendall_w(rep_ranks, pair_keys, _CONSENSUS_N_PERM,
                       _seed("model_similarity_kendall_w", "representation", base=base_seed))
    per_pair = _rank_summary_per_pair(ranks_by_metric, pair_keys, w_all["metrics_used"])

    return {
        "rank_table": ranks_by_metric, "extremes": extremes,
        "kendall_w_all": w_all, "kendall_w_representation": w_rep,
        "per_pair": per_pair,
        "note": ("With only a handful of model pairs, Kendall's W has low power: a "
                "non-significant p does NOT mean the metrics agree, and a "
                "significant one should still be read alongside the per-metric "
                "rank table above, not instead of it."),
    }


def _contrasts(metrics: dict, pair_keys: list, consensus: dict) -> list:
    """Large rank disagreements between two DIFFERENT metric families for the
    same pair -- `|rank_hi - rank_lo| >= n_pairs - 2` (a judgment call, cmp-B
    spec). `metric_hi`/`rank_hi` name the metric under which the pair ranks
    HIGH (most similar, a low rank number); `metric_lo`/`rank_lo` the metric
    under which it ranks LOW (least similar, a high rank number)."""
    n_pairs = len(pair_keys)
    threshold = n_pairs - 2
    ranks = consensus["rank_table"]
    families = {k: METRIC_DOCS[k]["family"] for k in METRIC_ORDER}
    metric_keys = [k for k in METRIC_ORDER if k in ranks]
    out = []
    for i in range(len(metric_keys)):
        for j in range(i + 1, len(metric_keys)):
            m1, m2 = metric_keys[i], metric_keys[j]
            if families[m1] == families[m2]:
                continue
            for pk in pair_keys:
                r1, r2 = ranks[m1].get(pk), ranks[m2].get(pk)
                if r1 is None or r2 is None:
                    continue
                gap = abs(r1 - r2)
                if gap < threshold:
                    continue
                if r1 < r2:
                    hi_m, hi_r, lo_m, lo_r = m1, r1, m2, r2
                else:
                    hi_m, hi_r, lo_m, lo_r = m2, r2, m1, r1
                out.append({"pair": pk, "metric_hi": hi_m, "rank_hi": hi_r,
                           "metric_lo": lo_m, "rank_lo": lo_r, "gap": gap})
    out.sort(key=lambda r: (-r["gap"], r["pair"]))
    return out


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def build_model_similarity(run_dir, cfg) -> dict:
    """`-> dict`. Every metric, for every unordered model pair (order taken
    from `cfg.models`, matching `cfg.comparison_pairs()`'s own `i < j`
    convention so `l1`/`l2`'s `__A__B`-suffixed keys are found directly),
    plus the cross-metric consensus and contrast reductions. Never raises on
    a missing stage artifact -- every metric degrades to `status: "not
    measured"` with a reason instead.
    """
    run_dir = Path(run_dir)
    names = [m.name for m in cfg.models]
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    pair_keys = [_pair_key(a, b) for a, b in pairs]
    n_boot = int(getattr(getattr(cfg, "stats", None), "n_boot", 500) or 500)
    base_seed = int(getattr(getattr(cfg, "run", None), "seed", 0) or 0)

    naive_by_id, naive_reason = _seasonal_naive_mase_by_series(cfg)

    metrics = {}
    for key in METRIC_ORDER:
        doc = METRIC_DOCS[key]
        metrics[key] = {"label": doc["label"], "family": doc["family"],
                       "what_it_measures": doc["what_it_measures"],
                       "evidence_class": doc["evidence_class"],
                       "higher_means": doc["higher_means"],
                       "reference_text": doc["reference_text"], "pairs": {}}

    for a, b in pairs:
        pk = _pair_key(a, b)
        metrics["error_agreement"]["pairs"][pk] = _error_agreement(
            run_dir, cfg, a, b, naive_by_id, naive_reason, n_boot, base_seed)
        metrics["cka_best"]["pairs"][pk] = _cka_best(run_dir, cfg, a, b)
        metrics["stitching_gain"]["pairs"][pk] = _stitching_gain(run_dir, cfg, a, b)
        metrics["cluster_ami"]["pairs"][pk] = _cluster_ami(run_dir, cfg, a, b)
        metrics["atlas_co_membership"]["pairs"][pk] = _atlas_co_membership(run_dir, cfg, a, b)
        metrics["input_transfer"]["pairs"][pk] = _input_transfer(run_dir, cfg, a, b)
        metrics["shared_concepts"]["pairs"][pk] = _shared_concepts(run_dir, cfg, a, b)

    consensus = _consensus(metrics, pair_keys, base_seed)
    contrasts = _contrasts(metrics, pair_keys, consensus)

    return {"schema_version": 1, "models": names,
           "pairs": [{"model_a": a, "model_b": b, "key": _pair_key(a, b)} for a, b in pairs],
           "metrics": metrics, "consensus": consensus, "contrasts": contrasts}


def write_model_similarity(run_dir, cfg) -> dict:
    run_dir = Path(run_dir)
    out = build_model_similarity(run_dir, cfg)
    out_path = run_dir / "report" / "model_similarity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    log.info("model similarity: wrote %s (%d pair(s), %d metric(s))",
             out_path, len(out["pairs"]), len(out["metrics"]))
    return out
