"""ROADMAP.md sec 37.9 P6a -- generator-side input counterfactuals, measured:
does an atlas concept's causal-feature score actually RESPOND to the
structural property its generator controls, rather than merely correlating
with it the way every other "about" label in this repo does?

`tsfm_benchmark/build_pipeline/counterfactual.py`'s draw-neutral knobs let a
`random_parametric` series be rebuilt with exactly one structural property
rescaled and everything else bit-identical -- a true input-side intervention.
This module drives that regeneration for every eligible series, encodes each
dose the same way the store itself was built (window-pooled activation ->
per-window SAE encode -> mean over windows, `sae/train.py::
encode_and_persist_features`'s own convention, NOT SAE-encoding an
already-pooled vector -- the two differ because the encoder is nonlinear),
and scores each atlas concept part's response to the dose ladder.

Scope: **input response only** (design items 1-4 of ROADMAP.md's "measurement
side"). Mediation (does the FORECAST'S response to the knob pass through the
concept?) and the cross-model comparison are P6b, gated on this step's
go/no-go result -- this module is deliberately not wired into the `concepts`
stage or the report yet.

Reuses rather than re-derives: `shared_input_agreement.py`'s `_atlas_parts`
(unit grouping), `TargetContext` (adapter/SAE/pooled-features/alive-mask
loading), `matched_null_sets` (the decile-matched random-alive-feature-set
floor) and `_spearman` (constant-input -> `None`, never `0`, CLAUDE.md sec
11.37); `transfer.py`'s `concept_scores`, `top_series`, `_seed` (sha256,
never Python's salted `hash()`) and `_exact_p`/`benjamini_hochberg` for the
response p-value and its within-target FDR control; `analysis/stats.py::
mean_ci` for the series-bootstrap CI on the aggregated response.

The encode-check that gates every target (does a fresh dose=1.0 recompute
land in the same space as the persisted store?) is per-concept-part Spearman
on `concept_scores`, not a whole-vector tolerance on raw SAE-feature values:
see `_concept_part_encode_checks` and `_ENCODE_CHECK_ATOL`'s docstring for
why (an absolute epsilon on a sparse dictionary cannot tell precision noise
from a wrong space, CLAUDE.md sec 8).

Evidence class: still correlational-with-a-planted-cause, not causal -- this
measures whether the CONCEPT'S OWN SCORE moves with the input property, never
whether the FORECAST does (`sae/response.py`'s ablation battery is the causal
half, unaffected by this module).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..analysis.stats import mean_ci
from ..extraction.alignment import align, pooling_matrix
from ..extraction.extract import capture_raw_tokens
from ..utils import batch_slices, load_json, log, sample_rows, save_json
from .shared_input_agreement import TargetContext, _atlas_parts, _spearman, matched_null_sets
from .transfer import _exact_p, _seed, benjamini_hochberg, concept_scores, top_series

__all__ = ["counterfactual_response_path", "run_counterfactual_response"]

# CLAUDE.md sec 6.2.1 Stage 3d / `sae/train.py::encode_and_persist_features`:
# the persisted series-level SAE feature is float16-quantized twice (once
# encoding, once pooling) before landing in the store, so a from-scratch
# recomputation cannot be bit-exact -- only close to float16's own precision
# (~3-4 significant decimal digits, machine eps ~9.8e-4). Measured against a
# real run (ROADMAP sec 37.9 P6a go/no-go, first attempt): a fresh recapture
# differed from the store by a 0.17% RELATIVE max diff on the raw activation
# itself (bf16-scale precision noise, not a wrong space), yet EVERY one of 4
# targets failed this whole-vector `atol`/`rtol` check on the raw SAE
# features by up to 0.8 absolute -- because a sparse TopK SAE's dictionary is
# mostly near-zero features, and CLAUDE.md sec 8's "absolute epsilons" lesson
# applies exactly: a fixed `atol` cannot survive many near-zero entries no
# matter how small the true perturbation is. This whole-vector check is kept
# below as `sae_feature_diagnostic` (reported, never used to gate); the gate
# itself is `_concept_part_encode_checks`' per-concept-part Spearman, which
# only looks at features a concept actually uses and asks the question that
# matters -- does this concept rank series the same way in both spaces --
# rather than whether the whole raw dictionary matches to a fixed tolerance.
_ENCODE_CHECK_ATOL = 1e-3
_ENCODE_CHECK_RTOL = 1e-2
_ENCODE_BATCH_SIZE = 64


def counterfactual_response_path(run_dir) -> Path:
    return Path(run_dir) / "sae" / "counterfactual_response.json"


def _load_counterfactual_module():
    """Lazy import: `tsfm_benchmark` is a separately installed package
    (CLAUDE.md sec 3), so its absence is a stated-reason degrade for THIS
    module, never an import-time crash of `tsfm_lens` itself. A seam so
    tests can monkeypatch either a stub or a forced `ImportError`."""
    from tsfm_benchmark.build_pipeline.counterfactual import KNOBS, regenerate
    from tsfm_benchmark.build_pipeline.seal import load_sealed
    return KNOBS, regenerate, load_sealed


def _load_samples_by_id(load_sealed_fn, corpus_path: str) -> dict:
    """`sample_id -> TimeSeriesSample`, full provenance intact (unlike
    `data.meta`, which keeps only scalar columns -- `ground_truth.py::
    load_ground_truth_table`'s own pattern, reused rather than re-derived)."""
    samples, _ = load_sealed_fn(corpus_path, verify=False)
    return {s.sample_id: s for s in samples}


def _sample_has_knob(sampled_params: dict, knob: str) -> bool:
    """Mirrors each `KNOBS[knob]` validator's own presence check (generator
    side), so a series is only offered to a knob it would actually accept."""
    if knob == "seasonal_amplitude":
        return bool(sampled_params.get("seasonalities"))
    if knob == "anomaly_magnitude":
        return bool(sampled_params.get("n_anomalies"))
    if knob == "heteroskedastic_depth":
        return bool(sampled_params.get("heteroskedastic"))
    if knob == "trend_scale":
        return bool((sampled_params.get("trend") or {}).get("scale"))
    if knob == "intermittency_rate":
        return bool(sampled_params.get("intermittency"))
    raise ValueError(f"unknown counterfactual knob {knob!r}")


def _doses_for_knob(knob: str, doses: list) -> list:
    """`trend_scale` refuses factor 0.0 (the mirror of its 0 -> nonzero
    refusal: both cross `parametric()`'s zero-scale branch and desync every
    later draw) -- dropped uniformly for every series rather than raising
    mid-measurement, and recorded in `params` for transparency."""
    if knob == "trend_scale":
        return [float(d) for d in doses if float(d) != 0.0]
    return [float(d) for d in doses]


def _eligible_series_idx(samples_by_id: dict, series_ids: np.ndarray, knob: str) -> np.ndarray:
    """Row indices into `data` (not corpus indices) whose sample is
    `random_parametric`, still carries `sampled_params`/`inner_seed`, and has
    the component `knob` targets."""
    idx = []
    for i, sid in enumerate(series_ids):
        s = samples_by_id.get(sid)
        if s is None or s.provenance.generator != "random_parametric":
            continue
        gp = s.provenance.generator_params or {}
        sp = gp.get("sampled_params")
        if sp is None or "inner_seed" not in gp:
            continue
        if _sample_has_knob(sp, knob):
            idx.append(i)
    return np.asarray(idx, dtype=int)


def _encode_series_level(adapter, sae, layer: str, contexts: np.ndarray, pool,
                         device, batch_size: int) -> np.ndarray:
    """Window-pooled activation -> per-window SAE encode -> mean over windows,
    `encode_and_persist_features`'s own convention (never SAE-encoding an
    already-pooled vector -- the encoder is nonlinear, so the two differ).
    Capture is `autocast=True` (`sae/real_data.py`'s convention): this cache
    is encoded and compared/concatenated with the store's own space, never
    patched, so it is NOT the patching regime (`autocast=False`, CLAUDE.md
    sec 11.49). Returns `[n_contexts, dict_size]` float64.
    """
    take = min(int(batch_size), int(getattr(adapter.cfg, "batch_size", batch_size)))
    out = []
    with torch.no_grad():
        for s, e in batch_slices(len(contexts), take):
            tokens = capture_raw_tokens(adapter, contexts[s:e], [layer], autocast=True)[layer]
            act = align(tokens, pool)
            b, w, d = act.shape
            flat = act.reshape(-1, d).to(device=device, dtype=torch.float32)
            features = sae.encode(flat).detach().cpu().numpy().astype(np.float16)
            pooled = features.reshape(b, w, -1).mean(axis=1)
            out.append(pooled.astype(np.float64))
    return np.concatenate(out, axis=0)


def _raw_pooled_activation(adapter, layer: str, contexts: np.ndarray, pool,
                           device, batch_size: int) -> np.ndarray:
    """Fresh window-pooled RAW activation (before any SAE), mean over
    windows -- exactly `store.py`'s own `pooled/{model}/{layer} =
    aligned.mean(axis=1)` convention, computed directly from a fresh
    capture. This is the encode-check's item (a): a scale-aware diagnostic
    of whether the fresh capture is in the same numeric regime as the store,
    decoupled entirely from the SAE encoder (so it cannot be confused by the
    encoder's own nonlinearity or sparsity). Returns `[n_contexts, dim]`
    float64.
    """
    take = min(int(batch_size), int(getattr(adapter.cfg, "batch_size", batch_size)))
    out = []
    with torch.no_grad():
        for s, e in batch_slices(len(contexts), take):
            tokens = capture_raw_tokens(adapter, contexts[s:e], [layer], autocast=True)[layer]
            act = align(tokens, pool)
            out.append(act.mean(dim=1).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def _concept_part_encode_checks(dose1_features: np.ndarray, persisted: np.ndarray,
                                concept_ids: list, parts: dict, target: str,
                                min_spearman: float) -> list:
    """Per atlas concept part at `target`: does the concept's OWN pooled
    score (the only thing this module reads downstream, `concept_scores`'
    mean over member features) rank the checked series the same way in the
    fresh recompute as in the persisted store?

    A pure function on top of two already-encoded `[n_series, dict_size]`
    arrays (never re-encodes), so it is directly unit-testable with
    hand-built arrays -- no adapter, SAE or store mock needed. `top_k`
    overlap is reported alongside Spearman as a second, coarser view of the
    same question (`transfer.py::top_series`, the atlas's own convention),
    never used to gate.
    """
    out = []
    n_series = dose1_features.shape[0]
    k = min(20, n_series)
    for concept_id in concept_ids:
        features = parts[(concept_id, target)]["features"]
        fresh_scores = concept_scores(dose1_features, features)
        stored_scores = concept_scores(persisted, features)
        r = _spearman(fresh_scores, stored_scores)
        overlap = None
        trivial_agreement = None
        if r is not None and k > 0:
            fresh_top = set(top_series(fresh_scores, k).tolist())
            stored_top = set(top_series(stored_scores, k).tolist())
            overlap = len(fresh_top & stored_top) / k
            passes = r >= min_spearman
        else:
            # Spearman is undefined only when a score has zero variance
            # (CLAUDE.md sec 11.37: constant input -> `None`, never `0`) --
            # which says nothing about a space mismatch by itself. The free
            # control this module already relies on elsewhere (patch a layer
            # into itself and require exactly 0.0): if fresh and stored are
            # THEMSELVES numerically identical, agreement is established
            # without rank information; if they are constant but disagree,
            # that IS evidence of a mismatch and must not pass silently.
            trivial_agreement = bool(np.allclose(fresh_scores, stored_scores,
                                                 atol=1e-6, rtol=1e-6))
            passes = trivial_agreement
        entry = {"concept": concept_id, "n_features": len(features), "n_series": n_series,
                 "spearman": r, "top_k": k, "top_k_overlap": overlap,
                 "trivial_agreement": trivial_agreement, "pass": passes}
        if r is None and passes:
            entry["reason"] = ("concept score is constant across checked series in both spaces, "
                                "and the constant values agree exactly (trivial pass)")
        elif r is None and not passes:
            entry["reason"] = ("concept score is constant across checked series (Spearman "
                                "undefined) and the fresh/stored constant values disagree -- "
                                "cannot confirm the same space")
        elif not passes:
            entry["reason"] = (f"fresh-vs-stored concept score Spearman {r!r} is below "
                                f"concepts.cf_encode_min_spearman={min_spearman!r}")
        out.append(entry)
    return out


def _spearman_rows(scores: np.ndarray, doses) -> np.ndarray:
    """Vectorized per-row Spearman of `scores[..., n_doses]` against
    `doses`: average ranks (ties as `scipy.stats.spearmanr` treats them),
    then Pearson. A row that is constant across doses is NaN (undefined,
    never 0 -- CLAUDE.md sec 11.37), matching `_spearman`'s `None`. Exists so
    the matched null can use enough draws for its BH family to be
    satisfiable: the per-series `spearmanr` loop cost one call per series
    per null set."""
    from scipy.stats import rankdata
    x = rankdata(np.asarray(scores, dtype=np.float64), axis=-1)
    y = rankdata(np.asarray(doses, dtype=np.float64))
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean()
    num = (xc * yc).sum(axis=-1)
    den = np.sqrt((xc ** 2).sum(axis=-1) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    const = np.ptp(np.asarray(scores, dtype=np.float64), axis=-1) == 0.0
    r[const] = np.nan
    return r


def run_counterfactual_response(cfg, run_dir, hub, store, data, device, targets=None) -> dict:
    """Score every atlas concept part's response to its target's targets'
    draw-neutral input knobs, and write `sae/counterfactual_response.json`.

    `targets`, when given, restricts to a subset of `"model/layer"` strings
    (the go/no-go step's "one target per model first" -- ROADMAP.md sec
    37.9). Series selection (per knob, stratified by archetype via
    `utils.sample_rows`, up to `concepts.cf_max_series`) and the counterfactual
    regeneration are computed ONCE and shared across every target/concept, so
    a cross-model comparison in P6b reads the same inputs everywhere
    (invariant 5's discipline applied to an input-side intervention).
    """
    t0 = time.monotonic()
    run_dir = Path(run_dir)
    out_path = counterfactual_response_path(run_dir)

    try:
        KNOBS, regenerate, load_sealed = _load_counterfactual_module()
    except ImportError as exc:
        out = {"status": "unsupported", "reason": f"tsfm_benchmark not importable ({exc})"}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(out_path, out)
        log.warning("counterfactual_response: %s", out["reason"])
        return out

    atlas_path = run_dir / "sae" / "concept_atlas.json"
    if not atlas_path.exists():
        out = {"status": "no_atlas", "reason": f"{atlas_path} does not exist -- run the "
              f"concepts stage (atlas_enabled: true) first"}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(out_path, out)
        log.warning("counterfactual_response: %s", out["reason"])
        return out
    atlas = load_json(atlas_path)
    parts = _atlas_parts(atlas)
    if not parts:
        out = {"status": "no_atlas_concepts", "reason": "concept_atlas.json has no rows"}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(out_path, out)
        return out

    c = cfg.concepts
    base_seed = int(cfg.run.seed)
    doses_cfg = [float(d) for d in c.cf_doses]
    n_series_cap = int(c.cf_max_series)
    n_null = int(c.cf_n_null)
    fdr_q = float(getattr(c, "transfer_fdr_q", 0.05))
    knob_names = sorted(KNOBS)

    all_targets = sorted({key[1] for key in parts})
    wanted = set(targets) if targets is not None else None
    use_targets = [t for t in all_targets if wanted is None or t in wanted]

    samples_by_id = _load_samples_by_id(load_sealed, cfg.data.path)
    series_ids = data.meta["series_id"].to_numpy()
    archetypes = data.meta["archetype"].to_numpy()
    context_len = int(data.context_len)
    contexts_native = data.contexts().astype(np.float32)

    # ------------------------------------------------------------------
    # Series selection and regeneration: per knob, shared across targets.
    # ------------------------------------------------------------------
    eligibility, doses_used, identity_checks = {}, {}, {}
    cf_contexts: dict = {}
    for knob in knob_names:
        elig = _eligible_series_idx(samples_by_id, series_ids, knob)
        seed = _seed("cf_series_select", knob, base=base_seed)
        strata = archetypes[elig] if elig.size else None
        chosen_local = sample_rows(elig.size, n_series_cap, seed, strata=strata)
        rows = elig[chosen_local]
        eligibility[knob] = rows
        doses = _doses_for_knob(knob, doses_cfg)
        doses_used[knob] = doses

        per_series, max_diff, n_checked = {}, 0.0, 0
        for row in rows:
            sample = samples_by_id[series_ids[row]]
            per_dose = {}
            for dose in doses:
                cf = regenerate(sample, knob, dose)
                ctx = np.asarray(cf.values[:context_len], dtype=np.float32)
                per_dose[dose] = ctx
                if dose == 1.0:
                    n_checked += 1
                    diff = float(np.max(np.abs(ctx - contexts_native[row])))
                    max_diff = max(max_diff, diff)
            per_series[int(row)] = per_dose
        cf_contexts[knob] = per_series
        identity_checks[knob] = {"n_series": int(rows.size), "n_checked_dose_1_0": n_checked,
                                 "max_abs_diff": max_diff, "bit_identical": max_diff == 0.0}
        if n_checked > 0 and max_diff != 0.0:
            raise AssertionError(
                f"counterfactual_response: knob {knob!r} dose=1.0 does not reproduce the "
                f"corpus series bit-identically (max abs diff {max_diff!r}) -- every "
                f"response for this knob would be measured against the wrong baseline")

    # ------------------------------------------------------------------
    # Per target: encode-path sanity check, then per (concept, knob) response.
    # ------------------------------------------------------------------
    target_records = []
    for target in use_targets:
        t_target0 = time.monotonic()
        model, layer = target.split("/", 1)
        ctx = TargetContext(cfg, hub, store, data, run_dir, model, layer, device)
        pool = pooling_matrix(ctx.adapter.token_time_spans(), context_len, cfg.alignment.window)

        dose1_rows = sorted({row for knob in knob_names for row in eligibility[knob].tolist()})
        record = {"target": target}
        concept_ids = sorted({cid for (cid, t) in parts if t == target})

        if dose1_rows:
            dose1_rows_arr = np.asarray(dose1_rows, dtype=int)
            dose1_features = _encode_series_level(
                ctx.adapter, ctx.sae, layer, contexts_native[dose1_rows_arr], pool, device,
                _ENCODE_BATCH_SIZE)
            persisted = np.asarray(
                store.load(model, layer, level="series", space="sae", rows=dose1_rows_arr),
                dtype=np.float64)
            sae_abs_diff = np.abs(dose1_features - persisted)
            sae_max_abs_diff = float(sae_abs_diff.max())
            sae_diag_ok = bool(np.allclose(dose1_features, persisted, atol=_ENCODE_CHECK_ATOL,
                                           rtol=_ENCODE_CHECK_RTOL))

            raw_fresh = _raw_pooled_activation(ctx.adapter, layer, contexts_native[dose1_rows_arr],
                                               pool, device, _ENCODE_BATCH_SIZE)
            raw_stored = np.asarray(
                store.load(model, layer, level="series", space="act", rows=dose1_rows_arr),
                dtype=np.float64)
            raw_abs_diff = np.abs(raw_fresh - raw_stored)
            raw_abs_mean_stored = float(np.abs(raw_stored).mean())
            raw_relative_max = (float(raw_abs_diff.max() / raw_abs_mean_stored)
                                if raw_abs_mean_stored > 0 else None)
            raw_relative_mean = (float(raw_abs_diff.mean() / raw_abs_mean_stored)
                                 if raw_abs_mean_stored > 0 else None)

            part_checks = _concept_part_encode_checks(
                dose1_features, persisted, concept_ids, parts, target,
                float(c.cf_encode_min_spearman))
            spearmans = [p["spearman"] for p in part_checks if p["spearman"] is not None]
            any_pass = any(p["pass"] for p in part_checks) if part_checks else True

            record["encode_check"] = {
                "n_series": len(dose1_rows),
                "raw_activation": {"max_abs_diff": float(raw_abs_diff.max()),
                                   "mean_abs_diff": float(raw_abs_diff.mean()),
                                   "stored_abs_mean": raw_abs_mean_stored,
                                   "relative_max_diff": raw_relative_max,
                                   "relative_mean_diff": raw_relative_mean},
                "concept_parts": part_checks,
                "min_spearman_observed": min(spearmans) if spearmans else None,
                "max_spearman_observed": max(spearmans) if spearmans else None,
                "sae_feature_diagnostic": {
                    "max_abs_diff": sae_max_abs_diff, "atol": _ENCODE_CHECK_ATOL,
                    "rtol": _ENCODE_CHECK_RTOL, "ok": sae_diag_ok,
                    "note": "diagnostic only, does not gate -- a whole-vector absolute/relative "
                            "tolerance on raw SAE-feature values cannot tell bf16 precision "
                            "noise from a wrong space (CLAUDE.md sec 8, 'Absolute epsilons'); "
                            "concept_parts' per-part Spearman is the actual gate"},
                "ok": any_pass,
            }
            ok = any_pass
        else:
            record["encode_check"] = {"n_series": 0, "concept_parts": [], "ok": True,
                                      "reason": "no eligible series for any knob"}
            ok = True

        if not ok:
            record["status"] = "encode_mismatch"
            record["reason"] = (f"no atlas concept part at {target} clears "
                                f"concepts.cf_encode_min_spearman="
                                f"{float(c.cf_encode_min_spearman)!r} between the fresh dose=1.0 "
                                f"recompute and the persisted store -- every response for "
                                f"{target} would be measured in the wrong space")
            log.warning("counterfactual_response: %s", record["reason"])
            target_records.append(record)
            continue

        excluded_parts = {p["concept"]: p for p in record["encode_check"]["concept_parts"]
                          if not p["pass"]}

        knob_feats: dict = {}
        for knob in knob_names:
            rows = eligibility[knob]
            doses = doses_used[knob]
            if rows.size == 0:
                knob_feats[knob] = None
                continue
            batch = np.stack([[cf_contexts[knob][int(r)][d] for d in doses] for r in rows])
            n, nd, cl = batch.shape
            feats = _encode_series_level(ctx.adapter, ctx.sae, layer,
                                         batch.reshape(n * nd, cl), pool, device,
                                         _ENCODE_BATCH_SIZE)
            knob_feats[knob] = feats.reshape(n, nd, -1)

        tests = []
        pvals: dict = {}
        for concept_id in concept_ids:
            if concept_id in excluded_parts:
                reason = excluded_parts[concept_id].get(
                    "reason", "encode-check failed for this concept part")
                for knob in knob_names:
                    tests.append({"concept": concept_id, "knob": knob,
                                 "status": "excluded_encode_mismatch", "reason": reason})
                continue
            features = parts[(concept_id, target)]["features"]
            for knob in knob_names:
                feats = knob_feats[knob]
                rows = eligibility[knob]
                doses = doses_used[knob]
                key = (concept_id, knob)
                if feats is None:
                    tests.append({"concept": concept_id, "knob": knob, "status": "undefined",
                                 "reason": f"no eligible series for knob {knob!r}"})
                    continue
                n, nd, _ = feats.shape
                real_scores = concept_scores(feats.reshape(n * nd, -1), features).reshape(n, nd)
                real_per_series = [_spearman(real_scores[i], doses) for i in range(n)]
                valid = np.array([v for v in real_per_series if v is not None], dtype=np.float64)
                n_undefined = int(sum(v is None for v in real_per_series))
                if valid.size == 0:
                    tests.append({"concept": concept_id, "knob": knob, "status": "undefined",
                                 "reason": "per-series response is constant for every series "
                                          "(no series has a defined Spearman correlation)",
                                 "n_series": int(n), "n_undefined": n_undefined})
                    continue

                boot_seed = _seed("cf_bootstrap", target, str(concept_id), knob, base=base_seed)
                agg = mean_ci(valid, seed=boot_seed)

                null_seed = _seed("cf_null", target, str(sorted(features)), knob, base=base_seed)
                null_sets, null_diag = matched_null_sets(ctx, features, rows, n_null, null_seed)
                null_means = []
                flat = feats.reshape(n * nd, -1)
                for null_set in null_sets:
                    r = _spearman_rows(concept_scores(flat, null_set).reshape(n, nd), doses)
                    r = r[np.isfinite(r)]
                    if r.size:
                        null_means.append(float(r.mean()))

                abs_null = np.abs(np.asarray(null_means, dtype=np.float64)) if null_means else \
                    np.empty(0)
                null_p95 = float(np.quantile(abs_null, 0.95)) if abs_null.size else None
                ci_excludes_zero = not (agg["lo"] <= 0.0 <= agg["hi"])
                responds = bool(null_p95 is not None and abs(agg["value"]) > null_p95
                               and ci_excludes_zero)
                p, hits_floor = ((None, None) if abs_null.size == 0 else
                                _exact_p(abs(agg["value"]), abs_null))
                if p is not None:
                    pvals[key] = p

                tests.append({
                    "concept": concept_id, "knob": knob, "status": "scored",
                    "n_series": int(n), "n_defined": int(valid.size), "n_undefined": n_undefined,
                    "doses_used": doses, "response_mean": agg["value"], "response_lo": agg["lo"],
                    "response_hi": agg["hi"], "ci_excludes_zero": ci_excludes_zero,
                    "null_n": int(abs_null.size), "null_p95_abs": null_p95,
                    "null_diag": null_diag, "p": p, "p_hits_floor": hits_floor,
                    "responds": responds,
                })

        bh = benjamini_hochberg(pvals, fdr_q) if pvals else {}
        m_tests = len(pvals)
        min_q_single = (m_tests / (n_null + 1)) if m_tests else None
        record["bh_family"] = {
            "n_tests": m_tests, "n_null": n_null, "p_floor": 1.0 / (n_null + 1),
            "min_attainable_q_single": min_q_single,
            "satisfiable": bool(min_q_single is not None and min_q_single <= fdr_q),
            "note": "a single test can survive BH only if n_tests/(n_null+1) <= fdr_q; "
                    "otherwise 'nothing survives' is the resampling resolution, not a "
                    "measurement (CLAUDE.md sec 6.5)"}
        if min_q_single is not None and min_q_single > fdr_q:
            log.warning("counterfactual_response: %s BH family unsatisfiable (%d tests, "
                        "n_null=%d, min attainable q %.4f > %.3f)", target, m_tests, n_null,
                        min_q_single, fdr_q)
        for test in tests:
            if test.get("status") != "scored":
                continue
            key = (test["concept"], test["knob"])
            b = bh.get(key)
            test["p_bh"] = b["p_bh"] if b else None
            test["survives_fdr"] = bool(b["survives"]) if b else False
            test["responds_fdr"] = bool(test["responds"] and test.get("survives_fdr"))

        record.update({"status": "ok", "concepts": concept_ids, "tests": tests,
                       "runtime_seconds": time.monotonic() - t_target0})
        target_records.append(record)

    out = {
        "schema_version": 1,
        "status": "ok",
        "params": {"knobs": knob_names, "doses": doses_cfg, "doses_used": doses_used,
                  "cf_max_series": n_series_cap, "cf_n_null": n_null, "fdr_q": fdr_q},
        "identity_checks": identity_checks,
        "targets": target_records,
        "runtime_seconds": time.monotonic() - t0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    log.info("counterfactual_response: wrote %s (%d target(s), %.1fs)",
             out_path, len(target_records), out["runtime_seconds"])
    return out
