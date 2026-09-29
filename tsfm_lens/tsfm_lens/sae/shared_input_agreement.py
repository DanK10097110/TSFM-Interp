"""ROADMAP.md sec 37.8 P5b -- cross-model causal agreement, measured on the
SAME series, with a floor every model can have.

sec 37.8's own Findings on `sae/matching.py::add_role_causal_agreement`
(sec 27) name three stacked defects: (a) no floor for 5 of 6 model pairs
(the untrained-twin floor is obtainable for only one pair, sec 29), (b) the
two models' effects are measured on PARTLY DIFFERENT series (each feature's
own top-firing rows), so a disagreement conflates "acts differently" with
"was measured on different inputs", and (c) `reach_probe`'s pre-P5a absolute
epsilon passed numerically dead models. P5a (merged) fixed (c). This module
fixes (a) and (b) together, for the FDR-surviving reciprocal atlas-transfer
tests (`sae/atlas_transfer.json`) rather than `add_role_causal_agreement`'s
activation-matched pairs -- an atlas-transfer test already establishes that
the two features/parts select an overlapping set of series, which is the
precondition this module's whole design leans on.

For each surviving test (source = the atlas concept's PART in the source
model at its own target; destination = the best destination feature, plus
its own atlas part if it has one), the shared series `U` is the union of the
two top-k series sets the ORIGINAL transfer test computed (recomputed here
via `sae/transfer.py`'s own `concept_scores`/`top_series`, never by hand, and
checked against the recorded AUC as a free correctness test). Both sides are
ablated on `U` (never on their own separate top-firing rows) and scored with
`sae/response.py`'s existing 9-channel battery, level and shape separated
(sec 37.7 P4) so a shared LEVEL move is never mistaken for a shared SHAPE
effect. Two agreement statistics answer different questions: (i) per-series
Spearman concordance of the signed level effect (do the forecasts move the
same way on the same inputs?); (ii) cosine of the null-normalized shape-
channel vectors, over channels that clear in either model (do they move the
same KIND of thing?).

**Two separate nulls, for two separate questions (review of the first
version tightened this; see the deviation note below).** Whether a side's
own effect is real at all -- `clears_null`, `not scorable`, and the
per-channel `null_p95` that normalizes each side's shape vector for (ii) --
is decided by that side's own **row-matched random-direction null on `U`**
(`own_effect_null`), exactly the mechanism `feature_ablation_fingerprints`
already uses: `concepts.n_null_directions` random unit directions injected
via `_direction_steered_replacement` at `-mean(nonzero abs activation of the
ablated set on U)` magnitude, scored with the same raw/level-removed
battery. This answers "is this ablation's effect distinguishable from
removing an arbitrary direction of the same size" -- the ordinary causal
bar every other fingerprint in this repo clears. Whether the AGREEMENT
between the two sides' effects is more than an artifact of matched-feature
noise is a different question, and keeps its own floor: each side draws
`concepts.shared_input_n_null` random ALIVE feature sets from its OWN
dictionary, matched to the real set's size and its decile of mean pooled
activation on `U`, and re-scores the SAME statistic (i or ii) against the
OTHER side's fixed, already-measured effect. A statistic clears when the
observed value beats BOTH sides' matched-null p95.

Deviation from sec 37.8 item 6, recorded here as the spec instructs: this
writes its OWN `sae/shared_input_agreement.json` rather than a
`causal_agreement_shared_input` block inside `sae/transfer.json`. Reasons:
(1) the two artifacts have different units -- `transfer.json`'s rows are
INPUT-selectivity tests with no forward pass, while this module's rows carry
forward-pass ablation results, a materially heavier and differently-shaped
record; (2) `sae/transfer.json` is written by `run_transfer`, which this
module never calls (it reads `sae/atlas_transfer.json`, a sibling artifact);
bolting a causal block onto the wrong producer's file would make `transfer.
json` depend on a stage it does not otherwise touch. `sae/matching.py::
add_role_causal_agreement`'s own 1-vs-18 output is untouched by any of this.

Second deviation, an ORCHESTRATOR decision made on review of the v1 run on
`full_report_run_4model` (288 tests: 125 `not scorable`, 109 `acts
differently`), refining design item 5: v1 scored each side's OWN effect
against the matched feature sets rather than a random-direction null, which
is the wrong null for "is this effect real" (feature sets are not
independent draws from "nothing", so a small dictionary starves the floor
and a highly-selective one over-clears it) -- fixed by the two-null split
above. Separately, v1's `else` branch called "neither statistic clears its
matched-null p95" `acts differently`, but failing to clear a floor is not
evidence of disagreement -- it is the ABSENCE of evidence of agreement
beyond what matched, equally-active features already produce by chance. v2
also records each floor's p05 and only calls the pair `acts differently`
when an observed statistic falls BELOW the p05 of both sides' floors (worse
than matched features agree by chance -- disagreement beyond noise); the
genuine middle ground -- scorable on both sides, neither clearing nor
falling below the floor -- is its own verdict, `no specific agreement`.

Evidence class: causal WITHIN each model (an ablation, scored against that
model's own random-direction null), compared ACROSS models only on the same
corpus inputs -- never a transplant of one model's activation into another
(invariant 5).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from ..analysis.response_reach import reach_probe
from ..extraction.extract import capture_raw_tokens
from ..extraction.hooks import token_patch
from ..utils import load_json, log, save_json
from .eval import _token_level_replacement
from .ground_truth import load_ground_truth_table
from .response import CHANNELS, _direction_steered_replacement, _feature_ablated_replacement, \
    _score_channel_against_null, alive_feature_mask, battery_statistics
from .train import load_all_windows, load_sae_checkpoint, sanitize
from .transfer import _seed, auc_from_ranks, concept_scores, top_series

__all__ = ["shared_input_agreement_path", "run_shared_input_agreement",
           "cap_units_per_pair"]

SHAPE_CHANNELS = tuple(c for c in CHANNELS if c != "level")
_VERDICTS = ("same causal effect", "level only", "shape only",
            "acts differently", "no specific agreement", "not scorable")
_N_DECILE_BINS = 10


def shared_input_agreement_path(run_dir) -> Path:
    return Path(run_dir) / "sae" / "shared_input_agreement.json"


# ---------------------------------------------------------------------------
# 1. Build the test units from the FDR-surviving reciprocal atlas-transfer
#    tests (ROADMAP.md sec 37.8 P5b item 1).
# ---------------------------------------------------------------------------

def _atlas_parts(atlas: dict) -> dict:
    """`(concept_id, "model/layer") -> {"model": m, "features": [sorted]}`.

    Mirrors `sae/transfer.py::_concept_source_parts`'s grouping exactly (that
    helper is not imported because it also filters against a `live` set this
    module has no equivalent of -- `atlas_transfer.json`'s own tests already
    restrict themselves to live targets)."""
    out: dict = {}
    for r in atlas.get("rows") or []:
        cid = r.get("concept")
        if cid is None:
            continue
        key = (int(cid), f"{r['model']}/{r['layer']}")
        entry = out.setdefault(key, {"model": r["model"], "features": set()})
        entry["features"].add(int(r["feature"]))
    return {k: {"model": v["model"], "features": sorted(v["features"])}
            for k, v in out.items()}


def _index_by_target_feature(atlas: dict) -> dict:
    """`("model/layer", feature) -> (concept_id, sorted member features at
    THAT target)` for every atlas row with a concept assignment -- decides
    `dst_set_kind` (item 1: is the transfer test's best destination feature
    itself a member of an atlas concept part at that target?)."""
    members_at: dict = {}
    for r in atlas.get("rows") or []:
        cid = r.get("concept")
        if cid is None:
            continue
        key = (f"{r['model']}/{r['layer']}", int(cid))
        members_at.setdefault(key, set()).add(int(r["feature"]))
    out: dict = {}
    for r in atlas.get("rows") or []:
        cid = r.get("concept")
        if cid is None:
            continue
        target = f"{r['model']}/{r['layer']}"
        out[(target, int(r["feature"]))] = (int(cid), sorted(members_at[(target, cid)]))
    return out


def build_units(atlas: dict, atlas_transfer: dict) -> list:
    """Every `reciprocal_fdr` test in `atlas_transfer`, resolved into a
    (source set, destination set) work unit. A test whose source concept-part
    is absent from `atlas["rows"]` (should not happen -- that part is exactly
    what produced the test) is skipped defensively rather than raising."""
    src_parts = _atlas_parts(atlas)
    by_target_feature = _index_by_target_feature(atlas)
    k_top = int(atlas_transfer.get("k_top_series", 20))
    units = []
    for t in atlas_transfer.get("tests") or []:
        if not t.get("reciprocal_fdr"):
            continue
        part = src_parts.get((int(t["concept"]), t["src_target"]))
        if not part:
            log.warning("shared_input_agreement: test for concept %s at %s has no atlas "
                       "source part; skipped", t.get("concept"), t.get("src_target"))
            continue
        dst_feature = int(t["feature"])
        hit = by_target_feature.get((t["dst_target"], dst_feature))
        if hit is not None:
            _dst_concept, dst_features = hit
            dst_kind = "atlas_part"
        else:
            dst_features = [dst_feature]
            dst_kind = "feature"
        units.append({
            "concept": int(t["concept"]), "src_target": t["src_target"],
            "src_model": t["src_model"], "src_features": part["features"],
            "dst_target": t["dst_target"], "dst_model": t["dst_model"],
            "dst_feature": dst_feature, "dst_features": dst_features,
            "dst_set_kind": dst_kind, "k_top_series": k_top,
            "test_auc": t.get("auc"),
            "transfer_margin": _transfer_margin(t),
        })
    return units


def _transfer_margin(test: dict) -> float | None:
    """The reciprocal transfer margin of one atlas-transfer test: the smaller
    of the forward and reverse `auc - null_p95`, so a test is only as strong
    as its weaker leg. None when either leg's fields are absent (a legacy
    artifact), which the cap ranks last."""
    legs = []
    for auc_key, null_key in (("auc", "null_p95"), ("rev_auc", "rev_null_p95")):
        if test.get(auc_key) is None or test.get(null_key) is None:
            return None
        legs.append(float(test[auc_key]) - float(test[null_key]))
    return min(legs)


def cap_units_per_pair(units: list, cap: int | None) -> tuple:
    """`(kept_units, record)` under `concepts.agreement_max_tests_per_pair`.

    `cap is None` returns `units` untouched and `record is None`, so the
    default path is byte-identical to a run without this knob. Otherwise each
    ordered (src_model, dst_model) pair keeps its `cap` units with the
    largest `transfer_margin` (None ranks last; ties break on concept, source
    target, destination target, destination feature so the selection is
    deterministic), and the original relative order of the kept units is
    preserved. The record carries per-pair `n_before`, `n_kept` and
    `n_dropped`, so the report can say the step was capped and by how much.
    """
    if cap is None:
        return units, None
    cap = int(cap)
    if cap < 1:
        raise ValueError(f"concepts.agreement_max_tests_per_pair must be >= 1 or null, got {cap}")
    by_pair: dict = {}
    for i, u in enumerate(units):
        by_pair.setdefault((u["src_model"], u["dst_model"]), []).append(i)

    def _rank(i: int) -> tuple:
        u = units[i]
        m = u.get("transfer_margin")
        return (m is None, -(m if m is not None else 0.0), u["concept"], u["src_target"],
                u["dst_target"], u["dst_feature"])

    keep: set = set()
    pairs: dict = {}
    for (src, dst), idx in sorted(by_pair.items()):
        kept = sorted(idx, key=_rank)[:cap]
        keep.update(kept)
        pairs[f"{src}->{dst}"] = {"n_before": len(idx), "n_kept": len(kept),
                                  "n_dropped": len(idx) - len(kept)}
    record = {"max_tests_per_pair": cap, "ranked_by": "reciprocal transfer margin "
              "(min of forward and reverse auc - null_p95)",
              "n_before": len(units), "n_kept": len(keep),
              "n_dropped": len(units) - len(keep), "pairs": pairs}
    return [u for i, u in enumerate(units) if i in keep], record


# ---------------------------------------------------------------------------
# 2. Per-target context: adapter, SAE, pooled features, alive mask, reach.
# ---------------------------------------------------------------------------

class TargetContext:
    """Everything one (model, layer) target needs, loaded once and cached by
    the driver across every unit that touches it."""

    def __init__(self, cfg, hub, store, data, run_dir, model: str, layer: str, device):
        self.cfg, self.device = cfg, device
        self.model, self.layer = model, layer
        ckpt_path = Path(run_dir) / "sae" / sanitize(model) / f"{sanitize(layer)}.pt"
        self.sae = load_sae_checkpoint(str(ckpt_path)).to(device)
        self.adapter = hub.get(model)
        self.adapter.ensure_loaded()
        self.pooled = np.asarray(store.load(model, layer, level="series", space="sae"),
                                 dtype=np.float64)
        bench = load_all_windows(store, model, layer)
        self.alive_mask = alive_feature_mask(self.sae, bench, device)
        self.reach = reach_probe(cfg, self.adapter, layer, data, device)
        self._ranks = None

    @property
    def ranks(self) -> np.ndarray:
        if self._ranks is None:
            self._ranks = rankdata(self.pooled, axis=0)
        return self._ranks


def _target_key(unit_target: str) -> tuple:
    model, layer = unit_target.split("/", 1)
    return model, layer


# ---------------------------------------------------------------------------
# 3. Series sets (item 2) and the free correctness check.
# ---------------------------------------------------------------------------

def series_sets(ctx_src: TargetContext, ctx_dst: TargetContext, unit: dict) -> tuple:
    """`-> (S_src, S_dst, score_src)`, recomputed with `transfer.py`'s own
    `concept_scores`/`top_series` (sec 37.8 P5b item 2), never by hand."""
    k_top = unit["k_top_series"]
    score_src = concept_scores(ctx_src.pooled, unit["src_features"])
    S_src = top_series(score_src, k_top)
    score_dst_feature = concept_scores(ctx_dst.pooled, [unit["dst_feature"]])
    S_dst = top_series(score_dst_feature, k_top)
    return S_src, S_dst, score_src


def verify_auc_reproduces(ctx_dst: TargetContext, S_src: np.ndarray, dst_feature: int,
                          expected_auc: float, tol: float = 1e-6) -> float:
    """The free correctness check item 2 asks for: the recomputed `S_src`
    against the destination's own rank matrix must reproduce the transfer
    test's recorded forward AUC at its own best feature."""
    recomputed = float(auc_from_ranks(ctx_dst.ranks, S_src)[dst_feature])
    if not np.isclose(recomputed, expected_auc, atol=tol, rtol=1e-6):
        raise AssertionError(
            f"shared_input_agreement: recomputed forward AUC {recomputed!r} does not "
            f"reproduce the transfer test's recorded {expected_auc!r} (tol={tol}) -- "
            f"S_src/top_series diverges from what run_atlas_transfer computed")
    return recomputed


# ---------------------------------------------------------------------------
# 4. Ablation batteries on a shared row set U (raw + level-removed).
# ---------------------------------------------------------------------------

_baseline_cache: dict = {}
_battery_cache: dict = {}
_own_null_cache: dict = {}


def reset_caches() -> None:
    """Tests call this between runs so module-level caches (keyed only by
    content, not by run) cannot leak state across fixtures."""
    _baseline_cache.clear()
    _battery_cache.clear()
    _own_null_cache.clear()


def _baseline_for_rows(ctx: TargetContext, contexts_u: np.ndarray, U_key: tuple,
                       seed: int) -> tuple:
    """Cached per `(model, layer, U)`, deliberately not per-seed: every caller
    at this `(model, layer, U)` must be seeding a comparison against the SAME
    baseline forecast (§11.49/§11.50), so a caller passing a different seed
    than the one that produced the cached baseline is a bug, not a cache
    miss to silently recompute -- it would mean scoring a real or null
    forward pass against a baseline sampled under a DIFFERENT seed (exactly
    P5b's own-effect-null defect, ROADMAP.md sec 37.8 "Sundial as a
    destination is almost never scorable"). The cache therefore records the
    seed it was built with and asserts every later call agrees."""
    key = (ctx.model, ctx.layer, U_key)
    if key in _baseline_cache:
        cached_seed, cached = _baseline_cache[key]
        if cached_seed != seed:
            raise AssertionError(
                f"_baseline_for_rows: cached baseline for {key} was produced under "
                f"seed {cached_seed!r}, but this call passed seed {seed!r} -- every "
                f"caller sharing this (model, layer, U) baseline must pass the same "
                f"seed the baseline was created with")
        return cached
    clean_tokens = capture_raw_tokens(ctx.adapter, contexts_u, [ctx.layer])[ctx.layer]
    full_recon = _token_level_replacement(clean_tokens, ctx.sae, ctx.device)
    with token_patch(ctx.adapter.module, ctx.layer, ctx.adapter.token_slice, full_recon):
        torch.manual_seed(seed)
        rec_full = ctx.adapter.predict(contexts_u, ctx.cfg.data.horizon, ctx.cfg.l0.quantiles)
    out = (clean_tokens, rec_full["point"], rec_full.get("quantiles"))
    _baseline_cache[key] = (seed, out)
    return out


def battery_for_set(ctx: TargetContext, features, U_key: tuple, contexts_u: np.ndarray,
                    targets_u: np.ndarray, periods_u: np.ndarray, seed: int) -> tuple:
    """Ablate `features` (an int or an iterable of ints) on the rows named by
    `U_key`, and score the raw and level-removed 9-channel battery against
    the SAME full-reconstruction baseline (cached per (model, layer, U), sec
    37.8 P5b item 3). Cached by `(model, layer, sorted(features), U)`, which
    is item 1's "deduplicate identical (source set, destination set) work"
    for BOTH real ablations and null draws -- any repeated feature-set/`U`
    pair, real or null, is computed once and read back."""
    feat_list = [int(features)] if isinstance(features, (int, np.integer)) else \
        sorted(int(f) for f in features)
    key = (ctx.model, ctx.layer, tuple(feat_list), U_key)
    if key in _battery_cache:
        return _battery_cache[key]
    clean_tokens, baseline_fc, baseline_q = _baseline_for_rows(ctx, contexts_u, U_key, seed)
    ablated = _feature_ablated_replacement(clean_tokens, ctx.sae, ctx.device, feat_list)
    with token_patch(ctx.adapter.module, ctx.layer, ctx.adapter.token_slice, ablated):
        torch.manual_seed(seed)
        rec = ctx.adapter.predict(contexts_u, ctx.cfg.data.horizon, ctx.cfg.l0.quantiles)
    stats_raw = battery_statistics(rec["point"], baseline_fc, targets_u, contexts_u, periods_u,
                                   steered_quantiles=rec.get("quantiles"),
                                   baseline_quantiles=baseline_q, remove_level=False)
    stats_shape = battery_statistics(rec["point"], baseline_fc, targets_u, contexts_u, periods_u,
                                     steered_quantiles=rec.get("quantiles"),
                                     baseline_quantiles=baseline_q, remove_level=True)
    out = (stats_raw, stats_shape)
    _battery_cache[key] = out
    return out


def own_effect_null(ctx: TargetContext, features, U_key: tuple, contexts_u: np.ndarray,
                    targets_u: np.ndarray, periods_u: np.ndarray, baseline_seed: int,
                    direction_seed: int, n_null: int) -> list:
    """The null that decides whether `features`' own effect on `U` is real at
    all (review of the v1 run, tightening design item 5's floor -- see the
    module docstring's second deviation note): `n_null` row-matched
    random-direction passes, exactly `feature_ablation_fingerprints`'s own
    mechanism, NEVER the matched candidate feature sets used for the
    AGREEMENT floor (`matched_null_sets`) -- those are a fixed, sparse
    handful of other dictionary atoms and are not independent draws from
    "no effect", so using them here starves a small dictionary's floor and
    over-clears a highly selective one.

    **Two seeds, deliberately not one (fix for ROADMAP.md sec 37.8's
    "Sundial as a destination is almost never scorable", diagnosed
    2026-09-27 as an instrument defect).** `baseline_seed` MUST be the same seed that
    produced (or will produce) this `(model, layer, U)`'s cached baseline
    forecast in `_baseline_for_rows` -- i.e. the side's `seed_src`/`seed_dst`
    from `battery_for_set` -- and is the ONLY seed used to reseed
    `torch.manual_seed` before each null `predict()` call below. For a
    *sampled* model (Sundial's flow-matching head; Chronos-T5 too), seeding
    the null forward pass with anything else injects fresh sampling noise
    into `null_delta = null_forecast - baseline_forecast` that the real
    ablation delta (scored under the SAME `baseline_seed` throughout,
    `battery_for_set`) never carries, inflating the null far above the real
    effect and making the side almost never scorable. `direction_seed` seeds
    ONLY `np.random.default_rng` for the null DIRECTION draws, which do not
    touch the model's own sampling.

    `magnitude` is `-mean(nonzero abs activation of `features` on U)`, the
    same convention `feature_ablation_fingerprints` uses for its own
    candidates -- "this effect" means more than removing that much of an
    arbitrary direction does. Returns a list of `(stats_raw, stats_shape)`
    tuples, the SAME shape `battery_for_set` returns for one candidate, so
    `_side_channel_scores` (unchanged) can score against either.

    Cached by `(model, layer, sorted(features), U)`: the magnitude is a
    property of the specific ablated set, so two different sets at the same
    (target, U) are two different nulls, but a repeated (set, U) -- e.g. the
    same source part scored against two different destinations that share a
    `U` -- is computed once. Both seeds are deterministic functions of
    exactly these same components (the driver's `_seed(...)` calls), so a
    repeated (set, U) always implies the same pair of seeds too."""
    feat_list = [int(features)] if isinstance(features, (int, np.integer)) else \
        sorted(int(f) for f in features)
    key = (ctx.model, ctx.layer, tuple(feat_list), U_key)
    if key in _own_null_cache:
        return _own_null_cache[key]
    clean_tokens, baseline_fc, baseline_q = _baseline_for_rows(ctx, contexts_u, U_key,
                                                               baseline_seed)
    with torch.no_grad():
        d_in = clean_tokens.shape[-1]
        enc = ctx.sae.encode(clean_tokens.reshape(-1, d_in).to(ctx.device))
        removed = [float(enc[:, f].abs().mean().cpu()) for f in feat_list]
    nonzero = [m for m in removed if m > 0]
    null_magnitude = -float(np.mean(nonzero) if nonzero else 1.0)

    rng = np.random.default_rng(direction_seed)
    out = []
    for _ in range(int(n_null)):
        direction = rng.normal(size=ctx.sae.dict_size)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        replacement = _direction_steered_replacement(
            clean_tokens, ctx.sae, ctx.device,
            torch.as_tensor(direction, dtype=torch.float32), null_magnitude)
        with token_patch(ctx.adapter.module, ctx.layer, ctx.adapter.token_slice, replacement):
            torch.manual_seed(baseline_seed)
            rec = ctx.adapter.predict(contexts_u, ctx.cfg.data.horizon, ctx.cfg.l0.quantiles)
        stats_raw = battery_statistics(rec["point"], baseline_fc, targets_u, contexts_u, periods_u,
                                       steered_quantiles=rec.get("quantiles"),
                                       baseline_quantiles=baseline_q, remove_level=False)
        stats_shape = battery_statistics(rec["point"], baseline_fc, targets_u, contexts_u, periods_u,
                                         steered_quantiles=rec.get("quantiles"),
                                         baseline_quantiles=baseline_q, remove_level=True)
        out.append((stats_raw, stats_shape))
    _own_null_cache[key] = out
    return out


# ---------------------------------------------------------------------------
# 5. The within-run floor: matched random alive feature sets (item 5).
# ---------------------------------------------------------------------------

def matched_null_sets(ctx: TargetContext, real_features: list, U: np.ndarray,
                      n_null: int, seed: int) -> tuple:
    """`-> (null_sets [list of int lists], diag {dict})`.

    Candidates are every ALIVE feature of `ctx`'s own dictionary except
    `real_features`, restricted to the same DECILE of mean pooled activation
    on `U` as `real_features`' own mean (item 5). `diag` records the
    achieved pool size and the achieved number of draws whenever either
    falls short of what was requested -- never a silently repeated draw.
    """
    alive_idx = np.flatnonzero(ctx.alive_mask)
    exclude = set(int(f) for f in real_features)
    candidates = np.array([i for i in alive_idx if i not in exclude], dtype=int)
    set_size = max(1, len(real_features))
    diag = {"requested_n_null": int(n_null), "requested_set_size": set_size,
           "pool_size_before_decile": int(candidates.size)}
    if candidates.size == 0:
        diag.update({"pool_size": 0, "achieved_set_size": 0, "achieved_n_null": 0})
        return [], diag

    pooled_u = ctx.pooled[U]
    act_bar_candidates = pooled_u[:, candidates].mean(axis=0)
    act_bar_real = float(pooled_u[:, sorted(exclude)].mean())
    if act_bar_candidates.size >= 2:
        edges = np.quantile(act_bar_candidates,
                            np.linspace(1.0 / _N_DECILE_BINS, 1.0 - 1.0 / _N_DECILE_BINS,
                                       _N_DECILE_BINS - 1))
    else:
        edges = np.array([])
    target_bin = int(np.searchsorted(edges, act_bar_real, side="right"))
    candidate_bins = np.searchsorted(edges, act_bar_candidates, side="right")
    pool = candidates[candidate_bins == target_bin]
    diag["pool_size"] = int(pool.size)

    rng = np.random.default_rng(seed)
    if pool.size < set_size:
        achieved_set_size = int(pool.size)
        diag["achieved_set_size"] = achieved_set_size
        if achieved_set_size == 0:
            diag["achieved_n_null"] = 0
            return [], diag
        diag["achieved_n_null"] = 1
        diag["note"] = (f"matched pool ({pool.size}) smaller than the requested set size "
                        f"({set_size}); ablated the whole pool once rather than repeating "
                        f"a draw")
        return [sorted(int(i) for i in pool)], diag

    diag["achieved_set_size"] = set_size
    draws = [sorted(int(i) for i in rng.choice(pool, size=set_size, replace=False))
            for _ in range(int(n_null))]
    diag["achieved_n_null"] = len(draws)
    return draws, diag


# ---------------------------------------------------------------------------
# 6. Per-side channel scoring and the two agreement statistics (item 4).
# ---------------------------------------------------------------------------

def _channel_deltas(stats: dict, channel: str, idx: np.ndarray | None = None) -> np.ndarray | None:
    rec = stats.get(channel)
    if rec is None or not rec.get("available") or rec.get("delta") is None:
        return None
    d = np.asarray(rec["delta"], dtype=np.float64)
    return d if idx is None else d[idx]


def _side_channel_scores(real_stats_raw: dict, real_stats_shape: dict,
                         own_null: list) -> dict:
    """`-> {"level": score_dict, "shape": {channel: score_dict}}`, each
    `score_dict` from `response.py::_score_channel_against_null` (reused
    directly, never re-derived -- the same gating arithmetic every other
    causal fingerprint in this repo uses). `own_null` is THIS side's own
    row-matched random-direction null (`own_effect_null`, review item 1) --
    it decides whether the real effect is distinguishable from removing an
    arbitrary direction at all, which is a different question from whether
    the two sides AGREE (the matched feature sets scored by `_statistic_i`/
    `_statistic_ii` answer that one, and never reach this function)."""
    level_real = _channel_deltas(real_stats_raw, "level")
    level_null = [np.abs(_channel_deltas(sr, "level")) for sr, _ in own_null
                 if _channel_deltas(sr, "level") is not None]
    level_score = (_score_channel_against_null(level_real, level_null)
                  if level_real is not None else
                  {"available": False, "effect": None, "signed_effect": None,
                   "null_p95": None, "clears_null": False, "reason": "level channel unavailable"})

    shape_scores = {}
    for ch in SHAPE_CHANNELS:
        real_delta = _channel_deltas(real_stats_shape, ch)
        if real_delta is None:
            shape_scores[ch] = {"available": False, "effect": None, "signed_effect": None,
                                "null_p95": None, "clears_null": False,
                                "reason": "channel unavailable on this row set"}
            continue
        null_draws = [np.abs(_channel_deltas(ss, ch)) for _, ss in own_null
                      if _channel_deltas(ss, ch) is not None]
        shape_scores[ch] = _score_channel_against_null(real_delta, null_draws)
    return {"level": level_score, "shape": shape_scores}


def _clearing_channels(side_scores: dict) -> list:
    out = []
    if side_scores["level"].get("clears_null"):
        out.append("level")
    out += [c for c in SHAPE_CHANNELS if side_scores["shape"].get(c, {}).get("clears_null")]
    return out


def _normalized_shape_vector(side_scores: dict, mask: list) -> np.ndarray:
    vals = []
    for c in mask:
        rec = side_scores["shape"].get(c, {})
        p95 = rec.get("null_p95")
        eff = rec.get("signed_effect")
        vals.append(eff / p95 if (p95 not in (None, 0.0) and eff is not None) else 0.0)
    return np.array(vals, dtype=np.float64)


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size == 0:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def _spearman(x, y) -> float | None:
    if x is None or y is None:
        return None
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    r, _p = spearmanr(x, y)
    return float(r) if np.isfinite(r) else None


def _quantile95(values: list) -> float | None:
    return float(np.quantile(values, 0.95)) if values else None


def _quantile05(values: list) -> float | None:
    return float(np.quantile(values, 0.05)) if values else None


def _statistic_i(level_a_real, level_b_real, null_stats_a: list, null_stats_b: list) -> dict:
    """Item 4(i): per-series Spearman concordance of the signed level effect,
    against BOTH sides' matched-null floors (item 5). `clears` (beats both
    p95s) and `below_floor` (falls below both p05s -- review's second
    deviation note: disagreement beyond what matched, equally-active
    features produce by chance, never merely "did not clear") are reported
    together; neither implies the other, and both can be false (the
    ordinary "no specific agreement" case)."""
    obs = _spearman(level_a_real, level_b_real)
    floor_src = []
    for stats_raw, _stats_shape in null_stats_a:
        d = _channel_deltas(stats_raw, "level")
        v = _spearman(d, level_b_real) if d is not None else None
        if v is not None:
            floor_src.append(v)
    floor_dst = []
    for stats_raw, _stats_shape in null_stats_b:
        d = _channel_deltas(stats_raw, "level")
        v = _spearman(level_a_real, d) if d is not None else None
        if v is not None:
            floor_dst.append(v)
    p95_src, p95_dst = _quantile95(floor_src), _quantile95(floor_dst)
    p05_src, p05_dst = _quantile05(floor_src), _quantile05(floor_dst)
    clears = bool(obs is not None and p95_src is not None and p95_dst is not None
                 and obs > p95_src and obs > p95_dst)
    below_floor = bool(obs is not None and p05_src is not None and p05_dst is not None
                       and obs < p05_src and obs < p05_dst)
    return {"observed": obs, "floor_p95_src": p95_src, "floor_p95_dst": p95_dst,
           "floor_p05_src": p05_src, "floor_p05_dst": p05_dst,
           "n_floor_src": len(floor_src), "n_floor_dst": len(floor_dst),
           "clears": clears, "below_floor": below_floor}


def _null_normalized_vector(null_stats_shape: dict, side_scores: dict, mask: list) -> np.ndarray:
    """A null draw's shape vector, normalized by the REAL side's own fixed
    per-channel null p95 (item 5: only the treatment side's ablation set
    changes across a floor draw; the channel-clearing decision that produced
    `mask` and each `null_p95` stays the one measured on the real run)."""
    vals = []
    for c in mask:
        p95 = side_scores["shape"][c].get("null_p95")
        d = _channel_deltas(null_stats_shape, c)
        if p95 in (None, 0.0) or d is None:
            vals.append(0.0)
        else:
            vals.append(float(np.nanmean(d)) / p95)
    return np.array(vals, dtype=np.float64)


def _statistic_ii(side_a: dict, side_b: dict, null_stats_a: list, null_stats_b: list,
                  mask: list) -> dict:
    """Item 4(ii): cosine of the null-normalized shape-channel vectors, over
    channels clearing in either model, against both sides' matched-null
    floors. See `_statistic_i` for `clears` vs `below_floor`."""
    if not mask:
        return {"observed": None, "floor_p95_src": None, "floor_p95_dst": None,
               "floor_p05_src": None, "floor_p05_dst": None,
               "n_floor_src": 0, "n_floor_dst": 0, "clears": False, "below_floor": False,
               "reason": "no shape channel is available and clears in either model"}
    vec_a = _normalized_shape_vector(side_a, mask)
    vec_b = _normalized_shape_vector(side_b, mask)
    obs = _cosine(vec_a, vec_b)

    floor_src = []
    for _stats_raw, stats_shape in null_stats_a:
        c = _cosine(_null_normalized_vector(stats_shape, side_a, mask), vec_b)
        if c is not None:
            floor_src.append(c)
    floor_dst = []
    for _stats_raw, stats_shape in null_stats_b:
        c = _cosine(vec_a, _null_normalized_vector(stats_shape, side_b, mask))
        if c is not None:
            floor_dst.append(c)
    p95_src, p95_dst = _quantile95(floor_src), _quantile95(floor_dst)
    p05_src, p05_dst = _quantile05(floor_src), _quantile05(floor_dst)
    clears = bool(obs is not None and p95_src is not None and p95_dst is not None
                 and obs > p95_src and obs > p95_dst)
    below_floor = bool(obs is not None and p05_src is not None and p05_dst is not None
                       and obs < p05_src and obs < p05_dst)
    return {"observed": obs, "floor_p95_src": p95_src, "floor_p95_dst": p95_dst,
           "floor_p05_src": p05_src, "floor_p05_dst": p05_dst,
           "n_floor_src": len(floor_src), "n_floor_dst": len(floor_dst),
           "clears": clears, "below_floor": below_floor}


# ---------------------------------------------------------------------------
# 7. Driver.
# ---------------------------------------------------------------------------

def run_shared_input_agreement(cfg, run_dir, hub, store, data, device, atlas: dict,
                               atlas_transfer: dict) -> dict:
    """Score every FDR-surviving reciprocal atlas-transfer test and write
    `sae/shared_input_agreement.json`. Gates each target's reach FIRST (P5a),
    then ablates on the shared series `U`, scores each side's own effect
    against its row-matched random-direction null (`own_effect_null`,
    review item 1), scores the two agreement statistics against each side's
    own matched-random-SET floor, and assigns a verdict from both floors'
    p95 (clears) and p05 (below_floor, review item 2). See the module
    docstring for both deviations from ROADMAP.md sec 37.8.
    """
    t0 = time.monotonic()
    c = cfg.concepts
    n_null = int(getattr(c, "shared_input_n_null", 50))
    base_seed = int(cfg.run.seed)

    units = build_units(atlas, atlas_transfer)
    units, cap_record = cap_units_per_pair(
        units, getattr(c, "agreement_max_tests_per_pair", None))
    if cap_record is not None:
        log.warning("shared_input_agreement: capped at %d test(s) per ordered model pair -- "
                    "%d of %d reciprocal-FDR test(s) not scored",
                    cap_record["max_tests_per_pair"], cap_record["n_dropped"],
                    cap_record["n_before"])

    periods_full = None
    try:
        gt = load_ground_truth_table(cfg.data.path)
        periods_full = gt.reindex(data.meta["series_id"].to_numpy())[
            "seasonal_period_dominant"].to_numpy(dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 -- degrade the seasonal channel, not the module
        log.warning("shared_input_agreement: no ground-truth periods (%s); seasonal "
                   "channel unavailable for every test", exc)

    contexts: dict = {}

    def _ctx(target: str) -> TargetContext:
        if target not in contexts:
            model, layer = _target_key(target)
            contexts[target] = TargetContext(cfg, hub, store, data, run_dir, model, layer, device)
        return contexts[target]

    verified_pairs = set()
    tests: list = []
    reach_records: dict = {}
    dst_kind_counts = {"feature": 0, "atlas_part": 0}
    n_short_pool = 0

    for unit in units:
        dst_kind_counts[unit["dst_set_kind"]] += 1
        ctx_src, ctx_dst = _ctx(unit["src_target"]), _ctx(unit["dst_target"])
        reach_records.setdefault(unit["src_target"], ctx_src.reach)
        reach_records.setdefault(unit["dst_target"], ctx_dst.reach)

        record = {"concept": unit["concept"], "src_target": unit["src_target"],
                  "src_model": unit["src_model"], "src_features": unit["src_features"],
                  "dst_target": unit["dst_target"], "dst_model": unit["dst_model"],
                  "dst_feature": unit["dst_feature"], "dst_features": unit["dst_features"],
                  "dst_set_kind": unit["dst_set_kind"]}

        if not ctx_src.reach["reachable"] or not ctx_dst.reach["reachable"]:
            bad_target, bad_reach = ((unit["src_target"], ctx_src.reach)
                                     if not ctx_src.reach["reachable"] else
                                     (unit["dst_target"], ctx_dst.reach))
            record["verdict"] = "not scorable"
            record["reason"] = (f"{bad_target} does not causally reach the forecast "
                                f"(reach_probe): {bad_reach['reason']}")
            tests.append(record)
            continue

        S_src, S_dst, _score_src = series_sets(ctx_src, ctx_dst, unit)
        pair_key = (unit["src_model"], unit["dst_model"])
        if pair_key not in verified_pairs and unit["test_auc"] is not None:
            verify_auc_reproduces(ctx_dst, S_src, unit["dst_feature"], unit["test_auc"])
            verified_pairs.add(pair_key)

        U = np.array(sorted(set(S_src.tolist()) | set(S_dst.tolist())), dtype=int)
        U_key = tuple(int(u) for u in U)
        contexts_u = data.contexts()[U]
        targets_u = data.targets()[U]
        periods_u = (periods_full[U] if periods_full is not None
                    else np.full(U.size, np.nan))

        seed_src = _seed("shared_input_baseline", unit["src_target"], U_key, base=base_seed)
        seed_dst = _seed("shared_input_baseline", unit["dst_target"], U_key, base=base_seed)
        real_raw_a, real_shape_a = battery_for_set(ctx_src, unit["src_features"], U_key,
                                                   contexts_u, targets_u, periods_u, seed_src)
        real_raw_b, real_shape_b = battery_for_set(ctx_dst, unit["dst_features"], U_key,
                                                   contexts_u, targets_u, periods_u, seed_dst)

        null_seed_a = _seed("shared_input_null", unit["src_target"],
                            tuple(sorted(unit["src_features"])), U_key, base=base_seed)
        null_seed_b = _seed("shared_input_null", unit["dst_target"],
                            tuple(sorted(unit["dst_features"])), U_key, base=base_seed)
        null_sets_a, diag_a = matched_null_sets(ctx_src, unit["src_features"], U, n_null, null_seed_a)
        null_sets_b, diag_b = matched_null_sets(ctx_dst, unit["dst_features"], U, n_null, null_seed_b)
        if (diag_a.get("pool_size", 0) < diag_a["requested_set_size"]
               or diag_b.get("pool_size", 0) < diag_b["requested_set_size"]):
            n_short_pool += 1

        null_stats_a = [battery_for_set(ctx_src, ns, U_key, contexts_u, targets_u, periods_u, seed_src)
                       for ns in null_sets_a]
        null_stats_b = [battery_for_set(ctx_dst, ns, U_key, contexts_u, targets_u, periods_u, seed_dst)
                       for ns in null_sets_b]

        # Review of the v1 run (ROADMAP.md sec 37.8 P5b review item 1):
        # whether a side's OWN effect is real is decided by its own
        # row-matched random-direction null, never the matched candidate
        # feature sets above (those are ONLY the floor for statistics
        # (i)/(ii) themselves -- see the module docstring).
        #
        # Fix (2026-09-27, ROADMAP sec 37.8 "Sundial as a destination is
        # almost never scorable"): the null forward pass must be reseeded
        # with the SAME seed that produced this side's cached baseline
        # (`seed_src`/`seed_dst`, already computed above for `battery_for_set`),
        # never `own_null_seed`, which now seeds ONLY the direction draws. See
        # `own_effect_null`'s docstring.
        own_null_seed_a = _seed("shared_input_own_null", unit["src_target"],
                                tuple(sorted(unit["src_features"])), U_key, base=base_seed)
        own_null_seed_b = _seed("shared_input_own_null", unit["dst_target"],
                                tuple(sorted(unit["dst_features"])), U_key, base=base_seed)
        n_null_directions = int(getattr(c, "n_null_directions", 16))
        own_null_a = own_effect_null(ctx_src, unit["src_features"], U_key, contexts_u, targets_u,
                                     periods_u, seed_src, own_null_seed_a, n_null_directions)
        own_null_b = own_effect_null(ctx_dst, unit["dst_features"], U_key, contexts_u, targets_u,
                                     periods_u, seed_dst, own_null_seed_b, n_null_directions)

        side_a = _side_channel_scores(real_raw_a, real_shape_a, own_null_a)
        side_b = _side_channel_scores(real_raw_b, real_shape_b, own_null_b)
        clearing_a, clearing_b = _clearing_channels(side_a), _clearing_channels(side_b)

        record.update({
            "U": list(U_key), "n_shared_series": int(U.size),
            "matched_null_diag": {"src": diag_a, "dst": diag_b},
            "own_effect_null": {"src": {"n_directions": n_null_directions, "n": len(own_null_a)},
                               "dst": {"n_directions": n_null_directions, "n": len(own_null_b)}},
            "side_src": {"clearing_channels": clearing_a, "level": side_a["level"],
                        "shape": side_a["shape"]},
            "side_dst": {"clearing_channels": clearing_b, "level": side_b["level"],
                        "shape": side_b["shape"]},
        })

        if not clearing_a or not clearing_b:
            empty_target = unit["src_target"] if not clearing_a else unit["dst_target"]
            record["verdict"] = "not scorable"
            record["reason"] = (f"{empty_target} has no channel (raw level or "
                                f"level-removed shape) clearing its own random-direction "
                                f"null on the shared series")
            tests.append(record)
            continue

        level_a_real = _channel_deltas(real_raw_a, "level")
        level_b_real = _channel_deltas(real_raw_b, "level")
        stat_i = _statistic_i(level_a_real, level_b_real, null_stats_a, null_stats_b)

        mask = [ch for ch in SHAPE_CHANNELS
               if side_a["shape"][ch]["available"] and side_b["shape"][ch]["available"]
               and (side_a["shape"][ch]["clears_null"] or side_b["shape"][ch]["clears_null"])]
        stat_ii = _statistic_ii(side_a, side_b, null_stats_a, null_stats_b, mask)

        record["statistic_i"] = stat_i
        record["statistic_ii"] = stat_ii
        record["shape_mask"] = mask

        # Review item 2: failing to clear the matched-feature floor is the
        # ABSENCE of evidence of agreement, not evidence of disagreement.
        # `acts differently` is reserved for an observed statistic falling
        # BELOW the p05 of BOTH sides' floors -- worse than matched,
        # equally-active features already agree by chance.
        if stat_i["clears"] and stat_ii["clears"]:
            verdict = "same causal effect"
        elif stat_i["clears"]:
            verdict = "level only"
        elif stat_ii["clears"]:
            verdict = "shape only"
        elif stat_i["below_floor"] or stat_ii["below_floor"]:
            verdict = "acts differently"
        else:
            verdict = "no specific agreement"
        record["verdict"] = verdict
        tests.append(record)

    verdict_counts: dict = {}
    pair_counts: dict = {}
    concept_counts: dict = {}
    for r in tests:
        v = r["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        pkey = f"{r['src_model']}->{r['dst_model']}"
        pair_counts.setdefault(pkey, {})
        pair_counts[pkey][v] = pair_counts[pkey].get(v, 0) + 1
        concept_counts.setdefault(str(r["concept"]), {})
        concept_counts[str(r["concept"])][v] = concept_counts[str(r["concept"])].get(v, 0) + 1

    out = {
        "schema_version": 1,
        "params": {"n_null": n_null, "shape_channels": list(SHAPE_CHANNELS),
                  "verdicts": list(_VERDICTS)},
        "n_tests": len(tests),
        "dst_set_kind_counts": dst_kind_counts,
        "n_short_matched_pool": n_short_pool,
        "tests": tests,
        "verdict_counts": verdict_counts,
        "pair_verdict_counts": pair_counts,
        "concept_verdict_counts": concept_counts,
        "reach": reach_records,
        "runtime_seconds": time.monotonic() - t0,
    }
    if cap_record is not None:
        out["agreement_cap"] = cap_record
    out_path = shared_input_agreement_path(run_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    log.info("shared input agreement: wrote %s (%d test(s), verdicts=%s, %.1fs)",
             out_path, len(tests), verdict_counts, out["runtime_seconds"])
    return out
