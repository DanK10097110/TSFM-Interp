"""ROADMAP.md sec 37 P2 -- seed stability of atlas concepts, and the
within-model transfer ceiling.

`sae/concept_atlas.py` pools every target's causal SAE features into one
cross-model space and clusters them into concepts. That answers "do TWO
MODELS group features the same way" (via `sae/transfer.py`'s reciprocal
test). It says nothing about whether a concept is a property of the MODEL
being decomposed or of the one stochastic SAE draw that happened to be
trained. This module answers that, by the same transfer test `transfer.py`
already uses -- run within a model, between two independently-seeded
dictionaries at the SAME layer, on the SAME activations, instead of between
two models.

EVIDENCE CLASS, stated once here because every consumer of this artifact
inherits it: "stable" means an atlas concept's series-grouping survives a
second, independent SAE training at the same target -- reproducibility of a
DECOMPOSITION, not confirmation that the underlying causal structure is
real (that is what the ablation battery's own null, sec 25.23, already
establishes). A concept can be causally real and still SAE-unstable (e.g.
one of two co-occurring structures wins the dictionary's limited capacity
on one draw and not the other), and it can be SAE-stable while still being
an artifact of this corpus. Read `stable` as "reproducible", not as
"real".

The within-model transfer CEILING this module also computes (sec 37 P2's
second question) is the natural denominator for reading any CROSS-model
transfer rate from `transfer.py`: a model that cannot reproducibly transfer
a concept to its own second SAE draw is not a fair target for a "does
another model transfer this" question either, and a cross-model rate should
be read relative to this ceiling, not to 1.0.

Mechanism, reusing `sae/transfer.py`'s own machinery throughout (never
reimplemented -- `CLAUDE.md` sec 2.2/sec 11.41): for one atlas concept's
PART at one target T (the subset of its member features that live at T),
`concept_scores` on the PRIMARY dictionary's pooled features gives a
per-series score, `top_series` gives its top-`k` set `S`, and
`matched_draws` builds the stratum-matched null for `S` exactly as
`run_transfer` does. `transfer_one` is then called with the REPLICATE
dictionary's `rankdata` in place of a second model's -- the forward leg
asks "does the replicate's best feature separate `S`", the reverse leg
asks "does that feature's own top-k, scored by the primary, come back
above chance". A part is STABLE only if the reciprocal leg clears at EVERY
replicate (all-of-`n_replicates`, pre-registered -- a one-of-two pass is a
coin flip away from failing, so a majority rule would silently relax the
bar the first time `n_sae_seeds` grows). A CONCEPT is stable only if every
one of its parts is.

Seeds are derived via `transfer.py::_seed`'s stable sha256 digest, NEVER
Python's builtin `hash()` (`CLAUDE.md` sec 11.2) -- `_seed(target, concept,
"@r{i}", base=transfer_seed)` for the reverse leg against replicate `i`, so
a run is reproducible run to run and the whole artifact moves as one when
`sae.transfer_seed` moves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from ..analysis.stats import mean_ci
from ..extraction.store import ActivationStore, load_meta
from ..utils import load_json, log, save_json
from .transfer import (_by_stratum, _seed, concept_scores, matched_draws,
                       series_strata, top_series, transfer_one)

__all__ = ["concept_stability"]


# ---------------------------------------------------------------------------
# Part extraction: an atlas concept, or an old per-target concept, split by
# the target(s) its member features actually live at.
# ---------------------------------------------------------------------------

def _atlas_parts(atlas: dict) -> list:
    """One row per (atlas concept, target it has members at):
    `{"concept", "target", "feature_ids"}`. `atlas["concepts"][*]["members"]`
    are row INDICES into `atlas["rows"]`; grouping those rows by their own
    `model`/`layer` is how one atlas concept -- which can span several
    targets and several models -- becomes several PARTS, one per target.
    """
    rows = atlas.get("rows", [])
    out: list = []
    for concept in atlas.get("concepts", []):
        cid = int(concept["concept"])
        by_target: dict = {}
        for idx in concept.get("members", []):
            r = rows[idx]
            key = f"{r['model']}/{r['layer']}"
            by_target.setdefault(key, []).append(int(r["feature"]))
        for target, feature_ids in sorted(by_target.items()):
            out.append({"concept": cid, "target": target,
                       "feature_ids": sorted(feature_ids)})
    return out


def _concepts_json_parts(concepts_doc: dict) -> list:
    """The equivalent PART list for the OLD, per-target `concepts.json`
    partition. Trivial (one part per concept, never split): a per-target
    concept, by construction of `concepts.py::run_concepts`, is clustered
    against its own target's peers only and can never span a second target.
    """
    out: list = []
    for key, t in sorted((concepts_doc.get("targets") or {}).items()):
        for concept in t.get("concepts") or []:
            out.append({"concept": int(concept["concept"]), "target": key,
                       "feature_ids": sorted(int(f) for f in concept["features"])})
    return out


# ---------------------------------------------------------------------------
# Store access: the primary dictionary's pooled features, and each
# replicate's rankdata, cached across parts/concepts.
# ---------------------------------------------------------------------------

class _ReplicateReader:
    """Caches per-target primary pooled features and per-(target, replicate)
    ranked replicate features -- the same caching shape
    `transfer.py::run_transfer`'s `_pooled`/`_ranks` closures use, generalized
    with a `replicate` axis."""

    def __init__(self, store: ActivationStore):
        self.store = store
        self._primary: dict = {}
        self._ranks: dict = {}

    def primary(self, target: str) -> np.ndarray:
        if target not in self._primary:
            model, layer = target.split("/", 1)
            self._primary[target] = self.store.load(model, layer, level="series", space="sae")
        return self._primary[target]

    def replicate_ranks(self, target: str, i: int) -> np.ndarray:
        cache_key = (target, i)
        if cache_key not in self._ranks:
            model, layer = target.split("/", 1)
            pooled = self.store.load(model, layer, level="series", space="sae", replicate=i)
            self._ranks[cache_key] = rankdata(pooled, axis=0)
        return self._ranks[cache_key]


def _part_reciprocal_at(reader: _ReplicateReader, target: str, feature_ids: list, seed_tag,
                        strata: np.ndarray, by_stratum: dict, k_top: int, n_null: int,
                        base_seed: int, n_seeds: int) -> tuple:
    """`-> (reciprocal_at [n_seeds - 1] of bool, src_scores)`. One forward-leg
    null (`fwd_draws`) is built ONCE, from the primary's own top-`k_top`
    series, and reused across every replicate `i` -- exactly `run_transfer`'s
    "build the matched draws once per source, reuse across destinations"
    shape, with "replicate" standing in for "destination target".
    """
    src_scores = concept_scores(reader.primary(target), feature_ids)
    S = top_series(src_scores, k=k_top)
    fwd_seed = _seed(target, seed_tag, base=base_seed)
    fwd_rng = np.random.default_rng(fwd_seed)
    fwd_draws = matched_draws(S, strata, by_stratum, n_null, fwd_rng)
    reciprocal_at = []
    for i in range(1, int(n_seeds)):
        dst_ranks = reader.replicate_ranks(target, i)
        rseed = _seed(target, seed_tag, f"@r{i}", base=base_seed)
        result = transfer_one(src_scores, dst_ranks, S, fwd_draws, strata, by_stratum,
                              k=k_top, n_draws=n_null, seed=rseed)
        reciprocal_at.append(bool(result["reciprocal"]))
    return reciprocal_at, src_scores


# ---------------------------------------------------------------------------
# The within-model ceiling: reciprocal rate, averaged over replicates, per
# target and per model (bootstrap CI over PARTS).
# ---------------------------------------------------------------------------

def _ceiling(parts_with_rates: list) -> dict:
    """`parts_with_rates`: `[{"target", "model", "rate"}]`, `rate` already
    being that part's own fraction of replicates at which it was reciprocal
    -- averaging that fraction over a target's (or a model's) parts is
    exactly "the fraction of parts reciprocal at replicate i, averaged over
    replicates" (order of averaging does not matter here: both reduce the
    same `(n_parts, n_replicates)` boolean table to one number). The
    model-level bootstrap CI resamples PARTS (never series, never
    replicates -- sec 37 P2's own instruction), via
    `analysis/stats.py::mean_ci`, the shared bootstrap machinery every other
    stage's CIs already go through (`CLAUDE.md` sec 2.2).
    """
    by_target: dict = {}
    by_model: dict = {}
    for p in parts_with_rates:
        by_target.setdefault(p["target"], []).append(p["rate"])
        by_model.setdefault(p["model"], []).append(p["rate"])
    target_out = {t: {"rate": float(np.mean(v)), "n_parts": len(v)}
                 for t, v in sorted(by_target.items())}
    model_out: dict = {}
    for m, v in sorted(by_model.items()):
        arr = np.asarray(v, dtype=np.float64)
        if arr.size >= 2:
            ci = mean_ci(arr, n_boot=1000, seed=0, unit="parts")
            model_out[m] = {"rate": ci["value"], "ci": [ci["lo"], ci["hi"]],
                            "n_parts": int(arr.size)}
        else:
            model_out[m] = {"rate": (float(arr[0]) if arr.size else float("nan")),
                            "ci": None, "n_parts": int(arr.size)}
    return {"by_target": target_out, "by_model": model_out}


def _universality_table(atlas_concepts: list, stable_ids: set) -> dict:
    """How many atlas concepts span 4/3/2/1 models -- over every concept,
    and again over only the ones `concept_stability` found stable."""
    def _hist(concepts: list) -> dict:
        hist: dict = {}
        for c in concepts:
            n = int(c.get("n_models", 0))
            hist[str(n)] = hist.get(str(n), 0) + 1
        return hist
    return {"all": _hist(atlas_concepts),
           "stable": _hist([c for c in atlas_concepts if int(c["concept"]) in stable_ids])}


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def concept_stability(run_dir, atlas: dict, cfg) -> dict:
    """Seed-stability of every atlas concept, plus the within-model transfer
    ceiling. Writes and returns `sae/concept_stability.json`.

    `atlas` is an already-loaded `concept_atlas.json` dict (this module does
    not re-derive or re-cluster anything from `concept_atlas.py`); `cfg` is
    the run's `PipelineConfig` (`cfg.concepts.n_sae_seeds`,
    `cfg.sae.transfer_top_k`/`transfer_n_null`/`transfer_seed`).

    `cfg.concepts.n_sae_seeds < 2` means no replicates were trained --
    returns/writes a `"measured": False` artifact with every concept's
    `stability` recorded as the string `"not measured"` (sec 37 P2's own
    instruction), never a partial or fabricated measurement.
    """
    run_dir = Path(run_dir)
    concepts_cfg = getattr(cfg, "concepts", None)
    sae_cfg = getattr(cfg, "sae", None)
    n_seeds = int(getattr(concepts_cfg, "n_sae_seeds", 1))
    atlas_concepts = atlas.get("concepts", [])
    out_path = run_dir / "sae" / "concept_stability.json"

    if n_seeds < 2:
        out = {"schema_version": 1, "measured": False, "n_sae_seeds": n_seeds,
              "reason": "concepts.n_sae_seeds < 2 -- replicate stability not measured",
              "concepts": [{"concept": int(c["concept"]), "stability": "not measured"}
                          for c in atlas_concepts],
              "ceiling": None, "old_ceiling": None, "universality": None}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(out_path, out)
        log.info("concept stability: not measured (n_sae_seeds=%d)", n_seeds)
        return out

    k_top = int(getattr(sae_cfg, "transfer_top_k", 20))
    n_null = int(getattr(sae_cfg, "transfer_n_null", 200))
    base_seed = int(getattr(sae_cfg, "transfer_seed", 0))

    meta = load_meta(run_dir)
    strata = series_strata(meta)
    by_stratum = _by_stratum(strata)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    reader = _ReplicateReader(store)

    atlas_parts = _atlas_parts(atlas)
    parts_by_concept: dict = {}
    for p in atlas_parts:
        parts_by_concept.setdefault(p["concept"], []).append(p)

    parts_with_rates: list = []
    concepts_out: list = []
    stable_by_concept: dict = {}
    for concept in atlas_concepts:
        cid = int(concept["concept"])
        part_records = []
        for p in parts_by_concept.get(cid, []):
            model = p["target"].split("/", 1)[0]
            try:
                reciprocal_at, _src = _part_reciprocal_at(
                    reader, p["target"], p["feature_ids"], cid, strata, by_stratum,
                    k_top, n_null, base_seed, n_seeds)
            except KeyError as e:
                log.warning("concept stability: run %s concept %d part at %s -- %s",
                           run_dir, cid, p["target"], e)
                continue
            rate = float(np.mean(reciprocal_at)) if reciprocal_at else float("nan")
            stable = bool(all(reciprocal_at)) if reciprocal_at else None
            part_records.append({"target": p["target"], "n_features": len(p["feature_ids"]),
                                 "reciprocal_at": reciprocal_at, "stable": stable})
            parts_with_rates.append({"target": p["target"], "model": model, "rate": rate})
        if part_records:
            concept_stable = all(pr["stable"] for pr in part_records)
            frac_parts_stable = sum(1 for pr in part_records if pr["stable"]) / len(part_records)
        else:
            concept_stable, frac_parts_stable = None, None
        stable_by_concept[cid] = concept_stable
        concepts_out.append({
            "concept": cid,
            "stability": {"replicates": n_seeds - 1, "parts": part_records,
                         "stable": concept_stable, "frac_parts_stable": frac_parts_stable},
        })

    ceiling = _ceiling(parts_with_rates)
    stable_ids = {cid for cid, s in stable_by_concept.items() if s}
    universality = _universality_table(atlas_concepts, stable_ids)
    n_stable = sum(1 for s in stable_by_concept.values() if s)
    n_scored = sum(1 for s in stable_by_concept.values() if s is not None)

    # The OLD per-target `concepts.json` ceiling, best-effort (sec 37 P2 step
    # 4's "if that is straightforward, labeled as such; skip it otherwise").
    old_ceiling = None
    concepts_path = run_dir / "sae" / "concepts.json"
    if concepts_path.exists():
        try:
            old_parts = _concepts_json_parts(load_json(concepts_path))
            old_rates: list = []
            for p in old_parts:
                model = p["target"].split("/", 1)[0]
                try:
                    reciprocal_at, _src = _part_reciprocal_at(
                        reader, p["target"], p["feature_ids"], f"old{p['concept']}",
                        strata, by_stratum, k_top, n_null, base_seed, n_seeds)
                except KeyError as e:
                    log.warning("concept stability: old-concepts part at %s -- %s",
                               p["target"], e)
                    continue
                rate = float(np.mean(reciprocal_at)) if reciprocal_at else float("nan")
                old_rates.append({"target": p["target"], "model": model, "rate": rate})
            old_ceiling = _ceiling(old_rates) if old_rates else None
        except Exception as e:  # pragma: no cover -- best-effort per spec
            log.warning("concept stability: old per-target ceiling skipped -- %s", e)
            old_ceiling = None

    out = {
        "schema_version": 1, "measured": True, "n_sae_seeds": n_seeds,
        "replicates": n_seeds - 1,
        "params": {"k_top_series": k_top, "n_null_draws": n_null, "transfer_seed": base_seed},
        "concepts": concepts_out,
        "n_concepts": len(atlas_concepts), "n_scored": n_scored, "n_stable": n_stable,
        "frac_stable": (n_stable / n_scored) if n_scored else None,
        "ceiling": ceiling, "old_ceiling": old_ceiling, "universality": universality,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    log.info("concept stability: %d/%d concept(s) stable across %d replicate(s)",
             n_stable, n_scored, n_seeds - 1)
    return out
