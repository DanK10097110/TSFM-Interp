"""The `concepts` pipeline stage (ROADMAP.md sec 37.4 P1): one entry point for
"what concepts does each model's dictionary hold, and does another model
group the same series".

Chains five steps that previously ran only as separate hand-invoked scripts:
  1. the ablation battery (`sae/ablation_run.py`, sec 27) -> `*_ablation.json`
  2. concept clustering in ablation space (`sae/concepts.py`) -> `concepts.json`
  3. cross-model transfer on shared inputs (`sae/transfer.py`) -> `transfer.json`
  4. deterministic descriptions (`sae/describe_run.py`) -> `descriptions.json`
  5. a cross-model concept ATLAS (`sae/concept_atlas.py`, ROADMAP.md sec
     37.15 q6 option b) -> `concept_atlas.json` -- ADDITIVE, pools every
     target's causal features (step 1's output, not step 2's per-target
     clusters) into one space and clusters ACROSS models, entirely
     independent of steps 2-4's own artifacts.
  6. seed STABILITY of the atlas's own concepts, and the within-model
     transfer ceiling (`sae/stability.py`, ROADMAP.md sec 37 P2) ->
     `concept_stability.json` -- ADDITIVE on top of step 5: when
     `concepts.n_sae_seeds >= 2`, trains `n_sae_seeds - 1` extra,
     independently-seeded SAE replicates per already-trained target (at each
     target's own fixed `dict_size`, never re-searched) and asks whether
     step 5's concepts survive a second SAE draw at the SAME target, via the
     same reciprocal transfer test step 4 uses -- within a model instead of
     between two. Runs only when the atlas itself ran, since stability is a
     property of the atlas's own concepts.

Having no production caller is what let `runs/full_report_run_4model`'s
`transfer.json` outlive two regenerations of the `concepts.json` it was built
from (sec 37.3 P0's Findings); as a stage, all five artifacts are produced
together under one config fingerprint, so a clustering change re-runs the
transfer that depends on it.

What it inherits from `sae`: `sae/meta.json` (which targets this config
trained) and each target's checkpoint. Transfer additionally needs the
persisted feature store (`sae.persist_features: true`), which
`run_pipeline`'s preflight enforces before any stage runs. The atlas needs
neither -- it reads the same raw ablation files step 1 already wrote.

Writes `sae/concept_stage.json`, the stage's own record: which targets were
ablated, withheld or skipped, the concept counts, and -- for a solo run or a
run whose targets span one model -- a `transfer` block stating why transfer
did not run, so an absent `transfer.json` is never mistaken for a failed one.
Same discipline for the atlas: a `concepts.atlas_enabled: false` config, or a
run with no pooled causal features to cluster (every target withheld/skipped/
empty), removes any `concept_atlas.json` left from an earlier run rather than
leaving it beside a `concept_stage.json` that says nothing ran
(`_drop_stale`, exactly `transfer.json`'s own pattern above).
"""

from __future__ import annotations

import time
from pathlib import Path

from ..utils import load_json, log, save_json, set_seed
from .ablation_run import run_ablation_all
from .concept_atlas import pooled_features, run_concept_atlas
from .concepts import run_concepts
from .stability import concept_stability
from .transfer import run_transfer


def concept_stage_path(cfg) -> Path:
    return cfg.run_dir() / "sae" / "concept_stage.json"


def preflight_problem(cfg) -> str | None:
    """Why this config cannot run the stage, or None. Checked by
    `run_pipeline` before the first stage, because the stage sits after
    `sae` and discovering this there would cost every stage before it
    (`CLAUDE.md` sec 11.40's cost asymmetry)."""
    if not cfg.sae.persist_features:
        return ("the `concepts` stage needs `sae.persist_features: true` -- cross-model "
                "transfer reads each target's persisted SAE features (space='sae'), "
                "and a concept run without them cannot produce its central artifact")
    return None


def _trained_targets(run_dir: Path) -> tuple[list, dict]:
    meta = load_json(run_dir / "sae" / "meta.json")
    targets = [tuple(key.split("/", 1)) for key in meta]
    return targets, meta


def _drop_stale(path: Path) -> bool:
    """Remove this stage's own artifact from an earlier run when this run
    does not rewrite it -- a `transfer.json` left beside a fresh
    `concepts.json` is precisely the state sec 37.3 P0 found on disk."""
    if path.exists():
        path.unlink()
        log.warning("concepts: removed stale %s (not rewritten by this run)", path)
        return True
    return False


def _transfer_skip_reason(cfg, concepts: dict, meta: dict) -> str | None:
    if not cfg.sae.transfer_enabled:
        return "sae.transfer_enabled is false"
    if cfg.run_shape() == "solo":
        return "run shape solo -- transfer compares two models' dictionaries"
    live = {k: t for k, t in concepts.get("targets", {}).items() if not t.get("withheld")}
    models = {t["model"] for t in live.values()}
    if len(models) < 2:
        return (f"the trained, non-withheld targets span {len(models)} model(s) "
                f"({', '.join(sorted(models)) or 'none'}); transfer needs two")
    if not any(t.get("concepts") for t in live.values()):
        return "no target has any concepts (every one is non-modular or withheld)"
    unpersisted = sorted(k for k in live if not (meta.get(k) or {}).get("features_persisted"))
    if unpersisted:
        return ("no persisted SAE features for " + ", ".join(unpersisted)
                + " -- re-run the sae stage with sae.persist_features: true")
    return None


def _atlas_skip_reason(cfg, n_pooled: int) -> str | None:
    """Why the atlas should not run/persist this time, or `None`. `n_pooled`
    is a cheap probe -- `len(pooled_features(run_dir)[0])` -- computed by the
    caller BEFORE deciding whether to pay for `run_concept_atlas`'s own null
    procedures, mirroring `_transfer_skip_reason`'s own precomputed-then-
    decide shape above."""
    if not cfg.concepts.atlas_enabled:
        return "concepts.atlas_enabled is false"
    if n_pooled == 0:
        return "no causal, non-withheld ablation candidates pooled across any target"
    return None


def run_concept_stage(cfg, hub, store, data, device) -> dict:
    """Run the five steps and write `sae/concept_stage.json`."""
    run_dir = cfg.run_dir()
    c = cfg.concepts
    set_seed(cfg.run.seed)
    targets, meta = _trained_targets(run_dir)
    log.info("concepts: %d trained target(s): %s", len(targets),
             ", ".join(f"{m}/{l}" for m, l in targets))

    by_model: dict = {}
    for model, layer in targets:
        by_model.setdefault(model, []).append((model, layer))
    written = []
    for model, model_targets in by_model.items():
        written += run_ablation_all(
            cfg, run_dir, hub, data, store, device, model_targets,
            top_k_series=c.top_k_series, n_null_directions=c.n_null_directions,
            max_series=c.max_series, keep_forecasts=c.keep_forecasts,
            n_features_per_rule=c.n_features_per_rule)
        if not cfg.run.keep_models_loaded:
            hub.release(model)

    ablation = {}
    for path in written:
        art = load_json(path)
        key = f"{art['model']}/{art['layer']}"
        state = ("skipped" if art.get("skipped") else
                 "withheld" if art.get("withheld") else "measured")
        ablation[key] = {"state": state, "path": str(path.relative_to(run_dir)),
                         "reason": art.get("reason") or (art.get("reach") or {}).get("reason", "")}

    concepts = run_concepts(run_dir, cfg)
    concept_counts = {k: len(t.get("concepts") or []) for k, t in concepts["targets"].items()}
    non_modular = sorted(k for k, t in concepts["targets"].items() if t.get("non_modular"))

    skip = _transfer_skip_reason(cfg, concepts, meta)
    if skip:
        log.warning("concepts: transfer skipped -- %s", skip)
        transfer_block = {"status": "skipped", "reason": skip,
                          "removed_stale": _drop_stale(run_dir / "sae" / "transfer.json")}
    else:
        tr = run_transfer(run_dir, concepts, cfg)
        transfer_block = {"status": "ran", "n_pairs": len(tr["pairs"]),
                          "n_forward_clear": sum(1 for p in tr["pairs"] if p["clears"]),
                          "n_reciprocal": sum(1 for p in tr["pairs"] if p["reciprocal"])}

    describe_block = {"status": "skipped", "reason": "concepts.describe is false"}
    if c.describe:
        from .describe_run import describe_run
        doc = describe_run(run_dir, cfg, top_features=c.describe_top_features)
        describe_block = ({"status": "ran", "n_targets": len(doc)} if doc is not None else
                          {"status": "empty", "reason": "no evidence packets could be built"})
    if describe_block["status"] != "ran":
        describe_block["removed_stale"] = _drop_stale(run_dir / "sae" / "descriptions.json")

    # ROADMAP.md sec 37.15 q6 (option b): ADDITIVE, entirely independent of
    # steps 2-4 above -- pools step 1's raw ablation candidates across every
    # model rather than reading `concepts`. A cheap probe (`pooled_features`
    # itself; no clustering/nulls yet) decides skip vs. run, exactly
    # `_transfer_skip_reason`'s own precomputed-then-decide shape.
    causal_only = bool(getattr(cfg.sae, "concept_causal_only", True))
    n_pooled = len(pooled_features(run_dir, causal_only=causal_only)[0])
    atlas_skip = _atlas_skip_reason(cfg, n_pooled)
    if atlas_skip:
        log.warning("concepts: atlas skipped -- %s", atlas_skip)
        atlas_block = {"status": "skipped", "reason": atlas_skip,
                       "removed_stale": _drop_stale(run_dir / "sae" / "concept_atlas.json")}
        stability_block = {
            "status": "skipped", "reason": f"atlas skipped -- {atlas_skip}",
            "removed_stale": _drop_stale(run_dir / "sae" / "concept_stability.json")}
    else:
        atlas = run_concept_atlas(run_dir, cfg)
        n_multi_model = sum(1 for c_rec in atlas["concepts"] if c_rec["n_models"] >= 2)
        atlas_block = {"status": "ran", "n_features": atlas["n_features"],
                       "n_assigned": atlas["n_assigned"], "n_concepts": len(atlas["concepts"]),
                       "n_multi_model": n_multi_model}

        # ROADMAP.md sec 37 P2: is an atlas concept a property of the model,
        # or of the one SAE draw that happened to be trained? Additive on
        # top of the atlas above -- runs only when the atlas itself ran,
        # since stability is a property of ITS concepts.
        n_sae_seeds = int(getattr(c, "n_sae_seeds", 1))
        replicate_seconds = None
        if n_sae_seeds >= 2:
            from .train import run_sae_replicates
            t0 = time.monotonic()
            run_sae_replicates(cfg, hub, store, data, device, n_sae_seeds)
            replicate_seconds = time.monotonic() - t0
        stability = concept_stability(run_dir, atlas, cfg)
        stability_block = {
            "status": "ran" if stability.get("measured") else "not_measured",
            "n_sae_seeds": n_sae_seeds, "n_concepts": stability.get("n_concepts"),
            "n_scored": stability.get("n_scored"), "n_stable": stability.get("n_stable"),
            "frac_stable": stability.get("frac_stable")}
        if replicate_seconds is not None:
            stability_block["replicate_train_seconds"] = replicate_seconds

    record = {"schema_version": 1, "targets": [f"{m}/{l}" for m, l in targets],
              "ablation": ablation, "concept_counts": concept_counts,
              "n_concepts": sum(concept_counts.values()), "non_modular": non_modular,
              "transfer": transfer_block, "describe": describe_block, "atlas": atlas_block,
              "stability": stability_block}
    save_json(concept_stage_path(cfg), record)
    log.info("concepts: %d concept(s) across %d target(s) (%d non-modular); transfer %s; atlas %s; "
             "stability %s",
             record["n_concepts"], len(concept_counts), len(non_modular),
             transfer_block["status"], atlas_block["status"], stability_block["status"])
    return record
