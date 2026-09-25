"""The `concepts` pipeline stage (ROADMAP.md sec 37.4 P1): one entry point for
"what concepts does each model's dictionary hold, and does another model
group the same series".

Chains the following steps (the first four ran only as separate hand-invoked
scripts before this stage existed; the rest are additive extensions added
since):
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
  7. per-concept PROFILES (`sae/concept_profiles.py`, ROADMAP.md sec 37 Spec
     A) -> `concept_profiles.json` -- ADDITIVE on top of steps 5-6: for every
     atlas concept and every one of its (model, layer) parts, what corpus
     inputs make it fire (correlational, against `ground_truth.py`'s own
     structural fields), whether models sharing the concept's EFFECT also
     share its INPUTS (a stratum-matched permutation test reusing
     `transfer.py`'s own machinery), and whether the model holding it
     forecasts better on its top-firing series BECAUSE of it (behavioral,
     upgraded to within-model causal only when the concept's own ablation
     battery already moved MASE there). Runs only when the atlas has
     concepts, since a profile is a property of an atlas concept.
  8. cross-model causal AGREEMENT on shared inputs (`sae/shared_input_
     agreement.py`, ROADMAP.md sec 37.8 P5b) -> `shared_input_agreement.json`
     -- ADDITIVE, runs after the atlas's own cross-model transfer (reusing
     its `sae/atlas_transfer.json`): for every FDR-surviving reciprocal
     transfer test, ablates the source concept-part and the destination
     feature (or its own atlas part) on the SAME shared series the transfer
     test already selected, and scores their agreement against each side's
     own matched-random-feature-set floor. Runs only when atlas transfer ran
     and at least one test survived FDR, since its unit of work is exactly
     that test.

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
from .shared_input_agreement import run_shared_input_agreement, shared_input_agreement_path
from .stability import concept_stability
from .transfer import run_atlas_transfer, run_transfer


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


def _atlas_transfer_skip_reason(cfg, atlas: dict | None) -> str | None:
    """ROADMAP.md sec 37 P3 item 6. Same shape as `_transfer_skip_reason`
    above, one artifact over: `run_atlas_transfer` needs the atlas itself
    (skipped above -> nothing to test) AND >=2 models (a solo run has no
    cross-model transfer question to ask), and honors the same
    `sae.transfer_enabled` toggle the per-target transfer above does, since
    both are the identical shared-inputs transfer TEST, just over a
    different source unit (an atlas concept's per-model part vs. a
    per-target concept)."""
    if not cfg.sae.transfer_enabled:
        return "sae.transfer_enabled is false"
    if cfg.run_shape() == "solo":
        return "run shape solo -- atlas transfer compares two models' dictionaries"
    if atlas is None:
        return "the atlas did not run (see the atlas block's own skip reason)"
    if not atlas.get("concepts"):
        return "the atlas has no concepts to test"
    return None


def _shared_input_skip_reason(cfg, atlas_transfer_skip: str | None, at: dict | None) -> str | None:
    """ROADMAP.md sec 37.8 P5b item 8: skip with a stated reason when atlas
    transfer did not run, or ran but no test survived reciprocal FDR --
    mirrors `_atlas_transfer_skip_reason`'s own precomputed-then-decide
    shape one artifact over."""
    if not getattr(cfg.concepts, "shared_input_enabled", True):
        return "concepts.shared_input_enabled is false"
    if atlas_transfer_skip:
        return f"atlas transfer did not run -- {atlas_transfer_skip}"
    if at is None or not any(t.get("reciprocal_fdr") for t in (at.get("tests") or [])):
        return "no atlas-transfer test survived reciprocal FDR"
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
    atlas = None
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

    # ROADMAP.md sec 37 P3 item 6: the atlas's own per-model PARTS, tested
    # against every OTHER model's targets -- reuses `run_atlas_transfer`,
    # which shares `transfer_one`/`matched_draws`/`_seed` with `run_transfer`
    # above rather than re-deriving them. Runs after the atlas so a fresh
    # clustering always re-runs the transfer built on it (the exact staleness
    # `_drop_stale` exists to prevent, `sec 37 P0`'s own motivating finding).
    atlas_transfer_skip = _atlas_transfer_skip_reason(cfg, atlas)
    at = None
    if atlas_transfer_skip:
        log.warning("concepts: atlas transfer skipped -- %s", atlas_transfer_skip)
        atlas_transfer_block = {
            "status": "skipped", "reason": atlas_transfer_skip,
            "removed_stale": _drop_stale(run_dir / "sae" / "atlas_transfer.json")}
    else:
        at = run_atlas_transfer(run_dir, atlas, cfg)
        n_recip_fdr = sum(1 for t in at["tests"] if t["reciprocal_fdr"])
        atlas_transfer_block = {
            "status": "ran", "n_tests": len(at["tests"]),
            "n_ordered_pairs": len(at["pair_summary"]),
            "n_uncorrected_reciprocal": sum(1 for t in at["tests"] if t["reciprocal"]),
            "n_fdr_reciprocal": n_recip_fdr}
        # sec 37 P3 item 7: the doctor's transfer-FDR-floor check, run here
        # (not from static preflight) because it needs the REAL per-pair,
        # per-leg test count `m`, which only exists once transfer has run --
        # see `check_transfer_fdr_budget`'s own docstring for why.
        from ..doctor import check_transfer_fdr_budget
        p_method = str(getattr(c, "transfer_p_method", "exact") or "exact")
        fdr_q = float(getattr(c, "transfer_fdr_q", 0.05) or 0.05)
        n_null = int(getattr(cfg.sae, "transfer_n_null", 200))
        for rec in at["pair_summary"]:
            fdr_check = check_transfer_fdr_budget(n_null, rec["n_tests"], fdr_q, p_method)
            if fdr_check.status != "pass":
                log.warning("concepts: transfer FDR floor (%s -> %s): %s -- %s",
                           rec["src_model"], rec["dst_model"], fdr_check.status,
                           fdr_check.detail)

    # ROADMAP.md sec 37.8 P5b: cross-model causal agreement, measured on the
    # SAME shared series, for every FDR-surviving reciprocal atlas-transfer
    # test above. Runs after atlas transfer, since its whole unit of work is
    # that test's own (source concept-part, destination feature) pair;
    # `_drop_stale` mirrors every other additive artifact's pattern here.
    shared_input_skip = _shared_input_skip_reason(cfg, atlas_transfer_skip, at)
    if shared_input_skip:
        log.warning("concepts: shared-input agreement skipped -- %s", shared_input_skip)
        shared_input_block = {
            "status": "skipped", "reason": shared_input_skip,
            "removed_stale": _drop_stale(shared_input_agreement_path(run_dir))}
    else:
        sia = run_shared_input_agreement(cfg, run_dir, hub, store, data, device, atlas, at)
        shared_input_block = {"status": "ran", "n_tests": sia["n_tests"],
                              "verdict_counts": sia["verdict_counts"],
                              "dst_set_kind_counts": sia["dst_set_kind_counts"],
                              "n_short_matched_pool": sia["n_short_matched_pool"],
                              "runtime_seconds": sia["runtime_seconds"]}

    # ROADMAP.md sec 37 Spec A: per-concept profiles (`sae/concept_profiles.py`)
    # -- what a concept's parts fire on, whether models sharing its effect
    # also share its inputs, and the link to each part's own causal MASE
    # effect. Runs only when the atlas itself ran, since a profile is a
    # property of an atlas concept (exactly `stability`'s own condition
    # above); `_drop_stale` mirrors every other additive artifact's pattern
    # in this stage when it does not rewrite it.
    if not c.profiles_enabled:
        profiles_reason = "concepts.profiles_enabled is false"
        profiles_block = {"status": "skipped", "reason": profiles_reason,
                          "removed_stale": _drop_stale(run_dir / "sae" / "concept_profiles.json")}
    elif atlas is None:
        profiles_reason = "the atlas did not run (see the atlas block's own skip reason)"
        profiles_block = {"status": "skipped", "reason": profiles_reason,
                          "removed_stale": _drop_stale(run_dir / "sae" / "concept_profiles.json")}
    elif not atlas.get("concepts"):
        profiles_reason = "the atlas has no concepts to profile"
        profiles_block = {"status": "skipped", "reason": profiles_reason,
                          "removed_stale": _drop_stale(run_dir / "sae" / "concept_profiles.json")}
    else:
        from .concept_profiles import run_concept_profiles
        profiles = run_concept_profiles(cfg.run_dir(), cfg)
        profiles_block = {"status": "ran", "n_concepts": len(profiles["concepts"]),
                          "sharing_class_counts": profiles["summary"]["sharing_class_counts"],
                          "sharing_class_counts_stable_only":
                              profiles["summary"]["sharing_class_counts_stable_only"],
                          "n_provenance_driven_parts": profiles["summary"]["n_provenance_driven_parts"],
                          "n_parts": profiles["summary"]["n_parts"]}

    record = {"schema_version": 1, "targets": [f"{m}/{l}" for m, l in targets],
              "ablation": ablation, "concept_counts": concept_counts,
              "n_concepts": sum(concept_counts.values()), "non_modular": non_modular,
              "transfer": transfer_block, "describe": describe_block, "atlas": atlas_block,
              "stability": stability_block, "atlas_transfer": atlas_transfer_block,
              "shared_input_agreement": shared_input_block,
              "profiles": profiles_block}
    save_json(concept_stage_path(cfg), record)
    log.info("concepts: %d concept(s) across %d target(s) (%d non-modular); transfer %s; "
             "atlas %s; stability %s; atlas transfer %s; shared-input agreement %s; profiles %s",
             record["n_concepts"], len(concept_counts), len(non_modular),
             transfer_block["status"], atlas_block["status"], stability_block["status"],
             atlas_transfer_block["status"], shared_input_block["status"], profiles_block["status"])
    return record
