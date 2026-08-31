"""Stage orchestration.

Stages form a short DAG (everything depends on extraction; the report reads
whatever exists). Each stage is gated by its config `enabled` flag, skipped
when its artifacts already exist unless forced, and selectable by name from
the CLI, so any subset of analyses composes into a valid run.
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from .analysis.attention import run_attention
from .analysis.clustering import run_clustering
from .analysis.exemplars import run_exemplars
from .analysis.confirm import run_confirm
from .analysis.hypotheses import run_register
from .analysis.internals import run_internals
from .analysis.layer_screen import run_layer_screen
from .analysis.l0_behavioral import run_l0
from .analysis.l1_geometry import run_l1
from .analysis.l2_stitching import run_l2
from .analysis.l3_perturbation import run_l3
from .analysis.lens import run_lens
from .analysis.frontend import run_frontend
from .analysis.model_budget import run_budget
from .config import PipelineConfig, dump_config
from .data import BenchmarkData, load_benchmark
from .extraction.extract import run_extraction
from .extraction.store import ActivationStore
from .manifest import (diff_resolved, fingerprint_stage, load_manifest,
                       record_extra, resolve_config_keys, save_manifest)
from .models import ModelHub
from .models.base import TIER_NAMES, NotTimeLocalized
from .report.report import run_report
from .sae.train import run_sae
from .utils import (log, resolve_device, resolve_dtype, run_provenance, save_json,
                    set_seed, setup_logging)


class Context:
    """Lazy shared state handed to every stage."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        # Which stage names this invocation was asked to re-run (`--force`).
        # `run_pipeline` fills it in; a stage that guards a *consumable*
        # resource of its own -- `confirm` and the private benchmark -- reads
        # it, because the pipeline-level skip predicate cannot express "the
        # artifacts exist AND re-running them costs something irreversible".
        # Without this the guard's own remediation ("rerun with --force
        # confirm") is unreachable: force bypasses the skip, the stage runs,
        # and the guard raises anyway.
        self.forced: set = set()
        self.device = resolve_device(cfg.run.device)
        self.dtype = resolve_dtype(cfg.run.dtype, self.device)
        self.hub = ModelHub(cfg.models, cfg.data, self.device, self.dtype)
        self._data: Optional[BenchmarkData] = None
        self._store: Optional[ActivationStore] = None

    @property
    def data(self) -> BenchmarkData:
        if self._data is None:
            self._data = load_benchmark(self.cfg.data, self.cfg.run.seed)
        return self._data

    @property
    def store(self) -> ActivationStore:
        if self._store is None:
            self._store = ActivationStore(self.cfg.run_dir() / "activations.zarr", mode="a")
        return self._store

    def reset_store(self) -> None:
        self._store = None


@dataclass
class Stage:
    name: str
    deps: list
    enabled: Callable[[PipelineConfig], bool]
    done: Callable[[PipelineConfig], bool]
    run: Callable[[Context], None]
    # Dotted config keys this stage actually consumes (`ROADMAP.md` sec 15 A3)
    # -- used to fingerprint whether an existing artifact still matches the
    # current config before trusting a skip. Explicit per-stage, not a
    # whole-config hash, so an edit unrelated to a stage doesn't invalidate it.
    config_keys: tuple = field(default_factory=tuple)


_EXTRACT_KEYS = ("data", "alignment", "extraction",
                 "models[*].checkpoint", "models[*].layer_regex",
                 "models[*].capture_layer_stride", "models[*].batch_size",
                 "models[*].kwargs", "run.seed", "run.dtype")


def _stages() -> list:
    """The canonical stage sequence."""
    return [
        Stage("extract", [],
              lambda c: c.extraction.enabled,
              lambda c: (c.run_dir() / "activations.zarr").exists()
              and (c.run_dir() / "meta.parquet").exists(),
              lambda ctx: (run_extraction(ctx.cfg, ctx.hub, ctx.data),
                           ctx.reset_store()),
              _EXTRACT_KEYS),
        # No `extract` dependency on purpose: a cost record needs only a
        # loaded model, so `--stages budget` is a valid standalone run. It is
        # placed here so that in a full run the models are already warm.
        Stage("budget", [],
              lambda c: c.budget.enabled,
              lambda c: (c.run_dir() / "budget" / "model_budget.json").exists(),
              lambda ctx: run_budget(ctx.cfg, ctx.hub, ctx.data, ctx.device),
              ("budget", "data.context_len", "data.horizon",
               "models[*].checkpoint", "models[*].layer_regex",
               "models[*].capture_layer_stride")),
        # No `extract` dependency on purpose, same reasoning as `budget`: a
        # front-end diagnostic needs only a loaded model and `predict()`, so
        # `--stages frontend` is a valid standalone run. Placed right after
        # `budget` so a full run reuses the already-warm models.
        Stage("frontend", [],
              lambda c: c.frontend.enabled,
              lambda c: (c.run_dir() / "frontend" / "frontend.json").exists(),
              lambda ctx: run_frontend(ctx.cfg, ctx.hub, ctx.data, ctx.device),
              ("frontend", "data.context_len", "data.horizon", "alignment.window",
               "models[*].checkpoint", "models[*].layer_regex",
               "models[*].capture_layer_stride")),
        Stage("layer_screen", ["extract"],
              lambda c: c.layer_screen.enabled,
              lambda c: (c.run_dir() / "layer_screen" / "selection.json").exists(),
              lambda ctx: run_layer_screen(ctx.cfg, ctx.hub, ctx.store, ctx.data, ctx.device),
              ("layer_screen",)),
        Stage("l0", ["extract"],
              lambda c: c.l0.enabled,
              lambda c: (c.run_dir() / "l0" / "summary.json").exists(),
              lambda ctx: run_l0(ctx.cfg, ctx.hub, ctx.data, ctx.store),
              ("l0",)),
        Stage("internals", ["extract"],
              lambda c: c.internals.enabled,
              lambda c: (c.run_dir() / "internals" / "profile.json").exists(),
              lambda ctx: run_internals(ctx.cfg, ctx.store, ctx.data, ctx.device),
              ("internals",)),
        Stage("lens", ["extract"],
              lambda c: c.lens.enabled,
              lambda c: (c.run_dir() / "lens" / "lens.json").exists(),
              lambda ctx: run_lens(ctx.cfg, ctx.hub, ctx.store, ctx.data, ctx.device),
              ("lens",)),
        Stage("l1", ["extract"],
              lambda c: c.l1.enabled,
              lambda c: (c.run_dir() / "l1" / "meta.json").exists(),
              lambda ctx: run_l1(ctx.cfg, ctx.store, ctx.device),
              ("l1",)),
        Stage("l2", ["extract"],
              lambda c: c.l2.enabled,
              lambda c: (c.run_dir() / "l2" / "stitching.json").exists(),
              lambda ctx: run_l2(ctx.cfg, ctx.store, ctx.data, ctx.device),
              ("l2",)),
        Stage("l3", ["extract"],
              lambda c: c.l3.enabled,
              lambda c: (c.run_dir() / "l3" / "meta.json").exists(),
              lambda ctx: run_l3(ctx.cfg, ctx.hub, ctx.store, ctx.data, ctx.device),
              ("l3",)),
        Stage("attention", ["extract"],
              lambda c: c.attention.enabled,
              lambda c: (c.run_dir() / "attention" / "meta.json").exists(),
              lambda ctx: run_attention(ctx.cfg, ctx.hub, ctx.store, ctx.data, ctx.device),
              ("attention",)),
        Stage("cluster", ["extract"],
              lambda c: c.clustering.enabled,
              lambda c: (c.run_dir() / "clustering" / "comparison.json").exists(),
              lambda ctx: run_clustering(ctx.cfg, ctx.store, ctx.data),
              ("clustering",)),
        Stage("sae", ["extract"],
              lambda c: c.sae.enabled,
              lambda c: (c.run_dir() / "sae" / "meta.json").exists(),
              lambda ctx: run_sae(ctx.cfg, ctx.hub, ctx.store, ctx.data, ctx.device),
              ("sae",)),
        Stage("exemplars", ["l0"],
              lambda c: c.exemplars.enabled,
              lambda c: (c.run_dir() / "exemplars" / "exemplars.json").exists(),
              lambda ctx: run_exemplars(ctx.cfg, ctx.hub, ctx.store, ctx.data, ctx.device),
              ("exemplars",)),
        Stage("register", ["l0"],
              lambda c: c.confirm.enabled,
              lambda c: (c.run_dir() / "hypotheses.json").exists(),
              lambda ctx: run_register(ctx.cfg),
              ()),
        Stage("confirm", ["register"],
              lambda c: c.confirm.enabled,
              lambda c: (c.run_dir() / "confirm" / "confirmation.json").exists(),
              lambda ctx: run_confirm(ctx.cfg, ctx.hub, forced=(
                  "confirm" in ctx.forced or "all" in ctx.forced)),
              ("confirm",)),
        Stage("report", [],
              lambda c: c.report.enabled,
              lambda c: False,
              lambda ctx: run_report(ctx.cfg),
              ("report",)),
    ]


# Stages that need only forecasts and a loaded model -- the whole analysis
# surface still available to a model whose tokens have no time->window map.
# `l0` is here despite declaring an `extract` dependency in the DAG: it reads
# no activations (it only *writes* its predictions into the store), so the
# dependency exists to order a full run, not because L0 needs a capture.
_L0_ONLY_STAGES = ("l0", "budget", "frontend", "report")

# The tier each stage needs from EVERY configured model (`ROADMAP.md` sec 19
# G1). Tier 0 is forecasts only; 1 adds readable activations; 2 adds
# single-pass patching; 3 adds attention introspection. Two deliberate
# choices:
#   * `attention` sits at 1, not 3. Its head-ablation half works from
#     `attention_info` alone and its pattern half already skips per-model
#     (`CLAUDE.md` sec 2.5) -- gating the whole stage at 3 would delete
#     working analyses for Chronos-Bolt and Sundial to protect one that
#     already degrades correctly.
#   * `confirm` sits at 0. Its behavioral half needs only `predict`, and its
#     CKA half is conditional on a registered geometric hypothesis that a
#     tier-0 run has no way to produce in the first place.
# A run is narrowed to the MINIMUM tier across its models, because every
# stage above tier 0 is either cross-model or feeds one that is.
_STAGE_MIN_TIER = {
    "l0": 0, "budget": 0, "frontend": 0, "report": 0, "register": 0, "confirm": 0,
    "extract": 1, "layer_screen": 1, "internals": 1, "l1": 1, "l2": 1,
    "cluster": 1, "sae": 1, "attention": 1, "exemplars": 1,
    "lens": 2, "l3": 2,
}


# ROADMAP.md sec 24.3 -- the run-shape gate, the model-count analogue of
# `_STAGE_MIN_TIER` above and deliberately built on the same machinery rather
# than as a second, parallel mechanism. A stage listed here is one whose
# ENTIRE product is a comparison between two models: `l1`'s CKA matrix, `l2`'s
# stitching directions, `cluster`'s cross-model AMI, `exemplars`' selection by
# MASE *gap*, and `confirm`'s hypothesis tests (all of which are "model A beats
# model B on family F"). In a `solo` run they have nothing to measure, so they
# are dropped WITH A STATED REASON rather than left to raise out of
# `comparison_pair()` mid-run or -- worse -- render as an empty section a
# reader cannot distinguish from a crashed one (`CLAUDE.md` invariant 8).
#
# Stages ABSENT from this map are the ones that survive a solo run unchanged
# or degrade internally, and the distinction is deliberate rather than an
# oversight: `l3` keeps its within-model patching (invariant 5 -- patching was
# never cross-model) and loses only its fingerprint-agreement number; `l0`
# keeps every per-family metric and calibration curve and loses only the
# paired tests; `sae`'s feature-space CKA already returns None and logs when
# it has nothing to compare. Those are internal degradations, not stage drops,
# and are tracked as their own sub-items of sec 24.3.
_STAGE_MIN_MODELS = {
    "l1": 2, "l2": 2, "cluster": 2, "exemplars": 2, "confirm": 2,
}


def _apply_shape(cfg: PipelineConfig, selected: set) -> dict:
    """Narrow this run's stages to what its MODEL COUNT can support.

    Runs beside `_apply_tiers` and for the same reason: a capability the run
    does not have should produce a decision with a recorded reason, not a
    crash and not a blank. The two gates are independent -- a tier-3 solo run
    is perfectly coherent, and so is a black-box panel -- so they are applied
    separately and each writes its own artifact.

    Like `_apply_tiers`, the dropped list is computed over every stage ENABLED
    in the config rather than over `selected`, so a `--stages report` rerun
    cannot rewrite the artifact to claim nothing was dropped and erase the
    reason the report prints beside each skipped section.
    """
    shape, n = cfg.run_shape(), len(cfg.models)
    dropped = sorted(name for name in stage_names()
                     if _stage_by_name(name).enabled(cfg)
                     and _STAGE_MIN_MODELS.get(name, 1) > n)
    if dropped:
        selected.difference_update(dropped)
        log.warning("RUN SHAPE -- this is a '%s' run (%d model%s); dropping %s. "
                    "Each dropped stage measures a comparison BETWEEN models and has "
                    "nothing to compare here; see shapes.json and the report.",
                    shape, n, "" if n == 1 else "s", ", ".join(dropped))
    return {"shape": shape, "n_models": n,
            "models": [m.name for m in cfg.models],
            "reference_model": cfg.models[0].name if cfg.models else None,
            "comparison_pairs": [[x.name, y.name] for x, y in cfg.comparison_pairs()],
            "dropped_stages": dropped}


def resolve_tiers(cfg: PipelineConfig, hub: ModelHub) -> dict:
    """Each configured model's declared capability tier -- no checkpoint loaded.

    `capability_tier` is a classmethod over what the adapter subclass
    implements, so this costs nothing and can therefore run on every stage
    selection, including a `--stages report` rerun. That matters: the tier
    decision must not depend on whether `extract` happened to be scheduled,
    since for a tier-0 model `extract` is precisely the stage being dropped.
    """
    return {mcfg.name: hub.get(mcfg.name).tier_report() for mcfg in cfg.models}


def resolve_routing(cfg: PipelineConfig, hub: ModelHub) -> dict:
    """Decide what each model in this run is eligible for, and record it.

    ROADMAP.md sec 16 E3(c)'s routing half. An adapter that *measures* its own
    token->time map (`GenericHFAdapter`) can find its impulse response
    diffuse, in which case pooling its activations onto the shared window
    axis would put every cross-model number on a fiction. Until now that
    surfaced as a `NotTimeLocalized` escaping mid-extraction and killing the
    run; here it is caught once, up front, and turned into a *decision*: that
    model is routed to L0 only, with the measured contrast that drove it
    written to `routing.json` and rendered in the report's fairness card.

    Only `NotTimeLocalized` is caught. Any other failure while resolving a
    span table is a real defect and still propagates -- a broad except here
    would silently relabel bugs as "this model is diffuse", which reads as a
    considered finding rather than a crash (`CLAUDE.md` sec 11.33's lesson
    about the cost of a false refusal).
    """
    routing = {}
    for mcfg in cfg.models:
        adapter = hub.get(mcfg.name)
        if not adapter.measures_own_spans:
            routing[mcfg.name] = {"eligible": "full", "measured": False}
            continue
        try:
            adapter.token_time_spans()
        except NotTimeLocalized as exc:
            log.warning("ROUTING -- model '%s' is not time-localized; it is restricted to "
                        "L0-only for this run (stages %s). %s",
                        mcfg.name, ", ".join(_L0_ONLY_STAGES), exc)
            routing[mcfg.name] = {"eligible": "l0_only", **exc.as_record()}
            continue
        finally:
            if not cfg.run.keep_models_loaded:
                hub.release(mcfg.name)
        loc = adapter.time_localization()
        routing[mcfg.name] = {
            "eligible": "full",
            "measured": loc is not None,
            **({"contrast": loc.get("contrast"), "min_contrast": loc.get("min_contrast"),
                "diffuseness": loc.get("diffuseness")} if loc else {}),
        }
    return routing


def _apply_routing(cfg: PipelineConfig, ctx: "Context", selected: set, force: set) -> dict:
    """Resolve routing, persist it, and narrow this run's stages to match.

    Deliberately only runs when `extract` is actually going to execute:
    resolving a span table loads a checkpoint, and a report-only rerun should
    not pay for a model load to rediscover a decision already on disk. When
    it is skipped, an existing `routing.json` is reused (and still narrows
    the run) so the decision survives a `--stages report` rerun rather than
    quietly reverting to "everything is eligible".
    """
    path = cfg.run_dir() / "routing.json"
    will_extract = "extract" in selected and (
        not _stage_by_name("extract").done(cfg) or "extract" in force or "all" in force)
    if will_extract:
        routing = resolve_routing(cfg, ctx.hub)
        save_json(path, routing)
    elif path.exists():
        from .utils import load_json
        routing = load_json(path)
    else:
        return {}

    l0_only = [m for m, r in routing.items() if r.get("eligible") == "l0_only"]
    if not l0_only:
        return routing
    dropped = sorted(selected - set(_L0_ONLY_STAGES))
    selected.intersection_update(_L0_ONLY_STAGES)
    log.warning("ROUTING -- %s not time-localized: this run is restricted to %s; "
                "dropping %s. Reason and measured contrast in %s, and rendered in the "
                "report's fairness card.",
                ", ".join(f"'{m}'" for m in l0_only), ", ".join(sorted(selected)),
                ", ".join(dropped) or "(nothing)", path)
    return routing


def _apply_tiers(cfg: PipelineConfig, ctx: "Context", selected: set) -> dict:
    """Narrow this run's stages to what every configured model can support.

    Runs BEFORE `_apply_routing` and unconditionally, because a tier-0 model
    cannot reach the span measurement routing depends on -- asking a black-box
    adapter for `token_time_spans` is the exact crash this gate exists to
    replace with a decision. Persisted into `routing.json` alongside the
    localization record so one artifact answers "what was this model eligible
    for, and why" (`CLAUDE.md` invariant 8: a dropped stage that leaves no
    trace in the deliverable is a silent degradation, not a loud one).
    """
    tiers = resolve_tiers(cfg, ctx.hub)
    run_tier = min(t["tier"] for t in tiers.values())
    # Computed over every stage ENABLED in the config, not over `selected`.
    # A `--stages report` rerun selects one stage, and a `dropped_stages` list
    # derived from that would come back empty -- rewriting the artifact to
    # claim the tier gate dropped nothing, and erasing the reason the report
    # prints beside each skipped section. The list is a property of the config
    # and the models, so it must not depend on which stages this invocation
    # happened to ask for.
    dropped = sorted(n for n in stage_names()
                     if _stage_by_name(n).enabled(cfg) and _STAGE_MIN_TIER.get(n, 1) > run_tier)
    if dropped:
        limiting = sorted(m for m, t in tiers.items() if t["tier"] == run_tier)
        selected.difference_update(dropped)
        log.warning("TIERS -- this run is capped at tier %d (%s) by %s; dropping %s. "
                    "Each dropped stage needs a capability those adapters do not "
                    "implement; see routing.json and the report's fairness card.",
                    run_tier, TIER_NAMES[run_tier],
                    ", ".join(f"'{m}'" for m in limiting), ", ".join(dropped))
    return {"models": tiers, "run_tier": run_tier, "dropped_stages": dropped}


def _stage_by_name(name: str) -> Stage:
    return {s.name: s for s in _stages()}[name]


def stage_names() -> list:
    return [s.name for s in _stages()]


def run_pipeline(cfg: PipelineConfig, stages: Optional[list] = None,
                 force: Optional[set] = None, allow_stale: bool = False) -> None:
    """Execute the selected (or all enabled) stages in canonical order.

    Before running anything, every stage's config fingerprint is computed
    (its own declared `config_keys` plus its dependencies' fingerprints, so
    an upstream change propagates without every downstream stage needing to
    declare it) and compared against `run_manifest.json` for any stage this
    run would otherwise skip. A mismatch means the artifacts on disk were
    produced by a different config than the one running now -- refused by
    default (`ROADMAP.md` sec 15 A3) rather than silently mixed in, since a
    stale-but-skipped stage is exactly the "looks complete, isn't" failure
    mode invariant 8 (`CLAUDE.md` sec 7) exists to prevent. `allow_stale`
    (CLI `--allow-stale`) downgrades this to a loud warning and proceeds.
    """
    setup_logging()
    set_seed(cfg.run.seed)
    cfg.run_dir().mkdir(parents=True, exist_ok=True)
    dump_config(cfg, cfg.run_dir() / "config_resolved.yaml")
    # Git/library/device/checkpoint provenance, recorded once per run start
    # (`ROADMAP.md` sec 15 A7) -- `extract`'s stage merges in `corpus_digest`
    # once the corpus is actually loaded, since which corpus a run used is
    # only known once a data-consuming stage runs, not at CLI invocation.
    record_extra(cfg.run_dir(), "provenance", run_provenance(cfg))
    force = force or set()
    all_stages = _stages()
    by_name = {s.name: s for s in all_stages}

    selected = set(stages) if stages else {s.name for s in all_stages if s.enabled(cfg)}
    unknown = selected - set(by_name)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; available: {stage_names()}")
    for name in list(selected):
        for dep in by_name[name].deps:
            if not by_name[dep].done(cfg) and dep not in selected:
                if not by_name[dep].enabled(cfg):
                    raise ValueError(
                        f"stage '{name}' needs '{dep}', which is disabled and has no artifacts")
                log.info("adding '%s' (required by '%s')", dep, name)
                selected.add(dep)

    ctx = Context(cfg)
    ctx.forced = set(force)
    save_json(cfg.run_dir() / "tiers.json", _apply_tiers(cfg, ctx, selected))
    save_json(cfg.run_dir() / "shapes.json", _apply_shape(cfg, selected))
    _apply_routing(cfg, ctx, selected, force)

    manifest = load_manifest(cfg.run_dir())
    prev_stages = manifest.get("stages", {})
    current_own, current_fp = {}, {}
    for stage in all_stages:  # already dependency-ordered: deps precede dependents
        own = resolve_config_keys(cfg, stage.config_keys)
        dep_fps = {d: current_fp[d] for d in stage.deps}
        current_own[stage.name] = own
        current_fp[stage.name] = fingerprint_stage(own, dep_fps)

    stale_messages = []
    # Every stage with artifacts on disk is checked, not only the selected
    # ones. `report` declares `deps=[]` on purpose -- it must render whatever
    # exists, which is what makes partial runs useful -- so a stale upstream
    # stage it *reads* is invisible to a dependency-based check. Before this,
    # `--stages report` after a config edit rendered new artifacts beside old
    # ones from a different config, with no warning: exactly the "looks
    # complete, isn't" failure sec 15 A3 exists to prevent, on the one path
    # that skipped the guard because it skipped the stage.
    renders_everything = any(s_.name == "report" for s_ in all_stages
                             if s_.name in selected)
    for stage in all_stages:
        if stage.name not in selected and not (renders_everything and stage.done(cfg)):
            continue
        will_skip = stage.done(cfg) and stage.name not in force and "all" not in force
        if not will_skip:
            continue
        prev = prev_stages.get(stage.name)
        if prev is None:
            log.warning("stage '%s': artifacts exist but no recorded config fingerprint "
                       "(likely written before this run had config-fingerprinting); "
                       "trusting them as-is", stage.name)
            continue
        if prev.get("fingerprint") == current_fp[stage.name]:
            continue
        own_diff = diff_resolved(prev.get("own_resolved", {}), current_own[stage.name])
        dep_diff = [d for d in stage.deps
                    if prev.get("dep_fingerprints", {}).get(d) != current_fp.get(d)]
        detail = "; ".join(f"{k}: {ov!r} -> {nv!r}" for k, ov, nv in own_diff) \
            or "(no change in this stage's own config keys)"
        dep_detail = f"; upstream stage(s) with changed config: {', '.join(dep_diff)}" \
            if dep_diff else ""
        msg = (f"stage '{stage.name}' would be skipped (artifacts exist) but its config "
              f"fingerprint no longer matches: {detail}{dep_detail}")
        if allow_stale:
            log.warning("STALE -- %s -- proceeding anyway (--allow-stale)", msg)
        else:
            stale_messages.append(msg)
    if stale_messages:
        names = ", ".join(m.split("'")[1] for m in stale_messages)
        # `--force X` only bypasses the skip for a stage that is SELECTED, so
        # for a stale stage outside the selection the remediation has to name
        # `--stages` too. Getting this wrong sends the reader in a circle:
        # they force the stage, nothing re-runs, and the same error returns.
        raise ValueError(
            "refusing to run with stale artifacts (ROADMAP.md sec 15 A3):\n  "
            + "\n  ".join(stale_messages)
            + f"\nRerun with --stages {names},report --force {names},report (and "
              f"anything downstream that depends on "
              f"{'it' if len(stale_messages) == 1 else 'them'}) to regenerate under the "
              f"current config, or pass --allow-stale to proceed anyway at your own risk.")

    for stage in all_stages:
        if stage.name not in selected:
            continue
        if stage.done(cfg) and stage.name not in force and "all" not in force:
            log.info("stage '%s': artifacts exist, skipping (--force %s to rerun)",
                     stage.name, stage.name)
            continue
        log.info("stage '%s': running", stage.name)
        t0 = time.time()
        stage.run(ctx)
        prev_stages[stage.name] = {
            "fingerprint": current_fp[stage.name],
            "own_resolved": current_own[stage.name],
            "dep_fingerprints": {d: current_fp[d] for d in stage.deps},
            "timestamp": datetime.datetime.now().isoformat(),
            "wall_clock_seconds": round(time.time() - t0, 2),
        }
        extra = {k: v for k, v in load_manifest(cfg.run_dir()).items()
                if k not in ("version", "stages")}
        save_manifest(cfg.run_dir(), {"version": 1, "stages": prev_stages, **extra})
    log.info("pipeline finished: %s", cfg.run_dir())
