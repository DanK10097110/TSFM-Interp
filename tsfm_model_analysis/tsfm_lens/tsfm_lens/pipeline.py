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
from .analysis.model_budget import run_budget
from .config import PipelineConfig, dump_config
from .data import BenchmarkData, load_benchmark
from .extraction.extract import run_extraction
from .extraction.store import ActivationStore
from .manifest import (diff_resolved, fingerprint_stage, load_manifest,
                       record_extra, resolve_config_keys, save_manifest)
from .models import ModelHub
from .report.report import run_report
from .sae.train import run_sae
from .utils import log, resolve_device, resolve_dtype, run_provenance, set_seed, setup_logging


class Context:
    """Lazy shared state handed to every stage."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
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
              lambda ctx: run_confirm(ctx.cfg, ctx.hub),
              ("confirm",)),
        Stage("report", [],
              lambda c: c.report.enabled,
              lambda c: False,
              lambda ctx: run_report(ctx.cfg),
              ("report",)),
    ]


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

    manifest = load_manifest(cfg.run_dir())
    prev_stages = manifest.get("stages", {})
    current_own, current_fp = {}, {}
    for stage in all_stages:  # already dependency-ordered: deps precede dependents
        own = resolve_config_keys(cfg, stage.config_keys)
        dep_fps = {d: current_fp[d] for d in stage.deps}
        current_own[stage.name] = own
        current_fp[stage.name] = fingerprint_stage(own, dep_fps)

    stale_messages = []
    for stage in all_stages:
        if stage.name not in selected:
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
        raise ValueError(
            "refusing to run with stale artifacts (ROADMAP.md sec 15 A3):\n  "
            + "\n  ".join(stale_messages)
            + f"\nRerun with --force {names} (and everything downstream that depends on "
              f"{'it' if len(stale_messages) == 1 else 'them'}) to regenerate under the "
              f"current config, or pass --allow-stale to proceed anyway at your own risk.")

    ctx = Context(cfg)
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
