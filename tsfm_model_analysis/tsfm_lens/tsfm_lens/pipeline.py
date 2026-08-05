"""Stage orchestration.

Stages form a short DAG (everything depends on extraction; the report reads
whatever exists). Each stage is gated by its config `enabled` flag, skipped
when its artifacts already exist unless forced, and selectable by name from
the CLI, so any subset of analyses composes into a valid run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .analysis.attention import run_attention
from .analysis.clustering import run_clustering
from .analysis.exemplars import run_exemplars
from .analysis.confirm import run_confirm
from .analysis.internals import run_internals
from .analysis.l0_behavioral import run_l0
from .analysis.l1_geometry import run_l1
from .analysis.l2_stitching import run_l2
from .analysis.l3_perturbation import run_l3
from .analysis.lens import run_lens
from .config import PipelineConfig, dump_config
from .data import BenchmarkData, load_benchmark
from .extraction.extract import run_extraction
from .extraction.store import ActivationStore
from .models import ModelHub
from .report.report import run_report
from .sae.train import run_sae
from .utils import log, resolve_device, resolve_dtype, set_seed, setup_logging


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


def _stages() -> list:
    """The canonical stage sequence."""
    return [
        Stage("extract", [],
              lambda c: c.extraction.enabled,
              lambda c: (c.run_dir() / "activations.zarr").exists()
              and (c.run_dir() / "meta.parquet").exists(),
              lambda ctx: (run_extraction(ctx.cfg, ctx.hub, ctx.data),
                           ctx.reset_store())),
        Stage("l0", ["extract"],
              lambda c: c.l0.enabled,
              lambda c: (c.run_dir() / "l0" / "summary.json").exists(),
              lambda ctx: run_l0(ctx.cfg, ctx.hub, ctx.data, ctx.store)),
        Stage("internals", ["extract"],
              lambda c: c.internals.enabled,
              lambda c: (c.run_dir() / "internals" / "profile.json").exists(),
              lambda ctx: run_internals(ctx.cfg, ctx.store, ctx.data, ctx.device)),
        Stage("lens", ["extract"],
              lambda c: c.lens.enabled,
              lambda c: (c.run_dir() / "lens" / "lens.json").exists(),
              lambda ctx: run_lens(ctx.cfg, ctx.hub, ctx.store, ctx.data, ctx.device)),
        Stage("l1", ["extract"],
              lambda c: c.l1.enabled,
              lambda c: (c.run_dir() / "l1" / "meta.json").exists(),
              lambda ctx: run_l1(ctx.cfg, ctx.store, ctx.device)),
        Stage("l2", ["extract"],
              lambda c: c.l2.enabled,
              lambda c: (c.run_dir() / "l2" / "stitching.json").exists(),
              lambda ctx: run_l2(ctx.cfg, ctx.store, ctx.data, ctx.device)),
        Stage("l3", ["extract"],
              lambda c: c.l3.enabled,
              lambda c: (c.run_dir() / "l3" / "meta.json").exists(),
              lambda ctx: run_l3(ctx.cfg, ctx.hub, ctx.store, ctx.data, ctx.device)),
        Stage("attention", ["extract"],
              lambda c: c.attention.enabled,
              lambda c: (c.run_dir() / "attention" / "meta.json").exists(),
              lambda ctx: run_attention(ctx.cfg, ctx.hub, ctx.store, ctx.data, ctx.device)),
        Stage("cluster", ["extract"],
              lambda c: c.clustering.enabled,
              lambda c: (c.run_dir() / "clustering" / "comparison.json").exists(),
              lambda ctx: run_clustering(ctx.cfg, ctx.store, ctx.data)),
        Stage("sae", ["extract"],
              lambda c: c.sae.enabled,
              lambda c: (c.run_dir() / "sae" / "meta.json").exists(),
              lambda ctx: run_sae(ctx.cfg, ctx.hub, ctx.store, ctx.data, ctx.device)),
        Stage("exemplars", ["l0"],
              lambda c: c.exemplars.enabled,
              lambda c: (c.run_dir() / "exemplars" / "exemplars.json").exists(),
              lambda ctx: run_exemplars(ctx.cfg, ctx.hub, ctx.store, ctx.data, ctx.device)),
        Stage("confirm", ["l0"],
              lambda c: c.confirm.enabled,
              lambda c: (c.run_dir() / "confirm" / "confirmation.json").exists(),
              lambda ctx: run_confirm(ctx.cfg, ctx.hub)),
        Stage("report", [],
              lambda c: c.report.enabled,
              lambda c: False,
              lambda ctx: run_report(ctx.cfg)),
    ]


def stage_names() -> list:
    return [s.name for s in _stages()]


def run_pipeline(cfg: PipelineConfig, stages: Optional[list] = None,
                 force: Optional[set] = None) -> None:
    """Execute the selected (or all enabled) stages in canonical order."""
    setup_logging()
    set_seed(cfg.run.seed)
    cfg.run_dir().mkdir(parents=True, exist_ok=True)
    dump_config(cfg, cfg.run_dir() / "config_resolved.yaml")
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
    for stage in all_stages:
        if stage.name not in selected:
            continue
        if stage.done(cfg) and stage.name not in force and "all" not in force:
            log.info("stage '%s': artifacts exist, skipping (--force %s to rerun)",
                     stage.name, stage.name)
            continue
        log.info("stage '%s': running", stage.name)
        stage.run(ctx)
    log.info("pipeline finished: %s", cfg.run_dir())
