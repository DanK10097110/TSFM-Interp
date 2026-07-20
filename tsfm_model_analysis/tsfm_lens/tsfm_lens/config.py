"""Typed pipeline configuration with YAML loading and reasonable defaults.

Every analysis stage carries an `enabled` flag so users can compose exactly
the comparison they want; disabled stages are skipped and simply absent from
the final report.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "model"
    adapter: str = "mock_patch"
    checkpoint: str = ""
    layer_regex: str = ""
    batch_size: int = 32
    capture_layer_stride: int = 1
    kwargs: dict = field(default_factory=dict)


@dataclass
class DataConfig:
    source: str = "smoke"
    path: str = ""
    context_len: int = 512
    horizon: int = 64
    max_series: Optional[int] = None
    family_key: str = "auto"
    smoke_series_per_family: int = 20


@dataclass
class AlignmentConfig:
    window: int = 32
    sanity_check: bool = False


@dataclass
class ExtractionConfig:
    enabled: bool = True
    store_dtype: str = "float16"


@dataclass
class L0Config:
    enabled: bool = True
    quantiles: list = field(default_factory=lambda: [0.1, 0.5, 0.9])


@dataclass
class L1Config:
    enabled: bool = True
    layer_stride: int = 1
    max_rows: int = 100_000
    family_conditioned: bool = True
    min_family_series: int = 15
    rsa: bool = True
    rsa_max_series: int = 600


@dataclass
class L2Config:
    enabled: bool = True
    layer_stride: int = 2
    lambdas: list = field(default_factory=lambda: [1e-3, 1e-2, 1e-1, 1.0])
    val_frac: float = 0.25
    max_rows: int = 60_000
    input_baseline: bool = True


@dataclass
class PatchingConfig:
    enabled: bool = True
    layer_stride: int = 2
    corruptions: list = field(default_factory=lambda: [
        "deseasonalize", "noise", "level_shift", "spike", "warp", "dropout"])
    max_series: int = 64
    num_samples: int = 8
    per_window: bool = True
    window_stride: int = 1


@dataclass
class LensConfig:
    enabled: bool = True
    layer_stride: int = 1
    max_series: int = 64
    tuned: bool = True
    tuned_max_series: int = 2000
    val_frac: float = 0.25
    lambdas: list = field(default_factory=lambda: [1e-3, 1e-2, 1e-1, 1.0])
    crystallization_tol: float = 0.1


@dataclass
class AttentionConfig:
    enabled: bool = True
    patterns: bool = True
    max_series: int = 128
    batch_series: int = 16
    max_lag_tokens: int = 256
    ablation: bool = True
    ablation_max_series: int = 128
    head_layer_stride: int = 1
    top_k: int = 5


@dataclass
class ExemplarsConfig:
    enabled: bool = True
    per_family: int = 2
    max_families: int = 6


@dataclass
class L3Config:
    enabled: bool = True
    corruptions: dict = field(default_factory=lambda: {
        "noise": {"snr_db": 6.0},
        "detrend": {},
        "deseasonalize": {"top_k": 2},
        "frequency_shift": {"factor": 2.0},
        "level_shift": {"position_frac": 0.6, "scale": 3.0},
        "spike": {"count": 3, "scale": 6.0},
        "smooth": {"kernel": 9},
        "warp": {"strength": 0.15},
        "dropout": {"frac": 0.15, "n_blocks": 3},
    })
    max_series: int = 512
    patching: PatchingConfig = field(default_factory=PatchingConfig)


@dataclass
class StatsConfig:
    enabled: bool = True
    n_boot: int = 500
    n_boot_heavy: int = 200
    ci: float = 0.95
    alpha: float = 0.05
    min_series: int = 8


@dataclass
class InternalsConfig:
    enabled: bool = True
    layer_stride: int = 1
    max_rows: int = 40_000
    probe_pca_dim: int = 50


@dataclass
class ConfirmConfig:
    enabled: bool = False
    source: str = "sealed"
    path: str = ""
    max_series: int = 1024
    require_seal: bool = True
    alpha: float = 0.05


@dataclass
class ClusteringConfig:
    enabled: bool = True
    layer: str = "auto"
    k: str = "auto"
    pca_dim: int = 50
    max_series: int = 5000
    use_umap: bool = True


@dataclass
class SAEConfig:
    enabled: bool = False
    checkpoints: dict = field(default_factory=dict)


@dataclass
class ReportConfig:
    enabled: bool = True
    title: str = "TSFM Comparison Report"


@dataclass
class RunConfig:
    name: str = "run"
    out_dir: str = "runs"
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 0
    keep_models_loaded: bool = False


@dataclass
class PipelineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: list = field(default_factory=list)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    l0: L0Config = field(default_factory=L0Config)
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config = field(default_factory=L2Config)
    l3: L3Config = field(default_factory=L3Config)
    lens: LensConfig = field(default_factory=LensConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    exemplars: ExemplarsConfig = field(default_factory=ExemplarsConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    internals: InternalsConfig = field(default_factory=InternalsConfig)
    confirm: ConfirmConfig = field(default_factory=ConfirmConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def run_dir(self) -> Path:
        """Directory holding every artifact for this run."""
        return Path(self.run.out_dir) / self.run.name

    def comparison_pair(self) -> tuple:
        """The two models all cross-model analyses compare (first two configured)."""
        if len(self.models) < 2:
            raise ValueError("cross-model analyses require at least two models in config")
        return self.models[0], self.models[1]

    def validate(self) -> None:
        """Fail fast on structurally invalid configurations."""
        if self.data.context_len % self.alignment.window != 0:
            raise ValueError("data.context_len must be a multiple of alignment.window")
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
        if len(self.models) > 2:
            import logging
            logging.getLogger("tsfm_lens").warning(
                "more than two models configured; comparisons use the first two, "
                "extraction and L0 run for all")


def _build(cls: type, data: dict):
    """Recursively construct a dataclass from a plain dict, keeping defaults for absent keys."""
    if data is None:
        return cls()
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        if dataclasses.is_dataclass(f.type) if isinstance(f.type, type) else False:
            kwargs[f.name] = _build(f.type, v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


_NESTED = {
    "run": RunConfig, "data": DataConfig, "alignment": AlignmentConfig,
    "extraction": ExtractionConfig, "l0": L0Config, "l1": L1Config, "l2": L2Config,
    "clustering": ClusteringConfig, "sae": SAEConfig, "report": ReportConfig,
    "stats": StatsConfig, "internals": InternalsConfig, "confirm": ConfirmConfig,
    "lens": LensConfig, "attention": AttentionConfig, "exemplars": ExemplarsConfig,
}


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML config into a validated PipelineConfig."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return config_from_dict(raw)


def config_from_dict(raw: dict) -> PipelineConfig:
    """Construct and validate a PipelineConfig from a plain dict."""
    cfg = PipelineConfig()
    for key, cls in _NESTED.items():
        if key in raw:
            setattr(cfg, key, _build(cls, raw[key]))
    if "l3" in raw:
        l3raw = dict(raw["l3"])
        patch = _build(PatchingConfig, l3raw.pop("patching", None))
        cfg.l3 = _build(L3Config, l3raw)
        cfg.l3.patching = patch
    if "models" in raw:
        cfg.models = [_build(ModelConfig, m) for m in raw["models"]]
    cfg.validate()
    return cfg


def dump_config(cfg: PipelineConfig, path: Path) -> None:
    """Persist the fully resolved config next to the run artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
