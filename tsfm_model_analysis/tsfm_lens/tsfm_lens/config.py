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
    sanity_check: bool = True
    min_diagonal_frac: float = 0.5   # required hit fraction at the shallowest probed layer
    on_failure: str = "fail"         # fail | warn -- ROADMAP.md sec 15 A2


@dataclass
class ExtractionConfig:
    enabled: bool = True
    # "float16" (default, half the disk/VRAM of float32) or "float32" for a
    # model whose activations run hot enough to overflow float16's ~65504
    # max (`root.attrs["nonfinite"]`, written by `extraction/store.py::
    # finalize_layer`, sec 15 A19, says which layers actually do on a given
    # run -- check it before assuming this knob is needed for a new model).
    store_dtype: str = "float16"


@dataclass
class L0Config:
    enabled: bool = True
    quantiles: list = field(default_factory=lambda: [0.1, 0.5, 0.9])
    # MASE denominator (`ROADMAP.md` sec 15 A11): "mean_abs_diff" (default,
    # every previously-recorded number keeps reproducing) or
    # "seasonal_naive" (mean absolute error of a period-m seasonal-naive
    # forecast on the context -- the standard fix for scale degeneracy
    # under heavy intermittency or a near-flat context; period estimated
    # per series via autocorrelation when not otherwise available).
    scale: str = "mean_abs_diff"
    # Reliability guard, independent of `scale`: a series whose MASE
    # denominator is below this fraction of the target's own mean absolute
    # level is flagged `mase_reliable: false` and excluded from MASE
    # aggregates (never from smape/pinball, which don't share this
    # degeneracy). 0.0 disables the guard (matches pre-A11 behavior
    # exactly); >0 is the recommended setting for any corpus containing
    # `intermittent_bursts` or similarly zero-heavy families.
    min_scale_frac: float = 0.05
    # Repeat-run noise floor (`ROADMAP.md` sec 15 A13): how much a model's
    # own MASE varies between two calls that should, in the absence of
    # sampling, be identical -- every ΔMASE elsewhere in the repo (head/MLP
    # ablation, SAE forecast-preservation, L3 restoration) is otherwise
    # compared against zero instead of against this. `noise_floor_repeats:
    # 0` disables it (no extra predict() calls, matches pre-A13 runtime);
    # >=2 measures it on a small `noise_floor_series`-sized subset. Cheap
    # relative to what it re-contextualizes, so on by default with a small
    # repeat count (3) rather than requiring opt-in.
    noise_floor_repeats: int = 3
    noise_floor_series: int = 16


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
    # `ROADMAP.md` sec 15 A12: corruption strengths are set independently
    # per corruption, so raw "behavioral sensitivity" bars aren't comparable
    # across corruptions (`level_shift` dominates by construction). `"none"`
    # (default) keeps every recorded number reproducing byte-for-byte.
    # `"input_energy"` re-solves each calibratable corruption's magnitude
    # parameter (pure numpy, no model) to hit a common per-series
    # perturbation-energy budget derived from this battery's own configured
    # strengths (the median of their natural energies) -- every corruption
    # is still run and reported either way (§5's decision to keep the
    # low-signal end of the contrast stands).
    calibrate: str = "none"


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
class LayerScreenConfig:
    enabled: bool = True
    method: str = "work_bend"       # work_bend | coverage | factor_emergence
    budget_frac: float = 0.25
    min_budget: int = 2
    max_series: int = 100_000       # cap on rows fed to the screen; effectively "all" by default
    use_curvature: bool = True      # work_bend only
    seed: Optional[int] = None      # falls back to run.seed
    stride: int = 1                 # screening capture stride, independent of models[*].capture_layer_stride
    require_full_capture: bool = True   # run a dedicated stride-1 screening extraction (ROADMAP.md sec 15 A1)
    keep_store: bool = False        # keep screen_activations.zarr after selection instead of deleting it


@dataclass
class SAEConfig:
    enabled: bool = False
    checkpoints: dict = field(default_factory=dict)
    targets: list = field(default_factory=list)  # [{"model": "...", "layer": "..."}]
    dict_size_mult: int = 8
    k: int = 32
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 4096
    resample_dead_every_epochs: int = 0
    forecast_preservation_max_series: int = 64
    ground_truth_max_series: int = 2000
    real_data_enabled: bool = False
    real_data_source: str = "Monash-University/monash_tsf"
    real_data_n_windows: int = 20_000
    real_data_pool_limit: int = 2000


@dataclass
class ReportConfig:
    enabled: bool = True
    title: str = "TSFM Comparison Report"
    verbose: bool = True
    verbose_series: int = 3
    # A section builder raising is a bug (ROADMAP.md sec 15 A5), so the
    # default is to surface it as a hard failure (non-zero exit) rather than
    # let a report with a silently-dropped section report success. CLI
    # --allow-partial-report (or setting this true) downgrades that to a
    # loud warning, matching --allow-stale's precedent.
    allow_partial: bool = False


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
    layer_screen: LayerScreenConfig = field(default_factory=LayerScreenConfig)
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
    "clustering": ClusteringConfig, "layer_screen": LayerScreenConfig,
    "sae": SAEConfig, "report": ReportConfig,
    "stats": StatsConfig, "internals": InternalsConfig, "confirm": ConfirmConfig,
    "lens": LensConfig, "attention": AttentionConfig, "exemplars": ExemplarsConfig,
}


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML config into a validated PipelineConfig."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
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
    path.write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False), encoding="utf-8")
