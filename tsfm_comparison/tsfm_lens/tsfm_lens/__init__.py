"""tsfm_lens: layered, architecture-agnostic comparison of time-series foundation models."""

__version__ = "0.1.0"

from .config import PipelineConfig, config_from_dict, load_config
from .pipeline import run_pipeline

__all__ = ["PipelineConfig", "config_from_dict", "load_config", "run_pipeline", "__version__"]
