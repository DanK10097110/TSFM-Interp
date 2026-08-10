"""Adapter registry and the lazy-loading ModelHub.

Register a new model by adding one entry to ADAPTERS; everything downstream
consumes the ModelAdapter interface only.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..config import DataConfig, ModelConfig
from ..utils import log
from .base import ModelAdapter
from .mock import MockPatchAdapter, MockStepAdapter, MockWaveAdapter

ADAPTERS: Dict[str, type] = {
    "mock_patch": MockPatchAdapter,
    "mock_step": MockStepAdapter,
    "mock_wave": MockWaveAdapter,
}


def _register_optional() -> None:
    """Register real-model adapters; their heavy imports happen only at load()."""
    from .chronos_adapter import ChronosAdapter
    from .chronos_bolt_adapter import ChronosBoltAdapter
    from .chronos2_adapter import Chronos2Adapter
    from .timesfm_adapter import TimesFMAdapter
    ADAPTERS["chronos"] = ChronosAdapter
    ADAPTERS["chronos_bolt"] = ChronosBoltAdapter
    ADAPTERS["chronos2"] = Chronos2Adapter
    ADAPTERS["timesfm"] = TimesFMAdapter


_register_optional()


def build_adapter(mcfg: ModelConfig, data_cfg: DataConfig,
                  device: torch.device, dtype: torch.dtype) -> ModelAdapter:
    """Construct (without loading) the adapter named in a model config."""
    if mcfg.adapter not in ADAPTERS:
        raise ValueError(f"unknown adapter '{mcfg.adapter}'; available: {sorted(ADAPTERS)}")
    return ADAPTERS[mcfg.adapter](mcfg, data_cfg, device, dtype)


class ModelHub:
    """Caches adapters per model name and releases weights when asked."""

    def __init__(self, model_cfgs: list, data_cfg: DataConfig,
                 device: torch.device, dtype: torch.dtype):
        self._cfgs = {m.name: m for m in model_cfgs}
        self._adapters: Dict[str, ModelAdapter] = {}
        self._data_cfg, self._device, self._dtype = data_cfg, device, dtype

    def get(self, name: str) -> ModelAdapter:
        """Return the adapter for a model, constructing it on first request."""
        if name not in self._adapters:
            self._adapters[name] = build_adapter(self._cfgs[name], self._data_cfg,
                                                 self._device, self._dtype)
        return self._adapters[name]

    def release(self, name: str) -> None:
        """Unload one model's weights to free accelerator memory between stages."""
        if name in self._adapters and self._adapters[name]._loaded:
            log.info("releasing model '%s'", name)
            self._adapters[name].unload()
