"""Adapter registry and the lazy-loading ModelHub.

Register a new model by adding one entry to ADAPTERS; everything downstream
consumes the ModelAdapter interface only. Contributed adapters (`ROADMAP.md`
sec 34.6 Item E1) live in `models/contrib/` instead and never touch this
dict directly -- they are discovered into `CONTRIB_REGISTRY` below, by
scanning source text rather than importing it, so a heavy or broken contrib
file cannot affect `import tsfm_lens.models` for anyone else.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from ..config import DataConfig, ModelConfig
from ..utils import log
from . import contrib
from .base import ModelAdapter
from .mock import (MockBlackBoxAdapter, MockEncDecAdapter, MockPatchAdapter,
                   MockStepAdapter, MockWaveAdapter)

ADAPTERS: Dict[str, type] = {
    "mock_patch": MockPatchAdapter,
    "mock_step": MockStepAdapter,
    "mock_wave": MockWaveAdapter,
    "mock_encdec": MockEncDecAdapter,
    "mock_blackbox": MockBlackBoxAdapter,
}


def _register_optional() -> None:
    """Register real-model adapters; their heavy imports happen only at load()."""
    from .chronos_adapter import ChronosAdapter
    from .generic_hf_adapter import GenericHFAdapter
    from .chronos_bolt_adapter import ChronosBoltAdapter
    from .chronos2_adapter import Chronos2Adapter
    from .sundial_adapter import SundialAdapter
    from .timesfm_adapter import TimesFMAdapter
    ADAPTERS["chronos"] = ChronosAdapter
    ADAPTERS["generic_hf"] = GenericHFAdapter
    ADAPTERS["chronos_bolt"] = ChronosBoltAdapter
    ADAPTERS["chronos2"] = Chronos2Adapter
    ADAPTERS["sundial"] = SundialAdapter
    ADAPTERS["timesfm"] = TimesFMAdapter


_register_optional()


# `CONTRIB_REGISTRY`: name -> (dotted_module_path, class_name), discovered by
# AST scan only (`contrib.discover_contrib_adapters`, no import). `errors`
# from that scan, PLUS any contrib name colliding with a name already in
# ADAPTERS above (checked here, since only this module knows the built-in
# names), land in `discovery_errors` -- printed by `run.py --doctor`
# (ROADMAP.md sec 34.6 E1.2: silent skipping is forbidden).
CONTRIB_REGISTRY: Dict[str, Tuple[str, str]] = {}
discovery_errors: List[dict] = []


def _discover_contrib() -> None:
    registry, errors = contrib.discover_contrib_adapters()
    discovery_errors.extend(errors)
    for name, (module_path, class_name) in registry.items():
        if name in ADAPTERS:
            filename = module_path.rsplit(".", 1)[-1] + ".py"
            log.warning("contrib adapter '%s' declared in %s collides with the built-in "
                       "adapter of the same name -- rename its ADAPTER_NAME", name, filename)
            discovery_errors.append({
                "file": filename, "collides_with": name,
                "error": f"ADAPTER_NAME '{name}' collides with a built-in adapter "
                        f"registered by tsfm_lens.models -- rename it"})
            continue
        CONTRIB_REGISTRY[name] = (module_path, class_name)
    for err in errors:
        log.warning("contrib adapter discovery: %s: %s", err["file"], err["error"])


_discover_contrib()


def resolve_adapter_class(name: str) -> type:
    """The class registered under `name`, built-in or contrib -- no instance.

    Split out of `build_adapter` (ROADMAP.md sec 34.6 Item E4, found while
    verifying E3's own scaffolded config against real preflight output) so a
    caller that only needs a class-level fact -- `capability_tier()` is a
    classmethod, per `pipeline.py::resolve_tiers`'s own docstring "costs
    nothing" -- is not forced to also supply `data_cfg`/`device`/`dtype` and
    construct a full instance just to ask it. One lookup, used by both
    `build_adapter` here and `doctor.py`'s batch-cap preflight check, so the
    built-in-then-contrib-then-collision resolution order can never drift
    between the two call sites (`CLAUDE.md` sec 11.34's tie-break lesson).
    """
    if name in ADAPTERS:
        return ADAPTERS[name]
    if name in CONTRIB_REGISTRY:
        module_path, class_name = CONTRIB_REGISTRY[name]
        try:
            return contrib.import_contrib_class(module_path, class_name)
        except Exception as exc:
            raise ValueError(
                f"contrib adapter '{name}' ({module_path}.{class_name}) failed to "
                f"import: {type(exc).__name__}: {exc}") from exc
    collision = next((e for e in discovery_errors if e.get("collides_with") == name), None)
    if collision is not None:
        raise ValueError(
            f"adapter name '{name}' collides with a built-in adapter of the same "
            f"name, declared by contrib file {collision['file']} -- rename the contrib "
            f"ADAPTER_NAME to something else")
    raise ValueError(f"unknown adapter '{name}'; available: "
                     f"{sorted(list(ADAPTERS) + list(CONTRIB_REGISTRY))}")


def build_adapter(mcfg: ModelConfig, data_cfg: DataConfig,
                  device: torch.device, dtype: torch.dtype) -> ModelAdapter:
    """Construct (without loading) the adapter named in a model config.

    Checks the built-in registry first, then contrib (by lazy import, which
    is the first point a contrib file's own top-level code -- and any heavy
    library it imports -- actually runs). A name that collided with a
    built-in at discovery time was excluded from `CONTRIB_REGISTRY`, so it
    raises here naming both, rather than silently resolving to whichever one
    registration order happened to favor (`CLAUDE.md` sec 11.34).
    """
    cls = resolve_adapter_class(mcfg.adapter)
    return cls(mcfg, data_cfg, device, dtype)


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
