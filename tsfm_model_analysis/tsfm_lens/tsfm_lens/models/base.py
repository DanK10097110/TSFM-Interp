"""The architecture-agnostic model contract.

Every model enters the pipeline through a `ModelAdapter`, which owns four
responsibilities: loading, running a forward pass that fires capture hooks,
mapping internal tokens to time spans (the key to cross-architecture
alignment), and producing forecasts. New models plug in by subclassing this
and registering in `models/__init__.py`; nothing downstream changes.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from ..config import DataConfig, ModelConfig
from ..utils import log


class ModelAdapter(ABC):
    """Wraps one model behind a uniform capture/align/predict interface."""

    default_layer_regex: str = r".*"

    def __init__(self, cfg: ModelConfig, data_cfg: DataConfig,
                 device: torch.device, dtype: torch.dtype):
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.device = device
        self.dtype = dtype
        self._loaded = False

    @property
    def name(self) -> str:
        return self.cfg.name

    @abstractmethod
    def load(self) -> None:
        """Instantiate the underlying model on the target device."""

    @property
    @abstractmethod
    def module(self) -> nn.Module:
        """The torch module hooks attach to."""

    @abstractmethod
    def prepare(self, contexts: np.ndarray) -> Any:
        """Convert raw contexts [B, context_len] into model-ready inputs."""

    @abstractmethod
    def forward(self, prepared: Any) -> None:
        """Run one forward pass over prepared inputs so capture hooks fire."""

    @abstractmethod
    def token_time_spans(self) -> np.ndarray:
        """Per-token (start, end) time coverage in context steps, shape [n_tokens, 2].

        This is the single piece of information that makes representations
        from patch-based and per-step tokenizations comparable.
        """

    @abstractmethod
    def predict(self, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
        """Forecast, returning {'point': [B, H], 'quantiles': [B, H, Q]}."""

    def ensure_loaded(self) -> None:
        """Load lazily on first use."""
        if not self._loaded:
            log.info("loading model '%s' (%s)", self.name, self.cfg.adapter)
            self.load()
            self.module.eval()
            self._loaded = True

    def unload(self) -> None:
        """Release the model and free accelerator memory."""
        self._release()
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release(self) -> None:
        """Drop references to the underlying model; subclasses extend as needed."""

    def layer_names(self) -> list:
        """Capture points: named modules matching the layer regex, in forward order."""
        self.ensure_loaded()
        pattern = re.compile(self.cfg.layer_regex or self.default_layer_regex)
        names = [n for n, _ in self.module.named_modules() if pattern.search(n)]
        if not names:
            raise ValueError(
                f"model '{self.name}': layer regex matched nothing; "
                f"run discover_layers() to inspect module names")
        return names[:: max(1, self.cfg.capture_layer_stride)]

    def discover_layers(self, contains: str = "") -> list:
        """List candidate module names, for choosing a layer regex on a new checkpoint."""
        self.ensure_loaded()
        return [n for n, _ in self.module.named_modules() if contains in n]

    def postprocess_tokens(self, layer_name: str, hidden: torch.Tensor) -> torch.Tensor:
        """Strip special or padded positions so tokens match `token_time_spans`.

        The default trusts the raw capture; adapters whose sequence carries
        EOS or padding override this. A shape mismatch here is loud on
        purpose: silent misalignment would corrupt every downstream analysis.
        """
        expected = self.token_time_spans().shape[0]
        if hidden.shape[1] == expected:
            return hidden
        raise ValueError(
            f"model '{self.name}' layer '{layer_name}': captured {hidden.shape[1]} tokens, "
            f"alignment expects {expected}; override postprocess_tokens for this adapter")

    def find_module(self, name: str) -> nn.Module:
        """Fetch a submodule by its qualified name."""
        return dict(self.module.named_modules())[name]

    def token_slice(self, live_len: int) -> slice:
        """Positions of the postprocessed tokens within a live captured sequence.

        The inverse of `postprocess_tokens` for patching: activation patching
        writes cached clean token states back into a running forward, and this
        says where they belong. Default assumes tokens lead the sequence with
        any specials trailing; adapters with front padding override.
        """
        expected = self.token_time_spans().shape[0]
        if live_len < expected:
            raise ValueError(f"model '{self.name}': live sequence {live_len} shorter "
                             f"than expected {expected} tokens")
        return slice(0, expected)

    def hidden_size(self) -> Optional[int]:
        """Best-effort hidden dimension, resolved from the first capture on the fly if unknown."""
        return None
