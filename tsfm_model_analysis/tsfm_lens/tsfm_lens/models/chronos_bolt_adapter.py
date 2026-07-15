"""Adapter for Chronos-Bolt (T5 encoder over input patches, direct quantile head).

Bolt patches the context (default 16-step non-overlapping patches read from
the checkpoint's chronos_config) and decodes all quantiles in one forward
pass, so unlike sampled Chronos its forecasts are deterministic — L3
restoration curves for Bolt carry no sampling noise. Capture rides
`predict_quantiles` so hooks see the production path. Bolt left-pads the
context to a whole number of patches and may append a [REG] token; spans and
postprocessing account for both, and overlapping strides are handled
naturally by the overlap-weighted pooling.

Written against chronos-forecasting>=1.4 with chronos-bolt-* checkpoints;
run --check-alignment on your installed version before trusting results.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ..utils import log
from .base import ModelAdapter


class ChronosBoltAdapter(ModelAdapter):

    default_layer_regex = r"encoder\.block\.\d+$"

    def load(self) -> None:
        """Load a Bolt pipeline and read patching geometry from its config."""
        from chronos import BaseChronosPipeline
        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.cfg.checkpoint or "amazon/chronos-bolt-small",
            device_map=str(self.device), torch_dtype=self.dtype)
        self._inner = self.pipeline.model
        ccfg = getattr(self._inner.config, "chronos_config", {}) or {}
        self._patch = int(ccfg.get("input_patch_size", 16))
        self._stride = int(ccfg.get("input_patch_stride", self._patch))
        self._model_context = int(ccfg.get("context_length", 2048))
        self._use_reg = bool(ccfg.get("use_reg_token", True))
        log.info("chronos_bolt '%s': patch=%d stride=%d model_context=%d reg=%s",
                 self.name, self._patch, self._stride, self._model_context, self._use_reg)

    @property
    def module(self):
        return self._inner

    def _release(self) -> None:
        self.pipeline = None
        self._inner = None

    def _geometry(self):
        """(kept steps, token count, front pad) for the current data context."""
        kept = min(self.data_cfg.context_len, self._model_context)
        if kept <= self._patch:
            n_tokens = 1
        else:
            n_tokens = 1 + math.ceil((kept - self._patch) / self._stride)
        front_pad = (n_tokens - 1) * self._stride + self._patch - kept
        return kept, n_tokens, front_pad

    def prepare(self, contexts: np.ndarray) -> Any:
        return torch.from_numpy(np.ascontiguousarray(contexts)).float()

    def forward(self, prepared: Any) -> None:
        """One quantile forecast pass; encoder hooks capture patch-token states."""
        self.pipeline.predict_quantiles(prepared, prediction_length=self.data_cfg.horizon,
                                        quantile_levels=[0.5])

    def token_time_spans(self) -> np.ndarray:
        """Patch spans, right-aligned and clipped where front padding covers no data."""
        kept, n_tokens, front_pad = self._geometry()
        offset = self.data_cfg.context_len - kept
        starts = np.arange(n_tokens) * self._stride - front_pad
        spans = np.stack([starts, starts + self._patch], axis=1).astype(np.float64)
        return np.clip(spans, 0, kept) + offset

    def postprocess_tokens(self, layer_name: str, hidden: torch.Tensor) -> torch.Tensor:
        """Drop the trailing [REG] token when present."""
        _, n_tokens, _ = self._geometry()
        if hidden.shape[1] == n_tokens + 1 and self._use_reg:
            return hidden[:, :n_tokens]
        return super().postprocess_tokens(layer_name, hidden)

    def predict(self, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
        """Deterministic quantile forecast via the Bolt head."""
        q, mean = self.pipeline.predict_quantiles(
            self.prepare(contexts), prediction_length=horizon,
            quantile_levels=list(quantiles))
        return {"point": mean.float().cpu().numpy(),
                "quantiles": q.float().cpu().numpy()}
