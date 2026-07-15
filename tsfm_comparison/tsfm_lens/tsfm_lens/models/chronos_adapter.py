"""Adapter for Amazon Chronos (T5 encoder-decoder over quantized per-step tokens).

Representations are captured from the encoder blocks over the context: that
is where the series is read, and it keeps capture cost to a single encoder
pass. The tokenizer maps one context timestep to one token (plus an optional
trailing EOS, stripped in postprocess), and left-truncates contexts longer
than the model limit, which `token_time_spans` accounts for.

Tested against chronos-forecasting>=1.2 with chronos-t5-* checkpoints.
Chronos-Bolt uses a patch-based architecture and needs its own adapter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..utils import log
from .base import ModelAdapter


class ChronosAdapter(ModelAdapter):

    default_layer_regex = r"encoder\.block\.\d+$"

    def load(self) -> None:
        """Load a ChronosPipeline and cache tokenizer limits."""
        from chronos import ChronosPipeline
        self.pipeline = ChronosPipeline.from_pretrained(
            self.cfg.checkpoint or "amazon/chronos-t5-small",
            device_map=str(self.device), torch_dtype=self.dtype)
        self._t5 = self.pipeline.model.model
        tok_cfg = getattr(self.pipeline.tokenizer, "config", None)
        self._model_context = int(getattr(tok_cfg, "context_length", self.data_cfg.context_len))
        self._use_eos = bool(getattr(tok_cfg, "use_eos_token", True))
        if self._model_context < self.data_cfg.context_len:
            log.warning("chronos '%s': model context %d < data context %d; "
                        "representations cover only the most recent %d steps",
                        self.name, self._model_context, self.data_cfg.context_len,
                        self._model_context)

    @property
    def module(self):
        return self._t5

    def _release(self) -> None:
        self.pipeline = None
        self._t5 = None

    def _kept(self) -> int:
        """Number of context steps the tokenizer actually keeps."""
        return min(self.data_cfg.context_len, self._model_context)

    def prepare(self, contexts: np.ndarray) -> Any:
        """Tokenize contexts into (token_ids, attention_mask) on device."""
        tensor = torch.from_numpy(np.ascontiguousarray(contexts)).float()
        token_ids, attention_mask, _state = self.pipeline.tokenizer.context_input_transform(tensor)
        return token_ids.to(self.device), attention_mask.to(self.device)

    def forward(self, prepared: Any) -> None:
        """Run the encoder only; hooks on encoder blocks capture per-token states."""
        token_ids, attention_mask = prepared
        self._t5.encoder(input_ids=token_ids, attention_mask=attention_mask)

    def token_time_spans(self) -> np.ndarray:
        """One token per kept context step, right-aligned to the end of the context."""
        kept = self._kept()
        offset = self.data_cfg.context_len - kept
        starts = offset + np.arange(kept)
        return np.stack([starts, starts + 1], axis=1).astype(np.float64)

    def postprocess_tokens(self, layer_name: str, hidden: torch.Tensor) -> torch.Tensor:
        """Drop the trailing EOS token so the sequence matches declared spans."""
        kept = self._kept()
        if hidden.shape[1] == kept + 1 and self._use_eos:
            return hidden[:, :kept]
        return super().postprocess_tokens(layer_name, hidden[:, :kept]
                                          if hidden.shape[1] > kept else hidden)

    def predict(self, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
        """Sample forecast trajectories and reduce to point (median) and quantiles."""
        num_samples = int(self.cfg.kwargs.get("num_samples", 20))
        tensor = torch.from_numpy(np.ascontiguousarray(contexts)).float()
        samples = self.pipeline.predict(tensor, prediction_length=horizon,
                                        num_samples=num_samples,
                                        limit_prediction_length=False).float()
        q = torch.quantile(samples, torch.tensor(quantiles, dtype=torch.float32), dim=1)
        return {"point": samples.median(dim=1).values.cpu().numpy(),
                "quantiles": q.permute(1, 2, 0).cpu().numpy()}
