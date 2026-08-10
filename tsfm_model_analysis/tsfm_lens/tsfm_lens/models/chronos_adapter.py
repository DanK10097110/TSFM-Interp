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
from .base import ModelAdapter, random_init_like


class ChronosAdapter(ModelAdapter):

    default_layer_regex = r"encoder\.block\.\d+$"

    def load(self) -> None:
        """Load a ChronosPipeline and cache tokenizer limits.

        The tokenizer itself is never randomized even when `random_init` is
        set -- its quantization bin edges are architecture/checkpoint
        metadata, not learned weights, and the null baseline this flag
        exists for (ROADMAP.md sec 16 E9) is meant to isolate "untrained
        transformer weights", not "a different tokenizer".
        """
        from chronos import ChronosPipeline
        self.pipeline = ChronosPipeline.from_pretrained(
            self.cfg.checkpoint or "amazon/chronos-t5-small",
            device_map=str(self.device), torch_dtype=self.dtype)
        self._t5 = self.pipeline.model.model
        if self.cfg.random_init:
            log.warning("chronos '%s': random_init=True -- discarding pretrained "
                        "weights, using an architecture-matched random-weight twin "
                        "(ROADMAP.md sec 16 E9)", self.name)
            self._t5 = random_init_like(self._t5).to(self.device, dtype=self.dtype)
            self.pipeline.model.model = self._t5
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

    def attention_info(self) -> list:
        """T5 encoder blocks expose SelfAttention.o, a plain Linear over head concat."""
        self.ensure_loaded()
        cfg = self._t5.config
        return [{"block": f"encoder.block.{i}",
                 "o_proj": f"encoder.block.{i}.layer.0.SelfAttention.o",
                 "n_heads": int(cfg.num_heads), "head_dim": int(cfg.d_kv)}
                for i in range(int(cfg.num_layers))]

    def mlp_info(self) -> dict:
        """DenseReluDense outputs the pre-residual feed-forward contribution."""
        self.ensure_loaded()
        return {f"encoder.block.{i}": f"encoder.block.{i}.layer.1.DenseReluDense"
                for i in range(int(self._t5.config.num_layers))}

    def attention_patterns(self, prepared: Any) -> dict:
        """Encoder self-attention probabilities with the EOS row/column stripped."""
        token_ids, attention_mask = prepared
        kept = self._kept()
        with torch.no_grad():
            out = self._t5.encoder(input_ids=token_ids, attention_mask=attention_mask,
                                   output_attentions=True)
        pats = {}
        for i, att in enumerate(out.attentions):
            if att.shape[-1] == kept + 1 and self._use_eos:
                att = att[..., :kept, :kept]
            pats[f"encoder.block.{i}"] = att.float().cpu()
        return pats

    def cross_attention_patterns(self, prepared: Any) -> torch.Tensor:
        """First-step decoder cross-attention over encoder tokens, [L, B, H, T_enc]."""
        token_ids, attention_mask = prepared
        kept = self._kept()
        start = torch.full((token_ids.shape[0], 1),
                           int(self._t5.config.decoder_start_token_id),
                           dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self._t5(input_ids=token_ids, attention_mask=attention_mask,
                           decoder_input_ids=start, output_attentions=True)
        cross = torch.stack([att[..., 0, :kept].float().cpu()
                             for att in out.cross_attentions])
        return cross
