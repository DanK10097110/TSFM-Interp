"""Adapter for Chronos-2 (encoder-only, patch-based, with a second
"group" attention axis for cross-series in-context learning).

Architecturally distinct from both existing Chronos adapters despite the
shared lineage/naming: each encoder block runs **time** self-attention
(within one series, the axis this adapter captures/exposes) followed by
**group** self-attention (across series sharing a `group_id`, along the
*batch* axis -- not a token-token pattern, and out of scope for
`attention_patterns`/`attention_info` here) and a feed-forward block. The
model reads the full context *and* a placeholder for the forecast horizon
in one non-causal encoder pass -- there is no separate decoder step. A
`[REG]` token sits between the two. `token_time_spans` only covers the
leading context patches; `postprocess_tokens` drops the trailing `[REG]` +
forecast-placeholder positions the same way `ChronosBoltAdapter` drops its
own trailing `[REG]`.

Each item's `group_id` defaults to a value unique to that item (see
`Chronos2Model.forward`), so a plain batch of otherwise-unrelated series
(this repo's usual case) gets independent per-series group attention with
no cross-series leakage -- `cross_learning=True`-style batch mixing is
never requested here.

Written against chronos-forecasting>=2.0 with the `amazon/chronos-2`
checkpoint; run --check-alignment on your installed version before
trusting results, per every other adapter in this file.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ..utils import log
from .base import ModelAdapter, _scan_attention, _scan_mlp, random_init_like


class Chronos2Adapter(ModelAdapter):

    default_layer_regex = r"encoder\.block\.\d+$"

    def load(self) -> None:
        """Load a Chronos2Pipeline and read patch/quantile geometry from its config.

        `random_init` (ROADMAP.md sec 16 E9) reconstructs the underlying HF
        model from its own config -- same patch geometry and chronos_config
        metadata, but untrained weights -- identical mechanism to every
        other Chronos adapter in this file.
        """
        from chronos import Chronos2Pipeline
        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.cfg.checkpoint or "amazon/chronos-2",
            device_map=str(self.device), torch_dtype=self.dtype)
        self._inner = self.pipeline.model
        if self.cfg.random_init:
            log.warning("chronos2 '%s': random_init=True -- discarding pretrained "
                        "weights, using an architecture-matched random-weight twin "
                        "(ROADMAP.md sec 16 E9)", self.name)
            self._inner = random_init_like(self._inner).to(self.device, dtype=self.dtype)
            self.pipeline.model = self._inner
        ccfg = self._inner.chronos_config
        self._patch = int(ccfg.input_patch_size)
        self._stride = int(ccfg.input_patch_stride)
        self._model_context = int(ccfg.context_length)
        self._use_reg = bool(ccfg.use_reg_token)
        self._levels = list(ccfg.quantiles)
        log.info("chronos2 '%s': patch=%d stride=%d model_context=%d reg=%s n_levels=%d",
                 self.name, self._patch, self._stride, self._model_context,
                 self._use_reg, len(self._levels))

    @property
    def module(self):
        return self._inner

    def _release(self) -> None:
        self.pipeline = None
        self._inner = None

    def _geometry(self):
        """(kept steps, context-patch count, front pad) -- identical formula to
        ChronosBoltAdapter's, since both use the same `chronos.chronos_bolt.Patch`
        front-padding convention."""
        kept = min(self.data_cfg.context_len, self._model_context)
        if kept <= self._patch:
            n_tokens = 1
        else:
            n_tokens = 1 + math.ceil((kept - self._patch) / self._stride)
        front_pad = (n_tokens - 1) * self._stride + self._patch - kept
        return kept, n_tokens, front_pad

    def _num_output_patches(self, horizon: int) -> int:
        return max(1, math.ceil(horizon / self._patch))

    def prepare(self, contexts: np.ndarray) -> Any:
        return torch.from_numpy(np.ascontiguousarray(contexts)).float().to(self.device)

    def forward(self, prepared: Any) -> None:
        """One non-causal encoder pass over context + [REG] + forecast placeholders."""
        with torch.no_grad():
            self._inner(context=prepared, num_output_patches=self._num_output_patches(self.data_cfg.horizon))

    def token_time_spans(self) -> np.ndarray:
        """Context-patch spans, right-aligned and clipped where front padding covers no data."""
        kept, n_tokens, front_pad = self._geometry()
        offset = self.data_cfg.context_len - kept
        starts = np.arange(n_tokens) * self._stride - front_pad
        spans = np.stack([starts, starts + self._patch], axis=1).astype(np.float64)
        return np.clip(spans, 0, kept) + offset

    def postprocess_tokens(self, layer_name: str, hidden: torch.Tensor) -> torch.Tensor:
        """Keep only the leading context-patch positions -- the trailing
        [REG] token and forecast-horizon placeholders that follow in the
        same encoder pass carry no time-span meaning."""
        _, n_tokens, _ = self._geometry()
        if hidden.shape[1] >= n_tokens:
            return hidden[:, :n_tokens]
        return super().postprocess_tokens(layer_name, hidden)

    def predict(self, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
        """Direct quantile forecast via `Chronos2Model.forward` (already
        unscaled to data units); interpolated onto the requested quantile
        levels with the library's own `interpolate_quantiles` (exact at
        levels the checkpoint was trained on, per its own predict_quantiles)."""
        from chronos.utils import interpolate_quantiles
        tensor = self.prepare(contexts)
        n_out = self._num_output_patches(horizon)
        with torch.no_grad():
            out = self._inner(context=tensor, num_output_patches=n_out)
        preds = out.quantile_preds[..., :horizon].float().cpu()  # [B, n_levels, horizon]
        preds_hq = preds.permute(0, 2, 1)                        # [B, horizon, n_levels]
        q_out = interpolate_quantiles(list(quantiles), self._levels, preds_hq)
        point = interpolate_quantiles([0.5], self._levels, preds_hq)[..., 0]
        return {"point": point.numpy(), "quantiles": q_out.numpy()}

    def attention_info(self) -> list:
        """Best-effort head map via the shared block scan -- resolves to each
        block's TIME self-attention (the first attention submodule found);
        GROUP self-attention (a batch-axis pattern, not token-token) is out
        of scope for this mechanism and is never returned."""
        self.ensure_loaded()
        infos = []
        for block in self.all_layer_names():
            info = _scan_attention(self.module, block)
            if info is None:
                log.info("chronos2 '%s': no head map for block %s; "
                         "head-level analyses disabled", self.name, block)
                return None
            infos.append(info)
        return infos

    def mlp_info(self) -> dict:
        """Best-effort block -> pre-residual MLP map via the shared block scan."""
        self.ensure_loaded()
        mapping = {}
        for block in self.all_layer_names():
            name = _scan_mlp(self.module, block)
            if name is None:
                log.info("chronos2 '%s': no MLP module for block %s; "
                         "MLP ablation disabled", self.name, block)
                return None
            mapping[block] = name
        return mapping

    def attention_patterns(self, prepared: Any) -> dict:
        """TIME self-attention probabilities per block, restricted to the
        context-patch positions (query and key) -- GROUP self-attention is
        excluded, matching `attention_info`'s scope."""
        _, n_tokens, _ = self._geometry()
        n_out = self._num_output_patches(self.data_cfg.horizon)
        with torch.no_grad():
            out = self._inner(context=prepared, num_output_patches=n_out, output_attentions=True)
        pats = {}
        for i, att in enumerate(out.enc_time_self_attn_weights):
            pats[f"encoder.block.{i}"] = att[..., :n_tokens, :n_tokens].float().cpu()
        return pats
