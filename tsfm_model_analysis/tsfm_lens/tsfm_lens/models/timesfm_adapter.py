"""Adapter for Google TimesFM 2.5 (decoder-only transformer over input patches).

Capture calls the library's own `model.decode` path (the eager method behind
both the high-level `forecast()` API and, for horizons within one output
patch, the entire computation TimesFM does in production) so hooks see
exactly the tensors the model uses. `torch_compile` is disabled at load time
because torch.compile can fuse submodule calls in ways that silently drop
Python-level forward hooks; every other adapter in this repo keeps the same
eager-only discipline for the same reason.

Tested against the `timesfm` PyPI package's 2.5-only API (>=2.0, which
dropped the old `TimesFmHparams`/`TimesFmCheckpoint`/`TimesFm` class
entirely) with `google/timesfm-2.5-200m-pytorch`. That package version has no
compatible release for Python >=3.12 that still exposes the old API, so this
adapter targets the new one; an older `timesfm` install with the Hparams API
would need the class-based loader instead.

Attention runs through a fused SDPA kernel by default with no exposed
intermediate weights; `attention_patterns` temporarily swaps in the
library's own unfused dot-product math (a drop-in replacement operating on
the same already-processed query/key/value, restored right after) to
recover them without changing what the model computes. The feed-forward
block is two bare `nn.Linear`s (`ff0`/`ff1`) with no wrapping MLP submodule,
so `mlp_info` finds nothing; head-level ablation still works because the
attention output projection (`attn.out`) is a plain hookable Linear.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from ..utils import log
from .base import ModelAdapter, _scan_attention, _scan_mlp


class TimesFMAdapter(ModelAdapter):

    default_layer_regex = r"stacked_xf\.\d+$"
    _patch_len = 32

    def load(self) -> None:
        """Instantiate TimesFM 2.5, eager (no compile) so capture hooks fire."""
        from timesfm import TimesFM_2p5_200M_torch
        if self.data_cfg.context_len % self._patch_len:
            raise ValueError("data.context_len must be a multiple of TimesFM's input patch (32)")
        repo = self.cfg.checkpoint or "google/timesfm-2.5-200m-pytorch"
        self.tfm = TimesFM_2p5_200M_torch.from_pretrained(repo, torch_compile=False)
        self.tfm.model.device = self.device
        self.tfm.model.to(self.device)
        self.tfm.model.eval()
        # Native quantile head: channel 0 is an unlabeled extra/mean channel,
        # channels 1..9 are these 9 quantiles in order (verified against the
        # checkpoint config: `aridx`/decode_index == 5 == channel for 0.5).
        self._native_quantiles = list(self.tfm.model.config.quantiles)
        self._point_idx = int(self.tfm.model.aridx)

    @property
    def module(self) -> nn.Module:
        return self.tfm.model

    def _release(self) -> None:
        self.tfm = None

    def prepare(self, contexts: np.ndarray) -> Any:
        """Fixed-length contexts need no padding: an all-False mask throughout."""
        inputs = torch.from_numpy(np.ascontiguousarray(contexts, dtype=np.float32)).to(self.device)
        masks = torch.zeros_like(inputs, dtype=torch.bool)
        return inputs, masks

    def forward(self, prepared: Any) -> None:
        """One real decode pass (the library's own production path) so hooks fire."""
        inputs, masks = prepared
        with torch.no_grad():
            self.tfm.model.decode(self.data_cfg.horizon, inputs, masks)

    def token_time_spans(self) -> np.ndarray:
        """One token per 32-step input patch, contiguous and non-overlapping."""
        n_tokens = self.data_cfg.context_len // self._patch_len
        starts = np.arange(n_tokens) * self._patch_len
        return np.stack([starts, starts + self._patch_len], axis=1).astype(np.float64)

    def predict(self, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
        """Decode and map requested quantiles onto the checkpoint's native quantile head."""
        inputs, masks = self.prepare(contexts)
        with torch.no_grad():
            renormed_outputs, _, ar_outputs = self.tfm.model.decode(horizon, inputs, masks)
        full = renormed_outputs[:, -1, ...]
        if ar_outputs is not None:
            extra = ar_outputs.reshape(inputs.shape[0], -1, self.tfm.model.q)
            full = torch.cat([full, extra], dim=1)
        full = full[:, :horizon, :].float().cpu().numpy()
        point = full[:, :, self._point_idx]
        idx = [1 + int(np.argmin(np.abs(np.array(self._native_quantiles) - q))) for q in quantiles]
        matched = [self._native_quantiles[i - 1] for i in idx]
        if not np.allclose(matched, quantiles):
            log.warning("timesfm '%s': requested quantiles %s mapped to native %s",
                        self.name, quantiles, matched)
        return {"point": point, "quantiles": full[:, :, idx]}

    def attention_patterns(self, prepared: Any) -> dict:
        """Per-head attention weights, captured by swapping the fused SDPA kernel.

        `MultiHeadAttention` takes its dot-product implementation as a
        swappable `attention_fn(query, key, value, mask) -> output`
        attribute; the default is a fused kernel with no exposed
        intermediate weights, but the query/key/value it receives are
        already fully processed (post rotary embedding, post qk-norm), so a
        drop-in replacement that runs the identical (unscaled) dot-product
        and softmax math — the library's own `_dot_product_attention`,
        inlined here only to also stash the intermediate weights — computes
        exactly the same output while exposing them. Swapped back
        immediately after the one forward call this needs.
        """
        inputs, masks = prepared
        blocks = self.all_layer_names()
        attn_modules = {b: self.find_module(f"{b}.attn") for b in blocks}
        originals = {b: m.attention_fn for b, m in attn_modules.items()}
        captured: dict = {}

        def make_fn(block_name: str):
            def fn(query, key, value, mask=None):
                weights = torch.einsum("...qhd,...khd->...hqk", query, key)
                if mask is not None:
                    weights = torch.where(mask, weights, -torch.finfo(weights.dtype).max / 2)
                weights = torch.softmax(weights, dim=-1)
                captured[block_name] = weights.detach()
                return torch.einsum("...hqk,...khd->...qhd", weights, value)
            return fn

        for b, m in attn_modules.items():
            m.attention_fn = make_fn(b)
        try:
            with torch.no_grad():
                self.tfm.model.decode(self.data_cfg.horizon, inputs, masks)
        finally:
            for b, m in attn_modules.items():
                m.attention_fn = originals[b]
        return captured

    def attention_info(self) -> list:
        """Best-effort head map resolved by scanning each block's submodules."""
        self.ensure_loaded()
        blocks = self.all_layer_names()
        infos = []
        for block in blocks:
            info = _scan_attention(self.module, block)
            if info is None:
                log.info("timesfm '%s': no head map for block %s; "
                         "head-level analyses disabled", self.name, block)
                return None
            infos.append(info)
        return infos

    def mlp_info(self) -> dict:
        """Best-effort block -> pre-residual MLP map via submodule scan."""
        self.ensure_loaded()
        mapping = {}
        for block in self.all_layer_names():
            name = _scan_mlp(self.module, block)
            if name is None:
                log.info("timesfm '%s': no MLP module for block %s; "
                         "MLP ablation disabled", self.name, block)
                return None
            mapping[block] = name
        return mapping
