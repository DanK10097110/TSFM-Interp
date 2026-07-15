"""Mock adapters: two deliberately different toy architectures.

They make the full pipeline runnable end-to-end with no downloads, no GPU,
and in seconds, which is how the analysis code, alignment, patching, and
report are validated. One tokenizes in patches (TimesFM-like), the other per
timestep (Chronos-like), so the cross-architecture alignment path is
genuinely exercised. Forecasts flow through the transformer blocks, so
activation patching has a real causal pathway to the output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .base import ModelAdapter


class _MockNet(nn.Module):
    """Tiny patch-embedding MLP stack with a mean-pooled forecast head."""

    def __init__(self, patch: int, dim: int, n_layers: int, horizon: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.patch = patch
        self.embed = nn.Linear(patch, dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, dim), nn.Tanh()) for _ in range(n_layers))
        self.head = nn.Linear(dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map [B, T] series to [B, horizon] forecasts through all blocks."""
        b, t = x.shape
        h = self.embed(x.view(b, t // self.patch, self.patch))
        for block in self.blocks:
            h = block(h)
        return self.head(h.mean(dim=1))


class _MockAdapterBase(ModelAdapter):

    default_layer_regex = r"blocks\.\d+$"
    patch = 32
    dim = 64
    n_layers = 6
    seed = 7

    def load(self) -> None:
        self._net = _MockNet(self.patch, self.dim, self.n_layers,
                             self.data_cfg.horizon, self.seed).to(self.device)

    @property
    def module(self) -> nn.Module:
        return self._net

    def _release(self) -> None:
        self._net = None

    def prepare(self, contexts: np.ndarray) -> Any:
        return torch.from_numpy(np.ascontiguousarray(contexts)).float().to(self.device)

    def forward(self, prepared: Any) -> None:
        self._net(prepared)

    def token_time_spans(self) -> np.ndarray:
        n_tokens = self.data_cfg.context_len // self.patch
        starts = np.arange(n_tokens) * self.patch
        return np.stack([starts, starts + self.patch], axis=1).astype(np.float64)

    def predict(self, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
        """Deterministic point forecast through the net, quantiles as scaled offsets."""
        with torch.no_grad():
            point = self._net(self.prepare(contexts)).cpu().numpy()[:, :horizon]
        scale = contexts.std(axis=1, keepdims=True) + 1e-6
        offsets = np.array([_normal_ppf(q) for q in quantiles], dtype=np.float32)
        q = point[:, :, None] + offsets[None, None, :] * scale[:, :, None]
        return {"point": point, "quantiles": q}


def _normal_ppf(q: float) -> float:
    """Standard normal inverse CDF via scipy, used for mock quantile spread."""
    from scipy.stats import norm
    return float(norm.ppf(q))


class MockPatchAdapter(_MockAdapterBase):
    """Patch-tokenizing mock (TimesFM-shaped): 32-step patches, 6 blocks, d=64."""
    patch, dim, n_layers, seed = 32, 64, 6, 7


class MockStepAdapter(_MockAdapterBase):
    """Per-step-tokenizing mock (Chronos-shaped): 1-step tokens, 4 blocks, d=48."""
    patch, dim, n_layers, seed = 1, 48, 4, 23
