"""Mock adapters: two deliberately different toy architectures.

They make the full pipeline runnable end-to-end with no downloads, no GPU,
and in seconds, which is how the analysis code, alignment, patching, lens,
attention analyses, and report are validated. One tokenizes in patches
(TimesFM-like), the other per timestep (Chronos-like), so the
cross-architecture alignment path is genuinely exercised. Blocks are real
pre-norm-free residual attention+MLP blocks with hookable projections, so
forecast-lens skip patching, per-head ablation, and attention-pattern
capture all have honest causal pathways to the output.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from .base import ModelAdapter


class _MockSelfAttention(nn.Module):
    """Minimal multi-head self-attention with a hookable output projection.

    Hand-rolled instead of nn.MultiheadAttention because torch's fused
    implementation applies the output projection functionally, so hooks on
    `out_proj` never fire; here `o_proj` is a real Linear whose input is the
    head concatenation, exactly the surface head ablation needs.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o_proj = nn.Linear(dim, dim)
        self.last_pattern: Optional[torch.Tensor] = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Standard scaled dot-product attention, stashing the last pattern."""
        b, t, d = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (b, t, self.n_heads, self.head_dim)
        q, k, v = (x.view(shape).transpose(1, 2) for x in (q, k, v))
        attn = torch.softmax(q @ k.transpose(-1, -2) / self.head_dim ** 0.5, dim=-1)
        self.last_pattern = attn.detach()
        z = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return self.o_proj(z)


class _MockBlock(nn.Module):
    """One residual attention + MLP block; its output is the residual stream."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = _MockSelfAttention(dim, n_heads)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(h)
        return h + self.mlp(h)


class _MockNet(nn.Module):
    """Tiny patch-embedding transformer stack with a mean-pooled forecast head."""

    def __init__(self, patch: int, dim: int, n_layers: int, n_heads: int,
                 horizon: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.patch = patch
        self.embed = nn.Linear(patch, dim)
        self.blocks = nn.ModuleList(_MockBlock(dim, n_heads) for _ in range(n_layers))
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
    n_heads = 2
    seed = 7

    # Mocks have no real "pretrained" state -- `seed` alone determines their
    # fixed weights. `random_init` (ROADMAP.md sec 16 E9) is emulated as a
    # different, offset seed: same architecture, deliberately different
    # (still fixed, still reproducible) weights, exactly analogous to a real
    # adapter's untrained-vs-pretrained twin, so config wiring / pipeline
    # tests can exercise the flag with no GPU or download involved.
    _RANDOM_INIT_SEED_OFFSET = 999_983

    def load(self) -> None:
        seed = self.seed + self._RANDOM_INIT_SEED_OFFSET if self.cfg.random_init else self.seed
        self._net = _MockNet(self.patch, self.dim, self.n_layers, self.n_heads,
                             self.data_cfg.horizon, seed).to(self.device)

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

    def attention_info(self) -> list:
        """Every block's o_proj is a plain Linear, so head slicing is exact."""
        self.ensure_loaded()
        return [{"block": f"blocks.{i}", "o_proj": f"blocks.{i}.attn.o_proj",
                 "n_heads": self.n_heads, "head_dim": self.dim // self.n_heads}
                for i in range(self.n_layers)]

    def mlp_info(self) -> dict:
        """The mlp submodule outputs the pre-residual MLP contribution."""
        self.ensure_loaded()
        return {f"blocks.{i}": f"blocks.{i}.mlp" for i in range(self.n_layers)}

    def attention_patterns(self, prepared: Any) -> dict:
        """One forward, then read each block's stashed [B, H, T, T] pattern."""
        with torch.no_grad():
            self._net(prepared)
        return {f"blocks.{i}": block.attn.last_pattern
                for i, block in enumerate(self._net.blocks)}

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
    patch, dim, n_layers, n_heads, seed = 32, 64, 6, 2, 7


class MockStepAdapter(_MockAdapterBase):
    """Per-step-tokenizing mock (Chronos-shaped): 1-step tokens, 4 blocks, d=48."""
    patch, dim, n_layers, n_heads, seed = 1, 48, 4, 3, 23


class MockWaveAdapter(_MockAdapterBase):
    """Third mock shape (8-step patches, 5 blocks, d=40): exercises the pipeline
    with >2 configured models without adding a real third dependency."""
    patch, dim, n_layers, n_heads, seed = 8, 40, 5, 4, 41
