"""MOMENT's published reconstruction head as a zero-shot point forecaster.

The upstream short_forecast path normalizes the original observed context, rolls
it left by whole patches, inserts future mask tokens, and denormalizes the
reconstruction. The interpretation interface exposes only fully observed
context patch positions. The checkpoint also emits hidden states for padding
and the forecast placeholders, which remain available on the live encoder.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import log
from .base import ModelAdapter, random_init_like


class MomentAdapter(ModelAdapter):
    """Expose the same official short-forecast call to capture, prediction and gradients."""

    default_layer_regex = r"encoder\.block\.\d+$"
    published_revisions = {
        "AutonLab/MOMENT-1-small": "411e288267f82cce86296dbe4d6c8bc533cc162f",
        "AutonLab/MOMENT-1-base": "5e44b0ea26376a176360f87831124e018f876d96",
        "AutonLab/MOMENT-1-large": "ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc",
    }

    def load(self) -> None:
        """Strictly load the checkpoint's reconstruction head, preserving float32 master weights."""
        from momentfm import MOMENTPipeline

        checkpoint = self.cfg.checkpoint or "AutonLab/MOMENT-1-large"
        unsupported = set(self.cfg.kwargs) - {"revision"}
        if unsupported:
            raise ValueError(f"Unsupported MOMENT adapter options: {sorted(unsupported)}")
        revision = self.cfg.kwargs.get("revision", self.published_revisions.get(checkpoint))
        inner = MOMENTPipeline.from_pretrained(
            checkpoint, revision=revision, strict=True,
            model_kwargs={"task_name": "reconstruction"},
        )
        if self.cfg.random_init:
            log.warning("MOMENT '%s' uses a freshly initialized architecture-matched model", self.name)
            inner = random_init_like(inner)
        self._inner = inner.to(self.device).eval()
        log.info("MOMENT '%s' has a point reconstruction head; reported quantiles repeat the point", self.name)
        self._width = int(inner.config.seq_len)
        self._patch = int(inner.config.patch_len)
        if inner.task_name != "reconstruction" or int(inner.config.patch_stride_len) != self._patch:
            raise ValueError("MOMENT short forecasts require a reconstruction head and nonoverlapping patches")
        if self._width % self._patch or self._width < 2 * self._patch:
            raise ValueError("MOMENT's fixed context must contain at least two whole patches")

    @property
    def module(self):
        return self._inner

    def _release(self) -> None:
        self._inner = None

    def prepare(self, contexts: np.ndarray):
        """Right-align observed history in the native window and exclude padding from RevIN."""
        self.ensure_loaded()
        x = torch.as_tensor(np.ascontiguousarray(contexts), dtype=torch.float32, device=self.device)
        if x.ndim != 2 or not x.shape[1]:
            raise ValueError("MOMENT expects a nonempty two-dimensional [batch, time] history")
        x = x[:, -self._width:]
        present = torch.isfinite(x)
        if not present.any(dim=1).all():
            raise ValueError("MOMENT requires at least one finite observed value per history")
        pad = self._width - x.shape[1]
        values = torch.where(present, x, torch.nan)
        return F.pad(values, (pad, 0)).unsqueeze(1), F.pad(present.long(), (pad, 0))

    def _first_horizon(self, horizon: int) -> int:
        """Use native lengths throughout its domain and half-window chunks for longer rollouts."""
        if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 1:
            raise ValueError("MOMENT forecast horizon must be a positive integer")
        if horizon <= self._width:
            return int(horizon)
        return (self._width // (2 * self._patch)) * self._patch

    def _short(self, prepared, horizon):
        x, mask = prepared
        return self._inner.short_forecast(x_enc=x, input_mask=mask, forecast_horizon=horizon).forecast[:, 0]

    def _feedback_forecast(self, prepared, horizon):
        """Differentiate RevIN's dependence on earlier predictions in a recursive rollout.

        Native RevIN detaches the observed mean and scale. In later chunks the
        observations include our previous forecasts. Supply the missing chain-rule
        terms at its input and output using zero-valued gradient corrections.
        """
        x, mask = prepared
        if not x.requires_grad:
            return self._short(prepared, horizon)
        present = mask[:, None].bool()
        count = present.sum(dim=-1, keepdim=True)
        finite = torch.where(present, x, 0.)
        mean = finite.sum(dim=-1, keepdim=True) / count
        centered = torch.where(present, finite - mean, 0.)
        # vector_norm supplies the zero subgradient for an exactly constant history.
        scale = torch.linalg.vector_norm(centered, dim=-1, keepdim=True) / count.sqrt()
        scale = scale + self._inner.normalizer.eps
        d_mean = mean - mean.detach()
        d_scale = scale - scale.detach()
        normalized_x = torch.nan_to_num((x.detach() - mean.detach()) / scale.detach())
        result = self._short((x - d_mean - normalized_x * d_scale, mask), horizon)
        normalized_y = (result.detach() - mean[:, 0].detach()) / scale[:, 0].detach()
        return result + d_mean[:, 0] + normalized_y * d_scale[:, 0]

    def forward(self, prepared) -> None:
        """Capture the first actual forecast call, before any recursive extension."""
        self._short(prepared, self._first_horizon(self.data_cfg.horizon))

    def predict_differentiable(self, contexts: np.ndarray, horizon: int):
        """Return native point forecasts in the input's units, preserving the intervention graph."""
        prepared = self.prepare(contexts)
        chunk = self._first_horizon(horizon)
        if chunk == horizon:
            return self._short(prepared, chunk)
        parts = []
        remaining = horizon
        while remaining:
            count = min(chunk, remaining)
            pred = self._feedback_forecast(prepared, count)
            parts.append(pred)
            remaining -= count
            if remaining:
                values, mask = prepared
                prepared = (torch.cat([values[:, :, count:], pred[:, None]], dim=-1),
                            torch.cat([mask[:, count:], torch.ones_like(pred, dtype=mask.dtype)], dim=-1))
        return torch.cat(parts, dim=1)

    def predict(self, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
        """Populate the benchmark's quantile array with point replicas, as for other point heads."""
        with torch.no_grad():
            point = self.predict_differentiable(contexts, horizon).float().cpu().numpy()
        return {"point": point, "quantiles": np.repeat(point[:, :, None], len(quantiles), axis=2)}

    def patch_layout(self, context_len: int, horizon: int) -> tuple[slice, slice]:
        """Live observed and future token positions for a particular first forecast call."""
        self.ensure_loaded()
        if context_len < 1:
            raise ValueError("MOMENT requires a positive context length")
        patch_count = self._width // self._patch
        future = math.ceil(self._first_horizon(horizon) / self._patch)
        stop = patch_count - future
        observed_count = min(stop, context_len // self._patch)
        return slice(stop - observed_count, stop), slice(stop, patch_count)

    def token_time_spans(self) -> np.ndarray:
        """Original half-open input coordinates; partial padded and future tokens are excluded."""
        length = self.data_cfg.context_len
        observed, _ = self.patch_layout(length, self.data_cfg.horizon)
        count = observed.stop - observed.start
        edges = length - self._patch * np.arange(count, -1, -1)
        return np.column_stack((edges[:-1], edges[1:])).astype(np.int64)

    def token_slice(self, live_len: int) -> slice:
        """Locate the reported observed patch tokens between front padding and masked future."""
        observed, _ = self.patch_layout(self.data_cfg.context_len, self.data_cfg.horizon)
        expected = self._width // self._patch
        if live_len != expected:
            raise ValueError(f"MOMENT expected {expected} native patch tokens, saw {live_len}")
        return observed

    def postprocess_tokens(self, layer_name: str, hidden: torch.Tensor) -> torch.Tensor:
        """Extract only the observed slots of this configured forecast run."""
        return hidden[:, self.token_slice(hidden.shape[1])]

    def forecast_reads_patched_positions(self) -> bool:
        """The reconstruction head reads the future slots; earlier blocks can relay context into them."""
        return False
