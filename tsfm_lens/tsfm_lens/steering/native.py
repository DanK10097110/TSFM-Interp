"""Audited native-head interface for Chronos-Bolt steering experiments.

The exact head is replayed in normalized units for cheap Jacobians. Full-model
interventions are separately available to verify that this is the actual causal
path. This interface deliberately restricts context, horizon and normalization
instead of silently approximating autoregressive or nonlinear inverse scaling.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch


class BoltHead:
    """A direct quantile head with immutable pretrained weights."""

    def __init__(self, checkpoint="amazon/chronos-bolt-small", horizon=32, device="cpu"):
        from chronos import BaseChronosPipeline
        self.pipeline = BaseChronosPipeline.from_pretrained(
            checkpoint, device_map=device, torch_dtype=torch.float32)
        self.model = self.pipeline.model.eval()
        self.model.requires_grad_(False)
        self.head = self.model.output_patch_embedding
        self.quantiles = np.asarray(self.pipeline.quantiles)
        self.horizon = horizon
        self.device = torch.device(device)
        self.native_horizon = int(self.model.chronos_config.prediction_length)
        if not 1 <= horizon <= self.native_horizon:
            raise ValueError("steering requires horizon within one native output patch")
        if self.model.instance_norm.use_arcsinh:
            raise ValueError("this normalized-head interface requires affine instance normalization")

    def tensor(self, value):
        """Create a float32 tensor on the model's device."""
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def forward(self, hidden):
        """Map [..., hidden] to normalized [..., horizon, quantiles]."""
        raw = self.head(hidden)
        return raw.reshape(*hidden.shape[:-1], len(self.quantiles), self.native_horizon)[
            ..., :self.horizon].transpose(-1, -2)

    def capture(self, contexts, batch_size=16):
        """Capture true head states and check normalized replay against full inference."""
        x = np.asarray(contexts, dtype=np.float32)
        if (x.ndim != 2 or not np.isfinite(x).all()
                or x.shape[1] > self.pipeline.model_context_length or batch_size < 1):
            raise ValueError("need finite contexts within native context length and positive batch size")
        states, forecasts, locations, scales = [], [], [], []
        max_error = 0.
        for start in range(0, len(x), batch_size):
            batch = self.tensor(x[start:start+batch_size])
            hits = []
            handle = self.head.register_forward_pre_hook(
                lambda module, args: hits.append(args[0].detach().clone()))
            try:
                with torch.no_grad():
                    full = self.model(context=batch).quantile_preds[:, :, :self.horizon].transpose(1, 2)
            finally:
                handle.remove()
            if len(hits) != 1 or hits[0].ndim != 3 or hits[0].shape[:2] != (len(batch), 1):
                raise RuntimeError("expected one [batch,1,hidden] native head input")
            with torch.no_grad():
                loc, scale = self.model.instance_norm(batch)[1]
                norm = self.forward(hits[0][:, 0])
                replay = norm*scale[:, None]+loc[:, None]
            err = float((replay-full).abs().max())
            max_error = max(max_error, err)
            if not torch.allclose(replay, full, atol=2e-6, rtol=2e-6):
                raise RuntimeError(f"native head replay mismatch: {err}")
            states.append(hits[0][:, 0].cpu().numpy())
            forecasts.append(norm.cpu().numpy())
            locations.append(loc.cpu().numpy())
            scales.append(scale.cpu().numpy())
        return dict(hidden=np.concatenate(states), normalized=np.concatenate(forecasts),
                    location=np.concatenate(locations), scale=np.concatenate(scales),
                    replay_max_abs=max_error)

    def jacobian(self, hidden):
        """Differentiate the complete quantile/time tensor at one original state."""
        h = self.tensor(hidden).detach().requires_grad_(True)
        return torch.autograd.functional.jacobian(
            lambda value: self.forward(value).flatten(), h, vectorize=True).detach().cpu().numpy()

    @contextmanager
    def _addition(self, delta):
        """Install an additive prehook and always remove it, including on failure."""
        change = self.tensor(delta)
        hits = []
        def hook(module, args):
            value = args[0]
            if value.ndim != 3 or value.shape[1] != 1 or value[:, 0].shape != change.shape:
                raise ValueError("delta must match [batch,hidden] exactly")
            hits.append(1)
            return (value + change[:, None],)
        handle = self.head.register_forward_pre_hook(hook)
        try:
            yield hits
        finally:
            handle.remove()

    def predict_edited(self, contexts, delta):
        """Run the full frozen model with a residual-preserving internal addition."""
        with self._addition(delta) as hits, torch.no_grad():
            result = self.model(context=self.tensor(contexts)).quantile_preds[
                :, :, :self.horizon].transpose(1, 2).cpu().numpy()
        if len(hits) != 1:
            raise RuntimeError("expected exactly one intervention")
        return result
