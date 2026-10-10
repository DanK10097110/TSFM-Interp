"""`ModelAdapter.predict_differentiable` on the generic_hf adapter and the sampled-head adapters (ROADMAP.md sec 42.1).

Planted answers, no checkpoint needed:
  * the differentiable forecast equals `predict()`'s point forecast (same forward, autograd on),
    including under `input_normalization: zscore`, where the de-normalization constants must be
    the same ones `predict` uses;
  * a gradient reaches a hook-patched layer through it (the property the attribution rule needs);
  * a head that resolved to sampled `generate()` is refused with a stated reason, as are the
    Sundial and Chronos-T5 adapters, whose sampled heads have no seeded differentiable path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_generic_hf_adapter as g  # noqa: E402

from tsfm_lens.config import DataConfig, ModelConfig  # noqa: E402
from tsfm_lens.extraction.hooks import token_patch  # noqa: E402
from tsfm_lens.models import build_adapter  # noqa: E402
from tsfm_lens.models.base import CapabilityUnavailable  # noqa: E402


def _contexts(n=4, seed=0):
    return np.random.default_rng(seed).normal(size=(n, g._CONTEXT)).astype(np.float32) * 3.0 + 10.0


def test_differentiable_forecast_equals_predict_on_a_forward_field_head():
    a = g._adapter(g._PatchNet)
    a.ensure_loaded()
    x = _contexts()
    diff = a.predict_differentiable(x, 4)
    assert diff.requires_grad
    np.testing.assert_allclose(diff.detach().numpy(), a.predict(x, 4, [0.5])["point"], atol=1e-6)


def test_zscore_denormalization_uses_the_same_constants_as_predict():
    a = g._clamp_adapter("zscore")
    a.ensure_loaded()
    x = g._ramp_contexts() * 50.0 + 7.0
    diff = a.predict_differentiable(x, 4)
    np.testing.assert_allclose(diff.detach().numpy(), a.predict(x, 4, [0.5])["point"], rtol=1e-5, atol=1e-5)
    assert np.abs(diff.detach().numpy()).max() > 5.0


def test_gradient_reaches_a_patched_layer_through_the_differentiable_forecast():
    a = g._adapter(g._PatchNet)
    a.ensure_loaded()
    x = _contexts()
    from tsfm_lens.extraction.extract import capture_raw_tokens
    clean = capture_raw_tokens(a, x, ["layers.1"])["layers.1"].clone().requires_grad_(True)
    with torch.enable_grad():
        with token_patch(a.module, "layers.1", a.token_slice, clean):
            point = a.predict_differentiable(x, 4)
        grad, = torch.autograd.grad(point.sum(), clean)
    assert torch.isfinite(grad).all() and float(grad.abs().sum()) > 0.0


def test_a_sampled_generate_head_is_refused_with_a_reason():
    a = g._adapter(g._GenerateNet)
    a.ensure_loaded()
    with pytest.raises(CapabilityUnavailable, match="generate"):
        a.predict_differentiable(_contexts(), 4)


@pytest.mark.parametrize("adapter,checkpoint", [("sundial", "thuml/sundial-base-128m"),
                                                ("chronos", "amazon/chronos-t5-small")])
def test_sampled_head_adapters_state_why_they_are_not_differentiable(adapter, checkpoint):
    a = build_adapter(ModelConfig(name="m", adapter=adapter, checkpoint=checkpoint, batch_size=4),
                      DataConfig(context_len=64, horizon=4), torch.device("cpu"), torch.float32)
    with pytest.raises(CapabilityUnavailable, match="not differentiable"):
        a.predict_differentiable(np.zeros((1, 64), dtype=np.float32), 4)
