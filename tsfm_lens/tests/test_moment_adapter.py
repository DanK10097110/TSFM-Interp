"""MOMENT's external window, native-horizon and observed-token contracts without model downloads."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.models import build_adapter
from tsfm_lens.models.moment_adapter import MomentAdapter


class RecordingReconstructor(nn.Module):
    """Native-shaped deterministic oracle whose forecast depends on the entire rolling prefix."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def short_forecast(self, *, x_enc, input_mask, forecast_horizon):
        self.calls.append((x_enc.clone(), input_mask.clone(), forecast_horizon))
        increment = torch.arange(1, forecast_horizon + 1, dtype=x_enc.dtype, device=x_enc.device)
        mean = torch.where(input_mask[:, None].bool(), x_enc, 0.).sum(-1, keepdim=True) / input_mask.sum(-1)[:, None, None]
        return SimpleNamespace(forecast=mean + increment)


def adapter(length=29, horizon=7):
    """A registered adapter using a 32-sample fixture window and four-sample patches."""
    result = build_adapter(ModelConfig(name="moment-oracle", adapter="moment"),
                           DataConfig(context_len=length, horizon=horizon),
                           torch.device("cpu"), torch.float32)
    assert isinstance(result, MomentAdapter)
    result._loaded = True
    result._inner = RecordingReconstructor()
    result._width = 32
    result._patch = 4
    return result


def test_padding_missing_values_and_raw_scale_reach_native_call():
    a = adapter()
    x = np.array([[100., 102., np.nan, 112.], [81., 82., 83., 84.]])
    out = a.predict(x, 3, [.1, .5, .9])
    values, mask, h = a.module.calls[0]
    assert h == 3
    torch.testing.assert_close(values[:, 0, -4:], torch.tensor([[100., 102., torch.nan, 112.], [81., 82., 83., 84.]]), equal_nan=True)
    torch.testing.assert_close(mask[:, -4:], torch.tensor([[1, 1, 0, 1], [1, 1, 1, 1]]))
    assert not mask[:, :-4].any()
    expected = np.nanmean(x, axis=1)[:, None] + [1., 2., 3.]
    np.testing.assert_allclose(out["point"], expected, rtol=2e-7)
    np.testing.assert_allclose(out["quantiles"], np.repeat(expected[:, :, None], 3, axis=2), rtol=2e-7)
    with pytest.raises(ValueError, match="finite"):
        a.prepare(np.full((1, 29), np.nan))


@pytest.mark.parametrize("length,horizon,expected_live,expected_spans", [
    (13, 3, (4, 7), [[1,5], [5,9], [9,13]]),
    (24, 7, (0, 6), [[0,4], [4,8], [8,12], [12,16], [16,20], [20,24]]),
    (35, 5, (0, 6), [[11,15], [15,19], [19,23], [23,27], [27,31], [31,35]]),
    (31, 32, (0, 0), []),
])
def test_history_geometry_matches_real_live_position_and_excludes_future(length, horizon, expected_live, expected_spans):
    a = adapter(length, horizon)
    live = torch.arange(8, dtype=torch.float32)[None, :, None]
    observed, future = a.patch_layout(length, horizon)
    assert (observed.start, observed.stop) == expected_live
    assert future == slice(8 - ((horizon + 3) // 4), 8)
    assert a.token_slice(8) == observed
    np.testing.assert_array_equal(a.token_time_spans(), np.asarray(expected_spans).reshape(-1, 2))
    torch.testing.assert_close(a.postprocess_tokens("block", live), live[:, slice(*expected_live)])
    with pytest.raises(ValueError, match="native patch"):
        a.token_slice(7)


def test_long_rollout_updates_the_observed_history_and_matches_native_call_boundaries():
    a = adapter()
    x = np.arange(40, dtype="f4")[None]
    y = a.predict_differentiable(x, 35)
    assert [call[2] for call in a.module.calls] == [16, 16, 3]
    history = x[0, -32:].astype(float)
    blocks = []
    for n in (16, 16, 3):
        block = history.mean() + np.arange(1, n + 1)
        blocks.extend(block)
        history = np.concatenate((history[n:], block))
    np.testing.assert_allclose(y.numpy()[0], blocks)
    np.testing.assert_array_equal(a.module.calls[0][0].numpy()[0, 0], x[0, -32:])
    assert a.module.calls[-1][1].all()
    with pytest.raises(ValueError, match="positive integer"):
        a.predict(x, 0, [.5])


def test_load_requests_the_pinned_pretrained_reconstruction_head_and_strict_state(monkeypatch):
    import sys

    calls = []

    class Factory:
        @staticmethod
        def from_pretrained(checkpoint, **options):
            calls.append((checkpoint, options))
            model = RecordingReconstructor()
            model.task_name = "reconstruction"
            model.config = SimpleNamespace(seq_len=32, patch_len=4, patch_stride_len=4)
            return model

    monkeypatch.setitem(sys.modules, "momentfm", SimpleNamespace(MOMENTPipeline=Factory))
    cfg = ModelConfig(adapter="moment", checkpoint="test/snapshot", kwargs={"revision": "pinned"})
    a = build_adapter(cfg, DataConfig(context_len=24, horizon=8), torch.device("cpu"), torch.float32)
    a.ensure_loaded()
    assert calls == [("test/snapshot", {"revision": "pinned", "strict": True,
                                        "model_kwargs": {"task_name": "reconstruction"}})]
    assert not a.module.training
    default = build_adapter(ModelConfig(adapter="moment"), DataConfig(context_len=24, horizon=8),
                            torch.device("cpu"), torch.float32)
    default.ensure_loaded()
    assert calls[-1][0] == "AutonLab/MOMENT-1-large"
    assert calls[-1][1]["revision"] == "ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc"


def test_rollout_gradient_follows_effect_of_first_chunk_through_later_normalization():
    """The upstream model detaches RevIN even when its context is a previous generated chunk."""
    class DetachedRevINForecast(nn.Module):
        def __init__(self, dose):
            super().__init__()
            self.normalizer = SimpleNamespace(eps=1e-5)
            self.dose = dose
            self.first = True

        def short_forecast(self, *, x_enc, input_mask, forecast_horizon):
            valid = input_mask[:, None].bool()
            count = valid.sum(-1, keepdim=True)
            mean = (torch.where(valid, x_enc, 0.).sum(-1, keepdim=True) / count).detach()
            sd = (torch.where(valid, (x_enc - mean).square(), 0.).sum(-1, keepdim=True) / count).sqrt().detach()
            sd = sd + self.normalizer.eps
            z = (x_enc - mean) / sd
            step = torch.arange(1, forecast_horizon + 1, dtype=x_enc.dtype)
            result = mean + sd * (.04*step + .15*z[..., -1:])
            if self.first:
                self.first = False
                result = result + self.dose * (.3 + .2*step)
            return SimpleNamespace(forecast=result)

    x = np.cos(np.arange(40) / 5)[None].astype("f4")

    def milestones(dose):
        a = adapter()
        a._inner = DetachedRevINForecast(dose)
        forecast = a.predict_differentiable(x, 35)[0]
        return torch.stack((forecast[:16].mean(), forecast[16:32].mean(), forecast[-1]))

    dose = torch.tensor(0., requires_grad=True)
    values = milestones(dose)
    analytic = torch.stack([torch.autograd.grad(v, dose, retain_graph=True)[0] for v in values])
    step = .001
    observed = (milestones(step) - milestones(-step)) / (2 * step)
    assert observed.abs().min() > .1
    torch.testing.assert_close(analytic, observed, rtol=3e-3, atol=2e-4)
