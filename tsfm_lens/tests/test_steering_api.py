"""Public forecasts retain original units and fail loudly on path mismatch."""

import numpy as np
import pytest
import torch

from tsfm_lens.steering import ForecastSteerer


class LinearHead:
    """Planted affine model with exact, independent output coordinates."""
    quantiles=np.array([.1,.5,.9])
    horizon=2

    def tensor(self,x):
        return torch.as_tensor(x,dtype=torch.float64)

    def forward(self,h):
        return h.reshape(2,3)

    def capture(self,x):
        hidden=np.tile([1.,2.,3.,2.,3.,4.],(len(x),1))
        return dict(hidden=hidden,normalized=hidden.reshape(-1,2,3),
                    scale=np.full((len(x),1),2.),location=np.full((len(x),1),5.))

    def predict_edited(self,x,delta):
        cache=self.capture(x)
        return (cache['hidden']+delta).reshape(-1,2,3)*2+5


def test_public_width_control_preserves_median_in_original_units():
    result=ForecastSteerer(LinearHead()).steer(np.ones((2,32)),task='interval',strength=.2,radius=1)
    assert np.allclose(result.forecast[:,:,1],result.baseline[:,:,1])
    assert np.allclose(result.forecast[:,:,-1]-result.forecast[:,:,0],4.8,atol=1e-4)
    assert all(d['achieved'] for d in result.diagnostics)


def test_public_season_strength_is_original_unit_amplitude():
    x=np.sin(2*np.pi*np.arange(32)/8)[None]
    result=ForecastSteerer(LinearHead()).steer(x,task='seasonal',period=8,strength=.4,radius=1)
    expected=.4*np.sin(2*np.pi*np.arange(32,34)/8)
    assert np.allclose(result.requested_change[0],expected[:,None])
    assert np.allclose(result.forecast-result.baseline,result.requested_change,atol=1e-4)


def test_zero_request_and_mismatched_native_path_are_explicit():
    head=LinearHead()
    result=ForecastSteerer(head).steer(np.ones((1,32)),strength=0)
    assert result.diagnostics[0]['zero_target']
    assert np.array_equal(result.forecast,result.baseline)
    original=head.predict_edited
    head.predict_edited=lambda x,d:original(x,d)+.3
    with pytest.raises(RuntimeError,match='differs from full-model'):
        ForecastSteerer(head).steer(np.ones((1,32)),strength=.2)
