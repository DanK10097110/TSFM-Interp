"""Known-answer fake normalizers exercise the REAL locking helper, without weights."""
from types import SimpleNamespace
import contextlib
import numpy as np
import pytest
import torch
from experiments.feature_search import postrelease_weather_lock as study


def revin(x,mu,sigma,reverse=False):
    """Small affine fake used by the actual TimesFM helper's module-level override."""
    return x*sigma+mu if reverse else (x-mu)/sigma


class TFModel:
    """Its module deliberately owns the fake revin used by coherence's helper."""
    pass


class BoltNorm:
    """Expose Bolt's forward-returned statistics and matching inverse contract."""
    def forward(self,x,loc_scale=None):
        if loc_scale is None:
            mu=x.mean(-1,keepdim=True);sigma=((x-mu)**2).mean(-1,keepdim=True).sqrt()
        else:
            mu,sigma=loc_scale
        return (x-mu)/sigma,(mu,sigma)

    def inverse(self,x,stats):
        return x*stats[1]+stats[0]


class FakeNative:
    """Known nonlinear response z_last squared ensures normalization changes matter."""
    def __init__(self,name):
        self.name=name;self.model=SimpleNamespace(instance_norm=BoltNorm()) if name=='bolt' else TFModel()

    def predict(self,x,replacement=None):
        value=torch.tensor(x,dtype=torch.float32)
        if self.name=='bolt':
            norm,stats=self.model.instance_norm.forward(value)
            y=self.model.instance_norm.inverse(norm[:,-1:].square().repeat(1,32),stats)
        else:
            mu=value.mean(-1,keepdim=True);sigma=((value-mu)**2).mean(-1,keepdim=True).sqrt()
            norm=revin(value,mu,sigma)
            y=revin(norm[:,-1:].square().repeat(1,32),mu,sigma,reverse=True)
        return y.numpy(),np.zeros((1,1,1),np.float32)


def contexts():
    """Exact integer answers: full native12, residual ordinary4, residual locked28."""
    return np.tile([8.,12.],64).astype(np.float32)[None],np.tile([2.,4.],64).astype(np.float32)[None]


def test_real_helper_locks_forward_inverse_and_restores():
    full,residual=contexts()
    for model in ['bolt','timesfm']:
        native=FakeNative(model)
        with study.normalization_mode(native,full,False):
            np.testing.assert_array_equal(study.prediction(native,residual),np.full(32,4.))
        with study.normalization_mode(native,full,True):
            np.testing.assert_array_equal(study.prediction(native,full),np.full(32,12.))
            np.testing.assert_array_equal(study.prediction(native,residual),np.full(32,28.))
        np.testing.assert_array_equal(study.prediction(native,residual),np.full(32,4.))


def fixture():
    """One accepted extraction and one abstention provide distinct gate branches."""
    full,residual=contexts()
    data=dict(full=np.concatenate([full,full]),residual=np.concatenate([residual,full]),
              tones=np.stack([np.full(32,2.),np.zeros(32)]))
    plans=[dict(accepted=True),dict(accepted=False)]
    reference=dict(native=np.full((2,32),12.,np.float32),residual=np.array([np.full(32,4.),np.full(32,12.)],np.float32),
                   hybrid=np.array([np.full(32,6.),np.full(32,12.)]))
    return data,plans,reference


def test_known_mechanism_forecasts_and_call_budget():
    data,plans,reference=fixture()
    for model,calls in [('bolt',9),('timesfm',11)]:
        result,gates=study.measure(FakeNative(model),data,plans,reference)
        np.testing.assert_array_equal(result['locked_hybrid'],np.array([np.full(32,30.),np.full(32,12.)]))
        np.testing.assert_array_equal(result['ordinary_hybrid'],reference['hybrid'])
        assert gates['actual_native_calls']==calls and gates['locked_full_anchor']==2
        assert gates['locked_residual_repeat']==1 and gates['abstention']==1


def test_full_forecast_mismatch_and_ignored_lock_fail(monkeypatch):
    data,plans,reference=fixture();reference['native']=np.nextafter(reference['native'],np.float32(np.inf))
    with pytest.raises(ValueError,match='full forecast replay'):
        study.measure(FakeNative('bolt'),data,plans,reference)
    monkeypatch.setattr(study,'normalization',lambda *args:contextlib.nullcontext())
    full,_=contexts()
    with pytest.raises(ValueError,match='Ignored'):
        with study.normalization_mode(FakeNative('bolt'),full,True):pass


def test_slot_hold_before_any_native_work(tmp_path):
    with pytest.raises(PermissionError,match='HOLD'):
        study.run(tmp_path,'bolt',None)
    with pytest.raises(PermissionError,match='HOLD'):
        study.worker(tmp_path,'timesfm')
