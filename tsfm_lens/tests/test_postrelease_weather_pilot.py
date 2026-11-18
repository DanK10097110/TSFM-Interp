"""Known-answer weather forecasts, grouping decoys and instrumentation failures."""
import numpy as np
import pytest
from experiments.feature_search import postrelease_weather_pilot as pilot
from experiments.feature_search import postrelease_weather_worker as worker
from experiments.feature_search import postrelease_weather_score as score
from experiments.feature_search.persistence_protocol import write, sha


def test_fixed_tone_and_seasonal():
    t=np.arange(-128,0);x=10+3*np.sin(2*np.pi*t/24)
    residual,tone,season,info=pilot.plan_history(x)
    assert info['accepted']
    np.testing.assert_allclose(residual,10,atol=1e-6)
    np.testing.assert_allclose(tone,3*np.sin(2*np.pi*np.arange(32)/24),atol=1e-6)
    np.testing.assert_array_equal(season,x[-24:][np.arange(32)%24])
    r,z,_,a=pilot.plan_history(np.full(128,7.))
    assert not a['accepted'];np.testing.assert_array_equal(r,np.full(128,7.));assert np.all(z==0)


def test_reconstruction_gate(monkeypatch):
    def broken(x,depth):
        return [[x],[np.ones(128),x]],[]
    monkeypatch.setattr(pilot,'decompose',broken)
    with pytest.raises(AssertionError):
        pilot.plan_history(np.arange(128.))


class FakeNative:
    """Exact batch1 last-value model; optional nonfinite or repeat drift faults."""
    def __init__(self,fault=None):
        self.calls=0;self.fault=fault

    def predict(self,x):
        assert x.shape==(1,128)
        self.calls+=1
        y=np.repeat(x[:,-1:],32,axis=1).astype(float)
        if self.fault=='repeat' and self.calls>2:
            y=np.nextafter(y,np.inf)
        if self.fault=='finite':
            y[:]=np.nan
        return y,np.zeros((1,1,1))


def fixture_bank():
    return dict(bank=np.array([np.full(128,7.),np.full(128,2.)]),
                tones=np.array([np.zeros(32),np.full(32,3.)]),
                seasonal24=np.full((2,32),5.),sentinel_tone=np.zeros(32)),[
                dict(native=0,residual=0,accepted=False),dict(native=0,residual=1,accepted=True)]


def test_known_forecasts_and_gates():
    data,plans=fixture_bank();native=FakeNative()
    _,forecasts,gates=worker.predict_bank(native,data,plans,0)
    assert native.calls==4 and gates['singleton_repeat']==2 and gates['abstentions']==1
    assert np.all(forecasts['native']==7) and np.all(forecasts['hybrid'][0]==7)
    assert np.all(forecasts['hybrid'][1]==5) and np.all(forecasts['tones_zero'][1]==3)
    assert np.all(forecasts['seasonal24']==5)
    for fault in ['repeat','finite']:
        with pytest.raises(ValueError):
            worker.predict_bank(FakeNative(fault),data,plans,0)
    data['tones'][0,0]=1
    with pytest.raises(ValueError,match='abstention'):
        worker.predict_bank(FakeNative(),data,plans,0)


def test_station_equal_weight_normalize_first():
    rows=[dict(station=s) for s in pilot.STATIONS for _ in range(2 if s=='KJFK' else 4)]
    scale=np.arange(1,31,dtype=float)
    error=np.array([9. if r['station']=='KJFK' else 1. for r in rows])
    prediction=np.sqrt(error*scale)[:,None]*np.ones((30,32))
    window,station=score.station_loss(prediction,np.zeros_like(prediction),scale,rows)
    np.testing.assert_allclose(window,error,rtol=1e-14)
    np.testing.assert_allclose(station,[1,1,1,1,1,9,1,1],rtol=1e-14)
    assert station.mean()==pytest.approx(2.) and window.mean()!=pytest.approx(2.)
    with pytest.raises(ValueError):
        score.station_loss(prediction,np.zeros_like(prediction),np.zeros(30),rows)
    gain=np.arange(8,dtype=float)
    got=score.bootstrap(gain,np.tile(np.arange(8),(4,1)))
    assert got['mean_gain']==3.5 and got['percentile95']==[3.5,3.5]


def test_hold_and_protocol_binding(tmp_path):
    with pytest.raises(PermissionError,match='HOLD'):
        worker.run(tmp_path,'bolt',None)
    with pytest.raises(PermissionError,match='HOLD'):
        worker.require_grant(tmp_path,'bolt')
    write(tmp_path/'protocol.json',{})
    write(tmp_path/'bolt_slot_grant.json',dict(coordinator_grant='fixture',protocol_sha256='stale'))
    with pytest.raises(PermissionError,match='Stale'):
        worker.require_grant(tmp_path,'bolt')
    (tmp_path/'inference').mkdir();write(tmp_path/'inference/plans.json',[])
    write(tmp_path/'inference/protocol.json',dict(sources={},hashes={'plans.json':sha(tmp_path/'inference/plans.json')}))
    write(tmp_path/'protocol.json',dict(inference_sha256=sha(tmp_path/'inference/protocol.json')))
    write(tmp_path/'seal.json',dict(sha256=sha(tmp_path/'protocol.json')))
    pilot.binding(tmp_path)
    write(tmp_path/'inference/plans.json',['changed'])
    with pytest.raises(ValueError,match='input changed'):
        pilot.binding(tmp_path)
