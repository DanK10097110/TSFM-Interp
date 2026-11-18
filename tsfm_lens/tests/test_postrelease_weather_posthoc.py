"""Known-answer level and geometry controls for explicitly posthoc diagnostics."""
import numpy as np
import pytest
from experiments.feature_search import postrelease_weather_posthoc as post


def test_level_baselines_preserve_level_and_do_not_refit():
    details=[dict(frequency=8.,coefficients=[3.,2.])]
    t=np.arange(-128,0);past=3*np.sin(2*np.pi*8*t/128)+2*np.cos(2*np.pi*8*t/128)
    t=np.arange(32);future=3*np.sin(2*np.pi*8*t/128)+2*np.cos(2*np.pi*8*t/128)
    residual=np.linspace(5.,15.,128);history=past+residual
    mean,last=post.level_baselines(history,details)
    np.testing.assert_allclose(mean,future+10.,atol=1e-12)
    np.testing.assert_allclose(last,future+15.,atol=1e-12)
    shifted=post.level_baselines(history+100.,details)
    np.testing.assert_allclose(shifted,np.array([mean,last])+100.,atol=1e-12)
    plain=post.level_baselines(residual,[])
    np.testing.assert_allclose(plain,np.array([np.full(32,10.),np.full(32,15.)]),atol=1e-12)


def test_known_orthogonal_gain_and_future_poison():
    t=np.arange(32);basis=np.column_stack([np.sin(2*np.pi*8*t/128),np.cos(2*np.pi*8*t/128)])
    q,_=np.linalg.qr(basis);other=np.ones(32)/np.sqrt(32)
    native=2*q[:,0]+3*other;hybrid=q[:,0]+other;target=np.zeros(32)
    details=[dict(frequency=8.)]
    r=post.location(native,hybrid,target,details,2.)
    assert r['gain']==pytest.approx(11/64) and r['tone_gain']==pytest.approx(3/64) and r['other_gain']==pytest.approx(8/64)
    assert r['tone_update']==pytest.approx(1/64) and r['other_update']==pytest.approx(4/64)
    poisoned=post.location(native,hybrid,np.arange(32.)**2,details,2.)
    for k in ['tone_update','other_update','update_energy']:assert r[k]==poisoned[k]
    assert all(v==0 for v in post.location(native,native,target,[],2.).values())
    with pytest.raises(ValueError,match='abstention'):
        post.location(native,hybrid,target,[],2.)


def test_equal_station_zero_abstention_accounting():
    rows=[dict(station=s) for s in post.STATIONS for _ in range(2 if s=='KJFK' else 4)]
    records=[{k:9. if row['station']=='KJFK' else 0. for k in post.FIELDS} for row in rows]
    got=post.grouped(records,rows)
    for values in got.values():
        np.testing.assert_array_equal(values,[0,0,0,0,0,9,0,0])
        assert values.mean()==1.125
