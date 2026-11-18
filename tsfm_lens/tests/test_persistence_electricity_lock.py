"""Known-answer reuse and explicit distinction between singleton and old batch4 gates."""
import numpy as np
import pytest
from test_postrelease_weather_lock import FakeNative,fixture,contexts
from experiments.feature_search import persistence_electricity_lock as study
from experiments.feature_search.persistence_electricity_lock_analysis import drift_bound


def test_fresh_singletons_and_old_batch_diagnostic():
    data,plans,old=fixture();old['native']=old['native']+.125
    for name,calls in [('bolt',9),('timesfm',10)]:
        forecasts,gates,differences=study.measure(FakeNative(name),data,plans,old,[0])
        assert gates['actual_native_calls']==calls and gates['ordinary_full_repeat']==2
        assert gates['locked_full_anchor']==2 and gates['locked_residual_repeat']==1 and gates['abstention']==1
        np.testing.assert_array_equal(forecasts['locked_hybrid'],[np.full(32,30.),np.full(32,12.)])
        assert differences['native']['bit_identical'] is False
        assert differences['native']['max_absolute']==.125
        assert differences['native']['max_relative_linf']==.125/12.125
        assert differences['ordinary_hybrid']['bit_identical'] is True


def test_exact_implicit_repeat_and_slot_hold(tmp_path):
    full,_=contexts();counter=study.Counted(FakeNative('timesfm'))
    counter.trace_expected=np.nextafter(np.full(32,12.,np.float32),np.float32(np.inf))
    with pytest.raises(ValueError,match='implicit TF'):
        counter.predict(full)
    with pytest.raises(PermissionError,match='HOLD'):
        study.run(tmp_path,'bolt',None)
    with pytest.raises(PermissionError,match='HOLD'):
        study.worker(tmp_path,'timesfm')


def test_old_loss_drift_bound_known_answer():
    old=np.full((2,32),2.);new=np.array([np.full(32,3.),np.full(32,1.)]);target=np.zeros((2,32))
    result=drift_bound(new,old,target)
    assert result['old_loss']==[4.,4.] and result['new_loss']==[9.,1.]
    assert result['absolute_drift']==[5.,3.] and result['per_source_upper_bound']==[5.,5.]
    tiny=study.batch_difference(old+1e-10,old)
    assert not tiny['bit_identical'] and tiny['different_elements']==64
