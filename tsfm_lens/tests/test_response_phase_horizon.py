"""Known-answer horizon/frequency fits and direct-head capability failures."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from experiments.feature_search import response_phase_horizon as driver
from experiments.feature_search import response_phase_horizon_analysis as analysis


def fixture(horizon,period,h=None):
    """Exact source-replicated odd response with a declared output frequency."""
    h=np.array([[.7,.2],[-.1,1.1]]) if h is None else h
    y=analysis.PHASE@(analysis.design(period,horizon,'tone_only')@h).T
    return np.repeat(y[None],16,axis=0)


def test_direct_head_gate_rejects_autoregression_and_nonidentical_prefix():
    """No tolerance or matching final shape may hide autoregression/prefix drift."""
    full=np.arange(4*64,dtype=np.float32).reshape(4,64);prefix=full[:,:32].copy()
    driver.check_pair(full,prefix,[1,1],64)
    for calls,head in [([2,1],64),([1,2],64),([1,1],32)]:
        with pytest.raises(ValueError,match='capability gate'):driver.check_pair(full,prefix,calls,head)
    wrong=prefix.copy();wrong[0,0]=np.nextafter(wrong[0,0],np.float32(1))
    with pytest.raises(ValueError,match='exact parity'):driver.check_pair(full,wrong,[1,1],64)


def test_static_head_mismatch_and_absent_slot_fail_before_native():
    """A32-step head or missing grant cannot qualify for64 capture."""
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory);driver.write(p/'config.json',dict(architectures=['ChronosBoltModelForForecasting'],chronos_config={'prediction_length':32}))
        (p/'implementation.py').write_text('',encoding='utf-8')
        with pytest.raises(ValueError,match='direct64'):driver.static_capability(p,p/'implementation.py')
        with patch.object(driver,'DirectBolt',side_effect=AssertionError('Native loaded without grant')):
            with pytest.raises(PermissionError):driver.worker(p,None)


def test_frequency_shift_and_phase_only_at_both_horizons():
    """A genuine24 output is recovered at32/64; a phase rotation of19.2 stays19.2."""
    for horizon in (32,64):
        shifted=analysis.curve(fixture(horizon,24.),19.2,'tone_only')
        assert shifted['selected_period']==24.
        assert shifted['cv_losses'][shifted['selected']].max()<1e-25
        h=np.array([[.4,.8],[-.8,.4]])
        phase=analysis.curve(fixture(horizon,19.2,h),19.2,'tone_only')
        assert phase['selected_period']==19.2
        assert phase['cv_losses'][phase['selected']].max()<1e-25
        np.testing.assert_allclose(phase['coefficients'][phase['selected']],h,atol=1e-13)


def test_64_design_uses_all_steps_and_dc_trend_is_matched():
    """A signal in the second half is not silently truncated; joint nuisance removes bias."""
    for horizon in (32,64):
        d=analysis.design(19.2,horizon,'tone_dc_trend');assert d.shape==(horizon,4)
        y=fixture(horizon,19.2)+analysis.PHASE@(d[:,2:]@np.array([[3.,-2.],[4.,5.]])).T
        fitted=analysis.curve(y,19.2,'tone_dc_trend');assert fitted['selected_period']==19.2
        np.testing.assert_allclose(fitted['coefficients'][fitted['selected']][2:],[[3,-2],[4,5]],atol=1e-12)
        assert fitted['cv_losses'][fitted['selected']].max()<1e-24
    y=np.zeros((16,4,64));y[:,:,32:]=1.
    loss=analysis.loss(y,np.zeros((64,2)),19.2)
    np.testing.assert_allclose(loss,np.full(16,1.),atol=1e-14)


def test_source_cv_and_waveform_capacity_are_preserved():
    """A held receiver cannot alter its own map, and mean waveform retains non-tone structure."""
    y=fixture(64,24.);first=analysis.curve(y,19.2,'tone_only');changed=y.copy();changed[3,0]+=500;changed[3,2]-=500
    second=analysis.curve(changed,19.2,'tone_only')
    np.testing.assert_array_equal(first['cv_waveforms'][:,3],second['cv_waveforms'][:,3])
    assert np.linalg.norm(first['cv_waveforms'][:,0]-second['cv_waveforms'][:,0])>1
    t=np.arange(64);other=np.column_stack([np.sin(2*np.pi*t/7),np.cos(2*np.pi*t/9)])
    y+=analysis.PHASE@other.T;fitted=analysis.curve(y,19.2,'tone_only')
    assert fitted['mean_cv_losses'].max()<1e-25 and fitted['cv_losses'].min()>.1


def test_paired_source_intervals_keep_family_four():
    """Amplitude/horizon cells share sixteen source draws rather than expanding n."""
    types=np.tile(np.arange(4),4);draws=analysis.draws(types);assert draws.shape==(16384,16)
    for kind in range(4):assert np.all(np.sum(types[draws]==kind,axis=1)==4)
    x=np.linspace(-1,2,16);s=analysis.summary(x,draws)
    np.testing.assert_allclose(s['family4_percentile'],np.quantile(x[draws].mean(1),[.00625,.99375]))
    assert s['family4_percentile'][0]<s['percentile95'][0]
