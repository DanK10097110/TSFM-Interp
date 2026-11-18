"""Known-answer frequency shift, phase-only, nuisance, capacity and source-holdout tests."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from experiments.feature_search import response_phase_frequency as study


def cohort(period,h=None,n=16):
    """A repeated exact conditional response, with both input axes observable."""
    h=np.array([[.7,.3],[-.2,1.1]]) if h is None else h
    waveform=study.design(period,'tone_only')@h
    return np.repeat((study.PHASE@waveform.T)[None],n,axis=0)


def test_known_frequency_shift_selects_output24_not_input19point2():
    """A shifted-frequency response is recovered exactly, even beside phase-even error."""
    y=cohort(24.);y[:,[0,2]]+=.2
    result=study.fit_curve(y,19.2,'tone_only')
    assert result['selected_period']==24.
    same=study.PERIODS.index(19.2);best=result['selected_index']
    assert result['cv_losses'][same].mean()-result['cv_losses'][best].mean()>.1
    np.testing.assert_allclose(result['coefficients'][best],[[.7,.3],[-.2,1.1]],atol=1e-13)
    assert result['cv_losses'][best].mean()>0


def test_pure_phase_rotation_keeps_input_frequency_and_rank_one_visible():
    """A strong phase lag alone needs no frequency shift; rank-one collapse remains explicit."""
    theta=-1.1;h=.6*np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    result=study.fit_curve(cohort(19.2,h),19.2,'tone_only')
    assert result['selected_period']==19.2
    assert result['cv_losses'][result['selected_index']].max()<1e-26
    fitted,_=study.fit_map(cohort(24.,np.diag([2.,0.])),24.,'tone_only')
    assert np.linalg.matrix_rank(fitted)==1
    np.testing.assert_allclose(np.linalg.svd(fitted,compute_uv=False),[2,0],atol=1e-13)


def test_joint_dc_trend_prevents_false_frequency_shift():
    """An odd offset/trend can bias plain frequency fits, but the matched joint model separates it."""
    nuisance=study.design(19.2,'tone_dc_trend')[:,2:]@np.array([[8.,-4.],[5.,3.]])
    y=cohort(19.2)+study.PHASE@nuisance.T
    plain=study.fit_curve(y,19.2,'tone_only')
    matched=study.fit_curve(y,19.2,'tone_dc_trend')
    assert plain['selected_period']!=19.2
    assert matched['selected_period']==19.2
    assert matched['cv_losses'][matched['selected_index']].max()<1e-25
    np.testing.assert_allclose(matched['coefficients'][matched['selected_index']][2:],[[8,-4],[5,3]],atol=1e-12)


def test_input_energy_denominator_is_shared_across_output_candidates():
    """Candidate frequency changes cannot improve the score by changing its denominator."""
    y=cohort(24.);wave=study.design(32.,'tone_only')@np.diag([.1,1.3])
    pred=study.PHASE@wave.T;ideal=study.PHASE@study.design(19.2,'tone_only').T
    expected=np.sum((pred-y[0])**2)/np.sum(ideal**2)
    np.testing.assert_allclose(study.loss_per_source(y,wave,19.2),np.full(16,expected),atol=1e-14)
    phase_ratios=np.mean(np.sum((pred-y[0])**2,axis=1)/np.sum(ideal**2,axis=1))
    assert abs(expected-phase_ratios)>1e-3


def test_source_loso_excludes_all_receiver_phases():
    """Changing all receiver phases cannot change the predictor used for that receiver."""
    y=cohort(24.);y+=np.arange(16)[:,None,None]*.01
    before=study.cv_predictions(y,24.,'tone_only');changed=y.copy()
    changed[3,0]+=1000;changed[3,2]-=1000
    after=study.cv_predictions(changed,24.,'tone_only')
    np.testing.assert_array_equal(before[3],after[3])
    assert np.linalg.norm(after[0]-before[0])>1


def test_mean_waveform_capacity_is_explicit_and_preserves_non_tone_shape():
    """An additional out-of-grid oscillation defeats every single-tone candidate."""
    t=np.arange(32);extra=np.column_stack([np.sin(2*np.pi*t/7),.4*np.cos(2*np.pi*t/9)])
    y=cohort(24.)+study.PHASE@extra.T
    result=study.fit_curve(y,19.2,'tone_only')
    assert result['mean_waveform_cv_losses'].max()<1e-26
    assert result['cv_losses'].min()>.01
    np.testing.assert_allclose(result['mean_waveform'],study.design(24.,'tone_only')@np.array([[.7,.3],[-.2,1.1]])+extra,atol=1e-13)


def test_amplitude_normalization_and_source_bootstrap():
    """Unequal physical stimulus amplitudes do not change conditional response weights."""
    y=cohort(24.);a=np.geomspace(.01,2.,16);base=np.arange(16*32,dtype=float).reshape(16,32)/100
    observed=base[:,None,:]+a[:,None,None]*y
    np.testing.assert_allclose(study.normalize_delta(observed,base,a),y,atol=1e-13)
    types=np.tile(np.arange(4),4);draws=study.bootstrap_sources(types,100)
    assert draws.shape==(100,16)
    for kind in range(4): assert np.all(np.sum(types[draws]==kind,axis=1)==4)


def test_test_evaluation_rejects_changed_fit_before_loading_test():
    """A changed predictor file cannot trigger TEST analysis or silent reselection."""
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory);study.write(out/'protocol.json',{})
        study.write(out/'fit_freeze.json',dict(protocol_sha256=study.sha(out/'protocol.json'),files={'fit_selection.json':'wrong'}))
        study.write(out/'fit_selection.json',[dict(model='bolt',input_period=19.2,variant='tone_only')])
        np.savez(out/'fit_arrays.npz')
        with patch.object(study,'validate',return_value={}),patch.object(study,'load_cohort',side_effect=AssertionError('TEST read before freeze gate')):
            with pytest.raises(ValueError,match='Frozen FIT changed'):study.evaluate(out)
