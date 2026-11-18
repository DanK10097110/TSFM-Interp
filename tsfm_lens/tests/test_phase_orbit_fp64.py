"""Load-bearing precision regression and complete-family retention tests."""
import numpy as np
import pytest
from experiments.feature_search import phase_orbit as orbit
from experiments.feature_search import phase_orbit_fp64 as replay


def test_fp32_saved_forecasts_are_explicitly_upcast_before_phase_fft():
    _,future,rows=orbit.make_orbit((32,16),bundles=(0,))
    truth=future.reshape(8,8,64)
    prediction=(1.234*truth+.2*np.random.default_rng(5).normal(size=truth.shape)).astype(np.float32)
    row=rows[0];origins=(row['origin_phi'],row['origin_psi'])
    corrected=replay.corrected_budget(prediction,truth,origins,row['scale'],32)
    assert corrected['error_Parseval_absolute_error']<1e-12
    assert corrected['primary_D']<=corrected['scaled_MSE']+1e-12
    assert corrected['expected_axis_gain_rotation_error']>=-1e-12
    # Current NumPy behavior is the confusable precision decoy.
    if np.fft.fft2(prediction,axes=(0,1)).dtype==np.complex64:
        original=orbit.waveform_budget(prediction,truth,origins,row['scale'],32)
        assert original['error_Parseval_absolute_error']>1e-10


def test_failed_c2_cannot_be_dropped_from_descriptive_six_comparisons():
    records={model:{stem:[dict(primary_D=value)]*8 for stem in ('p32_16','p40_20')}
             for model,value in [('timesfm',.1),('bolt',.2),('chronos2',.001)]}
    result=replay.descriptive_candidates(records)
    assert len(result)==6
    assert result['timesfm-bolt/p32_16']['second_larger_count']==8
    assert result['timesfm-chronos2/p40_20']['first_larger_count']==8
    del records['chronos2']
    with pytest.raises(ValueError,match='all three'):replay.descriptive_candidates(records)
