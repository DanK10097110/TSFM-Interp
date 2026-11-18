"""Known-answer phase geometry, rank, even-response, leakage and source-holdout fixtures."""
import numpy as np
from experiments.feature_search.response_phase_study import (
    basis,split_matrix,response_matrix,loso_mean,prediction,waveform_loss,bootstrap_indices)


def test_rotation_has_gain_phase_without_anisotropy_or_even_response():
    """A rotation/gain is complex-linear in the declared sine/cos orientation."""
    angle=.6;gain=1.7;a=.1;f=128/24
    h=gain*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    base=9+np.linspace(-2,2,32);y=base+prediction(h,a,f)
    score,raw=response_matrix(y,base,a,f)
    np.testing.assert_allclose(score['H'],h,atol=1e-13)
    np.testing.assert_allclose(score['complex_gain'],gain,atol=1e-13)
    np.testing.assert_allclose(score['phase_radians'],angle,atol=1e-13)
    assert score['anisotropy_frobenius_fraction']<1e-24 and score['even_raw_rms']<1e-13
    np.testing.assert_allclose(score['singular_values'],[gain,gain],atol=1e-13)


def test_rank_one_and_reflection_do_not_conflate_ellipse_shape_and_phase_conjugation():
    """A rank-one response collapses an axis; a reflection is anisotropic but remains circular."""
    rank_one=split_matrix(np.diag([2.,0.]))
    assert rank_one['numerical_rank']==1 and rank_one['relative_rank_at_10pct']==1
    assert rank_one['singular_ratio']==0 and rank_one['participation_rank']==1
    np.testing.assert_allclose(rank_one['symmetric_traceless'],[[1,0],[0,-1]])
    assert rank_one['anisotropy_frobenius_fraction']==.5
    reflected=split_matrix(np.diag([1.,-1.]))
    assert reflected['singular_ratio']==1 and reflected['anisotropy_frobenius_fraction']==1
    assert reflected['complex_gain']==0 and reflected['determinant']==-1


def test_off_frequency_and_nonlinear_even_energy_close_exactly():
    """A quadratic even term and an orthogonal odd term survive separately beside a linear tone."""
    f=128/19.2;a=.1;b=basis(f);base=np.full(32,41.)
    h=np.array([[.7,.2],[-.3,1.1]])
    nuisance=np.linspace(-1,1,32);off=nuisance-b@np.linalg.lstsq(b,nuisance,rcond=None)[0]
    y=base+prediction(h,a,f)
    y[0]+=a*off;y[2]-=a*off
    even_s=.3*np.cos(np.arange(32));even_c=.2*np.ones(32)
    y[[0,2]]+=a*a*even_s;y[[1,3]]+=a*a*even_c
    score,raw=response_matrix(y,base,a,f)
    np.testing.assert_allclose(score['H'],h,atol=1e-12)
    np.testing.assert_allclose(raw['even'],2*a*a*np.column_stack([even_s,even_c]),atol=1e-13)
    loss=waveform_loss(prediction(h,a,f),raw['delta'],raw['ideal'])
    np.testing.assert_allclose(loss,score['off_frequency_nmse_contribution']+score['even_nmse_contribution'],atol=1e-13)
    assert score['off_frequency_nmse_contribution']>.01 and score['even_nmse_contribution']>0


def test_finite_window_dc_leakage_is_explicitly_detected():
    """A constant odd forecast can project into the primary band but has zero DC-adjusted tone."""
    base=np.full(32,7.);a=.1;y=np.tile(base,(4,1));y[0]+=2*a;y[2]-=2*a
    score,raw=response_matrix(y,base,a,128/24)
    assert np.linalg.norm(score['H'])>.1
    np.testing.assert_allclose(score['H_dc_adjusted'],np.zeros((2,2)),atol=1e-13)
    np.testing.assert_allclose(score['dc_coefficients'],[2.,0.],atol=1e-13)
    assert score['dc_matrix_shift_frobenius']>.1


def test_loso_matrix_excludes_whole_receiver_source():
    """A huge receiver-only edit cannot change the matrix used to predict that receiver."""
    matrices=np.arange(16*4,dtype=float).reshape(16,2,2)/20
    first=loso_mean(matrices);changed=matrices.copy();changed[3]+=10000
    second=loso_mean(changed)
    np.testing.assert_array_equal(first[3],second[3])
    np.testing.assert_allclose(first[3],np.delete(matrices,3,axis=0).mean(0))
    assert np.linalg.norm(second[0]-first[0])>100


def test_bootstrap_resamples_sources_with_type_balance_not_phases():
    """Each draw contains sixteen source IDs with four draws per background type."""
    types=np.tile(np.arange(4),4);draws=bootstrap_indices(types,100)
    assert draws.shape==(100,16) and draws.max()<16 and draws.min()>=0
    for kind in range(4):assert np.all(np.sum(types[draws]==kind,axis=1)==4)
    assert any(len(set(row))<16 for row in draws)
    np.testing.assert_array_equal(draws,bootstrap_indices(types,100))


def test_phase_energy_pooled_score_is_not_mean_of_phase_ratios():
    """Unequal finite-horizon phase energies must be weighted by their actual energy."""
    ideal=prediction(np.eye(2),.1,128/24);actual=np.zeros_like(ideal);pred=ideal.copy();pred[0]*=3
    expected=np.sum(pred**2)/np.sum(ideal**2)
    assert abs(expected-np.mean(np.sum(pred**2,1)/np.sum(ideal**2,1)))>.01
    np.testing.assert_allclose(waveform_loss(pred,actual,ideal),expected)
