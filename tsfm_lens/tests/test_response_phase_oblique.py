"""Angular interpolation, cardinal harmonic aliasing and matched-source instrumentation."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from experiments.feature_search import response_phase_oblique as study


def test_linear_phase_law_has_equal_pooled_error_without_any_fit():
    """Balanced cardinal/oblique grids give the same error for a truly linear phase law."""
    truth=np.array([[.7,.3],[-.4,.9]]);frozen=np.array([[.2,-.1],[.1,.6]])
    with patch.object(np.linalg,'lstsq',side_effect=AssertionError('No fitting authorized')):
        values=[]
        for angles in (study.CARDINAL,study.ANGLES):
            y=study.prediction(truth,.1,angles)
            result,_=study.score(y,.1,angles,frozen);values.append(result['nmse'])
            assert result['even_nmse']<1e-25
            exact,_=study.score(y,.1,angles,truth);assert exact['nmse']==0
    np.testing.assert_allclose(values[0],values[1],atol=1e-14)
    assert values[0]>.1


def test_cardinal_third_harmonic_alias_breaks_on_oblique_holdout():
    """cos3phi/sin3phi masquerade as a reflected linear map at the four cardinal angles."""
    h=np.diag([1.,-1.]);t=np.arange(32)
    b=np.column_stack([np.sin(2*np.pi*t/study.PERIOD),np.cos(2*np.pi*t/study.PERIOD)])
    scores=[]
    for angles in (study.CARDINAL,study.ANGLES):
        y=.1*np.column_stack([np.cos(3*angles),np.sin(3*angles)])@b.T
        result,_=study.score(y,.1,angles,h);scores.append(result)
    assert scores[0]['nmse']<1e-25
    assert scores[1]['nmse']>1
    assert scores[1]['even_nmse']<1e-25
    np.testing.assert_allclose(scores[1]['odd_residual_nmse'],scores[1]['nmse'],atol=1e-13)


def test_even_response_and_phase_energy_close():
    """A second angular harmonic is even under opposite-phase pairs and remains scored."""
    a=.1;h=np.array([[.8,.1],[-.2,.6]]);y=study.prediction(h,a,study.ANGLES)
    even=.03*np.cos(2*study.ANGLES)[:,None]*np.linspace(1,2,32)
    y+=even
    score,raw=study.score(y,a,study.ANGLES,h)
    np.testing.assert_allclose(raw['even'],2*even[:4],atol=1e-14)
    np.testing.assert_allclose(score['nmse'],score['even_nmse'],atol=1e-14)
    assert score['odd_residual_nmse']<1e-25
    y[0]+=np.linspace(-.1,.3,32)
    score,raw=study.score(y,a,study.ANGLES,h)
    expected=np.sum((y-raw['prediction'])**2)/np.sum(raw['ideal']**2)
    np.testing.assert_allclose(score['nmse'],expected,atol=1e-14)
    assert abs(expected-np.mean(score['per_angle_nmse']))>1e-4


def fixture_prior():
    """Small synthetic metadata with sixteen independent source IDs; no native dependency."""
    rng=np.random.default_rng(7);inputs=rng.normal(size=(16,128)).astype(np.float32)
    groups=[dict(source=f'syn:{4954300+i}',source_id=i,background_type=i%4,center=128/19.2,offset=0.,
                 base=i,indices=[i]*4,amplitude=.1,input_scale=1.) for i in range(16)]
    return inputs,groups


def test_oblique_bank_reuses_exact_backgrounds_and_unseen_angles():
    """144 inputs contain exact existing backgrounds and all eight offset phase angles."""
    old,groups=fixture_prior();bank,nodes,new,gates=study.bank_from_prior(old,groups)
    assert bank.shape==(144,128) and len(nodes)==144 and len(new)==16
    np.testing.assert_allclose(study.ANGLES,np.pi/8+np.arange(8)*np.pi/4,atol=1e-15)
    assert all(not np.any(np.isclose((angle-study.CARDINAL)%(2*np.pi),0)) for angle in study.ANGLES)
    for g in new:
        np.testing.assert_array_equal(bank[g['base']],old[g['cardinal_base']])
        assert len(g['indices'])==8 and len(set(g['indices']))==8
        t=np.arange(-128,0)
        expected=np.array([old[g['cardinal_base']].astype(float)+.1*np.sin(2*np.pi*t/19.2+angle) for angle in study.ANGLES],np.float32)
        np.testing.assert_array_equal(bank[g['indices']],expected)
    assert gates['nominal_rms_over_peak_max']<1e-5


def test_source_inference_and_family_four_do_not_count_angles_as_n():
    """All dependent phase losses reduce to sixteen paired source observations."""
    types=np.tile(np.arange(4),4);draws=study.source_draws(types)
    assert draws.shape==(16384,16)
    for kind in range(4):assert np.all(np.sum(types[draws]==kind,axis=1)==4)
    x=np.linspace(-.2,.7,16);result=study.interval(x,draws)
    np.testing.assert_allclose(result['family4_percentile'],np.quantile(x[draws].mean(1),[.00625,.99375]))
    assert result['family4_percentile'][0]<result['percentile95'][0]


def test_changed_background_and_absent_slot_fail_closed():
    """Background replay drift or a missing external grant cannot silently qualify."""
    study.assert_background_parity(np.zeros(32),np.zeros(32))
    with pytest.raises(ValueError,match='parity'):study.assert_background_parity(np.zeros(32),np.ones(32)*1e-8)
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(PermissionError):study.capture_driver.worker(Path(directory),'bolt',None)
