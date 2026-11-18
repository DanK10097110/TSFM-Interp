"""Baseline interference known answers with genuine varying nuisance concepts."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

path = Path(__file__).resolve().parents[1]/'experiments/feature_search/mixture_interference.py'
spec = importlib.util.spec_from_file_location('mixture_interference',path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    """Build variable-amplitude factorial truths and a coupled nonlinear predictor."""
    rng = np.random.default_rng(401)
    rows,truth,coupled = [],[],[]
    t = np.arange(128,160)
    for seed in range(12):
        p,q = rng.uniform(-np.pi,np.pi,2)
        a,b = rng.uniform(.5,2,2)
        offset = rng.uniform(-2,2)
        for cp,cq in [(0,0),(0,1),(1,0),(1,1)]:
            phi,psi = p+cp*np.pi/2,q+cq*np.pi/2
            rows.append(dict(seed=seed,split='test',cell_phi=cp,cell_psi=cq,phi=phi,psi=psi,a=a,b=b,offset=offset,scale=np.sqrt((a*a+b*b)/2)))
            truth.append(offset+a*np.sin(2*np.pi*t/32+phi)+b*np.sin(2*np.pi*t/16+psi))
            coupled.append(offset+a*np.sin(2*np.pi*t/32+phi+.3*np.sin(psi))+b*(1+.2*np.cos(phi))*np.sin(2*np.pi*t/16+psi+.4*np.sin(phi)))
    return np.array(truth),np.array(coupled),rows


def test_independent_vs_coupled_forecaster_and_target_phase_decoy():
    """Changing the target byπ/2 alone must not count as untouched interference."""
    truth,coupled,rows = fixture()
    for target in ('phi','psi'):
        independent = audit.input_interference(truth,truth,rows,target)
        interference = audit.input_interference(coupled,truth,rows,target)
        assert independent['comparisons_per_seed'] == 4
        assert len(independent['seeds']) == 12
        assert independent['metrics']['other_phase_change_radians'].max() < 1e-12
        assert interference['metrics']['other_phase_change_radians'].mean() > .2
    receiver,donor,_ = audit.matched_pairs(rows,'phi')
    components = audit.harmonic_components(truth)['coefficients']
    target_angles = np.arctan2(components[:,1],components[:,0])
    assert np.abs(np.angle(np.exp(1j*(target_angles[donor]-target_angles[receiver])))).mean() > 1


def test_exact_frequency_amplitude_phase_interaction_and_residual_decomposition():
    """An off-basis wave is retained, and signed interaction reconstructs MSE."""
    truth,_,_ = fixture()
    t = np.arange(128,160)
    prediction = .5*truth+.3*np.sin(2*np.pi*t/8)[None,:]
    result = audit.forecast_error_decomposition(prediction,truth)
    np.testing.assert_allclose(result['total_mse'],result['reconstructed_mse'],atol=1e-12)
    np.testing.assert_allclose(result['residual_mse'],.3**2/2,atol=1e-12)
    shifted = np.roll(.5*truth,3,axis=1)
    phase_result = audit.forecast_error_decomposition(shifted,truth)
    assert phase_result['frequencies']['phi']['amplitude_phase_interaction'].mean() < -.01
    for values in phase_result['frequencies'].values():
        np.testing.assert_allclose(values['coefficient_mse'],values['amplitude_mse']+values['phase_mse']+values['amplitude_phase_interaction'],atol=1e-12)


def test_scaling_truth_and_unidentifiable_phase_guards():
    """Relative changes survive scale, while vanishing forecasts have undefined phase."""
    truth,coupled,rows = fixture()
    original = audit.input_interference(coupled,truth,rows,'phi')
    scaled_rows = [dict(row,**{key:row[key]*10 for key in ('a','b','offset','scale')}) for row in rows]
    scaled = audit.input_interference(coupled*10,truth*10,scaled_rows,'phi')
    np.testing.assert_allclose(original['metrics']['other_phase_change_radians'],scaled['metrics']['other_phase_change_radians'],atol=1e-12)
    np.testing.assert_allclose(original['metrics']['other_amplitude_change_relative'],scaled['metrics']['other_amplitude_change_relative'],atol=1e-12)
    with pytest.raises(ValueError,match='RMS'):
        bad = [dict(row,scale=row['scale']*2) for row in rows]
        audit.input_interference(coupled,truth,bad,'phi')
    with pytest.raises(ValueError,match='Truth untouched'):
        altered = truth.copy()
        altered[1] += .3*np.sin(2*np.pi*np.arange(128,160)/16)
        audit.input_interference(coupled,altered,rows,'phi')
    unavailable = audit.analyze({'bolt':np.zeros_like(truth),'timesfm':np.zeros_like(truth)},truth,rows,n_boot=49,n_permutations=49)
    assert set(unavailable['crossmodel_phase_change_timesfm_minus_bolt']) == {'phi','psi'}
    for contrast in unavailable['crossmodel_phase_change_timesfm_minus_bolt'].values():
        assert contrast['p_holm'] == 1
        assert 'untestable' in contrast
