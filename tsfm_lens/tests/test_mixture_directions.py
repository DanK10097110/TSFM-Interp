"""Known-answer filters-versus-patterns checks with a correlated noise decoy."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

path = Path(__file__).resolve().parents[1]/'experiments/feature_search/mixture_directions.py'
spec = importlib.util.spec_from_file_location('mixture_directions',path)
directions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(directions)


def fixture():
    """Plant two harmonic concepts plus target-free channels that cancel noise."""
    rng = np.random.default_rng(510)
    metadata = []
    for seed in range(90):
        phi,psi = rng.uniform(-np.pi,np.pi,2)
        a,b = rng.uniform(.5,2,2)
        offset = rng.uniform(-1,1)
        for p,q in [(0,0),(0,1),(1,0),(1,1)]:
            metadata.append(dict(seed=seed,split='train' if seed<60 else 'test',cell_phi=p,cell_psi=q,
                                 phi=phi+p*np.pi/2,psi=psi+q*np.pi/2,a=a,b=b,offset=offset,scale=np.sqrt((a*a+b*b)/2)))
    y, nuisance = directions.semantic_design(metadata,[0,8])
    flat = y.reshape(-1,4)
    design = np.column_stack((np.ones(len(flat)),flat,nuisance.reshape(-1,3)))
    noise = rng.normal(size=(len(flat),2))
    train = np.repeat([row['split']=='train' for row in metadata],2)
    noise[train] -= design[train]@np.linalg.lstsq(design[train],noise[train],rcond=None)[0]
    h = np.column_stack((flat[:,:2]+2*flat[:,2:]+3*noise,noise,flat[:,2:]))
    return h.reshape(len(metadata),2,6),metadata


def projection(frame):
    """Compare subspaces without relying on signs of basis vectors."""
    return frame@frame.T


def test_conditional_patterns_capture_signal_not_noise_cancelling_filters():
    """A decodable target does not make the raw filter its generating direction."""
    tokens, metadata = fixture()
    result = directions.fit_direction_frames(tokens,metadata,[0,8],alpha=.001)
    frames = result['frames']
    truth = np.eye(6)[:,:2]
    np.testing.assert_allclose(projection(frames['encoding2']),projection(truth),atol=1e-10)
    np.testing.assert_allclose(projection(frames['haufe2']),projection(truth),atol=.001)
    assert np.linalg.norm(frames['decoder2'][2:4]) > 1
    assert np.linalg.norm(frames['encoding2'][2:]) < 1e-10
    assert np.linalg.norm(projection(frames['decoder2'])-projection(truth)) > 1
    nuisance = directions.fit_direction_frames(tokens,metadata,[0,8],target='psi',alpha=.001)
    nuisance_truth,_ = np.linalg.qr(2*np.eye(6)[:,:2]+np.eye(6)[:,4:])
    np.testing.assert_allclose(projection(nuisance['frames']['encoding2']),projection(nuisance_truth),atol=1e-10)
    assert result['diagnostics']['training_seeds'] == 60


def test_physical_phase_alignment_and_heldout_fit_invariance():
    """Token phase is physical; held-out activation/label changes cannot fit frames."""
    tokens, metadata = fixture()
    labels,_ = directions.semantic_design(metadata,[0,8],amplitude_weighted=False)
    np.testing.assert_allclose(labels[:,1,0],labels[:,0,1],atol=1e-14)
    np.testing.assert_allclose(labels[:,1,1],-labels[:,0,0],atol=1e-14)
    before = directions.fit_direction_frames(tokens,metadata,[0,8])
    changed = tokens.copy()
    changed_metadata = [dict(row) for row in metadata]
    for i,row in enumerate(changed_metadata):
        if row['split'] != 'train':
            changed[i] = 1e8
            row['phi'] += 100
            row['psi'] -= 100
            row['a'] = 1000
    after = directions.fit_direction_frames(changed,changed_metadata,[0,8])
    for key in before['frames']:
        np.testing.assert_array_equal(before['frames'][key],after['frames'][key])
    assert before['diagnostics'] == after['diagnostics']


def test_haufe_prediction_covariance_and_factorial_guards():
    """Predicted-factor covariance gives exact OLS reconstruction of those factors."""
    rng = np.random.default_rng(88)
    h = rng.normal(size=(200,5))@np.diag([1,2,3,4,5])
    h -= h.mean(0)
    decoder = rng.normal(size=(5,2))@np.diag([.3,8])
    patterns = directions.haufe_patterns(h,decoder)
    expected = np.linalg.lstsq(h@decoder,h,rcond=None)[0].T
    np.testing.assert_allclose(patterns,expected,atol=1e-12)
    tokens,metadata = fixture()
    with pytest.raises(ValueError,match='complete four-cell'):
        directions.fit_direction_frames(tokens[1:],metadata[1:],[0,8])
    with pytest.raises(ValueError,match='training boundary'):
        mask = np.array([row['split']=='train' for row in metadata])
        mask[0] = False
        directions.fit_direction_frames(tokens,metadata,[0,8],train_mask=mask)
