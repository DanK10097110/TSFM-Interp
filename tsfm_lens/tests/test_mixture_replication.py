"""Frozen mixed-versus-pure replication with shared-energy and bundle controls."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

path = Path(__file__).resolve().parents[1]/'experiments/feature_search/mixture_replication.py'
spec = importlib.util.spec_from_file_location('mixture_replication',path)
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)


def fixture():
    """A coupled model loses on mixtures but wins on both matched pure signals."""
    protocol = rep.load_protocol()
    protocol['n_bundles'] = 16
    protocol['inference'].update(n_boot=199,n_permutations=199)
    context,truth,rows = rep.make_corpus(protocol)
    bolt,timesfm = truth.copy(),truth.copy()
    t = np.arange(128,160)
    for i,row in enumerate(rows):
        if row['condition']=='mixed':
            timesfm[i] = row['offset']+row['a']*np.sin(2*np.pi*t/32+row['phi']+.6*np.sin(row['psi']))+row['b']*np.sin(2*np.pi*t/16+row['psi']+.7*np.sin(row['phi']))
        else:
            bolt[i] += .5*row['scale']
    return protocol,context,truth,rows,{'bolt':bolt,'timesfm':timesfm}


def test_frozen_four_claims_detect_coupling_and_shared_energy_ranking_flip():
    """Pure errors use mixed energy; sixteen bundles remain sixteen units."""
    protocol,_,truth,rows,predictions = fixture()
    result = rep.analyze(predictions,truth,rows,protocol=protocol)
    assert result['n_seeds']==16
    assert result['n_contexts']==128
    assert result['family_size']==4
    assert result['all_four_pass']
    assert result['ranking_flip_descriptive']
    np.testing.assert_allclose(result['pure_gap_descriptive_interval']['value'],-.25,atol=1e-12)
    for effect in result['primary'].values():
        assert effect['n_seeds']==16
        assert effect['p_holm']>=effect['p']
    permutation = np.random.default_rng(18).permutation(len(rows))
    reordered = rep.analyze({key:value[permutation] for key,value in predictions.items()},truth[permutation],[rows[i] for i in permutation],protocol=protocol)
    for claim in result['primary']:
        np.testing.assert_allclose(result['primary'][claim]['value'],reordered['primary'][claim]['value'],atol=1e-12)


def test_unidentifiable_phase_and_duplicate_or_rescaled_cells_fail_honestly():
    """Untestable phase claims stay in Holm4; duplicated rows cannot inflate n."""
    protocol,_,truth,rows,predictions = fixture()
    result = rep.analyze({key:np.zeros_like(truth) for key in predictions},truth,rows,protocol=protocol)
    assert len(result['primary'])==4
    for name in ['mixed_phi_untouched_phase_timesfm_minus_bolt','mixed_psi_untouched_phase_timesfm_minus_bolt']:
        assert result['primary'][name]['p_holm']==1
        assert 'untestable' in result['primary'][name]
    with pytest.raises(ValueError,match='Duplicate'):
        rep.analyze({key:np.concatenate((value,value[:1])) for key,value in predictions.items()},np.concatenate((truth,truth[:1])),rows+rows[:1],protocol=protocol)
    altered = [dict(row) for row in rows]
    altered[4]['scale'] *= 2
    with pytest.raises(ValueError,match='mixed-energy'):
        rep.analyze(predictions,truth,altered,protocol=protocol)


def test_fresh_protocol_corpus_has_eight_cells_and_unrescaled_pure_amplitudes():
    """Pure controls remove one frequency and preserve the other's original energy."""
    protocol,context,truth,rows,_ = fixture()
    assert context.shape==(128,128)
    assert rows[0]['generator_seed']==1215000
    assert {row['condition'] for row in rows}=={'mixed','pure_phi','pure_psi'}
    coefficients = rep.harmonic_components(truth)['coefficients']
    for i,row in enumerate(rows):
        if row['condition']=='pure_phi':
            np.testing.assert_allclose(np.linalg.norm(coefficients[i,:2]),row['a'],atol=1e-12)
            np.testing.assert_allclose(coefficients[i,2:4],0,atol=1e-12)
        elif row['condition']=='pure_psi':
            np.testing.assert_allclose(np.linalg.norm(coefficients[i,2:4]),row['b'],atol=1e-12)
            np.testing.assert_allclose(coefficients[i,:2],0,atol=1e-12)
        np.testing.assert_allclose(row['scale']**2,(row['a']**2+row['b']**2)/2,atol=1e-12)
    assert protocol['claims']['model_by_composition_interaction']['minimum_practical']==.05
