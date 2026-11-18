"""Shared-noise composition known answers and clean future-target guards."""
import importlib.util
from pathlib import Path
import numpy as np
from experiments.feature_search.mixture_replication import make_corpus as make_clean

path=Path(__file__).resolve().parents[1]/'experiments/feature_search/mixture_noise_background.py'
spec=importlib.util.spec_from_file_location('mixture_noise_background',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture():
    protocol=audit.load_protocol();protocol['n_bundles']=16;protocol['inference'].update(n_boot=99,n_permutations=199)
    context,truth,rows,noise=audit.make_corpus(protocol)
    return protocol,context,truth,rows,noise


def test_actual_background_forecast_cancels_affine_bias_and_shared_noise():
    """The oracle-offset shortcut fails even when the model is exactly affine."""
    _,context,_,metadata,_=fixture()
    prediction=np.repeat((.7*context[:,-1].astype(float)+.3*context.mean(axis=1,dtype=float)+.2)[:,None],32,axis=1)
    for level in (0.,.1):
        rows=audit.matched_rows(metadata,level);composed=audit.compose_prediction(prediction,rows)
        np.testing.assert_allclose(composed,prediction[rows['mixed']],atol=1e-6)
        offset=np.array([metadata[i]['offset'] for i in rows['mixed']])[:,None]
        wrong=prediction[rows['pure_phi']]+prediction[rows['pure_psi']]-offset
        assert np.mean(np.abs(wrong-prediction[rows['mixed']]))>.1


def test_nine_cells_share_context_noise_and_all_future_truths_stay_clean():
    """Context noise is shared; the future target is its zero conditional mean."""
    protocol,context,truth,rows,noise=fixture()
    validation=audit.validate_corpus(context,truth,rows,protocol)
    assert validation['max_input_additivity_roundoff']<2e-6
    assert len(rows)==16*18
    lookup={(r['seed'],r['noise_level'],r['condition'],r['cell_phi'],r['cell_psi']):i for i,r in enumerate(rows)}
    for level in (0.,.1):
        matched=audit.matched_rows(rows,level)
        assert len(np.unique(matched['background']))==16
        assert all(np.sum(matched['groups']==s)==4 for s in matched['seeds'])
    for i,row in enumerate(rows):
        if row['noise_level']==0:
            j=lookup[row['seed'],.1,row['condition'],row['cell_phi'],row['cell_psi']]
            np.testing.assert_array_equal(truth[i],truth[j])
        if row['condition']=='background_only':
            expected=row['offset']+row['noise_level']*row['scale']*noise[row['seed']]
            np.testing.assert_allclose(context[i],expected,atol=3e-7)
    clean_context,clean_truth,clean_rows=make_clean(protocol)
    for i,row in enumerate(clean_rows):
        j=lookup[row['seed'],0.,row['condition'],row['cell_phi'],row['cell_psi']]
        np.testing.assert_allclose(context[j],clean_context[i],atol=1e-6)
        np.testing.assert_allclose(truth[j],clean_truth[i],atol=1e-12)


def test_complete_holm_four_uses_generating_seed_not_noise_levels_or_rows():
    """Known nonlinear mixture suppression gives four positive seed-paired effects."""
    protocol,_,truth,metadata,_=fixture();timesfm=truth.copy();bolt=truth.copy()
    for i,row in enumerate(metadata):
        if row['condition']=='mixed':timesfm[i]=row['offset']+.4*(truth[i]-row['offset'])
    result=audit.analyze({'timesfm':timesfm,'bolt':bolt},truth,metadata,protocol=protocol)
    assert len(result['primary'])==4
    assert result['n_seeds']==16
    assert result['n_contexts']==16*18
    assert result['all_four_pass']
    for effect in result['primary'].values():
        assert effect['n_seeds']==16
        np.testing.assert_allclose(effect['value'],.36,atol=1e-12)
        np.testing.assert_allclose(effect['p_holm'],4*effect['p'],atol=1e-12)
    np.testing.assert_allclose(result['exploratory_timesfm_noisy_minus_clean_mixed_mse']['value'],0,atol=1e-12)
    flat=audit.analyze({'timesfm':np.zeros_like(truth),'bolt':np.zeros_like(truth)},truth,metadata,protocol=protocol)
    assert len(flat['primary'])==4
    assert all('untestable' not in effect for effect in flat['primary'].values())
    assert 'untestable' in flat['models']['timesfm']['noise01']['behavioral_input_interference']['phi']['other_phase_change_radians']
