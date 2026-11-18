"""Right-alignment and nontiling fullGram known answers before native inference."""
import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'experiments/feature_search/patch_context_challenge.py'
spec=importlib.util.spec_from_file_location('patch_context_challenge',path)
challenge=importlib.util.module_from_spec(spec);spec.loader.exec_module(challenge)


def fixture():
    """Known context benefit only for the two neighboring nontiling periods."""
    protocol=challenge.load_protocol();protocol['n_bundles']=16;protocol['inference'].update(n_boot=99,n_permutations=199)
    contexts,truths,rows=challenge.make_corpora(protocol);scale=np.array([row['scale'] for row in rows])[:,None]
    predictions={}
    for (period,context),truth in truths.items():
        bias=.3 if period==32 else .4 if context==128 else 0
        predictions['timesfm',period,context]=truth+bias*scale
        predictions['bolt',period,context]=truth.copy()
    return protocol,contexts,truths,rows,predictions


def test_right_aligned_suffix_and_identical_future_for_nontiling_periods():
    """Prefix matching would silently change the future-relative phase at31/33."""
    _,contexts,truths,rows,_=fixture()
    assert rows[0]['generator_seed']==1548000
    for period in (31,32,33):
        np.testing.assert_array_equal(contexts[period,128],contexts[period,512][:,-128:])
        np.testing.assert_array_equal(truths[period,128],truths[period,512])
    assert not np.array_equal(contexts[31,128],contexts[31,512][:,:128])
    assert not np.array_equal(contexts[33,128],contexts[33,512][:,:128])


def test_nontiling_gram_cross_terms_reconstruct_full_mse():
    """Dropping cross terms fails even though harmonic OLS itself fits correctly."""
    _,_,truths,_,_=fixture()
    t=np.arange(32)
    for period in (31,33):
        truth=truths[period,128]
        prediction=truth+.4+.7*np.sin(2*np.pi*t/period+.3)[None,:]+.3*np.cos(2*np.pi*t/16)[None,:]
        result=challenge.error_decomposition(prediction,truth,period)
        np.testing.assert_allclose(result['reconstructed_mse'],result['total_mse'],atol=1e-12)
        assert np.max(np.abs(result['gram_cross_terms']))>.001
        gram=result['gram']
        assert np.max(np.abs(gram-np.diag(np.diag(gram))))>.001
    tiled=challenge.error_decomposition(truths[32,128],truths[32,128],32)['gram']
    assert np.max(np.abs(tiled-np.diag(np.diag(tiled))))<1e-12


def test_frozen_neighbor_average_minus_tiled_context_benefit():
    """The one prespecified contrast uses seed-paired neighbors versus tiled control."""
    protocol,_,truths,rows,predictions=fixture()
    result=challenge.analyze(predictions,truths,rows,protocol=protocol)
    np.testing.assert_allclose(result['primary']['value'],.16,atol=1e-12)
    assert result['primary']['passed']
    assert result['primary']['n_seeds']==16
    assert result['primary']['family_size']==1
    assert result['conditions']['32']['timesfm']['native_output_change_mse_common_theoretical_energy']['mixed']['value']==0
    assert result['conditions']['31']['timesfm']['mixed_context_benefit']['value']>.15
