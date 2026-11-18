"""Context expansion checks preserve physical signals and fixed inference families."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'experiments/feature_search/mixture_context_expansion.py'
spec=importlib.util.spec_from_file_location('mixture_context_expansion',path)
expansion=importlib.util.module_from_spec(spec);spec.loader.exec_module(expansion)


def fixture():
    """A composition defect disappears at512 while pure controls remain fixed."""
    protocol=expansion.load_protocol();protocol['n_bundles']=16;protocol['inference'].update(n_boot=99,n_permutations=199)
    contexts,truths,rows=expansion.make_context_corpora(protocol)
    predictions={};scale=np.array([row['scale'] for row in rows])[:,None]
    offset=np.array([row['offset'] for row in rows])[:,None]
    mixed=np.array([row['condition']=='mixed' for row in rows])
    for context,truth in truths.items():
        bolt=truth.copy()+np.where(mixed[:,None],.2,.4)*scale
        timesfm=truth.copy();timesfm[mixed]=offset[mixed]+(1-{128:.6,256:.3,512:0}[context])*(truth[mixed]-offset[mixed])
        predictions[context]=dict(timesfm=timesfm,bolt=bolt,chronos2=truth+.1*scale)
    return protocol,contexts,truths,rows,predictions


def test_contexts_share_fresh_seeds_global_phases_and_full_period_endings():
    """Longer contexts extend exactly the same physical signals."""
    protocol,contexts,truths,rows,_=fixture()
    assert rows[0]['generator_seed']==1326000
    np.testing.assert_array_equal(contexts[256][:,:128],contexts[128])
    np.testing.assert_array_equal(contexts[512][:,:256],contexts[256])
    np.testing.assert_allclose(truths[128],truths[512],atol=1e-12)
    bad=dict(protocol,contexts=[129,256,512])
    with pytest.raises(ValueError,match='divisible'):
        expansion.make_context_corpora(bad)


def test_frozen_global_six_and_nine_families_retain_vanished_context_effect():
    """An effect vanishing at512 remains a negative claim in the complete family."""
    protocol,_,truths,rows,predictions=fixture()
    result=expansion.analyze(predictions,truths,rows,protocol=protocol)
    assert len(result['primary'])==6
    assert len(result['secondary_native_mixed_loss'])==9
    np.testing.assert_allclose(result['primary']['timesfm_oracle_gain_context128']['value'],.36,atol=1e-12)
    np.testing.assert_allclose(result['primary']['timesfm_oracle_gain_context256']['value'],.09,atol=1e-12)
    assert not result['primary']['timesfm_oracle_gain_context512']['passed']
    np.testing.assert_allclose(result['primary']['timesfm_oracle_gain_context512']['value'],0,atol=1e-20)
    for value in result['primary'].values():
        assert value['n_seeds']==16
        assert value['p_holm']>=value['p']
    gap=result['secondary_native_mixed_loss']['timesfm_minus_chronos2_mixed_mse_context128']
    np.testing.assert_allclose(gap['value'],.35,atol=1e-12)


def test_forecast_start_and_metadata_truth_cannot_drift_with_context():
    """Explicit physical forecast starts recover phases at every declared context."""
    protocol,_,truths,rows,predictions=fixture()
    report=expansion.analyze(predictions,truths,rows,protocol=protocol)
    for context in protocol['contexts']:
        expansion._validate_compositions(truths[context],rows,'condition',forecast_start=context)
        assert report['descriptive'][str(context)]['timesfm']['forecast_start']==context
    altered={key:value.copy() for key,value in truths.items()}
    altered[512]=np.roll(altered[512],1,axis=1)
    with pytest.raises(ValueError,match='physical components'):
        expansion.analyze(predictions,altered,rows,protocol=protocol)
    with pytest.raises(ValueError,match='all three'):
        expansion.analyze({128:predictions[128]},truths,rows,protocol=protocol)
