"""Known-answer controls for the opt-in cached-window diagnostic."""
from pathlib import Path
from itertools import combinations

import numpy as np
import pytest

from tsfm_lens.analysis.window_robustness import (WindowTarget,available_rows,centered_kernel,
    compare_windows,kernel_similarity,load_native_tokens,physical_extent,project_rank,validate_rows,window_features)
from tsfm_lens.extraction.store import ActivationStore


def test_coarse_patch_isometry_crop_and_resolution():
    """Preserving ordered native content keeps identical geometry across tilings."""
    signal=np.random.default_rng(2).normal(size=(30,128)).astype(np.float32)
    coarse=signal.reshape(30,4,32);fine=signal[:,:,None]
    for window in (32,64,128):
        a=window_features(coarse,32,window,'native_concat',32).reshape(30,-1)
        b=window_features(fine,1,window,'native_concat',32).reshape(30,-1)
        np.testing.assert_equal(a,signal);np.testing.assert_equal(b,signal)
    assert physical_extent(480,[32,64,128],(96,480))==(96,480)
    with pytest.raises(ValueError,match='explicit common crop'):physical_extent(480,[32,64,128])
    with pytest.raises(ValueError,match='below native resolution'):window_features(coarse,32,16,'mean',32)


def test_group_null_is_invariant_to_duplicating_dependent_windows():
    """Duplicating a series eight times must not manufacture more null evidence."""
    rng=np.random.default_rng(1);x=rng.normal(size=(10,3));y=x+.2*rng.normal(size=(10,3))
    a=kernel_similarity(centered_kernel(x),centered_kernel(y),10,1,99,0)
    b=kernel_similarity(centered_kernel(np.repeat(x,8,0)),centered_kernel(np.repeat(y,8,0)),10,8,99,0)
    np.testing.assert_allclose([a[k] for k in ('value','null_p95','null_p_plusone')],
                               [b[k] for k in ('value','null_p95','null_p_plusone')],atol=1e-14)


def test_pca_is_fitted_on_train_only():
    """A heldout outlier cannot rotate the previously fitted training basis."""
    rng=np.random.default_rng(4);train=rng.normal(size=(40,12));test=rng.normal(size=(12,12))
    ordinary,_=project_rank(train,test,4)
    appended,_=project_rank(train,np.concatenate((test,np.full((1,12),1e8))),4)
    np.testing.assert_allclose(ordinary,appended[:len(test)],atol=1e-12)


def test_store_ids_refuse_missing_and_duplicate_requests(tmp_path):
    """Distinct cached row sets cannot silently turn into an intersected study."""
    store=ActivationStore.create(tmp_path/'activations.zarr',21,4,32,128)
    store.init_fine_layer('a','layer.0',1,32,rows=np.arange(20))
    store.init_fine_layer('b','layer.0',32,1,rows=np.arange(1,21))
    targets=[WindowTarget('a','layer.0',32),WindowTarget('b','layer.0',1)]
    with pytest.raises(ValueError,match='inconsistent IDs'):
        validate_rows(store,targets,np.arange(12),np.arange(6))
    with pytest.raises(ValueError,match='unique integer'):
        validate_rows(store,targets,np.array([1,1,2,3,4,5]),np.array([1,2]))
    store.root['fine/b/layer.0'].attrs['rows']=[1]*20
    with pytest.raises(ValueError,match='duplicate stored'):available_rows(store,targets[1])


def test_cached_end_to_end_isometry_and_visible_skip(tmp_path):
    """A dotted layer fixture verifies native reconstruction and unsupported W16."""
    store=ActivationStore.create(tmp_path/'activations.zarr',20,4,32,128)
    signal=np.random.default_rng(8).normal(size=(20,128)).astype(np.float32)
    store.init_fine_layer('coarse','layer.0',1,32,rows=np.arange(20))
    store.init_fine_layer('fine','layer.0',32,1,rows=np.arange(20))
    store.write_fine_batch('coarse','layer.0',0,signal.reshape(20,4,1,32))
    store.write_fine_batch('fine','layer.0',0,signal.reshape(20,4,32,1))
    loaded=load_native_tokens(store,WindowTarget('fine','layer.0',1),np.arange(20),(0,128))
    np.testing.assert_equal(loaded[:,:,0],signal.astype(np.float16).astype(np.float32))
    result=compare_windows(store,[WindowTarget('coarse','layer.0',32),WindowTarget('fine','layer.0',1)],
                           np.arange(20),np.arange(10),[16,32],ranks=(2,),permutations=9)
    assert result['sequence_partition_max_difference']=={'coarse/layer.0':0.,'fine/layer.0':0.}
    assert result['effective_grid_width']==32 and result['requested_grid_width'] is None and result['grid_width_source']=='auto_lcm'
    assert result['cells']['32_native_concat']['pairs']['coarse/layer.0|fine/layer.0']['raw']['value']==pytest.approx(1)
    assert result['cells']['16_mean']['targets']['coarse/layer.0']['status']=='skipped'


def test_subset_cache_without_row_ids_cannot_be_treated_as_full_store(tmp_path):
    """A plausible contiguous subset is a decoy for unrelated full-store IDs."""
    store=ActivationStore.create(tmp_path/'activations.zarr',21,4,32,128)
    store.init_fine_layer('subset','layer.0',1,4,rows=np.arange(1,21))
    del store.root['fine/subset/layer.0'].attrs['rows']
    with pytest.raises(ValueError,match='refusing to guess identities'):
        available_rows(store,WindowTarget('subset','layer.0',32))
    store.init_fine_layer('full','layer.0',1,4)
    np.testing.assert_equal(available_rows(store,WindowTarget('full','layer.0',32)),np.arange(21))


def test_fractional_source_ids_are_not_coerced_to_plausible_integer_rows(tmp_path):
    """Fractional IDs truncate to valid-looking distinct rows, a deliberate identity decoy."""
    store=ActivationStore.create(tmp_path/'activations.zarr',20,4,32,128)
    store.init_fine_layer('model','layer.0',1,4,rows=np.arange(20))
    store.root['fine/model/layer.0'].attrs['rows']=(np.arange(20)+.5).tolist()
    with pytest.raises(ValueError,match='refusing to coerce identities'):
        available_rows(store,WindowTarget('model','layer.0',32))


def test_report_supports_five_targets_and_states_changing_row_estimand(tmp_path):
    """A five-model panel has ten pairs; ordinary two-model spacing is a decoy."""
    from tsfm_lens.analysis.window_robustness import render_window_report
    targets=[f'model{i}/layer.0' for i in range(5)]
    pairs={'|'.join(pair):{'raw':dict(value=.5,null_p95=.6,null_p_plusone=.5,n_permutations=9)} for pair in combinations(targets,2)}
    result=dict(cells={'32_mean':dict(n_test_series=5,n_windows_per_series=4,targets={},pairs=pairs)},sequence_native_cka={})
    protocol=dict(source='cached/source',rows=[10,11,12],source_sha256='known-source-hash',grid_width=None,targets=[['coarse','layer.0',32],['fine','layer.0',16]])
    path=tmp_path/'report.html';render_window_report(result,path,protocol)
    html=path.read_text(encoding='utf-8')
    assert html.endswith('</body></html>')
    assert all(pair in html for pair in pairs)
    assert 'Whole-series permutations handle dependence; they do not make these scientific units identical.' in html
    assert 'A higher CKA need not mean more shared task information.' in html
    detail=html.index('<details');collapsed_end=html.index('</details>')
    assert '<details open' not in html
    assert 'known-source-hash' in html[detail:collapsed_end]
    assert html.index('<strong>Source:</strong>')<detail<html.index('<h2>Shared geometry across windows</h2>')
    assert 'auto LCM = 32 steps' in html[:detail]


def test_effective_grid_legacy_auto_and_explicit_override():
    """A plausible explicit64 grid is a decoy for blindly replacing every request with LCM32."""
    from tsfm_lens.analysis.window_robustness import effective_grid_metadata
    widths={'coarse/layer.0':32,'fine/layer.0':16}
    value,note=effective_grid_metadata({},dict(grid_width=None),widths)
    assert value==32 and 'auto LCM' in note and 'legacy' in note
    value,note=effective_grid_metadata({},dict(grid_width=64),widths)
    assert value==64 and 'explicit' in note
    value,note=effective_grid_metadata({},dict(grid_width=None),None)
    assert value is None and 'unavailable' in note
    value,note=effective_grid_metadata({},dict(methods=['common32_concat']),widths)
    assert value==32 and 'legacy common32_concat' in note and 'auto LCM = 32' in note and 'request not recorded' in note
    value,note=effective_grid_metadata({},dict(),widths)
    assert value is None and 'unavailable' in note
