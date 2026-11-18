"""Known isometric signals, private highfrequency decoys, and lag nuisance."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

_path=Path(__file__).parents[1]/'experiments/feature_search/window_alignment.py'
_spec=importlib.util.spec_from_file_location('window_alignment',_path)
w=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(w)


def test_native_concat_isometry_and_complete_context_partition_invariance():
    """Coarse patch embeds every sample; mean pooling of fine tokens destroys it."""
    signal=np.random.default_rng(2).normal(size=(80,128))
    coarse=signal.reshape(80,4,32);fine=signal[:,:,None]
    for window in (32,64,128):
        a=w.represent(coarse,32,window,'native_concat').reshape(-1,window)
        b=w.represent(fine,1,window,'native_concat').reshape(-1,window)
        assert w.linear_cka(a,b)==pytest.approx(1)
        np.testing.assert_equal(a.reshape(80,128),signal)
    coarse_mean=w.represent(coarse,32,32,'mean').reshape(-1,32)
    fine_mean=w.represent(fine,1,32,'mean').reshape(-1,1)
    assert w.linear_cka(coarse_mean,fine_mean)<.3
    with pytest.raises(ValueError,match='unsupported'):
        w.represent(coarse,32,16,'mean')


def test_common_grid_removes_private_highfrequency_and_retains_chronology():
    """Cancellation can help shared lowfrequency geometry while losing other tasks."""
    rng=np.random.default_rng(3)
    low=rng.normal(size=(80,4,1));private=rng.normal(size=(80,4,1))
    fine=np.repeat(low,32,axis=1)+np.repeat(private*10,32,axis=1)*np.tile([-1,1],64)[None,:,None]
    pooled=w.represent(fine,1,128,'common32_concat')
    np.testing.assert_allclose(pooled[:,0],low[:,:,0],atol=1e-12)
    assert w.linear_cka(pooled[:,0],low[:,:,0])==pytest.approx(1)
    assert w.linear_cka(fine.reshape(80,-1),low.reshape(80,-1))<.3


def test_series_null_keeps_adjacency_and_phase_jitter_is_not_identity():
    """Within-series random shifts preserve content but change aligned geometry."""
    rng=np.random.default_rng(1)
    rows=w.series_permutation(20,4,rng).reshape(20,4)
    np.testing.assert_equal(np.diff(rows,axis=1),1)
    assert len(set(rows[:,0]//4))==20
    x=rng.normal(size=(80,16));jitter=np.array([np.roll(row,i%16) for i,row in enumerate(x)])
    assert w.linear_cka(x,jitter)<.5
    np.testing.assert_allclose(np.sort(x,axis=1),np.sort(jitter,axis=1))


def test_pca_basis_cannot_see_test_distribution():
    """An extreme heldout decoy must not rotate the training representation."""
    rng=np.random.default_rng(7);train=rng.normal(size=(40,12));test=rng.normal(size=(10,12))
    a,b,_=w.matched_rank(train,test,rank=4)
    changed=test.copy();changed[:,0]*=10000
    c,d,_=w.matched_rank(train,changed,rank=4)
    np.testing.assert_equal(a,c)
    assert a.shape==(40,4) and b.shape==(10,4)
    assert not np.allclose(b,d)
