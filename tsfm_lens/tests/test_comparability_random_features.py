"""Known-answer controls for full-input random maps and old-FIT-only selection."""
import numpy as np
import pytest
from experiments.feature_search import comparability_random_features as r
from experiments.feature_search import comparability_core as c


def fixture():
    """Random descriptors with a genuine held-out affine midpoint probe."""
    x=np.random.default_rng(10).normal(size=(24,258));x[18]=(x[16]+x[17])/2
    return x,np.arange(16),np.ones(16)/16


def test_full_input_and_normalizer_complete():
    """The random map must receive every time coordinate and both normalizer fields."""
    x,_,_=fixture();full=r.full_input(x)
    assert full.shape==(24,130)
    np.testing.assert_array_equal(full[:,:128],x[:,:128])
    np.testing.assert_array_equal(full[:,128:],x[:,234:236])
    with pytest.raises(ValueError):r.full_input(x[:,:257])


@pytest.mark.parametrize('family',r.FAMILIES)
def test_seeded_nonlinear_maps(family):
    """Equal seeds reproduce, different seeds differ, and each map is nonlinear."""
    x,tr,w=fixture();a,meta=r.random_features(x,tr,w,family,1,64)
    again,_=r.random_features(x,tr,w,family,1,64);different,_=r.random_features(x,tr,w,family,2,64)
    np.testing.assert_array_equal(a,again)
    assert not np.allclose(a,different)
    assert a.shape==(24,64) and meta['random_parameters']==64*131
    assert np.max(abs(a[18]-(a[16]+a[17])/2))>1e-4


@pytest.mark.parametrize('family',r.FAMILIES)
def test_random_map_fit_excludes_test(family):
    """Mutating all TEST descriptors cannot change FIT features or bandwidth."""
    x,tr,w=fixture();a,meta=r.random_features(x,tr,w,family,1,64)
    changed=x.copy();changed[16:]+=10000
    b,other=r.random_features(changed,tr,w,family,1,64)
    np.testing.assert_array_equal(a[tr],b[tr]);assert meta==other


def test_projection_rank_and_rotation():
    """The readout budget counts PCA features, with orthogonal-coordinate invariance."""
    x,tr,w=fixture();h,_=r.random_features(x,tr,w,'rff',3,64)
    q,_=np.linalg.qr(np.random.default_rng(9).normal(size=(64,64)))
    k=c.pca_kernel(h@h.T,tr,w,8);rot=c.pca_kernel((h@q)@(h@q).T,tr,w,8)
    np.testing.assert_allclose(k,rot,atol=1e-10)
    assert np.linalg.matrix_rank(k,tol=1e-8)<=8


def test_grouped_selection_signal_and_test_label_isolation():
    """Known recoverable signal survives the matched readout; TEST labels cannot choose it."""
    rng=np.random.default_rng(6);h=rng.normal(size=(100,6));x=rng.normal(size=(100,4));y=h[:,:1]
    rows=[dict(source=f's{i}',domain='plant') for i in range(100)];tr=np.arange(72);te=np.arange(72,100)
    candidates=[(rank,alpha) for rank in (8,16) for alpha in r.ALPHAS];banks=[]
    for t,v in c.folds(rows,tr):
        w=c.weights(rows,t);base=c.pca_kernel(x@x.T,t,w,4);extra={rank:c.pca_kernel(h@h.T,t,w,rank) for rank in (8,16)};banks.append((t,v,base,extra))
    chosen=r.choose(candidates,y,rows,banks)
    assert len(chosen['all_cv_losses'])==8
    changed=y.copy();changed[te]=1e9
    assert r.choose(candidates,changed,rows,banks)==chosen
    w=c.weights(rows,tr);base=c.pca_kernel(x@x.T,tr,w,4);hidden=c.pca_kernel(h@h.T,tr,w,chosen['rank'])
    pred=c.ridge((2*base+hidden)/3,y,tr,te,w,chosen['alpha'])
    assert np.mean((pred-y[te])**2)<.01
    decoy=c.ridge(base,y,tr,te,w,chosen['alpha'])
    with pytest.raises(AssertionError):assert np.mean((decoy-y[te])**2)<.01


def test_prepared_budget_and_native_hold(tmp_path):
    """The frozen design must match the trained arm budget and expose no native request."""
    out=tmp_path/'prepared';r.prepare(out);p=r.read(out/'protocol.json');proposal=r.read(out/'new_condition_proposal.json')
    assert p['budget']['PCA_ranks']==[8,16]
    assert p['budget']['alpha_grid']==[.001,.01,.1,1.]
    assert p['budget']['candidates_per_init']==8
    assert p['budget']['widths']==dict(bolt=4608,timesfm=5120)
    assert len(p['initialization_seeds'])==len(set(p['initialization_seeds']))==3
    assert 'HOLD' in p['constraints']['native_launch'] and 'HOLD' in proposal['status']
