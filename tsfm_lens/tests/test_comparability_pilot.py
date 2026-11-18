"""Known-answer positives, decoys and explicit regression plants for comparability.

Each safety assertion is exercised against a deliberately broken counterpart;
these tests run before opening the saved study's outcomes.
"""
import numpy as np
import pytest
from experiments.feature_search import comparability_core as c


def fixture(seed=19, n=100, repeats=2):
    """Independent source latent signal plus input-only and hidden nuisance decoys."""
    rng = np.random.default_rng(seed)
    source_x = rng.normal(size=(n, 6))
    latent = rng.normal(size=(n, 2))
    x = np.repeat(source_x, repeats, axis=0)
    h = np.repeat(np.column_stack([latent, rng.normal(size=(n, 4))]), repeats, axis=0)
    y = np.repeat((3*latent[:, :1] + source_x[:, :1]), repeats, axis=0)
    rows = [dict(source=f's{i}',domain='plant',case=f'c{j}',run='plant',split='fit' if i<72 else 'test') for i in range(n) for j in range(repeats)]
    train=np.array([i for i,r in enumerate(rows) if r['split']=='fit'])
    test=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    return x,h,y,rows,train,test


def fit(x,h,y,rows,tr,te,arm='input+A_hidden'):
    """Exercise the production nested choice and final refit."""
    grams={'A_hidden':h@h.T}
    banks=[(a,b,c.kernel_bank(x,grams,rows,a)) for a,b in c.folds(rows,tr)]
    final=c.kernel_bank(x,grams,rows,tr)
    return c.select_predict(banks,final,y,rows,tr,te,arm)


def assert_signal(pred,y):
    """The shared-signal plant must recover the observable, not just geometry."""
    assert np.mean((pred-y)**2)<.08


def test_shared_signal_and_deleted_signal_regression():
    x,h,y,rows,tr,te=fixture()
    pred,_=fit(x,h,y,rows,tr,te)
    assert_signal(pred,y[te])
    decoy,_=fit(x,np.zeros_like(h),y,rows,tr,te)
    with pytest.raises(AssertionError):
        assert_signal(decoy,y[te])
    input_only,_=fit(x,h,y,rows,tr,te,'input')
    assert np.mean((input_only-y[te])**2)>20*np.mean((pred-y[te])**2)
    same,_=fit(x,np.column_stack([y,h]),y,rows,tr,te)
    assert_signal(same,y[te])


def assert_rotation_equal(a,b):
    """Dense re-expression must preserve the production PCA/ridge predictions."""
    np.testing.assert_allclose(a,b,atol=1e-7,rtol=1e-7)


def test_dense_rotation_and_broken_coordinate_regression():
    x,h,y,rows,tr,te=fixture()
    q,_=np.linalg.qr(np.random.default_rng(31).normal(size=(h.shape[1],h.shape[1])))
    pred,_=fit(x,h,y,rows,tr,te)
    rotated,_=fit(x,h@q,y,rows,tr,te)
    assert_rotation_equal(pred,rotated)
    broken,_=fit(x,(h@q)[:, :1],y,rows,tr,te)
    with pytest.raises(AssertionError):
        assert_rotation_equal(pred,broken)


def test_general_invertible_coordinates_are_not_promised_invariant():
    rng=np.random.default_rng(5)
    h=rng.normal(size=(60,6));w=np.ones(40)/40;tr=np.arange(40)
    q,_=np.linalg.qr(rng.normal(size=(6,6)))
    good=c.pca_kernel(h@h.T,tr,w,2)
    rotated=c.pca_kernel((h@q)@(h@q).T,tr,w,2)
    np.testing.assert_allclose(good,rotated,atol=1e-10)
    stretched=h@q@np.diag([.01,.1,1,10,100,1000])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(good,c.pca_kernel(stretched@stretched.T,tr,w,2),atol=1e-5)


def assert_disjoint(rows,tr,te):
    """Independent units must remain independent after splitting."""
    assert not ({rows[i]['source'] for i in tr}&{rows[i]['source'] for i in te})


def test_group_folds_and_row_split_regression():
    _,_,_,rows,tr,_=fixture()
    for a,b in c.folds(rows,tr):
        assert_disjoint(rows,a,b)
    with pytest.raises(AssertionError):
        assert_disjoint(rows,tr[::2],tr[1::2])


def test_test_labels_and_test_feature_fit_regressions():
    x,h,y,rows,tr,te=fixture()
    p,choice=fit(x,h,y,rows,tr,te)
    altered=y.copy();altered[te]=1e6
    pp,cc=fit(x,h,altered,rows,tr,te)
    assert cc==choice
    np.testing.assert_array_equal(p,pp)
    with pytest.raises(AssertionError):
        bad,_=fit(x,h,altered,rows,np.arange(len(rows)),te)
        np.testing.assert_allclose(p,bad,atol=1e-5)
    w=c.weights(rows,tr)
    k=c.pca_kernel(h@h.T,tr,w,2)
    changed=h.copy();changed[te]*=1e6
    kk=c.pca_kernel(changed@changed.T,tr,w,2)
    np.testing.assert_allclose(k[np.ix_(tr,tr)],kk[np.ix_(tr,tr)],atol=1e-10)
    with pytest.raises(AssertionError):
        leaked=c.pca_kernel(changed@changed.T,np.arange(len(rows)),c.weights(rows,np.arange(len(rows))),2)
        np.testing.assert_allclose(k[np.ix_(tr,tr)],leaked[np.ix_(tr,tr)],atol=1e-5)


def test_source_shuffle_preserves_conditions_and_breaks_signal():
    x,h,y,rows,tr,te=fixture()
    perm=c.shuffled_indices(rows,tr,4951000)
    assert np.array_equal(perm[te],te)
    for i in tr[::2]:
        assert rows[perm[i]]['source']==rows[perm[i+1]]['source']
        assert rows[perm[i]]['case']==rows[i]['case']
    pred,_=fit(x,h[perm],y,rows,tr,te)
    with pytest.raises(AssertionError):
        assert_signal(pred,y[te])
    with pytest.raises(AssertionError):
        assert np.array_equal(perm,np.arange(len(rows)))


def test_source_bootstrap_repetition_and_row_mean_regression():
    rows=[dict(source='a'),dict(source='b')]
    expected=c.interval(list(c.group_values(np.array([0.,2.]),rows,[0,1]).values()))
    repeated=[dict(source='a')]*12+[dict(source='b')]
    values=np.array([0.]*12+[2.])
    actual=c.interval(list(c.group_values(values,repeated,np.arange(13)).values()))
    assert actual==expected
    with pytest.raises(AssertionError):
        assert c.interval(values)==expected
    assert np.isclose(c.weights(repeated,np.arange(13))[:12].sum(),.5)


def test_full_history_ridge_candidate_and_omission_regression():
    x,h,y,rows,tr,te=fixture()
    bank=c.kernel_bank(x,{'A_hidden':h@h.T},rows,tr)
    def check(b):
        """Full-history ridge cannot be silently replaced by rank-limited descriptors."""
        assert 'history_ridge' in b['input'] and 'descriptor_ridge' in b['input']
    check(bank)
    broken={'input':{k:v for k,v in bank['input'].items() if k!='history_ridge'}}
    with pytest.raises(AssertionError):
        check(broken)


def test_identical_normalization_and_wrong_inverse_regression():
    rng=np.random.default_rng(6)
    x=rng.normal(size=(12,128))*np.arange(1,13)[:,None]+np.arange(12)[:,None]*100
    physical=rng.normal(size=(12,32))+x.mean(1)[:,None]
    mean,sd=x.mean(1)[:,None],x.std(1)[:,None]
    normalized=(physical-mean)/sd
    inverse=mean+sd*normalized
    np.testing.assert_allclose(inverse,physical)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(normalized,physical)


def test_cv_roundoff_tie_and_unstable_argmin_regression():
    first = [.03475589842617981, .03475589842617982, .1]
    rotated = [.03475589842617988, .034755898426179876, .1]
    assert c.stable_argmin(first) == c.stable_argmin(rotated) == 0
    assert c.stable_argmin([.3,.2,.1]) == 2
    with pytest.raises(AssertionError):
        assert np.argmin(first) == np.argmin(rotated)
