"""Known answers and confusable cases for saved position-mean diagnostics."""
import numpy as np
import pytest
from experiments.feature_search import comparability_position_mean as m


def fixture():
    rows=[dict(source=s,split=split,domain='d') for s,split in [('a','fit'),('a','fit'),('b','fit'),('c','test')]]
    return np.array([[[0.],[10.]],[[0.],[10.]],[[4.],[14.]],[[3.],[13.]]]),rows


def test_known_position_structure():
    h,rows=fixture();g,p=m.fit_means(h,rows)
    np.testing.assert_array_equal(g,[7.]);np.testing.assert_array_equal(p,[[2.],[12.]])
    result=m.summarize(h[3:],{'global_fit_mean':g,'position_conditioned_fit_mean':p},rows[3:])
    assert result['mse']=={'global_fit_mean':26.,'position_conditioned_fit_mean':1.}
    assert result['test_variance_about_domain_test_global_mean']==25.


def test_test_poison_cannot_change_means():
    h,rows=fixture();before=m.fit_means(h,rows);h[3:]+=1e9
    after=m.fit_means(h,rows)
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)


def test_source_weighting_with_duplicate_cases():
    h,rows=fixture()
    np.testing.assert_array_equal(m.weights(rows[:3]),[.25,.25,.5])
    g,p=m.fit_means(h,rows)
    repeated=np.concatenate([h[:1].repeat(7,axis=0),h[1:]])
    metadata=[rows[0]]*7+rows[1:]
    gg,pp=m.fit_means(repeated,metadata)
    np.testing.assert_allclose(g,gg);np.testing.assert_allclose(p,pp)


def test_source_overlap_rejected():
    h,rows=fixture();rows[-1]['source']='a'
    with pytest.raises(ValueError,match='Source overlap'):m.fit_means(h,rows)


def test_pca_is_adaptive_signed_topk():
    artifact=dict(center=np.zeros(3,np.float32),scale=np.array(1.),basis=np.eye(3,dtype=np.float32),k=np.array(1))
    h=np.array([[[1.,3.,-4.]]],np.float32)
    rec,_,_=m.sparse_pca(h,artifact)
    np.testing.assert_array_equal(rec,[[[0.,0.,-4.]]])
