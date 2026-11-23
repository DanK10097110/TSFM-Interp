import numpy as np
from research.activation_transfer.shape_transfer.cross_seed_groups.study import orthogonal,scale_like
from research.activation_transfer.shape_transfer.cross_seed_groups.score import summaries


def test_global_control_preserves_pairwise_geometry_but_changes_orientation():
    axes=np.array([[1.,.4,0.],[.7,-.2,.1],[0.,.1,.6]])
    q=orthogonal(3,11);r=axes@q
    np.testing.assert_allclose(r@r.T,axes@axes.T,atol=1e-12)
    assert not np.allclose(r,axes)
    np.testing.assert_allclose(r,axes@orthogonal(3,11))


def test_rotation_null_uses_specific_group_norm():
    v=np.array([[1.,2.],[-1.,0.]])
    budget=np.array([[.5,0.],[0.,3.]])
    actual=scale_like(v,budget)
    np.testing.assert_allclose(np.linalg.norm(actual,axis=1),[.5,3.])
    np.testing.assert_allclose(actual[0]/np.linalg.norm(actual[0]),v[0]/np.linalg.norm(v[0]))


def test_equal_and_pooled_responses_differ_when_donor_energy_differs():
    mat=np.array([[[.5,.5,.5,1.],[4.,0.,0.,1.]],[[.5,.5,.5,1.],[5.,0.,0.,1.]]])
    sw=np.array([[1,1],[2,0]]);fw=np.ones((2,2))
    eq,po,per=summaries(mat,sw,fw)
    np.testing.assert_allclose(per,[1.,0.]);np.testing.assert_allclose(eq[0],.5)
    np.testing.assert_allclose(po[0],.1)
