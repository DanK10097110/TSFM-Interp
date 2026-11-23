import numpy as np

from research.activation_transfer.shape_transfer.quantiles.score_interior import FRACTIONS, interpolated, learn_weights, observed


def test_interior_curvature_survives_median_and_signed_tail_removal():
    med=np.array([3.,-11.])[:,None,None]
    left=np.array([-2.,4.])[:,None,None]
    right=np.array([5.,-3.])[:,None,None]
    slopes=np.array([.9,.52,.12,.35,.57,.8])
    x=np.zeros((2,1,1,9))
    x[...,0],x[...,4],x[...,8]=med+left,med,med+right
    for j,idx in enumerate((1,2,3,5,6,7)):
        x[...,idx]=med+slopes[j]*(left if j<3 else right)
    y,curve=observed(x)
    w,shared=learn_weights(x,np.ones((2,1),dtype=bool))
    np.testing.assert_allclose(w,slopes[None,:])
    np.testing.assert_allclose(shared,slopes)
    dy,dc=interpolated(x[...,[0,4,8]],w)
    np.testing.assert_allclose(dy,y)
    np.testing.assert_allclose(dc,curve)
    np.testing.assert_allclose(curve[0,0,0],(slopes-FRACTIONS)*[-2,-2,-2,5,5,5])
