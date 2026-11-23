import numpy as np
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.capacity import optimum


def test_spherical_bound_prefers_the_sensitive_axis_over_radial_shrinkage():
    a=np.diag([1.,3.])
    y=np.array([1.,1.])
    norm,realized,rank=optimum(a,y,.4)
    assert rank==2
    np.testing.assert_allclose(norm,np.sqrt(1+1/9))
    v=np.linalg.solve(a,realized)
    np.testing.assert_allclose(np.linalg.norm(v),.4,atol=1e-12)
    # KKT stationarity gives the same multiplier on both coordinates.
    lam=a.T@(y-realized)/v
    np.testing.assert_allclose(lam[0],lam[1],atol=1e-10)
    assert np.linalg.norm(y-realized)<np.linalg.norm(y-y*.4/norm)
