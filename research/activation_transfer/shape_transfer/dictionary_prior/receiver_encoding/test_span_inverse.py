import numpy as np

from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.span_inverse import constrained


def test_active_decoder_basis_can_match_goal_when_euclidean_projection_loses_it():
    j=np.array([[[1.,0.,1.],[0.,1.,-1.]]])
    v=np.array([[0.,0.,1.]])
    q=np.array([[[1.,0.,0.],[0.,1.,0.]]])
    # The original direction is perpendicular to every offered decoder axis.
    np.testing.assert_allclose(q@v[:,:,None],0.)
    actual=constrained(j,q,v,1.)
    np.testing.assert_allclose(actual,[[1/1.001,-1/1.001,0.]],atol=1e-10)
    np.testing.assert_allclose(j@actual[:,:,None],(j@v[:,:,None])/1.001,atol=1e-10)
