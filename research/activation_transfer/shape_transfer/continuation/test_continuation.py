import numpy as np

from research.activation_transfer.shape_transfer.continuation.run import corrected, wrong_prior


def test_prior_changes_nullspace_and_corrects_its_visible_residual():
    # Two recipients have different sensitivities to their first coordinate.
    design = np.array([[[1.,0.]], [[2.,0.]]])
    donor = np.array([[[4.]], [[-2.]]])
    prior = np.array([[1.],[3.]])
    actual = corrected(design,donor,prior)
    np.testing.assert_allclose(actual[:,0,0],[1+3/1.001,1-2/1.001])
    np.testing.assert_array_equal(actual[:,1,0],[3.,3.])


def test_wrong_prior_changes_identity_while_preserving_each_correct_norm():
    prior = np.array([[3.,0.,-5.],[0.,4.,0.]])
    got = wrong_prior(prior,np.array([1,2,0]))
    np.testing.assert_allclose(got,[[0.,-4.,5.],[3.,0.,0.]])
