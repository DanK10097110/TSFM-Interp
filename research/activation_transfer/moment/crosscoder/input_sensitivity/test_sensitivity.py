import numpy as np

from research.activation_transfer.moment.crosscoder.input_sensitivity.capture import pulses
from research.activation_transfer.moment.crosscoder.input_sensitivity.score import comparisons,summary


def test_physical_plus_minus_observed_patch_has_the_specified_amplitude():
    x=np.tile(np.arange(480,dtype='f4')/50,(2,1))
    edited,dose=pulses(x,.15)
    diff=(edited[0,:,0,-8:]-edited[0,:,1,-8:])/(2*dose[0])
    np.testing.assert_allclose(diff@diff.T/8,np.eye(8),atol=2e-6)
    np.testing.assert_allclose(diff[0],[2,-2,0,0,0,0,0,0],atol=2e-6)
    np.testing.assert_allclose(diff[7],1,atol=2e-6)
    np.testing.assert_equal(edited[0,7,1,:-8],x[0,:-8])


def test_inner_product_over_shape_basis_preserves_signed_physical_orientations():
    a=np.array([[[1,1,1,1,1,1,1,3],[1,-1,0,0,0,0,0,2],[-1,-1,-1,-1,-1,-1,-1,-4]]],dtype=float)
    value=comparisons(a,a)
    np.testing.assert_allclose(value['paired'],[[1,1,1]])
    np.testing.assert_allclose(value['other'],[[-.5,0,-.5]],atol=1e-15)
    np.testing.assert_allclose(value['other_level_sign'],[[.5,.5,0]])


def test_equal_feature_statistic_corrects_repeated_history_events():
    atoms=np.array([[0,1],[0,1],[0,2]])
    score=np.array([[0.,0.],[0.,0.],[0.,1.]])
    stat=summary({'margin':score},atoms)['statistics']['margin']
    assert stat['event_mean']==1/6
    assert stat['equal_atom']==1/3
