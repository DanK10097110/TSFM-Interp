import numpy as np

from research.activation_transfer.shape_transfer.interactions.run import common_scale


def test_factorial_cap_includes_both_singletons_and_the_joint_vector():
    a=np.array([[3.,4.],[3.,0.],[1.,0.]])
    b=np.array([[-3.,-4.],[3.,0.],[0.,1.]])
    scale=common_scale(a,b,2.)
    np.testing.assert_allclose(scale,[2/5,1/3,1])
    # A linear recipient must have zero apparent interaction, including cap-active and cancellation cases.
    readout=np.array([[2.,-1.],[1.,3.]])
    np.testing.assert_allclose(((a+b)*scale[:,None])@readout,(a*scale[:,None])@readout+(b*scale[:,None])@readout)
