import numpy as np

from research.activation_transfer.shape_transfer.interactions.conditional import mixed


def test_interior_mixed_difference_cancel_additive_and_recover_bilinear_interaction():
    def waveform(a,b):
        return 7.+np.array([3.,4.])*a**2+b**3*np.array([-2.,5.])+np.array([-8.,2.])*a*b
    actual=mixed(*(waveform(a,b) for a,b in ((.25,.25),(.25,.75),(.75,.25),(.75,.75))))
    np.testing.assert_array_equal(actual,[-8.,2.])
