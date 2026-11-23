import numpy as np
from scipy.special import ndtri

from research.activation_transfer.shape_transfer.quantiles.five_interp import HIDDEN,SLOTS,W,between,from_linear
from research.activation_transfer.shape_transfer.quantiles.normal_score import five_components


def test_constant_normal_scale_and_asymmetric_perturbation_roundtrip():
    z=ndtri(np.arange(1,10)/10)
    # An exact location-scale displacement has no departure at the intervening quantiles.
    y=-.8+1.7*z
    np.testing.assert_allclose(between(y[SLOTS],W),y[HIDDEN])
    np.testing.assert_allclose(five_components(y[SLOTS])[-2:],0.,atol=1e-13)
    # Both tails, the asymmetric midpoint and the departures vary independently.
    a=np.stack((y[SLOTS],np.array([-.4,.3,1.1,1.9,3.6])))
    c=five_components(a)
    np.testing.assert_allclose(from_linear(c),a)
    assert abs(c[1,-2])>.01 and abs(c[1,-1])>.01
