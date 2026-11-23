import numpy as np

from research.activation_transfer.shape_transfer.full_space.leads import partial_writes


def test_hidden_lead_values_cannot_influence_the_write_or_calibration_weight():
    rng = np.random.default_rng(533)
    jac = rng.normal(size=(3,8,11))
    delta = rng.normal(size=(3,8,2))
    cal = rng.normal(size=(6,8,10))
    odd = np.array([0,2,4,6])
    fit = partial_writes(jac,delta,cal,odd)
    jac[:,1::2] *= -40
    delta[:,1::2] += 9
    cal[:,1::2] *= 17
    np.testing.assert_array_equal(fit,partial_writes(jac,delta,cal,odd))
    assert np.linalg.norm(fit)>0
