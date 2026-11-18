"""Native population-scale and no-fit normalization side-channel checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1]/'experiments/feature_search/normalization_audit.py'
spec = importlib.util.spec_from_file_location('normalization_audit', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_native_population_stats_and_zero_scale_distinction():
    """Running patch moments use all observed values, not sample correction."""
    values = np.arange(128, dtype=np.float32)[None, :]
    for surface in ['chronos_bolt', 'timesfm_2p5_decode']:
        mu, sigma = audit.normalization_state(values, surface)
        np.testing.assert_allclose(mu, values.mean(axis=1,keepdims=True), atol=1e-6)
        np.testing.assert_allclose(sigma, values.std(axis=1,keepdims=True,ddof=0), atol=1e-5)
        assert not np.allclose(sigma, values.std(axis=1,keepdims=True,ddof=1))
    _, bolt_sigma = audit.normalization_state(np.ones((2,128)), 'chronos_bolt')
    _, timesfm_sigma = audit.normalization_state(np.ones((2,128)), 'timesfm_2p5_decode')
    np.testing.assert_allclose(bolt_sigma, 1e-5)
    np.testing.assert_array_equal(timesfm_sigma, 0)


def test_fixed_affine_sidechannel_explains_known_swap():
    """Different native output units cause hidden-donor forecast mismatch."""
    latent = np.array([[1.,-2.,3.],[-3.,1.,2.]])
    donor_mu, donor_sigma = np.array([[7.],[-2.]]), np.array([[4.],[2.]])
    receiver_mu, receiver_sigma = np.array([[1.],[8.]]), np.array([[.5],[3.]])
    donor = latent*donor_sigma+donor_mu
    actual = latent*receiver_sigma+receiver_mu
    effect = audit.compare_sidechannel(donor, actual, donor_mu, donor_sigma, receiver_mu, receiver_sigma)
    assert effect['raw_donor_discrepancy_mse'] > 10
    assert effect['affine_corrected_residual_mse'] == 0
    assert effect['discrepancy_fraction_explained'] == 1
    assert effect['fitted_parameters'] == 0


def test_unknown_or_unidentifiable_surfaces_fail_loudly():
    """Do not infer normalization rules for unsupported adapters or zero scales."""
    with pytest.raises(ValueError, match='Unsupported'):
        audit.normalization_state(np.ones((1,128)), 'unknown')
    with pytest.raises(ValueError, match='horizon'):
        audit.normalization_state(np.ones((1,128)), 'timesfm_2p5_decode', horizon=256)
    with pytest.raises(ValueError, match='unidentifiable'):
        audit.transplant_output_normalizer(np.ones((1,32)), np.zeros((1,1)), np.zeros((1,1)), np.zeros((1,1)), np.ones((1,1)))
