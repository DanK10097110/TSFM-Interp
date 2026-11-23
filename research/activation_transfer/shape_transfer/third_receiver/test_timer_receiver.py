"""Check that the Timer comparison alters only the prescribed donor/lead information."""
import numpy as np

from research.activation_transfer.shape_transfer.third_receiver.run import directions, requests


def test_mean_donor_uses_calibration_only_and_keeps_eight_leads():
    rng = np.random.default_rng(918)
    source = rng.normal(size=(5,8,10))
    source[2:] += 30
    decoder = rng.normal(size=(10,2))
    cal = np.array([True, True, False, False, False])
    local, mean, weight = requests(source, decoder, cal)
    np.testing.assert_allclose(local, source[2:] @ decoder)
    np.testing.assert_allclose(mean, source[:2].mean(axis=0) @ decoder)
    np.testing.assert_allclose(weight, np.linalg.norm(np.repeat(source[:2].mean(1,keepdims=True),8,axis=1)) /
                               np.linalg.norm(source[:2]-source[:2].mean(1,keepdims=True)))
    recipient = np.tile(np.eye(8), (3,1,1))
    writes = directions('mean_donor', recipient, local, mean, 1.)
    np.testing.assert_allclose(writes*1.001, np.tile(mean, (3,1,1)))


def test_wrong_lead_control_preserves_level_and_permutates_shape():
    source = np.arange(8, dtype=float).reshape(1,8,1)
    target = np.eye(8)[None]
    raw = directions('adaptive', target, source, 10*source[0], 1.)
    wrong = directions('wrong_leads', target, source, 10*source[0], 1.)
    np.testing.assert_allclose(raw*1.001, source)
    np.testing.assert_allclose(wrong.mean(axis=1), raw.mean(axis=1))
    expected = np.array([4,2,5,0,7,6,1,3]).reshape(1,8,1)
    np.testing.assert_allclose(wrong*1.001, expected)
