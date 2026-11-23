import numpy as np

from research.activation_transfer.shape_transfer.feature_groups.evaluate import choose
from research.activation_transfer.shape_transfer.feature_groups.fit import greedy


def test_greedy_adds_complementary_atom_and_stops_at_saturation():
    gains = np.array([[10., 9., 0.], [0., 0., 3.]])
    np.testing.assert_array_equal(greedy(gains), [0, 2])


def test_signed_selector_cannot_use_an_unlisted_or_unavailable_atom():
    active = np.array([100, 130, 200])
    gain = np.array([.5, 1., 20.])
    assert choose(gain, active, [130, 101]) == 1
    assert choose(gain, active, [101]) == -1
    assert choose(gain, active, None) == 2
