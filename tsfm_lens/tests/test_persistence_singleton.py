"""A planted position-biased native mock must be isolated into real singleton calls."""
import numpy as np
from experiments.feature_search.persistence_singleton import singleton_class


def test_singleton_wrapper_keeps_real_calls_and_replacement_rows():
    class FakeNative:
        def __init__(self):
            self.calls = []

        def predict(self, x, replacement=None):
            self.calls.append((x.copy(), None if replacement is None else replacement.copy()))
            value = x+np.arange(len(x))[:, None]
            return value, x if replacement is None else replacement

    native = singleton_class(FakeNative)()
    x = np.ones((2, 3))
    replacement = np.array([[7., 7., 7.], [9., 9., 9.]])
    prediction, state = native.predict(x, replacement)
    np.testing.assert_array_equal(prediction, x)
    np.testing.assert_array_equal(state, replacement)
    assert len(native.calls) == 2
    assert all(len(call[0]) == 1 for call in native.calls)
    np.testing.assert_array_equal(native.calls[1][1], replacement[1:2])
