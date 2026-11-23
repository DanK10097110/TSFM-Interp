import numpy as np
from research.activation_transfer.shape_transfer.rolling.data import LAGS, windows
from research.activation_transfer.shape_transfer.rolling.intervene import carried


def test_origins_use_observed_overlap_and_anchor_only_scaling():
    history = np.arange(600, dtype=float)
    starts, mu, sd, fixed, _ = windows(history)
    np.testing.assert_array_equal(starts, [80,81,84,96,112])
    assert mu == history[80:560].mean() and sd == history[80:560].std()
    for k,lag in enumerate(LAGS[1:],1):
        np.testing.assert_array_equal(fixed[0,lag:], fixed[k,:-lag])
    np.testing.assert_allclose(fixed[-1,-1], (history[-9]-mu)/sd, rtol=1e-6)


def test_blind_old_write_and_current_bound_are_distinct_after_dropout():
    assert carried(17, .8, False, 0.) == ((17,.8),(-1,0.))
    assert carried(17, -.8, True, .3) == ((17,-.8),(17,-.3))
