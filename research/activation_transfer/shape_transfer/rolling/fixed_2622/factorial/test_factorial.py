import numpy as np

from research.activation_transfer.shape_transfer.rolling.fixed_2622.factorial.score import calendar_contrasts


def test_interaction_is_change_in_halfday_effect_from_tail_to_prior_week():
    effects = calendar_contrasts(np.array([[3., 1., 8., 12.], [3., 1., 8., 6.]]))
    np.testing.assert_array_equal(effects["interaction"], [6., 0.])
    np.testing.assert_array_equal(effects["halfday_main"], [1., -2.])
    np.testing.assert_array_equal(effects["week_main"], [8., 5.])
