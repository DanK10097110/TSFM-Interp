import numpy as np

from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape, solve_adaptive, solve_fixed


def test_same_donor_write_can_require_opposite_recipient_axes_on_different_histories():
    # First coordinate is a shared level. Second is temporal contrast with a history-specific sign.
    target = np.array([[[1., 1., 0.], [1., -1., 0.]], [[1., -1., 0.], [1., 1., 0.]]])
    response = np.broadcast_to(np.eye(2), (2, 2, 2))
    per_item = solve_adaptive(target, response)
    fixed = solve_fixed(target, response)
    np.testing.assert_allclose(target @ per_item, response / 1.001, atol=1e-10)
    np.testing.assert_allclose(target @ fixed, np.full((2, 2, 2), .5 / 1.001), atol=1e-10)
    weighted = boost_shape(target, 4.)
    np.testing.assert_allclose(weighted.mean(axis=1), target.mean(axis=1))
    np.testing.assert_allclose(weighted[:, 0]-weighted[:, 1], 4*(target[:, 0]-target[:, 1]))
