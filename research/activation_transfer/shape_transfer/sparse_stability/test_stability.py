import numpy as np

from research.activation_transfer.shape_transfer.sparse_stability.score import partition


def test_distinguishes_exit_entry_and_reranking_among_survivors():
    # Three matched histories, each with one continuously firing source atom.
    before = np.array([[3], [3], [3]])
    after = np.array([[9], [9], [9]])
    clean = np.array([[3, 4], [3, 4], [3, 9]])
    perturbed = np.array([[9, 4], [3, 9], [3, 9]])
    result = partition(np.ones_like(before, dtype=bool), before, after, clean, perturbed)
    np.testing.assert_array_equal(result["lost"].ravel(), [1, 0, 0])
    np.testing.assert_array_equal(result["surviving_to_new"].ravel(), [0, 1, 0])
    np.testing.assert_array_equal(result["both_surviving"].ravel(), [0, 0, 1])
