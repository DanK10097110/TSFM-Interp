import numpy as np

from research.activation_transfer.shape_transfer.sparse_writes.run import pursuit, removal_gains


def test_literal_ablation_and_bounded_signed_pursuit_choose_different_atoms():
    profiles = np.array([[1.,0.],[-1.,0.],[0.,1.]])
    strengths = np.array([.7,4.,2.])
    goal = np.array([1.,.3])
    # Doubling the first atom is useful, but it cannot be fully removed to make this rise.
    assert removal_gains(profiles,strengths,goal).max()<0
    idx,changes = pursuit(profiles,strengths,goal,3)
    np.testing.assert_array_equal(idx[:2],[1,2])
    np.testing.assert_allclose(changes[:2],[-1.,.3])
    idx,changes = pursuit(profiles[[0,2]],strengths[[0,2]],goal,2)
    np.testing.assert_array_equal(idx,[0,1])
    np.testing.assert_allclose(changes,[.7,.3])
