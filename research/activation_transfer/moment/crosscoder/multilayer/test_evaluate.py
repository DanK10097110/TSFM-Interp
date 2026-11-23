import numpy as np

from research.activation_transfer.moment.crosscoder.multilayer.evaluate import pairs


def test_family_contrast_uses_both_seed_metrics_with_shared_draws():
    draws={'aligned1':np.array([1.,3.,5.]), 'aligned2':np.array([5.,0.,1.]),
           'shuffled_depth1':np.array([.5,1.,2.]), 'shuffled_depth2':np.array([1.5,1.,0.]),
           'masked1':np.zeros(3), 'masked2':np.zeros(3), 'raw':np.ones(3)}
    contrasts=pairs(draws)
    np.testing.assert_allclose(contrasts['aligned_mean_minus_shuffled_depth_mean'],[2.,.5,2.])
    np.testing.assert_allclose(contrasts['aligned2_minus_shuffled_depth2'],[3.5,-1.,1.])
    np.testing.assert_allclose(contrasts['aligned_mean_minus_raw'],[2.,.5,2.])
