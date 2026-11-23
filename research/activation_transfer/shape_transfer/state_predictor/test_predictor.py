import numpy as np

from research.activation_transfer.shape_transfer.state_predictor.fit import choice,rank_neighbours,response


def test_support_neighbor_ties_and_order_match_symmetric_difference():
    train=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,1],[1,1,0,0]],bool)
    query=np.array([[1,1,0,0],[0,0,0,0]],bool)
    order,d=rank_neighbours(train,query,sparse=True)
    np.testing.assert_array_equal(d,[[1,1,4,0],[1,1,2,2]])
    np.testing.assert_array_equal(order,[[3,0,1,2],[0,1,2,3]])


def test_shrinkage_uses_known_generic_validation_truth_and_clamps_negative_signal():
    # A good nearest retrieval explains exactly half the calibrated deviation.
    y=np.array([-2.,0.,2.])[:,None]
    val=np.array([[-1.],[1.]])
    n=np.array([[0,1],[2,1]])
    true_energy=np.mean(val**2)
    got=choice(y@y.T,val@y.T,true_energy,n,grid=(1,))
    np.testing.assert_allclose([got["mix"],got["relative_generic_cal_loss"]],[.5,0.])
    bad=choice(y@y.T,-val@y.T,true_energy,n,grid=(1,))
    assert bad["mix"]==0


def test_generic_neighbor_average_predicts_new_decoder_columns_linearly():
    y=np.array([[[1.,2.]],[[3.,6.]],[[8.,1.]]])
    idx=np.array([[0,1],[2,1]])
    settings=dict(k=2,mix=.5)
    expected=np.array([[[3.,3.5]],[[4.75,3.25]]])
    np.testing.assert_allclose(response(y,idx,settings),expected)
    dec=np.array([[2.,-.5],[1.,3.]])
    np.testing.assert_allclose(response(y@dec,idx,settings),expected@dec)


def test_fractional_boundary_shares_remaining_mass_and_is_permutation_invariant():
    from research.activation_transfer.shape_transfer.state_predictor.ties import fractional
    dist=np.array([[0.,1.,1.,1.,3.],[4.,4.,4.,4.,4.]])
    expected=np.array([[.5,1/6,1/6,1/6,0.],[.2,.2,.2,.2,.2]])
    np.testing.assert_allclose(fractional(dist,2),expected)
    perm=[4,2,0,3,1]
    np.testing.assert_allclose(fractional(dist[:,perm],2),expected[:,perm])
