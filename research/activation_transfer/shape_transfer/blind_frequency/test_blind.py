import numpy as np
from scipy.fft import dct,idct

from research.activation_transfer.shape_transfer.blind_frequency.geometry import bands,blinded,permute_histories,within_model
from research.activation_transfer.shape_transfer.blind_frequency.score import scores


def test_blind_mask_keeps_the_early_horizon_and_removes_only_fast_modes():
    x=np.random.default_rng(41).normal(size=(3,16,4))
    after=blinded(x)
    np.testing.assert_array_equal(after[:,:8],x[:,:8])
    original=dct(x[:,8:],axis=1,norm="ortho")
    got=dct(after[:,8:],axis=1,norm="ortho")
    np.testing.assert_allclose(got[:,:3],original[:,:3],atol=2e-14)
    np.testing.assert_allclose(got[:,3:],0.,atol=2e-14)
    sample=x.transpose(0,2,1)
    parts=bands(sample)
    np.testing.assert_allclose((parts["slow"]**2).sum(-1)+(parts["fast"]**2).sum(-1),
                               ((sample[...,8:]-sample[...,8:].mean(-1,keepdims=True))**2).sum(-1))
    np.testing.assert_allclose(parts["fast"],original[:,3:].transpose(0,2,1))


def test_own_history_regression_restores_channel_means_and_temporal_orientation():
    rng=np.random.default_rng(6)
    early=rng.normal(size=(100,8,5))+np.arange(5)[None,None,:]*7
    late=2*np.roll(early,3,axis=1)+np.arange(8)[None,:,None]
    cal=np.concatenate((early,late),axis=1)
    current=early[:4]+rng.normal(size=(4,8,5))
    np.testing.assert_allclose(within_model(cal,current),2*np.roll(current,3,axis=1)+np.arange(8)[None,:,None],atol=.025)


def test_controls_derange_entire_distinct_histories():
    idx=permute_histories(96)
    np.testing.assert_array_equal(np.sort(idx),np.arange(96))
    assert np.all(idx!=np.arange(96))
    np.testing.assert_array_equal(permute_histories(96),idx)


def test_crossed_bootstrap_draws_the_same_features_across_all_histories():
    # One feature is perfect; the other has four times the error of a zero forecast.
    ref=np.array([[[1.],[2.]],[[1.],[1.]]])
    eff=ref*np.array([1.,-1.])[None,:,None]
    m=np.ones((2,2),bool)
    series=np.array([[1,1],[2,0]])
    features=np.array([[2,0],[0,2]])
    point,sboot,cboot,_=scores(eff,ref,m,series,features)
    np.testing.assert_allclose(point,-1.)
    np.testing.assert_allclose(sboot,[-1,-1])
    np.testing.assert_allclose(cboot,[1,-3])
