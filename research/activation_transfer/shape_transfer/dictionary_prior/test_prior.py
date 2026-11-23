import numpy as np

from research.activation_transfer.shape_transfer.dictionary_prior.run import blind_match,correct,inverse


def test_equal_levels_distinguished_by_shape_and_fraction():
    waves=np.array([[1,0,-1,0]*2,[0,1,0,-1]*2,[1,-1,0,0]*2],dtype=float)+2.
    rank,_,_,levels=blind_match(waves,waves[2],3.)
    assert levels[0]==0 and rank[0]==2
    _,rank,frac,_=blind_match(waves,.25*waves[1],3.)
    assert rank[0]==1 and np.isclose(frac[1],.25)


def test_correction_preserves_prior_invisible_component_and_fits_early_residual():
    jac=np.zeros((8,10));jac[:,:8]=np.eye(8)
    a,lift=inverse(jac,1.)
    prior=np.arange(10,dtype=float)/8
    goal=np.full(8,-2.)
    result=correct(prior,a,lift,goal)
    np.testing.assert_array_equal(result[-2:],prior[-2:])
    np.testing.assert_allclose(a@result,goal,atol=.003)
