import numpy as np
from research.activation_transfer.shape_transfer.fresh_cohort.summarize import eligible, score_set


def test_conditional_subset_keeps_eight_firings_and_excludes_seven_and_an_inactive_atom():
    mask=np.array([[i<8,i<7,False] for i in range(12)])
    n,keep=eligible(mask,8)
    np.testing.assert_array_equal(n,[8,7,0])
    np.testing.assert_array_equal(keep,[True,False,False])
    atoms=np.array([412,772,155])
    native=mask[:,:,None]*np.arange(1.,9.)[None,None,:]
    pred=native.copy()
    pred[:,1]=-10*native[:,1]
    s=score_set(mask,atoms,native,{"adaptive":pred},keep,np.ones((6,12)))
    assert s["atom_ids"]==[412] and s["n_firings"]==8
    assert s["shape"]["methods"]["adaptive"]["skill"]["value"]==1
