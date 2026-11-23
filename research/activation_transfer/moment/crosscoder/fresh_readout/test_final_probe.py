import numpy as np

from research.activation_transfer.moment.crosscoder.fresh_readout import final_scores


def test_replacing_readout_compares_the_same_source_target_and_decoy_per_case():
    t=np.array([[1.,-1.,0.],[0.,1.,-1.]])
    wave={"probe":{"small":t,"paired":t+10,"decoy":-t-5},
          "native":{"small":t,"paired":np.stack([t[0],-t[1]]),"decoy":t}}
    w=np.array([[1.,1.],[2.,0.],[0.,2.]])
    out,n=final_scores.contrast(w,wave,np.array([2.,5.]))
    np.testing.assert_allclose(out["final_probe_minus_native_advantage"],[3.,2.,4.])
    np.testing.assert_allclose(out["fraction_probe_improves_advantage"],1.)
    assert n["final_probe_minus_native_advantage"]==2
    # The third draw contains the 5-degree history only.
    np.testing.assert_allclose(out["probe_small_centered_RMS_C"][2],np.std(t[1])*5)
