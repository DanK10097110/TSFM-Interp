import numpy as np

from research.activation_transfer.moment.crosscoder.fresh_readout import panel, score


def test_history_eligibility_excludes_origin_day_and_counts_consecutive_missing():
    values=np.ones(panel.CUTS[-1]+12)
    t=panel.CUTS[-1];values[t-480]=np.nan;values[t-9:t-1]=np.nan;values[t]=np.nan
    n,run=panel.missing(values)
    assert n[-1]==9 and run[-1]==8
    values[t-6]=1.;values[t]=1.
    n,run=panel.missing(values)
    assert n[-1]==8 and run[-1]==7


def test_scope_specificity_removes_constant_level_and_uses_the_matching_readout():
    a=np.tile(np.arange(8,dtype=float),(2,4,1))
    a[1]+=10
    true=a+25
    true[:,2]*=-1
    wrong=-a-90
    dense=a+18
    wave={"small":a,"paired":true,"decoy":wrong,"dense":dense}
    w=np.array([[1.,1.],[2.,0.],[0.,2.]])
    out,count=score.metrics(w,wave)
    np.testing.assert_allclose(out["edited_paired_minus_decoy"],2.)
    np.testing.assert_allclose(out["edited_minus_future1_paired_minus_decoy"],2.)
    np.testing.assert_allclose(out["edited_minus_future2_paired_minus_decoy"],0.)
    np.testing.assert_allclose(out["future1_paired_minus_dense"],-2.)
    np.testing.assert_allclose(out["edited_small_centered_rms_C"],np.std(np.arange(8)))
    assert np.all(out["edited_small_rms_C"]>out["edited_small_centered_rms_C"])
    assert count["edited_minus_future1_paired_minus_decoy"]==2
