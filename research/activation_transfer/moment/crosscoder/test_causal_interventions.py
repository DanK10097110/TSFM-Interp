import numpy as np
from research.activation_transfer.moment.crosscoder.causal_scores import crossed_sum,normalized_shape,score_opportunities,matching_mask_pairing
from research.activation_transfer.moment.crosscoder.interventions import pick_checks


def test_closed_gate_wrong_write_counts_as_error_and_atom_weight_is_equal():
    native=np.zeros((4,2,2))
    native[[0,1],[0,1]]=[[-1.,1.],[-10.,10.]]
    proposed=native.copy()
    proposed[3,0]=[-2.,2.]
    on=(native!=0).any(axis=-1)
    sd=np.array([2.,2.,2.,4.])
    original=normalized_shape(native,sd)
    predicted=normalized_shape(proposed,sd)
    total=lambda x:x.sum(axis=0,keepdims=True)
    s,o,h=score_opportunities(original,predicted,on,total)
    # Equal normalized signal/error for feature 0; the louder second feature gets one vote.
    np.testing.assert_allclose(s,[[0.,1.]])
    np.testing.assert_allclose([s.mean(),o.mean(),h.mean()],[.5,1.,.5])


def test_crossed_resampling_repeats_rows_and_dates_together_for_each_atom():
    data=np.arange(3*2*4).reshape(6,4)
    sites=np.array([[1,1,1],[0,2,1]])
    dates=np.array([[1,1],[2,0]])
    expected=[]
    for s,d in zip(sites,dates):
        x=np.repeat(np.repeat(data.reshape(3,2,4),s,axis=0),d,axis=1)
        expected.append(x.sum(axis=(0,1)))
    np.testing.assert_equal(crossed_sum(data,sites,dates),expected)


def test_native_replay_samples_open_closed_and_false_writes():
    scores=np.tile([0.,1.,2.,0.],(4,1)).T
    gates=np.tile([0.,0.,1.,1.],(4,1)).T
    p,q=pick_checks(scores,gates).T
    found=set(zip(scores[p,q]>0,gates[p,q]>0))
    assert found=={(False,False),(False,True),(True,False),(True,True)}


def test_pairing_comparison_holds_target_view_dropout_fixed():
    values={'joint':np.full(6,-.2),'masked':np.full(6,.5),'shuffled':np.full(6,.55)}
    controlled=matching_mask_pairing(values)
    np.testing.assert_allclose(controlled['value'],-.05)
    np.testing.assert_allclose(controlled['crossed95'],[-.05,-.05])
