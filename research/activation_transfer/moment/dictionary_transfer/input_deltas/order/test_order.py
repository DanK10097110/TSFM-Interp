"""Known answers for the equal-dose swap and the balanced, receiver-referenced skill."""
import numpy as np

from research.activation_transfer.moment.dictionary_transfer.input_deltas.new_dates.run import equal_length
from research.activation_transfer.moment.dictionary_transfer.input_deltas.order.run import measurements, PERMUTATION


def test_swaps_hold_the_current_length_and_keep_the_other_direction():
    ramp=np.linspace(-2,2,8)
    other=ramp[PERMUTATION]
    np.testing.assert_array_equal(ramp[[0,7]],other[[0,7]])
    np.testing.assert_allclose(np.sort(ramp),np.sort(other))
    positions=np.arange(8)-3.5
    np.testing.assert_allclose(other@positions/(ramp@positions),1/6)
    directions=np.array([[1.,2.],[-1.,0.]])
    values=equal_length(directions,np.array([[0.,5.],[2.,0.]]))
    np.testing.assert_allclose(values[0],np.array([1.,2.])*np.sqrt(5))
    np.testing.assert_allclose(values[1],[-2.,0.])


def test_balanced_pair_weights_observable_edits_equally():
    # The two reference waveforms differ tenfold in magnitude.
    # Pooling their energies before scoring would produce a different contrast.
    x=np.arange(16,dtype=float)-7.5
    x=np.broadcast_to(x,(6,16))
    waves=dict(ramp_reference=x,ramp_correct=x,ramp_swapped=0*x,
               permuted_reference=10*x,permuted_correct=5*x,permuted_swapped=-10*x)
    result=measurements(waves,["a","b"],["j","f","m"])
    for r in result.values():
        np.testing.assert_allclose(r["ramp"]["correct_minus_swapped"],[1.]*3)
        np.testing.assert_allclose(r["permuted"]["correct_minus_swapped"],[3.75]*3)
        np.testing.assert_allclose(r["mean_correct_minus_swapped"],[2.375]*3)
