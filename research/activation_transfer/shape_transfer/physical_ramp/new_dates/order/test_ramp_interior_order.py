import numpy as np
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.order.study import reverse_interior
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.order.score import balanced_advantage

def test_reverses_only_interior_with_identical_edit_statistics():
    original=np.broadcast_to(np.arange(480,dtype='f4')/40,(2,480)).copy()
    before=original.copy()
    alt=reverse_interior(original)
    np.testing.assert_equal(original,before)
    np.testing.assert_equal(alt[:,:-8],original[:,:-8])
    linear=np.linspace(-2.,2.,8)
    delta=alt[:,-8:]-original[:,-8:]
    np.testing.assert_allclose(delta[0,1:-1],linear[-2:0:-1],atol=5e-7)
    np.testing.assert_allclose(delta[:,[0,-1]],np.tile([-2.,2.],(2,1)))
    np.testing.assert_allclose(np.sort(delta[0]),linear,atol=5e-7)
    np.testing.assert_allclose(delta.mean(1),0.,atol=1e-7)
    np.testing.assert_allclose(delta.var(1),linear.var(),atol=5e-7)

def test_balanced_advantage_compares_the_same_target_for_both_terms():
    measured={'AA':np.array([.7, .8]), 'AB':np.array([.4,.7]),
              'BA':np.array([-.5,.4]),'BB':np.array([.2,.1])}
    np.testing.assert_allclose(balanced_advantage(measured),[.5,-.1],atol=1e-12)
