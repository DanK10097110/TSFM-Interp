import numpy as np

from research.activation_transfer.shape_transfer.quantiles.geometry import components
from research.activation_transfer.shape_transfer.quantiles.intervene import channels,stack,weights


def test_tail_swap_changes_only_width_and_calibration_balances_distinct_channels():
    # Median and midrange differ; a mean-of-three or reversed width would change the answer.
    q=np.array([[-4.,1.,8.],[3.,4.,7.]])
    np.testing.assert_array_equal(components(q),[[1.,6.,1.],[4.,2.,1.]])
    np.testing.assert_array_equal(components(q[:,::-1]),components(q)*[1.,-1.,1.])
    rng=np.random.default_rng(843)
    a=rng.normal(size=(6,8,3,11))
    j=channels(a)
    np.testing.assert_allclose(j[:,1],(a[:,:,2]-a[:,:,0])/2)
    w,g=weights(j)
    weighted=stack(j,w,g).reshape(j.shape)
    energy=np.mean(weighted*weighted,axis=(0,2,3))
    np.testing.assert_allclose(energy,np.full(3,energy[0]),rtol=1e-12)
