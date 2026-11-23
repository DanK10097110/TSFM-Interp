import numpy as np

from research.activation_transfer.forecast_requests.units.analyze import mean_sham
from research.activation_transfer.forecast_requests.units.run import normalize_dose


def test_native_norm_changes_size_and_preserves_mapped_and_sham_directions():
    mapped=np.array([[.5,0.],[0.,2.]])
    rotated=mapped[:,::-1]
    shams=np.stack([rotated,-mapped,-rotated,mapped],axis=1)
    native=np.array([[0.,3.],[.25,0.]])
    unit,ratio=normalize_dose(mapped,shams,native)
    original=np.concatenate([mapped[:,None],shams],axis=1)
    np.testing.assert_allclose(ratio,np.repeat([[6.],[.125]],5,axis=1))
    np.testing.assert_allclose(unit,original*ratio[:,:,None])


def test_sham_skill_averages_errors_before_combining_directions():
    target=np.array([-1.,1.,-1.,1.])
    effects=np.stack([target,-target])[None]
    scored=mean_sham(effects,target)
    np.testing.assert_allclose(scored['skill'],-1.)
    np.testing.assert_allclose(scored['gain'],0.)
