import numpy as np

from research.activation_transfer.shape_transfer.level_geometry.run import random_lifts,remove_dc,spectrum
from research.activation_transfer.shape_transfer.level_geometry.score import exact_zero_writes,item_metrics


def test_correction_cancels_the_measured_offset_at_fixed_norm():
    basis=np.eye(4)[:2];g=np.array([1.,2.]);raw=np.array([[1.,1.,3.,-2.],[-3.,1.,2.,4.]])
    # True level slope has an off-PC component. A projection-only implementation will miss it.
    full_g=np.array([1.,2.,.5,0.]);measured=raw@full_g
    corrected,adjustment,residual=remove_dc(raw,measured,g,basis)
    np.testing.assert_allclose(corrected@full_g,0.,atol=2e-15)
    np.testing.assert_allclose(np.linalg.norm(corrected,axis=1),np.linalg.norm(raw,axis=1))
    np.testing.assert_allclose(adjustment@np.array([-2.,1.,0.,0.]),0.,atol=1e-14)
    np.testing.assert_array_equal(adjustment[:,2:],0.)
    np.testing.assert_allclose(residual,0.,atol=1e-14)


def test_spectrum_separates_isotropic_energy_from_shared_response():
    jac=np.array([[[2.,1.],[2.,-1.]],[[2.,-1.],[2.,1.]]])
    s=spectrum(jac)
    np.testing.assert_allclose(s['dc_fraction_all'],.8)
    np.testing.assert_allclose(s['shared_dc_gradient_fraction'],1.)
    np.testing.assert_allclose(s['shared_shape_response_fraction'],0.)
    np.testing.assert_allclose(s['dc_fraction_mean'],1.)


def test_random_control_matches_budget_and_uses_fit_level_sign():
    g=np.array([1.,-3.,2.]);basis=np.eye(5)[:3]
    r,s=random_lifts(g,basis,4.)
    np.testing.assert_allclose(np.linalg.norm(r,axis=1),4.)
    assert (r[:,:3]@g>=0).all()
    assert len(np.unique(s))==2
    np.testing.assert_array_equal(r[:,3:],0.)


def test_decomposition_distinguishes_wrong_waveform_from_level_offset():
    target=np.array([-1.,1.]);y=np.array([[2.,4.],[1.,-1.]])
    d=item_metrics(y,target,np.broadcast_to(target,y.shape))
    np.testing.assert_allclose(d['dc_error'],[9.,0.]);np.testing.assert_allclose(d['shape_error'],[0.,4.])
    np.testing.assert_allclose(d['skill'],[-8.,-3.]);np.testing.assert_allclose(d['shape_cosine_to_request'],[1.,-1.])


def test_zero_scalar_replays_are_exactly_zero_for_directional_metrics():
    values=np.array([[[1e-6,-2e-6],[3.,-2.]]])
    fixed,floor=exact_zero_writes(values,np.array([0.,.5]))
    assert floor==2e-6
    np.testing.assert_array_equal(fixed[:,0],0.);np.testing.assert_array_equal(fixed[:,1],values[:,1])
