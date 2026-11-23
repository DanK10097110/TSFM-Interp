import numpy as np
from research.activation_transfer.replication.functional import fit_feature
from research.activation_transfer.replication.experiment import dose_scale, derangements


def test_signed_calibration_excludes_zero_and_clips():
    x = np.ones((5, 3, 4))
    x[:, 0] = 0
    y = x*np.array([0, -2, 20])[None, :, None]
    np.testing.assert_allclose(dose_scale(x, y), [0, -2, 8])
    assert all(np.all(p != np.arange(16)) for p in derangements(16))


def test_jacobian_write_uses_negative_native_amplitude_and_recovers_shape():
    rng = np.random.default_rng(8123)
    j = rng.normal(size=(10, 8, 4))
    z = rng.uniform(.2, 2, size=10)
    v = np.array([.1, -.3, .25, .2])
    truth = -z[:, None]*(j@v)
    fitted, _ = fit_feature(j, z, truth, radius=20)
    np.testing.assert_allclose(fitted, v, atol=.012)
    z[:] = 0
    fitted, template = fit_feature(j, z, truth, radius=20)
    assert np.all(fitted==0) and np.all(template==0)


def test_ridge_path_is_a_displacement_map_at_the_declared_scale():
    from research.activation_transfer.replication.penalty import PENALTIES, spectral_map
    rng=np.random.default_rng(90)
    x=rng.normal(size=(40, 5))
    y=rng.normal(size=(40, 7))
    q=rng.normal(size=(3, 5))
    pred=spectral_map(x+2, y-7, q)
    xc=x-x.mean(axis=0)
    yc=y-y.mean(axis=0)
    top=np.linalg.svd(xc,compute_uv=False)[0]**2
    truth=np.stack([q@np.linalg.solve(xc.T@xc+p*top*np.eye(5),xc.T@yc) for p in PENALTIES])
    np.testing.assert_allclose(pred, truth, atol=1e-8)
