import numpy as np

from research.activation_transfer.shape_transfer.response_subspace.run import rotated_remainder


def test_surrogate_preserves_all_norms_and_calibration_mean_but_changes_unseen_readout():
    rng=np.random.default_rng(2841)
    basis=np.linalg.qr(rng.normal(size=(11,3)))[0].T
    dirs=rng.normal(size=(5,11))
    new=rotated_remainder(dirs,basis,rng)
    np.testing.assert_allclose(new@basis.T,dirs@basis.T,atol=1e-11)
    np.testing.assert_allclose(new@new.T,dirs@dirs.T,atol=1e-11)
    assert np.linalg.norm(new-dirs)>1.
    held=rng.normal(size=11)
    assert np.linalg.norm((new-dirs)@held)>1.
