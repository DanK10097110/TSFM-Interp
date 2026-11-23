import numpy as np

from research.activation_transfer.shape_transfer.self_transfer.run import decompose


def test_missing_hidden_component_can_be_the_entire_continuation():
    rng=np.random.default_rng(13)
    basis=np.linalg.qr(rng.normal(size=(12,12)))[0].T
    # Eight early outputs in a rotated space, a distinct late readout, and an irrelevant direction.
    operators=np.stack((basis[:8],np.diag(np.arange(1,9))@basis[:8]))
    dec=np.stack((2*basis[2]+3*basis[9], -basis[4]+4*basis[10]),axis=1)
    seen,missing=decompose(operators,dec)
    expected=np.stack((2*basis[2],-basis[4]),axis=1)
    np.testing.assert_allclose(seen,np.stack((expected,expected)),atol=1e-12)
    np.testing.assert_allclose(operators@missing,0.,atol=1e-12)
    np.testing.assert_allclose(basis[[9,10]]@missing[0],np.diag([3.,4.]),atol=1e-12)
