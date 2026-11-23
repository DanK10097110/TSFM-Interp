import numpy as np
from research.activation_transfer.input_baseline.kernel import fit, raw_state


def test_affine_intercept_and_differences():
    rng=np.random.default_rng(234)
    x=rng.normal(size=(80,6))
    w=rng.normal(size=(6,9))
    y=x@w+73
    mapping,settings=fit(x,y)
    h=rng.normal(size=(12,6))
    d=rng.normal(size=(12,6))*.16
    np.testing.assert_allclose(mapping.predict(h),h@w+73,atol=4e-4)
    np.testing.assert_allclose((mapping.predict(h+d)-mapping.predict(h-d))/2,d@w,atol=2e-4)
    assert settings["family"]=="linear"


def test_raw_baseline_invariance():
    x=np.arange(128)[None,:]
    np.testing.assert_allclose(raw_state(x,True),raw_state(3.1*x+5,True))
    assert raw_state(x,True).shape==(1,160)
