import numpy as np

from research.activation_transfer.shape_transfer.tangents.capture import variations
from research.activation_transfer.shape_transfer.tangents.maps import SpectralRidge, normalize_training
from research.activation_transfer.shape_transfer.tangents.summary import energy, scores


def test_tangent_weight_balance_and_selected_penalty_predict_known_change():
    a,b=variations("a"),variations("b")
    np.testing.assert_array_equal(a,variations("a"))
    assert not np.array_equal(a,b)
    np.testing.assert_allclose(np.sqrt(np.mean(a.astype(float)**2,axis=1)),.06,rtol=1e-5)
    rng=np.random.default_rng(18)
    x=rng.normal(size=(95,9))
    truth=rng.normal(size=(9,13))
    y=x@truth
    normalized=[normalize_training(s*x,s*y) for s in (.05,55.)]
    np.testing.assert_allclose(normalized[0][0],normalized[1][0],atol=1e-12)
    reg=SpectralRidge(*normalized[0])
    held=rng.normal(size=(30,9))
    selected,_=reg.choose((held,held@truth))
    predicted=held@reg.map(selected)
    np.testing.assert_allclose(predicted,held@truth,atol=.001)
    w=np.ones((1,len(held)))
    assert scores(energy(predicted,held@truth),w)["recovery"][0]>.999999
