"""Known-answer checks for the counterfactual continuation-prior experiment."""
import numpy as np
from experiments.feature_search.prior_training import bank,expected_loss


def test_conditional_mean_is_optimal_and_risk_decomposes():
    """Distinct futures have irreducible risk, even with an oracle latent tone."""
    _,tone,_=bank(7,12)
    for p in [0.,.25,.5,1.]:
        oracle=p*tone
        floor=p*(1-p)*np.mean(tone.astype(float)**2,axis=1)
        np.testing.assert_allclose(expected_loss(oracle,tone,p),floor,rtol=1e-6,atol=1e-9)
        bad=oracle+.3
        np.testing.assert_allclose(expected_loss(bad,tone,p),floor+np.mean((bad-oracle).astype(float)**2,axis=1),rtol=1e-6,atol=1e-8)
        assert np.all(expected_loss(bad,tone,p)>expected_loss(oracle,tone,p))


def test_paired_histories_reproducible_but_holdout_distinct():
    """Training probability must not alter history generation or expose test rows."""
    a=bank(7,12);b=bank(7,12);c=bank(8,12)
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    assert not np.array_equal(a[0],c[0])
    assert a[0].shape==(12,128) and a[1].shape==(12,32)
