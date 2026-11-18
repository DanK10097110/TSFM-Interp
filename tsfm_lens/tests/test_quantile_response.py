"""Known additive quantile shifts and asymmetric-spread decoys."""
import numpy as np
from experiments.feature_search.quantile_response import summaries
from experiments.feature_search import quantile_response


def test_native_policy_never_calls_locking_helper(monkeypatch):
    """A mislabeled native arm must not accidentally use the anchored transform."""
    from contextlib import contextmanager
    calls=[]
    @contextmanager
    def fake(native,anchor,policy):
        calls.append(policy)
        yield
    monkeypatch.setattr(quantile_response,'normalization',fake)
    with quantile_response.policy_context(None,None,'native'):pass
    assert calls==[]
    with quantile_response.policy_context(None,None,'locked'):pass
    assert calls==['background']


def test_translation_and_spread_are_not_confused():
    signal=np.sin(np.arange(32)*.3)
    perfect=np.repeat(signal[:,None],9,axis=1)
    r=summaries(perfect,signal)
    assert r['median_nmse']==0
    assert r['quantile_shift_disagreement']==0
    assert r['interval_width_change_energy']==0
    spread=perfect+np.linspace(-2,2,9)[None]*signal[:,None]
    r=summaries(spread,signal)
    assert r['median_nmse']==0
    assert r['central_quantile_average_nmse']<1e-28
    assert np.isclose(r['interval_width_change_energy'],16)
    decoy=perfect.copy();decoy[:,4]=0
    r=summaries(decoy,signal)
    assert np.isclose(r['median_nmse'],1)
    assert r['oracle_best_quantile_nmse']==0
    assert np.isclose(r['central_quantile_average_nmse'],1/81)
