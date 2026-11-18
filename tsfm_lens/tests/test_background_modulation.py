"""A planted background-only change is distinct from level and tone changes."""
import numpy as np
from experiments.feature_search.background_modulation import partition


def test_nested_projection_recovers_known_background_and_rejects_permuted_decoy():
    t=np.arange(32);signal=np.sin(2*np.pi*5.35*t/128+.4)
    background=np.random.default_rng(9).normal(size=32)
    response=.6*signal+.2+1.7*background
    result,_=partition(response,signal,background,5.35,5)
    assert result['after_background']<1e-25
    assert result['background_explained']>1
    assert result['background_exceeds_permuted95']
    control,_=partition(.6*signal+.2,signal,background,5.35,5)
    assert control['after_level_trend']<1e-25
    assert abs(control['background_explained'])<1e-25
