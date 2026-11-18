"""Known-answer controls for locating a frontend discontinuity and chord shape."""
import numpy as np
from experiments.feature_search.token_boundary import boundary_pair,chord


def test_discontinuity_has_distinct_raw_endpoints_and_same_token_control():
    """A decoy second boundary must not replace the first reachable transition."""
    tokenize=lambda x:np.searchsorted([.123456789,.55],x,side='right')
    left,right,record=boundary_pair(np.array([.12345,.01]),np.array([1.,.01]),tokenize)
    assert np.array_equal(tokenize(left),[0,0])
    assert np.array_equal(tokenize(right),[1,0])
    assert 0<record['raw_rms_distance']<1e-7
    assert record['same_token_raw_rms_distance']>record['raw_rms_distance']


def test_chord_distinguishes_affine_curved_and_degenerate_maps():
    """Affine vector maps have zero bend; a quadratic contracts with radius."""
    affine=lambda x:np.array([2*x+5,3*x-2])
    assert chord(affine(-1),affine(0),affine(1))['bend_rms']==0
    curve=lambda x:np.array([x,x*x])
    wide=chord(curve(-.2),curve(0),curve(.2))
    narrow=chord(curve(-.1),curve(0),curve(.1))
    assert np.isclose(narrow['ratio'],wide['ratio']/2)
    assert chord([1],[2],[1])['ratio'] is None
