"""Contracts for small-pilot axis selection, matched budgets and two-way resampling."""
import numpy as np

from research.activation_transfer.shape_transfer.calibration_budget.study import draws,select_pilot
from research.activation_transfer.shape_transfer.calibration_budget.score import COST,two_way


def test_pilot_retains_signed_mean_output_span_and_ignores_canceling_variation():
    rng=np.random.default_rng(17531)
    clean=np.linalg.qr(rng.normal(size=(40,24)))[0].T
    signal=np.zeros((8,24));signal[:,8:16]=np.diag(np.linspace(1.,2.,8))
    variation=np.zeros_like(signal);variation[:,:8]=50*np.eye(8)
    observed=np.stack([signal+variation,signal-variation,signal+variation,signal-variation])
    axes,_=select_pilot(observed,clean,2.)
    np.testing.assert_allclose(axes.T@axes,clean[8:16].T@clean[8:16],atol=1e-10)


def test_draws_are_distinct_nested_rows_and_all_pilot_calls_are_charged():
    sampled=draws(96)
    assert sampled.shape==(3,96)
    assert len(set(tuple(p) for p in sampled))==3
    for perm in sampled:
        np.testing.assert_array_equal(np.sort(perm),np.arange(96))
        assert set(perm[:4]).isdisjoint(perm[4:40])
    assert COST['4x24']==COST['12x8']==192
    assert COST['16x24']==COST['48x8']==COST['pilot']==2*(4*24+36*8)


def test_seed_counts_are_paired_with_the_same_history_draw():
    scores=np.array([[1,2,3],[4,5,6],[7,8,9]])
    weights=np.array([[3,0,0],[0,1,2],[0,3,0]])
    np.testing.assert_allclose(two_way(scores,weights),[1,7,6])
