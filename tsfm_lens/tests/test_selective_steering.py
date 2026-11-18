"""Planted cancellation, rank barriers, trust budgets and target semantics."""

import numpy as np
import pytest
import torch

from tsfm_lens.steering import solve_direction, response_metrics, interval_shift, seasonal_shift, quantile_shift
from tsfm_lens.steering import trust_region_direction, steer_hidden


def test_mixing_cancels_a_channel_that_no_single_atom_can_protect():
    dictionary = np.array([[1., 1.], [1., -1.]])
    target = np.array([1., 0.])
    one = solve_direction(np.eye(2), target, dictionary, max_features=1, ridge=1e-9)
    two = solve_direction(np.eye(2), target, dictionary, max_features=2, ridge=1e-9)
    assert response_metrics(one.predicted, target)['relative_error'] > .7
    assert np.allclose(two.predicted, target, atol=1e-8)
    assert np.allclose(two.coefficients @ dictionary, two.delta)


def test_hidden_penalty_is_invariant_to_duplicate_and_rescaled_dictionary():
    rng = np.random.default_rng(4)
    j = rng.normal(size=(4, 6))
    target = rng.normal(size=4)
    raw = solve_direction(j, target, ridge=.07)
    redundant = solve_direction(j, target, np.vstack([np.eye(6)*3, np.eye(6)*.2]), ridge=.07)
    assert np.allclose(raw.delta, redundant.delta, atol=1e-10)


def test_rank_obstruction_is_not_hidden_by_a_good_target_scalar():
    j = np.array([[1.], [1.]])
    target = np.array([1., 0.])
    result = solve_direction(j, target, ridge=1e-12)
    assert np.allclose(result.predicted, [.5, .5])
    assert response_metrics(result.predicted, target)['relative_error'] == pytest.approx(np.sqrt(.5))


def test_radius_clips_coefficients_prediction_and_delta_together():
    result = solve_direction(np.eye(3), np.ones(3), max_norm=.1, ridge=1e-8)
    assert result.clipped
    assert np.linalg.norm(result.delta) == pytest.approx(.1)
    assert np.allclose(result.coefficients, result.delta)
    assert np.allclose(result.predicted, result.delta)


def test_zero_response_dictionary_returns_no_fake_control():
    result = solve_direction(np.zeros((2, 3)), np.ones(2), max_features=2)
    assert np.array_equal(result.delta, np.zeros(3))
    assert len(result.selected) == 0


def test_width_target_holds_entire_median_and_preserves_quantile_order():
    q = [.1, .5, .9]
    y = np.array([[1., 3., 4.], [2., 4., 8.]])
    target = interval_shift(y, q, -.2)
    edited = y+target
    assert np.array_equal(edited[:, 1], y[:, 1])
    assert np.allclose(edited[:, -1]-edited[:, 0], .8*(y[:, -1]-y[:, 0]))
    assert (np.diff(edited) > 0).all()


def test_off_grid_phase_estimation_excludes_trend_and_preserves_width():
    period = 13.7
    t = np.arange(128)
    x = 4+.07*t+2*np.sin(2*np.pi*t/period+.4)
    result = seasonal_shift(x, 32, [.1,.5,.9], period=period, amplitude=.3)
    expected = .3*np.sin(2*np.pi*np.arange(128,160)/period+.4)
    assert np.allclose(result, expected[:, None])
    assert np.allclose(result[:, -1]-result[:, 0], 0)


def test_explicit_phase_can_add_a_prior_to_nonseasonal_context():
    result=seasonal_shift(np.ones(32),8,[.1,.5,.9],period=8,amplitude=.3,phase=.4)
    expected=.3*np.sin(2*np.pi*np.arange(32,40)/8+.4)
    assert np.allclose(result,expected[:,None])


def test_invalid_requests_fail_before_optimization():
    with pytest.raises(ValueError):
        solve_direction(np.eye(2), [1, np.nan])
    with pytest.raises(ValueError):
        solve_direction(np.eye(2), [1, 0], max_features=1.5)
    with pytest.raises(ValueError):
        interval_shift(np.ones((2,3)), [.1,.5,.9], -1)
    with pytest.raises(ValueError):
        seasonal_shift(np.ones(32), 8, [.1,.5,.9], period=8, amplitude=1)
    with pytest.raises(ValueError):
        quantile_shift(8, [.1,.5,.9], channel=.8, amount=1)


def test_trust_solver_beats_radial_clipping_and_matches_kkt():
    j=np.diag([10.,.1])
    target=np.ones(2)
    delta,diagnostic=trust_region_direction(j,target,.2)
    clipped=solve_direction(j,target,ridge=1e-9,max_norm=.2)
    assert np.linalg.norm(j@delta-target)<np.linalg.norm(clipped.predicted-target)
    assert np.linalg.norm(delta)==pytest.approx(.2)
    assert np.allclose(j.T@(j@delta-target)+diagnostic['penalty']*delta,0,atol=1e-9)


def test_nonlinear_controller_improves_and_never_exceeds_total_budget():
    h=torch.tensor([.2,.4])
    forward=lambda z:torch.stack([z[0]+z[0]**3,z[1]])
    target=np.array([.5,0.])
    one=steer_hidden(forward,h,target,max_norm=.5,steps=1,ridge=1e-8,tolerance=0)
    many=steer_hidden(forward,h,target,max_norm=.5,steps=5,ridge=1e-8,tolerance=0)
    assert many.metrics['relative_error']<one.metrics['relative_error']
    assert many.metrics['relative_error']<1e-5
    errors=[item['relative_error'] for item in many.history]
    assert np.all(np.diff(errors)<=0)
    assert all(item['hidden_norm']<=.5+1e-10 for item in many.history)
    assert many.forecast[1]==pytest.approx(.4)


def test_restricted_controller_cannot_invent_missing_subspace_direction():
    h=torch.zeros(2)
    result=steer_hidden(lambda z:z,h,np.array([0.,1.]),max_norm=1.,basis=[[1.,0.]])
    assert np.array_equal(result.delta,np.zeros(2))
    assert result.metrics['relative_error']==1
