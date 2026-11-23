"""Small analytic controls for the new causal and forecast-scoring experiments."""
import numpy as np

from research.activation_transfer.moment.calendar.run import metrics
from research.activation_transfer.moment.dictionary_transfer.functional.run import fit_ball,local_svd
from research.activation_transfer.moment.dictionary_transfer.gain import evaluate
from research.activation_transfer.moment.dictionary_transfer.groups import pursuit
from research.activation_transfer.moment.input_paths.run import CELLS,shapley
from research.activation_transfer.moment.token_paths.protect import protect,solve


def test_calendar_metrics_use_symmetric_denominator_and_all_future_steps():
    y=np.array([[0.,4.],[2.,4.]])
    p=np.array([[0.,2.],[4.,4.]])
    smape,mase=metrics(p,y,np.array([1.,2.]))
    np.testing.assert_allclose(smape,[100/3,100/3])
    np.testing.assert_allclose(mase,[1.,.5])


def test_positive_gain_handles_zero_and_opposing_directions_and_crossfits():
    y=np.array([[1.,-1.],[2.,-2.],[3.,-3.],[4.,-4.]])
    fold=np.array([True,False,True,False])
    for key in ("optimistic_atom","optimistic_event","crossfit"):
        assert np.isclose(evaluate(y*3,y,fold)[key],1)
        assert np.isclose(evaluate(-y,y,fold)[key],0)
        assert np.isclose(evaluate(y*0,y,fold)[key],0)


def test_fixed_sparse_pursuit_can_use_opposite_sign_and_needs_two_axes():
    axes=np.eye(17)
    y=np.array([-2.,1.]+[0.]*15)
    f=pursuit(y,axes,np.arange(17))
    assert f[1][0][0]==0 and f[1][1][0]<0
    np.testing.assert_allclose(f[4][2],y)
    np.testing.assert_allclose(f[16][2],y)
    assert len(set(f[16][0]))==16


def test_guard_preserves_a_reachable_early_effect_and_removes_late_effect():
    # First eight directions are reachable; an unconstrained write also changes the ninth coordinate.
    j=np.concatenate([np.eye(8),np.ones((8,1))],axis=1)[None]
    l=np.zeros_like(j);l[:,:,8]=1
    y=np.arange(1.,9.)[None]
    original=solve(j,y,100.,1.)
    protected=solve(protect(j,l),y,100.,1.)
    assert np.linalg.norm(l@original[:,:,None])>1
    np.testing.assert_allclose(l@protected[:,:,None],0.,atol=1e-12)
    np.testing.assert_allclose((j@protected[:,:,None])[:,:,0],y/1.001,rtol=1e-8)


def test_frozen_functional_solve_enforces_euclidean_budget_and_zero_signal():
    a=np.diag([1.,2.,3.])
    y=np.array([[3.,0.],[0.,0.],[0.,0.]])
    x=fit_ball(local_svd(a),y,np.array([.5,1.]),.001)
    np.testing.assert_allclose(x,np.array([[.5,0.],[0.,0.],[0.,0.]]),atol=1e-12)


def test_shapley_splits_interaction_and_cancels_nonzero_baseline():
    example=np.array([[1.,2.,3.]])
    d={"true":example*7}
    for a,b,c in CELLS:d[f"cell_{a}{b}{c}"]=example*(4+a+2*b+4*a*c)
    parts=shapley(d)
    for result,scale in zip(parts,(3,2,2)):np.testing.assert_allclose(result,example*scale)
