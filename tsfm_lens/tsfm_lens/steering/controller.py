"""Trust-region response control with measured nonlinear acceptance.

The linear bound is exact for the supplied Jacobian and norm ball, not for
global nonlinear controllability. Iteration adapts to the context and is not
a universal semantic vector. SAE features are optional subspace restrictions.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy import linalg, optimize

from .solver import response_metrics


def trust_region_direction(jacobian, target, max_norm, *, ridge=0.):
    """Solve linear least squares inside an L2 ball via SVD and a scalar dual.

    Ridge penalizes the actual hidden edit. With ridge zero, the residual is
    the best possible first-order error under this norm cap.
    """
    j=np.asarray(jacobian,dtype=float)
    t=np.asarray(target,dtype=float)
    if (j.ndim!=2 or not j.size or t.shape!=(j.shape[0],)
            or not np.isfinite(j).all() or not np.isfinite(t).all()
            or not np.isfinite(max_norm) or max_norm<=0 or not np.isfinite(ridge) or ridge<0):
        raise ValueError('need finite compatible arrays, positive norm bound and nonnegative ridge')
    u,s,vh=linalg.svd(j,full_matrices=False,check_finite=False)
    keep=s > max(j.shape)*np.finfo(float).eps*s[0]
    projected=u[:,keep].T@t
    singular=s[keep]
    basis=vh[keep]
    def coordinates(penalty):
        return singular*projected/(singular**2+penalty)
    value=coordinates(ridge)
    penalty=ridge
    active=np.linalg.norm(value)>max_norm
    if active:
        high=max(1.,ridge*2)
        while np.linalg.norm(coordinates(high))>max_norm:
            high*=2
        penalty=optimize.brentq(lambda z:np.linalg.norm(coordinates(z))-max_norm,
                                ridge,high,xtol=1e-14)
        value=coordinates(penalty)
    delta=basis.T@value
    norm=float(np.linalg.norm(t))
    span_residual=t-u[:,keep]@projected
    return delta,dict(relative_error=float(np.linalg.norm(j@delta-t)/norm) if norm else 0.,
                       unrestricted_minimum_norm=float(np.linalg.norm(projected/singular)),
                       unreachable_fraction=float(np.linalg.norm(span_residual)/norm) if norm else 0.,
                       rank=int(keep.sum()),norm_constraint_active=bool(active),penalty=float(penalty))


@dataclass
class ControlledForecast:
    """Original-state addition, actual forecast, and accepted-step diagnostics."""
    delta: np.ndarray
    forecast: np.ndarray
    baseline: np.ndarray
    history: list
    metrics: dict


def steer_hidden(forward, hidden, target, *, max_norm, steps=4, ridge=.001,
                 tolerance=.02, basis=None):
    """Fit and verify a context-specific edit through a differentiable suffix.

    hidden is one tensor [D]; forward(hidden) has exactly target's shape.
    Optional basis rows restrict control to a chosen subspace. A full-rank SAE
    changes no reachable direction. Every iteration solves for the TOTAL edit
    in the same norm ball and backtracks against the actual model. No future
    labels are consumed. All edits preserve the original reconstruction residual.
    """
    if not isinstance(hidden,torch.Tensor) or hidden.ndim!=1:
        raise ValueError('hidden must be a one-dimensional torch tensor')
    if (not torch.isfinite(hidden).all() or isinstance(steps, bool)
            or not isinstance(steps, (int, np.integer)) or steps<1
            or not np.isfinite(tolerance) or tolerance<0):
        raise ValueError('need finite hidden state, positive steps and nonnegative tolerance')
    h=hidden.detach()
    target=np.asarray(target,dtype=float)
    with torch.no_grad():
        base=forward(h).detach().cpu().numpy()
    if base.shape!=target.shape or not np.isfinite(target).all() or np.linalg.norm(target)==0:
        raise ValueError('nonzero finite target must match the full forecast shape')
    if basis is None:
        q=None
    else:
        b=np.asarray(basis,dtype=float)
        if b.ndim!=2 or not b.size or b.shape[1]!=len(h) or not np.isfinite(b).all():
            raise ValueError('basis must be finite [directions,hidden]')
        _,s,vh=linalg.svd(b,full_matrices=False)
        q=vh[s>max(b.shape)*np.finfo(float).eps*s[0]]
    delta=np.zeros(len(h))
    current=base.copy()
    history=[]
    for step in range(steps):
        state=h+torch.as_tensor(delta,dtype=h.dtype,device=h.device)
        jac=torch.autograd.functional.jacobian(lambda z:forward(z).flatten(),
                                               state.requires_grad_(True),vectorize=True).detach().cpu().numpy().astype(float)
        effective=(base+target-current).flatten()+jac@delta
        matrix=jac if q is None else jac@q.T
        proposed,diagnostic=trust_region_direction(matrix,effective,max_norm,ridge=ridge)
        if q is not None:
            proposed=q.T@proposed
        old_error=np.linalg.norm(current-base-target)
        accepted=False
        for rate in (1.,.5,.25,.125,.0625,.03125):
            candidate=delta+rate*(proposed-delta)
            with torch.no_grad():
                forecast=forward(h+torch.as_tensor(candidate,dtype=h.dtype,device=h.device)).detach().cpu().numpy()
            if np.isfinite(forecast).all() and np.linalg.norm(forecast-base-target)<old_error:
                delta,current=candidate,forecast
                accepted=True
                break
        metrics=response_metrics(current-base,target)
        history.append(dict(step=step,accepted=accepted,step_fraction=rate if accepted else 0.,
                            hidden_norm=float(np.linalg.norm(delta)),**metrics,linear_solve=diagnostic))
        if not accepted or metrics['relative_error']<=tolerance:
            break
    return ControlledForecast(delta,current,base,history,response_metrics(current-base,target))
