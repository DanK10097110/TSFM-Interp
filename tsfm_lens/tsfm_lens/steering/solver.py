"""Combine measured feature effects while penalizing unwanted output changes.

The target is a full output tensor flattened in the same order as the Jacobian.
Zeros are protected coordinates, not missing observations. The solution adds a
direction to the ORIGINAL hidden state, preserving SAE reconstruction residuals.
An SAE supplies candidate decoder atoms, not assumed semantic labels. Dense
solutions depend only on their span, so an overcomplete full-rank SAE cannot
claim an advantage over direct hidden-space control just from reparameterizing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg


@dataclass
class SteeringSolution:
    """A proposed additive hidden edit and its first-order diagnostics."""

    delta: np.ndarray
    coefficients: np.ndarray
    selected: np.ndarray
    predicted: np.ndarray
    clipped: bool


def _finite(value, name, ndim):
    """Reject ambiguous shapes and nonfinite optimizer inputs."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite {ndim}D array")
    return array


def solve_direction(jacobian, target, dictionary=None, *, max_features=None,
                    ridge=1e-3, max_norm=None) -> SteeringSolution:
    """Minimize ||J delta - target||² + ridge ||delta||² in a decoder span.

    `jacobian` is [outputs, hidden]; `dictionary` is [atoms, hidden], with
    identity used when omitted. `target` is [outputs]. A positive ridge is in
    the actual hidden metric, making dense solutions invariant to rescaling or
    duplicating atoms. Sparse solutions use normalized-response greedy pursuit
    with a joint refit after each selection. This is a heuristic, not an exact
    best-subset solution. `max_norm` radially clips the proposed direction;
    it is a trust safeguard, not an exact constrained optimizer. Return values
    always reflect the clipped direction. No latent nonnegativity is imposed:
    this is signed decoder-direction addition, not replacement of SAE codes.
    """
    j = _finite(jacobian, "jacobian", 2)
    t = _finite(target, "target", 1)
    if j.shape[0] != t.size:
        raise ValueError("target and Jacobian output dimensions differ")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    if max_norm is not None and (not np.isfinite(max_norm) or max_norm <= 0):
        raise ValueError("max_norm must be finite and positive")
    d = np.eye(j.shape[1]) if dictionary is None else _finite(dictionary, "dictionary", 2)
    if d.shape[1] != j.shape[1]:
        raise ValueError("dictionary and Jacobian hidden dimensions differ")
    if max_features is not None and (isinstance(max_features, bool) or
            not isinstance(max_features, (int, np.integer)) or max_features < 1):
        raise ValueError("max_features must be a positive integer or None")

    if max_features is None:
        if dictionary is None:
            basis = d
        else:
            _, s, vh = linalg.svd(d, full_matrices=False, check_finite=False)
            cutoff = max(d.shape) * np.finfo(float).eps * s[0]
            basis = vh[s > cutoff]
        a = j @ basis.T
        gram = a @ a.T + ridge * np.eye(j.shape[0])
        delta = basis.T @ (a.T @ linalg.solve(gram, t, assume_a="pos"))
        coefficients = delta.copy() if dictionary is None else linalg.lstsq(
            d.T, delta, lapack_driver="gelsd")[0]
        selected = np.flatnonzero(np.abs(coefficients) > 1e-12)
    else:
        a = j @ d.T
        norms = np.linalg.norm(a, axis=0)
        valid = norms > np.finfo(float).eps * max(1., float(norms.max()))
        chosen = []
        residual = t.copy()
        coefficients = np.zeros(len(d))
        for _ in range(min(max_features, int(valid.sum()))):
            score = np.where(valid, np.abs(a.T @ residual) / np.maximum(norms, 1e-30), -1.)
            if chosen:
                score[chosen] = -1.
            idx = int(np.argmax(score))
            if score[idx] <= 1e-14:
                break
            chosen.append(idx)
            design = np.vstack([a[:, chosen], np.sqrt(ridge) * d[chosen].T])
            fit = linalg.lstsq(design, np.r_[t, np.zeros(j.shape[1])],
                               lapack_driver="gelsy")[0]
            residual = t - a[:, chosen] @ fit
            coefficients[chosen] = fit
        selected = np.asarray(chosen, dtype=int)
        delta = coefficients @ d

    norm = np.linalg.norm(delta)
    clipped = max_norm is not None and norm > max_norm
    if clipped:
        factor = max_norm / norm
        delta *= factor
        coefficients *= factor
    return SteeringSolution(delta, coefficients, selected, j @ delta, bool(clipped))


def response_metrics(actual, target, predicted=None) -> dict:
    """Measure target gain and all orthogonal collateral, without scalar hiding.

    Gain one and collateral zero reproduce the entire requested change.
    Relative error includes both gain error and collateral. A zero target is
    invalid here; identity checks should use absolute error instead.
    """
    a = np.asarray(actual, dtype=float)
    t = np.asarray(target, dtype=float)
    if a.shape != t.shape or not np.isfinite(a).all() or not np.isfinite(t).all():
        raise ValueError("actual and target must have equal finite shapes")
    energy = float(np.sum(t * t))
    if energy <= 0:
        raise ValueError("target must have nonzero energy")
    gain = float(np.sum(a * t) / energy)
    result = dict(gain=gain, collateral=float(np.sqrt(np.sum((a - gain*t)**2) / energy)),
                  relative_error=float(np.sqrt(np.sum((a-t)**2) / energy)))
    if predicted is not None:
        p = np.asarray(predicted, dtype=float)
        if p.shape != a.shape or not np.isfinite(p).all():
            raise ValueError("predicted must match actual's finite shape")
        result["linearization_error"] = float(np.linalg.norm(a-p) / np.sqrt(energy))
    return result
