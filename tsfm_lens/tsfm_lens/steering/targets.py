"""Explicit quantile-space steering requests; no future truth is consumed.

All targets use [horizon, quantiles]. These are requested output changes, not
claims about a model's uncertainty calibration or latent semantic concepts.
"""

from __future__ import annotations

import numpy as np


def _quantiles(quantiles):
    """Require unique increasing interior quantiles, including the median."""
    q = np.asarray(quantiles, dtype=float)
    if (q.ndim != 1 or len(q) < 3 or not np.isfinite(q).all()
            or np.any(np.diff(q) <= 0) or q[0] <= 0 or q[-1] >= 1
            or not np.any(q == .5)):
        raise ValueError("need increasing interior quantiles including exactly 0.5")
    return q


def interval_shift(baseline, quantiles, fraction):
    """Expand/contract each quantile's distance from the unchanged median.

    `fraction=.2` requests 20% wider intervals, and `-.2` requests 20%
    narrower intervals. The output-only oracle preserves ordering if the
    baseline is ordered. An internal steering approximation may not.
    """
    q = _quantiles(quantiles)
    y = np.asarray(baseline, dtype=float)
    if y.ndim != 2 or y.shape[1] != len(q) or not np.isfinite(y).all():
        raise ValueError("baseline must be finite [horizon, quantiles]")
    if not np.isfinite(fraction) or fraction <= -1:
        raise ValueError("fraction must be finite and greater than -1")
    return fraction * (y - y[:, q == .5])


def seasonal_shift(context, horizon, quantiles, *, period, amplitude, phase=None):
    """Add a context-phase seasonal prior equally to every quantile.

    Fit known-period sine/cosine jointly with intercept and slope on the
    context, then extrapolate its phase. `amplitude` sets the absolute signed
    amplitude in the caller's units; it does not multiply the fitted amplitude.
    No FFT-bin snapping is used. Supply `phase` in radians at context index zero
    to impose a prior even without an existing harmonic. Otherwise a nonexistent
    context harmonic raises instead of inventing a phase. Widths are preserved.
    """
    q = _quantiles(quantiles)
    x = np.asarray(context, dtype=float)
    if (x.ndim != 1 or len(x) < 5 or not np.isfinite(x).all()
            or not isinstance(horizon, (int, np.integer)) or horizon < 1
            or not np.isfinite(period) or period <= 2 or not np.isfinite(amplitude)):
        raise ValueError("need finite context, positive horizon and period > 2")
    if phase is not None:
        if not np.isfinite(phase):
            raise ValueError('phase must be finite when supplied')
        future = np.arange(len(x), len(x)+horizon)
        wave = np.sin(2*np.pi*future/period+phase)
        return np.repeat((amplitude*wave)[:, None], len(q), axis=1)
    t = np.arange(len(x))
    design = np.column_stack([np.ones(len(x)), t-t.mean(),
                              np.sin(2*np.pi*t/period), np.cos(2*np.pi*t/period)])
    coef = np.linalg.lstsq(design, x, rcond=None)[0][-2:]
    size = np.linalg.norm(coef)
    if size <= 1e-10 * max(1., np.std(x)):
        raise ValueError("context contains no resolvable harmonic at the supplied period")
    future = np.arange(len(x), len(x)+horizon)
    wave = (coef[0]*np.sin(2*np.pi*future/period) + coef[1]*np.cos(2*np.pi*future/period))/size
    return np.repeat((amplitude*wave)[:, None], len(q), axis=1)


def quantile_shift(horizon, quantiles, *, channel, amount):
    """Change one quantile channel uniformly while protecting every other value."""
    q = _quantiles(quantiles)
    if (not isinstance(horizon, (int, np.integer)) or horizon < 1
            or channel not in q or not np.isfinite(amount)):
        raise ValueError("need positive horizon, a native quantile, and finite amount")
    out = np.zeros((horizon, len(q)))
    out[:, q == channel] = amount
    return out
