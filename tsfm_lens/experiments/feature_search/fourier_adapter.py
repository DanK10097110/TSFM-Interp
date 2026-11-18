"""Past-only, reconstruction-preserving spectral tree for frozen forecasters.

The leaves partition real FFT bins, retaining level/trend once. Tree grouping
changes the forecast calls, not the reconstructed signal. No generator metadata
or future array is accepted by the decomposition API.
"""
from __future__ import annotations
import numpy as np

LEVELS = (1, 2, 4, 8)


def decompose(history, detrend='mean'):
    """Eight contiguous frequency leaves; keep all energy, including Nyquist/DC."""
    x = np.asarray(history, dtype=np.float64)
    if x.ndim != 1 or len(x) != 128 or not np.isfinite(x).all():
        raise ValueError('Expected one finite 128-step history')
    if detrend == 'mean':
        trend = np.full(len(x), x.mean())
    elif detrend == 'linear':
        basis = np.column_stack([np.ones(len(x)), np.linspace(-1, 0, len(x))])
        trend = basis @ np.linalg.lstsq(basis, x, rcond=None)[0]
    else:
        raise ValueError('Unknown detrending policy')
    spectrum = np.fft.rfft(x-trend)
    edges = [0, 3, 5, 9, 13, 21, 33, 49, 65]
    leaves = []
    for start, stop in zip(edges[:-1], edges[1:]):
        masked = np.zeros_like(spectrum)
        masked[start:stop] = spectrum[start:stop]
        leaves.append(np.fft.irfft(masked, n=len(x)))
    leaves = np.asarray(leaves)
    leaves[0] += trend
    tiny = np.linalg.norm(leaves, axis=1) < 1e-12 * max(np.linalg.norm(x), 1e-12)
    leaves[tiny] = 0
    leaves[0] += x-leaves.sum(axis=0)
    if np.max(abs(leaves.sum(axis=0)-x)) > 1e-12*max(1., np.max(abs(x))):
        raise ValueError('Decomposition failed closure')
    return leaves


def groups(order, count):
    """Nested equal-leaf-count partitions, with every leaf present exactly once."""
    order = np.asarray(order, dtype=int)
    if sorted(order.tolist()) != list(range(8)) or count not in LEVELS:
        raise ValueError('Need permutation of eight leaves and count 1/2/4/8')
    return [tuple(sorted(g.tolist())) for g in np.split(order, count)]


def grouping_orders(leaves, seed):
    """Distant-band contrast and a bounded random energy-matching control."""
    energy = np.mean(np.asarray(leaves)**2, axis=1)
    normalizer = max(float(energy.sum()), 1e-20)
    contiguous = np.arange(8)
    def signature(order):
        return np.concatenate([np.sort([energy[list(g)].sum()/normalizer for g in groups(order, k)])
                               for k in (2, 4)])
    target = signature(contiguous)
    candidates = []
    rng = np.random.default_rng(seed)
    for _ in range(128):
        candidate = rng.permutation(8)
        if {frozenset(g) for g in groups(candidate, 4)} == {frozenset(g) for g in groups(contiguous, 4)}:
            continue
        candidates.append((float(np.max(abs(signature(candidate)-target))), tuple(candidate.tolist())))
    mismatch, random_order = min(candidates)
    return dict(contiguous=contiguous.tolist(), interleaved=[0, 4, 1, 5, 2, 6, 3, 7],
                matched_random=list(random_order)), dict(energy_fraction_max_mismatch=mismatch,
                                                       energy_match_valid=mismatch <= .1,
                                                       candidate_draws=128)


def combine(predictions, zero, corrected=True):
    """Sum point forecasts; optionally subtract duplicate F(0) affine biases."""
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 2 or len(predictions) == 0:
        raise ValueError('Need component by horizon point forecasts')
    return predictions.sum(axis=0) - (len(predictions)-1)*np.asarray(zero) if corrected else predictions.sum(axis=0)


def interaction(parent, left, right, zero):
    """Nonadditivity is a waveform, never itself an assertion of harmfulness."""
    return np.asarray(parent, dtype=float)-left-right+zero


def error(prediction, target, scale):
    """Mean squared error in one fixed source-scale unit, never per-component scale."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite source scale required')
    return float(np.mean((np.asarray(prediction, dtype=float)-target)**2)/scale**2)


def first_crossing(losses, threshold=.25):
    """Report a protocol-specific first crossing, allowing nonmonotone later losses."""
    valid = [int(k) for k, v in losses.items() if v <= threshold]
    return min(valid) if valid else None


def classical(history, kind='spectral'):
    """Past-only forecasts with no privileged frequencies; fixed three peak bins."""
    x = np.asarray(history, dtype=float)
    if kind == 'seasonal':
        return x[-24:][np.arange(32) % 24]
    t = np.arange(-128, 32)
    trend_basis = np.column_stack([np.ones(128), np.arange(-128, 0)/128])
    trend = trend_basis @ np.linalg.lstsq(trend_basis, x, rcond=None)[0]
    strength = abs(np.fft.rfft(x-trend))
    peaks = np.argsort(-strength[1:], kind='stable')[:3]+1
    basis = np.column_stack([np.ones(160), t/128] +
                            [fn(2*np.pi*bin*t/128) for bin in peaks for fn in (np.sin, np.cos)])
    return basis[128:] @ np.linalg.lstsq(basis[:128], x, rcond=None)[0]


def conditional_interval(values, source_ids, seed=492900, draws=1999):
    """Bootstrap independent sources after averaging dependent rows within source.

Intervals are descriptive, unadjusted and conditional on this dataset/design;
no multiplicity-controlled significance or population-domain claims follow.
"""
    values = np.asarray(values, dtype=float)
    ids = np.asarray(source_ids)
    if len(values) != len(ids) or not np.isfinite(values).all():
        raise ValueError('Need finite paired source values')
    unique = np.unique(ids)
    grouped = np.array([values[ids == key].mean() for key in unique])
    rng = np.random.default_rng(seed)
    means = grouped[rng.integers(len(grouped), size=(draws, len(grouped)))].mean(1)
    return dict(mean=float(grouped.mean()), low=float(np.quantile(means, .025)),
                high=float(np.quantile(means, .975)), independent_sources=len(grouped),
                favorable_sources=int(np.sum(grouped > 0)),
                scope='Unadjusted descriptive source bootstrap; conditional on this dataset/design')
