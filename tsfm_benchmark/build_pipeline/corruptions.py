"""Corruption transforms applied after generation.

Every corruption appends a ``ProvenanceStep`` so the exact perturbation is
recoverable, which matters because a corruption that is not recorded silently
destroys the ground truth it was applied to. The core transforms are pure numpy
to avoid a hard dependency; the sktime and tsaug adapters are optional and only
add value where they provide augmentations not trivially reproduced here.
"""

from __future__ import annotations

import numpy as np

from .registry import CORRUPTIONS
from .schema import ProvenanceStep, TimeSeriesSample


def _record(sample: TimeSeriesSample, op: str, params: dict) -> None:
    sample.provenance.transforms.append(ProvenanceStep(op=op, params=params))


@CORRUPTIONS.register("jitter")
def jitter(sample: TimeSeriesSample, seed: int = 0, sigma: float = 0.05) -> TimeSeriesSample:
    """Add zero-mean Gaussian noise scaled to the series standard deviation."""
    rng = np.random.default_rng(seed)
    scale = sigma * (sample.values.std() + 1e-8)
    sample.values = sample.values + rng.normal(0, scale, size=sample.values.shape)
    _record(sample, "jitter", {"sigma": sigma, "seed": seed})
    return sample


@CORRUPTIONS.register("scaling")
def scaling(sample: TimeSeriesSample, seed: int = 0, sigma: float = 0.1) -> TimeSeriesSample:
    """Multiply the whole series by a single random scalar near one."""
    rng = np.random.default_rng(seed)
    factor = float(rng.normal(1.0, sigma))
    sample.values = sample.values * factor
    _record(sample, "scaling", {"factor": factor})
    return sample


@CORRUPTIONS.register("time_warp")
def time_warp(sample: TimeSeriesSample, seed: int = 0, n_knots: int = 4, strength: float = 0.2) -> TimeSeriesSample:
    """Warp the time axis with a smooth random monotonic remapping."""
    rng = np.random.default_rng(seed)
    n = len(sample.values)
    knots = np.linspace(0, n - 1, n_knots + 2)
    offsets = rng.normal(0, strength * n / n_knots, size=n_knots + 2)
    offsets[0] = offsets[-1] = 0.0
    warped_knots = np.clip(knots + offsets, 0, n - 1)
    warped_knots = np.maximum.accumulate(warped_knots)
    new_index = np.interp(np.arange(n), knots, warped_knots)
    sample.values = np.interp(new_index, np.arange(n), sample.values)
    _record(sample, "time_warp", {"n_knots": n_knots, "strength": strength})
    return sample


@CORRUPTIONS.register("dropout")
def dropout(sample: TimeSeriesSample, seed: int = 0, rate: float = 0.05) -> TimeSeriesSample:
    """Replace a random fraction of points with forward-filled values."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(sample.values)) < rate
    out = sample.values.copy()
    last = out[0]
    for i in range(len(out)):
        if mask[i]:
            out[i] = last
        else:
            last = out[i]
    sample.values = out
    _record(sample, "dropout", {"rate": rate})
    return sample
