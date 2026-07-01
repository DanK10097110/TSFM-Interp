"""Series generators.

Two leakage tiers live here. The parametric generator is the leakage-safe
backbone: it builds series from a known compositional process and records every
component as ground truth, so it touches no real data and gives exact
interpretability targets. The mixture generator is real-derived: it combines
real source series with known weights, which provides attribution ground truth
but inherits the source distribution, so it is for the realism-stress tier, not
the leakage-guaranteed tier.

The bootstrap and sequential generators wrap external libraries. Their call
sites are written against the documented APIs but should be pinned and verified
against the installed versions before trusting them in a build.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .registry import GENERATORS
from .schema import GroundTruth, Provenance, SourceRef, TimeSeriesSample

Generator = Callable[..., TimeSeriesSample]


@GENERATORS.register("parametric")
def parametric(
    length: int = 512,
    seed: int = 0,
    trend: dict[str, Any] | None = None,
    seasonalities: list[dict[str, Any]] | None = None,
    ar_coeffs: list[float] | None = None,
    noise_scale: float = 0.1,
    n_changepoints: int = 0,
    n_anomalies: int = 0,
    anomaly_magnitude: float = 4.0,
) -> TimeSeriesSample:
    """Build a synthetic series from explicit additive components.

    Composes a polynomial trend, any number of sinusoidal seasonalities, an
    autoregressive coloured-noise term, optional level-shift changepoints, and
    optional point anomalies. Each component and event is stored in the ground
    truth so an interpretability method can be scored against the exact
    structure that produced the series. Uses no real data.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=float)
    components: dict[str, list[float]] = {}

    trend = trend or {"order": 1, "scale": 0.0}
    coeffs = rng.normal(0, trend["scale"], size=trend["order"] + 1) if trend["scale"] else np.zeros(trend["order"] + 1)
    trend_signal = np.polyval(coeffs[::-1], (t / length))
    components["trend"] = trend_signal.tolist()

    seasonal_total = np.zeros(length)
    for spec in seasonalities or []:
        period = float(spec["period"])
        amp = float(spec.get("amplitude", 1.0))
        phase = float(spec.get("phase", 0.0))
        s = amp * np.sin(2 * np.pi * t / period + phase)
        seasonal_total += s
        components[f"seasonal_{int(period)}"] = s.tolist()

    noise = rng.normal(0, noise_scale, size=length)
    if ar_coeffs:
        colored = np.zeros(length)
        p = len(ar_coeffs)
        for i in range(length):
            past = sum(ar_coeffs[k] * colored[i - k - 1] for k in range(p) if i - k - 1 >= 0)
            colored[i] = past + noise[i]
        noise = colored
    components["noise"] = noise.tolist()

    values = trend_signal + seasonal_total + noise

    changepoints: list[int] = []
    if n_changepoints:
        cps = sorted(rng.choice(np.arange(length // 8, length, length // (n_changepoints + 1)), size=n_changepoints, replace=False).tolist())
        for cp in cps:
            shift = float(rng.normal(0, 1.0 + noise_scale))
            values[cp:] += shift
            changepoints.append(int(cp))

    anomalies: list[dict[str, Any]] = []
    if n_anomalies:
        idxs = rng.choice(length, size=n_anomalies, replace=False)
        for idx in idxs:
            mag = float(anomaly_magnitude * noise_scale * rng.choice([-1, 1]))
            values[idx] += mag
            anomalies.append({"index": int(idx), "type": "point", "magnitude": mag})

    gt = GroundTruth(
        components=components,
        changepoints=changepoints,
        anomalies=anomalies,
        generative_params={
            "trend": trend,
            "seasonalities": seasonalities or [],
            "ar_coeffs": ar_coeffs or [],
            "noise_scale": noise_scale,
        },
        notes="fully synthetic; leakage-safe tier",
    )
    prov = Provenance(generator="parametric", generator_params={"length": length}, seed=seed)
    return TimeSeriesSample(values=values, ground_truth=gt, provenance=prov)


@GENERATORS.register("mixture")
def mixture(
    sources: list[tuple[SourceRef, np.ndarray]],
    seed: int = 0,
    mode: str = "weighted_sum",
    weight_concentration: float = 1.0,
) -> TimeSeriesSample:
    """Combine real source series with recorded weights.

    Aligns the inputs to a common length, draws Dirichlet mixture weights, and
    either sums them or multiplies interaction terms to simulate external
    correlations between domains. The drawn weights are stored as
    ``source_weights`` so attribution methods can be scored against the true
    contribution of each source. Real-derived: realism-stress tier only.
    """
    rng = np.random.default_rng(seed)
    length = min(len(arr) for _, arr in sources)
    aligned = [np.asarray(arr[:length], dtype=float) for _, arr in sources]
    aligned = [(a - a.mean()) / (a.std() + 1e-8) for a in aligned]
    weights = rng.dirichlet([weight_concentration] * len(aligned))

    if mode == "weighted_sum":
        values = sum(w * a for w, a in zip(weights, aligned))
    elif mode == "multiplicative":
        base = sum(w * a for w, a in zip(weights, aligned))
        interaction = np.prod(np.stack([a for a in aligned[:2]]), axis=0) if len(aligned) >= 2 else np.zeros(length)
        values = base + 0.5 * interaction
    else:
        raise ValueError(f"unknown mixture mode '{mode}'")

    refs = [ref for ref, _ in sources]
    gt = GroundTruth(
        source_weights={ref.item_id: float(w) for ref, w in zip(refs, weights)},
        generative_params={"mode": mode},
        notes="real-derived; inherits source distribution, not leakage-safe",
    )
    prov = Provenance(generator="mixture", generator_params={"mode": mode}, seed=seed, source_refs=refs)
    return TimeSeriesSample(values=np.asarray(values, dtype=float), ground_truth=gt, provenance=prov)


@GENERATORS.register("block_bootstrap")
def block_bootstrap(source: tuple[SourceRef, np.ndarray], seed: int = 0, block_length: int = 24) -> TimeSeriesSample:
    """Resample one real series with a block bootstrap (tsbootstrap).

    Produces a series with the source's short-range dependence but a reordered
    realisation. Useful for the realism-stress tier; it offers no instance-level
    distance guarantee, so the leakage audit must still gate it.

    Verify the tsbootstrap API against the installed version before use.
    """
    try:
        from tsbootstrap import MovingBlockBootstrap
    except ImportError as exc:
        raise ImportError("install tsbootstrap to use block_bootstrap") from exc

    ref, arr = source
    arr = np.asarray(arr, dtype=float).reshape(-1, 1)
    bootstrap = MovingBlockBootstrap(n_bootstraps=1, block_length=block_length, rng=seed)
    resampled = next(iter(bootstrap.bootstrap(arr)))
    values = np.asarray(resampled).reshape(-1)
    gt = GroundTruth(notes="real-derived block bootstrap; not leakage-safe")
    prov = Provenance(generator="block_bootstrap", generator_params={"block_length": block_length}, seed=seed, source_refs=[ref])
    return TimeSeriesSample(values=values, ground_truth=gt, provenance=prov)


@GENERATORS.register("sequential_par")
def sequential_par(training: list[np.ndarray], seed: int = 0, epochs: int = 128) -> TimeSeriesSample:
    """Fit a PAR model on a small real cohort and sample a new sequence (SDV).

    Learns cross-time dependence from a group of similar real series and samples
    a fresh one. Real-derived and dependent on the cohort distribution; gate with
    the audit. Verify the SDV PARSynthesizer API against the installed version.
    """
    raise NotImplementedError(
        "wire SDV PARSynthesizer here; kept as an explicit integration point "
        "because the fit/sample API and the required metadata object change "
        "across SDV releases and must be pinned"
    )
