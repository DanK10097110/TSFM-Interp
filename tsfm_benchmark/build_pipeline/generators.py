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


_ARCHETYPES: dict[str, dict[str, tuple[float, float]]] = {
    # Each archetype is a qualitatively distinct structural regime. Ranges are
    # (low, high) and are sampled fresh per generated series, so two series
    # from the same archetype still differ in exact shape, and series across
    # archetypes differ in kind, not just noise realization.
    "trend_dominant": dict(
        trend_order=(1, 3), trend_scale=(0.8, 2.5),
        n_seasonal=(0, 1), period=(20, 60), amplitude=(0.05, 0.3),
        ar_order=(0, 1), ar_coeff=(-0.2, 0.2),
        noise_scale=(0.05, 0.15),
        n_changepoints=(0, 1), n_anomalies=(0, 2), anomaly_magnitude=(2.0, 4.0),
    ),
    "seasonal_dominant": dict(
        trend_order=(0, 1), trend_scale=(0.0, 0.3),
        n_seasonal=(1, 2), period=(12, 48), amplitude=(0.8, 2.0),
        ar_order=(0, 1), ar_coeff=(-0.2, 0.2),
        noise_scale=(0.05, 0.15),
        n_changepoints=(0, 0), n_anomalies=(0, 2), anomaly_magnitude=(2.0, 4.0),
    ),
    "multi_seasonal_complex": dict(
        trend_order=(0, 2), trend_scale=(0.0, 0.6),
        n_seasonal=(2, 4), period=(6, 200), amplitude=(0.3, 1.5),
        ar_order=(0, 2), ar_coeff=(-0.3, 0.3),
        noise_scale=(0.05, 0.2),
        n_changepoints=(0, 1), n_anomalies=(0, 2), anomaly_magnitude=(2.0, 5.0),
    ),
    "regime_switching": dict(
        trend_order=(0, 1), trend_scale=(0.0, 0.4),
        n_seasonal=(0, 2), period=(10, 80), amplitude=(0.3, 1.2),
        ar_order=(0, 1), ar_coeff=(-0.2, 0.2),
        noise_scale=(0.1, 0.3),
        n_changepoints=(3, 8), n_anomalies=(0, 2), anomaly_magnitude=(2.0, 4.0),
    ),
    "ar_colored_noise": dict(
        trend_order=(0, 1), trend_scale=(0.0, 0.3),
        n_seasonal=(0, 1), period=(12, 60), amplitude=(0.0, 0.5),
        ar_order=(2, 4), ar_coeff=(-0.6, 0.6),
        noise_scale=(0.1, 0.25),
        n_changepoints=(0, 1), n_anomalies=(0, 1), anomaly_magnitude=(2.0, 4.0),
    ),
    "anomaly_heavy": dict(
        trend_order=(0, 1), trend_scale=(0.0, 0.3),
        n_seasonal=(0, 2), period=(12, 100), amplitude=(0.2, 1.0),
        ar_order=(0, 1), ar_coeff=(-0.2, 0.2),
        noise_scale=(0.05, 0.15),
        n_changepoints=(0, 1), n_anomalies=(5, 12), anomaly_magnitude=(3.0, 8.0),
    ),
    "clean_low_noise": dict(
        trend_order=(0, 2), trend_scale=(0.0, 1.0),
        n_seasonal=(1, 3), period=(10, 150), amplitude=(0.3, 1.5),
        ar_order=(0, 0), ar_coeff=(0.0, 0.0),
        noise_scale=(0.01, 0.05),
        n_changepoints=(0, 1), n_anomalies=(0, 1), anomaly_magnitude=(2.0, 3.0),
    ),
    "noisy_chaotic": dict(
        trend_order=(0, 2), trend_scale=(0.0, 1.0),
        n_seasonal=(0, 2), period=(10, 150), amplitude=(0.1, 1.0),
        ar_order=(1, 3), ar_coeff=(-0.85, 0.85),
        noise_scale=(0.3, 0.6),
        n_changepoints=(0, 3), n_anomalies=(0, 4), anomaly_magnitude=(2.0, 6.0),
    ),
}


def _ri(rng: np.random.Generator, lo: float, hi: float) -> int:
    lo, hi = int(lo), int(hi)
    return lo if hi <= lo else int(rng.integers(lo, hi + 1))


def _ru(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(lo) if hi <= lo else float(rng.uniform(lo, hi))


def _stable_ar_coeffs(rng: np.random.Generator, order: int, max_reflection: float) -> list[float]:
    """Sample AR(order) coefficients that are guaranteed stationary, for any order.

    Sampling each coefficient independently within a fixed magnitude bound is
    NOT safe once order >= 2: stationarity depends on every root of the AR
    characteristic polynomial lying outside the unit circle, a joint
    condition that per-coefficient bounds do not enforce. Two coefficients
    each comfortably under 1 in magnitude can still combine into a process
    that explodes exponentially (observed in practice here: an AR(3) draw
    with |coeffs| <= 0.85 each produced values up to ~1e96 on a length-512
    series). Sampling reflection coefficients in (-max_reflection,
    max_reflection) and running the Levinson-Durbin recursion instead
    guarantees a stationary process by construction, for any order, with no
    rejection sampling needed.
    """
    if order <= 0:
        return []
    reflection = rng.uniform(-max_reflection, max_reflection, size=order)
    a = np.zeros(order)
    for m in range(order):
        prev = a.copy()
        k = reflection[m]
        a[m] = k
        for i in range(m):
            a[i] = prev[i] - k * prev[m - 1 - i]
    return [round(float(c), 4) for c in a]


def _sample_archetype_params(rng: np.random.Generator, spec: dict[str, tuple[float, float]]) -> dict[str, Any]:
    trend = {"order": _ri(rng, *spec["trend_order"]), "scale": _ru(rng, *spec["trend_scale"])}

    seasonalities = []
    for _ in range(_ri(rng, *spec["n_seasonal"])):
        seasonalities.append({
            "period": round(_ru(rng, *spec["period"]), 2),
            "amplitude": round(_ru(rng, *spec["amplitude"]), 4),
            "phase": round(float(rng.uniform(0, 2 * np.pi)), 4),
        })

    ar_order = _ri(rng, *spec["ar_order"])
    max_reflection = max(abs(spec["ar_coeff"][0]), abs(spec["ar_coeff"][1]))
    ar_coeffs = _stable_ar_coeffs(rng, ar_order, max_reflection) if ar_order else None

    return dict(
        trend=trend,
        seasonalities=seasonalities or None,
        ar_coeffs=ar_coeffs,
        noise_scale=round(_ru(rng, *spec["noise_scale"]), 4),
        n_changepoints=_ri(rng, *spec["n_changepoints"]),
        n_anomalies=_ri(rng, *spec["n_anomalies"]),
        anomaly_magnitude=round(_ru(rng, *spec["anomaly_magnitude"]), 4),
    )


@GENERATORS.register("random_parametric")
def random_parametric(
    seed: int = 0,
    length: int = 512,
    archetypes: list[str] | None = None,
    archetype_weights: list[float] | None = None,
) -> TimeSeriesSample:
    """Pick a random structural archetype per call and sample its hyperparameters.

    A plain ``parametric`` task run ``count`` times keeps every structural
    choice (trend order, seasonal periods, AR coefficients, ...) fixed across
    the whole task and only varies the noise draw, so all its series share one
    shape family; a model can shortcut on that shape rather than learning to
    read the interpretability structure generically. This generator instead
    redraws the entire structural recipe from one of several qualitatively
    different regimes (trend-dominant, seasonal, regime-switching, colored
    noise, anomaly-heavy, clean, chaotic, ...) on every call, so a task's
    ``count`` repeats span a genuinely varied population. Delegates the actual
    composition to ``parametric`` so ground truth and provenance stay exact;
    the chosen archetype and sampled recipe are recorded alongside it.
    """
    rng = np.random.default_rng(seed)
    names = archetypes or list(_ARCHETYPES)
    unknown = sorted(set(names) - set(_ARCHETYPES))
    if unknown:
        raise ValueError(f"unknown archetype(s) {unknown}; have {sorted(_ARCHETYPES)}")

    weights = None
    if archetype_weights is not None:
        w = np.asarray(archetype_weights, dtype=float)
        weights = w / w.sum()
    archetype = names[int(rng.choice(len(names), p=weights))]

    params = _sample_archetype_params(rng, _ARCHETYPES[archetype])
    inner_seed = int(rng.integers(0, 2**31 - 1))
    sample = parametric(length=length, seed=inner_seed, **params)

    sample.ground_truth.generative_params["archetype"] = archetype
    sample.ground_truth.notes = f"fully synthetic; leakage-safe tier; archetype={archetype}"
    sample.provenance.generator = "random_parametric"
    sample.provenance.generator_params = {"length": length, "archetype": archetype, "sampled_params": params, "inner_seed": inner_seed}
    sample.provenance.seed = seed
    return sample


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
