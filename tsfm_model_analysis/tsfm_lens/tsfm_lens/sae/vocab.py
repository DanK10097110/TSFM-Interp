"""Plain-English definitions for every term the SAE section renders.

ROADMAP.md sec 26 B1. The SAE section's axes and tables are labelled with the
identifiers the code uses -- `spectral_centroid`, `horizon_shape_far`,
`ar_coeff_sum`, `has_random_walk`. Those are precise and they are also
unreadable to anyone who did not write them, which was the single most
common complaint about that section: a heatmap whose x-axis is nine
undefined words is a picture of nothing.

Two rules this module exists to enforce, both learned the expensive way
elsewhere in this repo:

- **A definition lives next to the thing it defines.** `glossary.py` already
  holds the report's cross-cutting vocabulary, and it is the right home for
  terms that recur across sections. These do not: they are the SAE
  section's own axis ticks and column values, and a reader looking at a
  heatmap cell should not have to leave the figure. So these render as
  hover text on the axis and as an expandable legend under the figure,
  not as a lookup table three sections away.
- **Say what a HIGH value means, not just what the quantity is.** "Spectral
  centroid: the centre of mass of the forecast's frequency spectrum" is a
  definition that still leaves a reader unable to read the cell. The
  `high` field carries the direction, which is the half that makes the
  number actionable.

`CHANNEL_DEFS` keys must stay in sync with `response.py::CHANNELS`, and
`FIELD_DEFS` with the structural (non-provenance) columns of the corpus's
ground-truth table; `tests/test_sae_vocab.py` pins both, so a channel added
without a definition fails a test rather than rendering as a bare word.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TermDef:
    """One rendered identifier, in words a reader can act on.

    `label` is the short human name that replaces the raw identifier on an
    axis; `what` defines the quantity; `high` says what a large positive
    value means, which is what a heatmap cell actually shows.
    """
    label: str
    what: str
    high: str


# --- causal response channels (the heatmap's x-axis) ------------------------
# Each is a property of the model's FORECAST. A cell measures how much
# patching this feature moves that property, in units of the
# random-direction null's 95th percentile: 1.0 means "exactly as large as
# chance", so only |value| meaningfully above 1.0 is evidence of anything.

CHANNEL_DEFS: dict = {
    "trend": TermDef(
        "Trend",
        "Slope of the forecast -- whether it rises or falls across the horizon.",
        "Feature pushes the forecast to slope upward more steeply."),
    "seasonal": TermDef(
        "Seasonal strength",
        "Energy in the forecast at the series' own dominant seasonal frequency.",
        "Feature makes the forecast more strongly periodic."),
    "spectral_centroid": TermDef(
        "Frequency balance",
        "Centre of mass of the forecast's frequency spectrum -- low means the "
        "forecast is dominated by slow, smooth movement, high means by fast wiggles.",
        "Feature shifts the forecast toward higher-frequency, choppier movement."),
    "level": TermDef(
        "Level",
        "Mean value of the forecast -- its overall height.",
        "Feature raises the whole forecast."),
    "dispersion": TermDef(
        "Spread",
        "How much the forecast varies within itself (its standard deviation).",
        "Feature makes the forecast more variable rather than flat."),
    "horizon_shape_near": TermDef(
        "Near horizon",
        "Shape of the first part of the forecast -- the steps just after the context ends.",
        "Feature pushes the near-term forecast upward."),
    "horizon_shape_far": TermDef(
        "Far horizon",
        "Shape of the late part of the forecast -- the steps furthest into the future.",
        "Feature pushes the long-range forecast upward."),
    "mase": TermDef(
        "Accuracy (MASE)",
        "Forecast error against the true continuation, scaled by a naive baseline.",
        "Feature makes the forecast WORSE (higher error)."),
    "flatness": TermDef(
        "Flatness",
        "How close the forecast is to a constant line.",
        "Feature flattens the forecast toward a constant."),
}


# --- structural ground-truth fields (what a feature correlates with) --------
# These are real generative properties of each synthetic series, known
# exactly because this repo generated them. Correlating a feature against
# these is what makes "interpretable" checkable rather than a vibe.

FIELD_DEFS: dict = {
    "ar_coeff_sum": TermDef(
        "Autocorrelation strength",
        "Sum of the series' autoregressive coefficients -- how strongly each point "
        "depends on the ones before it.",
        "Series whose values carry over strongly from step to step (persistent, slow-drifting)."),
    "ar_order": TermDef(
        "Autocorrelation depth",
        "How many past steps the series' autoregressive component looks back over.",
        "Series with longer memory of their own past."),
    "has_heteroskedastic": TermDef(
        "Changing volatility",
        "Whether the series' noise level grows and shrinks over time rather than staying constant.",
        "Series with bursts of high volatility separated by calm stretches."),
    "has_intermittency": TermDef(
        "Intermittency",
        "Whether the series is mostly zeros punctuated by occasional activity "
        "(think retail demand for a rarely-bought item).",
        "Series that are largely flat at zero with sporadic spikes."),
    "has_random_walk": TermDef(
        "Random-walk drift",
        "Whether the series contains a unit-root wander -- a trend that drifts without "
        "returning to any fixed level, like a stock price.",
        "Series that wander persistently instead of reverting to a mean."),
    "n_anomalies": TermDef(
        "Anomaly count",
        "Number of injected point anomalies -- isolated, one-step outliers.",
        "Series with many sudden isolated spikes."),
    "n_changepoints": TermDef(
        "Changepoint count",
        "Number of abrupt level shifts, where the series jumps to a new baseline and stays there.",
        "Series that repeatedly shift to new levels."),
    "n_seasonalities": TermDef(
        "Seasonal component count",
        "How many distinct periodic cycles are superimposed in the series.",
        "Series with several overlapping cycles (e.g. daily plus weekly)."),
    "noise_scale": TermDef(
        "Noise level",
        "Amplitude of the random noise added on top of the series' structure.",
        "Noisier series where the underlying pattern is harder to see."),
    "seasonal_amplitude_max": TermDef(
        "Seasonal amplitude",
        "Size of the largest periodic swing in the series.",
        "Series with large, pronounced cycles."),
    "seasonal_period_dominant": TermDef(
        "Dominant period",
        "Length in timesteps of the series' strongest repeating cycle.",
        "Series whose main cycle is long (slow seasonality)."),
    "trend_order": TermDef(
        "Trend shape",
        "Polynomial order of the deterministic trend -- 1 is a straight line, "
        "2 a curve, and so on.",
        "Series with more sharply curving trends."),
    "trend_scale": TermDef(
        "Trend magnitude",
        "How large the deterministic trend is relative to the rest of the series.",
        "Series dominated by their trend rather than by cycles or noise."),
}


_PROVENANCE_EXPLAINER = (
    "A corpus bookkeeping label -- which generator script, tier, or archetype "
    "produced this series. A feature matching one of these is detecting how the "
    "benchmark was built, NOT a property of time series, so it is reported "
    "separately and never as an interpretability finding."
)


def describe_term(name: str) -> TermDef:
    """Definition for any rendered identifier, provenance dummies included.

    Never raises on an unknown name -- an axis must still render if a new
    field appears -- but returns a `TermDef` that says the term is
    undefined rather than inventing a plausible-sounding gloss, so a
    missing definition is visible instead of silently papered over.
    """
    if name in CHANNEL_DEFS:
        return CHANNEL_DEFS[name]
    if name in FIELD_DEFS:
        return FIELD_DEFS[name]
    if name.startswith(("tier_", "generator_", "archetype_")):
        kind, _, rest = name.partition("_")
        return TermDef(f"{rest.replace('_', ' ')} ({kind})", _PROVENANCE_EXPLAINER,
                       "Series carrying this corpus label.")
    return TermDef(name, f"No definition recorded for `{name}`.", "")


def pretty(name: str) -> str:
    """Short human label for an axis tick or table cell."""
    return describe_term(name).label
