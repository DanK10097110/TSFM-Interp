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

    `how` is optional and carries the ARITHMETIC -- what was divided by
    what, against which reference. It exists because the three-field form
    above was not enough for the health table's own columns (user review,
    2026-09-11: "I don't understand how change in mase sign or alignment
    mean abs rho is calculated"): for a channel, naming the forecast
    property is the whole definition, but for a derived statistic a reader
    cannot tell a ratio from a difference from a rank correlation without
    being told. Left empty for the channels and ground-truth fields, whose
    `what` already is the calculation.
    """
    label: str
    what: str
    high: str
    how: str = ""


# --- causal response channels (the heatmap's x-axis) ------------------------
# Each is a property of the model's FORECAST. A cell measures how much
# patching this feature moves that property, in units of the
# random-direction null's 95th percentile: 1.0 means "exactly as large as
# chance", so only |value| meaningfully above 1.0 is evidence of anything.

CHANNEL_DEFS: dict = {
    "trend": TermDef(
        "Trend",
        "Slope of the forecast -- whether it rises or falls as it continues forward.",
        "Feature pushes the forecast to slope upward more steeply."),
    "seasonal": TermDef(
        "Seasonal strength",
        "Energy in the forecast concentrated at the series' own dominant "
        "repeating rhythm.",
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
        "Feature makes the forecast swing more from step to step."),
    "horizon_shape_near": TermDef(
        "Near horizon",
        "Shape of the first part of the forecast -- the steps just after the context ends.",
        "Feature changes the near-term forecast's shape by a larger amount."),
    "horizon_shape_far": TermDef(
        "Far horizon",
        "Shape of the late part of the forecast -- its furthest-out steps.",
        "Feature changes the long-range forecast's shape by a larger amount."),
    "mase": TermDef(
        "Accuracy (MASE)",
        "Forecast error against the true continuation, scaled by a naive baseline.",
        "Feature makes the forecast WORSE (higher error)."),
    "flatness": TermDef(
        "Flatness",
        "How close the forecast is to a constant line.",
        "Feature makes the forecast closer to a flat, constant line."),
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


# --- the section's own derived statistics (table columns, not data) ---------
# Added 2026-09-11 on user review. Every entry below was a bare `<th>` in the
# rendered report with no definition anywhere in the 5.4 MB document -- and
# two of them (`ΔMASE sign`, `alignment mean abs rho`) were the ones the
# reader specifically could not act on. Unlike CHANNEL_DEFS and FIELD_DEFS
# these are not properties of the data, they are things this pipeline
# COMPUTED, so each carries `how`: what was divided by what, against which
# reference. Keys are the rendered column labels verbatim, because the
# renderer looks them up by the header it just printed -- a key that drifts
# from its header shows up as a missing row in the legend, which is visible,
# rather than as a wrong definition, which is not.

METRIC_DEFS: dict = {
    "target": TermDef(
        "Target",
        "One trained dictionary: a model and the single layer of it this "
        "dictionary was fitted on. Every other column is a property of that "
        "one dictionary, so two rows differ in the model, the layer and the "
        "fit at once.",
        "",
        "Named `model/layer` exactly as the `sae` stage resolved it."),
    "reconstruction fidelity": TermDef(
        "Reconstruction fidelity",
        "How faithfully the sparse dictionary can rebuild the layer's own "
        "activations from a handful of active features. This is the FLOOR "
        "on everything below it: a dictionary that cannot reproduce the "
        "layer is not describing the layer.",
        "",
        "1 - (sum of squared reconstruction error / sum of squared error of "
        "predicting the mean activation), over the activations the "
        "dictionary was fitted on. 1.0 is exact; 0.0 is no better than "
        "predicting the mean; negative is worse than that."),
    "reconstruction fidelity (held out)": TermDef(
        "Reconstruction fidelity (held out)",
        "The same quantity on SERIES the dictionary never saw during "
        "training. A large drop from the fitted value means the dictionary "
        "memorized these series rather than learning the layer's structure.",
        "",
        "Identical formula, evaluated on a held-out split of whole SERIES "
        "(never windows -- windows inside one series are dependent, so a "
        "window split would leak). Drawn as a diamond on the same bar as "
        "the fitted value, so the gap between them is a visible length."),
    "dead rate": TermDef(
        "Dead-feature share",
        "The fraction of the dictionary's features that never activate on "
        "any input. A dead feature is capacity that was allocated and never "
        "used, so a high rate means every feature shown was drawn from a "
        "small alive minority rather than from the whole dictionary.",
        "",
        "(features whose activation is exactly zero on every probed row) / "
        "(dictionary size)."),
    "dead-rate gate": TermDef(
        "Dead-feature gate",
        "The pass/fail verdict on the dead share, printed together with the "
        "bar it was compared against rather than only as a word.",
        "",
        "`dead rate` compared to the configured acceptance threshold. A "
        "target with no threshold configured reads `no gate configured` -- "
        "its rate was recorded but never checked, which is not the same as "
        "passing."),
    "ΔMASE (window)": TermDef(
        "Forecast damage, window granularity",
        "How much the model's forecast error changes when the layer's real "
        "activations are replaced by the dictionary's reconstruction of "
        "them. Near zero means the dictionary is a faithful enough "
        "stand-in that the model still forecasts the same way; large "
        "positive means the reconstruction threw away something the "
        "forecast needed.",
        "",
        "MASE(forecast from the reconstructed activations) - MASE(forecast "
        "from the real activations), with the reconstruction done on the "
        "pooled WINDOW axis and then broadcast back over the tokens each "
        "window covers. That broadcast is lossy for any model whose tokens "
        "are narrower than the window, which is why the token column beside "
        "it exists."),
    "ΔMASE (token)": TermDef(
        "Forecast damage, token granularity",
        "The same measurement with no window pooling: every raw token is "
        "encoded and decoded on its own. This is the column to read as "
        "dictionary quality -- it has no architecture confound in it.",
        "",
        "MASE(reconstructed) - MASE(real), reconstructing each token "
        "independently."),
    "ΔMASE (token, held out)": TermDef(
        "Forecast damage (held out)",
        "Token-granularity forecast damage measured on series the "
        "dictionary was never fitted on.",
        "",
        "Same difference of MASEs, evaluated on the held-out series split."),
    "granularity gap": TermDef(
        "Granularity gap",
        "The distance between the two ΔMASE columns. It measures an "
        "ARCHITECTURE difference, not a difference in dictionary quality: "
        "it is exactly 0 for a model whose token width equals the "
        "alignment window (nothing is broadcast, so the two are the same "
        "computation) and grows with the mismatch.",
        "",
        "|ΔMASE (window) - ΔMASE (token)|. A large value says the window "
        "number is inflated by pooling loss and should not be read as "
        "dictionary damage."),
    "ΔMASE vs own floor": TermDef(
        "Forecast damage against this model's own noise floor",
        "Whether the forecast damage is larger than the amount this model's "
        "forecast moves between two identical runs. A model that samples "
        "its forecast has a floor above zero; a deterministic one has a "
        "floor of exactly zero, so any nonzero damage clears it by "
        "arithmetic and the comparison says nothing.",
        "",
        "|ΔMASE (token)| divided by that model's measured repeat-run MASE "
        "spread. Reads `no floor (deterministic)` when the model's repeat "
        "spread is exactly zero, rather than reporting a division by it."),
    "ΔMASE sign": TermDef(
        "Is the ΔMASE sign resolvable?",
        "Whether the SIGN of the forecast damage survives retraining the "
        "dictionary at a different random seed. Training an SAE is "
        "stochastic; if refitting it moves ΔMASE by more than ΔMASE itself, "
        "then 'the reconstruction made the forecast better' and 'worse' are "
        "the same measurement and neither should be quoted.",
        "",
        "The dictionary is retrained at several seeds against a FROZEN "
        "activation store (so seed is the only thing that varies) and the "
        "standard deviation of ΔMASE across those seeds is recorded. "
        "`resolvable` means |ΔMASE| exceeds that spread; `NOT resolvable` "
        "means it does not. The spread and the seed count are printed in "
        "the cell."),
    "alignment mean abs rho": TermDef(
        "Ground-truth alignment",
        "How strongly this dictionary's features correspond to real, "
        "labelled properties of the input series -- the closest thing here "
        "to 'are these features interpretable'. It is a different question "
        "from reconstruction fidelity, and the two can disagree: a "
        "dictionary can rebuild a layer almost perfectly while none of its "
        "features corresponds to anything nameable, and the whole point of "
        "the section is reading features, not rebuilding layers.",
        "",
        "Each probed feature's activation is correlated (Spearman, across "
        "series) against every candidate ground-truth field; the feature "
        "keeps its single best |ρ|; the column is the MEAN of those best "
        "|ρ| over all matched features. Because each feature keeps its best "
        "of many candidates, this number is inflated above zero by the "
        "search alone -- which is exactly why it is never read against "
        "zero, only against the permutation null beside it."),
    "permutation null p95": TermDef(
        "Label-permutation null (95th percentile)",
        "How large the alignment number gets by pure chance, for THIS "
        "dictionary, at this feature count, under this same best-of-many "
        "search. It is the reference the measured alignment has to beat.",
        "",
        "The ground-truth labels are shuffled across series and the entire "
        "matching procedure is re-run many times; this column is the 95th "
        "percentile of the resulting mean |ρ|. Every ingredient except the "
        "label-to-series correspondence is held fixed, so the difference "
        "between the two columns is the part the labels explain."),
    "alignment vs null": TermDef(
        "Alignment verdict",
        "The signed margin of the measured alignment over this target's own "
        "permutation null, with the verdict it implies.",
        "",
        "`alignment mean abs rho` - `permutation null p95`. Positive means "
        "the feature names carry signal the same search on shuffled labels "
        "would not have produced; zero or negative means they do not."),
    # The seed-floor table names two quantities the health table above also
    # names, under different headers -- `dead-feature rate` for `dead rate`,
    # and a whole question for `ΔMASE sign`. Both are defined here rather
    # than renamed: the header a reader is looking at is the key they can
    # look up, and silently renaming a column would move a recorded number's
    # label (§2.1). The definitions say they are the same quantity.
    "seeds": TermDef(
        "Seeds",
        "How many times this dictionary was retrained from scratch, varying "
        "nothing but the random seed, to size the spread every other number "
        "in this table carries.",
        "",
        "Retrained against a FROZEN activation store, so the model, the "
        "layer and the rows are bit-identical across seeds and training "
        "stochasticity is the only variable."),
    "dead-feature rate": TermDef(
        "Dead-feature share",
        "Same quantity as `dead rate` in the dictionary-health table above, "
        "here as a mean across seeds with its spread.",
        "",
        "(features that never activate) / (dictionary size), averaged over "
        "the retrained seeds."),
    "is that ΔMASE resolvable at one seed?": TermDef(
        "Is the ΔMASE sign resolvable?",
        "Same question as `ΔMASE sign` in the dictionary-health table above: "
        "whether the forecast damage is larger than the amount retraining "
        "the dictionary moves it. Where it is not, 'the reconstruction made "
        "the forecast better' and 'worse' are the same measurement.",
        "",
        "|mean ΔMASE across seeds| compared against the standard deviation "
        "of ΔMASE across those same seeds."),
    "admission": TermDef(
        "Admission",
        "Whether this dictionary was admitted for feature-level reading at "
        "all, and on which bar it failed if not.",
        "",
        "Every configured acceptance check (reconstruction fidelity, "
        "forecast damage) evaluated together. A bar that was never MEASURED "
        "leaves the verdict `undecidable` rather than `passed` -- an "
        "unmeasured check is not a cleared one."),
    # --- roles / causal battery -------------------------------------------
    "layer": TermDef(
        "Layer",
        "Which layer of this model the dictionary was fitted on. Rows are "
        "ordered shallowest-to-deepest, so reading down the column is "
        "reading down the model.",
        "",
        "The layer name the adapter itself reports, not a depth fraction."),
    "role": TermDef(
        "Role",
        "A cluster of sparse features that behave the same way causally: "
        "steering along any of them pushes the same forecast properties in "
        "the same direction. A role is a group of FEATURES, not a group of "
        "series and not a layer.",
        "",
        "Features are clustered on their null-normalized response "
        "fingerprint -- the vector of what steering that feature does to "
        "each forecast channel. The role's NAME is derived from its own "
        "mean effect vector (dominant channel plus sign), never authored."),
    "atoms": TermDef(
        "Atoms",
        "How many individual sparse features are in this role. 'Atom' and "
        "'feature' mean the same thing here: one column of the dictionary.",
        "",
        "Count of member features."),
    "dominant channel": TermDef(
        "Dominant channel",
        "The forecast property this role moves most. A CHANNEL is one "
        "measurable property of the model's forecast (its slope, its "
        "seasonal strength, its level, its error, ...) -- the list is fixed "
        "and each one is defined in the legend under the heatmap.",
        "",
        "The channel with the largest mean |effect| across the role's "
        "member features."),
    "effect (× null p95)": TermDef(
        "Effect, in null units",
        "How far the dominant channel moved, measured in units of what a "
        "RANDOM steering direction of the identical magnitude achieves. "
        "1.0 means exactly as much as chance, so only values meaningfully "
        "above 1.0 are evidence of anything.",
        "",
        "(role's mean signed effect on its dominant channel) / (that "
        "channel's own random-direction null at the 95th percentile). "
        "Channels sit on wildly different natural scales, so dividing each "
        "by its own null is what makes two columns comparable at all."),
    "members clearing it": TermDef(
        "Members clearing that channel",
        "How many of the role's own features individually beat the null on "
        "the dominant channel. A role can be labelled as clearing a null "
        "because SOME member cleared SOME channel while its own mean on its "
        "dominant channel sits below that null -- averaging a signed effect "
        "over many members dilutes it -- so this column is what separates "
        "the two.",
        "",
        "Count of member features whose own |effect| on the dominant "
        "channel exceeds that channel's null p95, out of the role's size."),
    "also moves (× null p95)": TermDef(
        "Also moves",
        "The other forecast channels this role moves past their own nulls, "
        "besides the dominant one. A role that moves several channels is "
        "less specific than one that moves exactly one.",
        "",
        "Every non-dominant channel whose role-mean |effect| exceeds its "
        "own null p95, each in the same null units."),
    "structural correlate": TermDef(
        "Structural correlate",
        "The labelled property of the input series this role's features "
        "correlate with. This is CORRELATIONAL -- it says the features fire "
        "on series with that property, not that they carry it causally. "
        "Only the channel columns are causal evidence.",
        "",
        "Spearman ρ of member activations against the field, after "
        "regressing out corpus provenance (which generator wrote the "
        "series), so a match cannot be a fact about how the benchmark was "
        "built."),
    "silhouette": TermDef(
        "Clustering quality (silhouette)",
        "How cleanly the features actually separate into the roles they "
        "were assigned to. Near 0 means the dictionary is not modular at "
        "this granularity -- which is a real result about the dictionary, "
        "not a failure of the clustering.",
        "",
        "Mean silhouette coefficient over the clustered response "
        "fingerprints: for each feature, (distance to the nearest other "
        "cluster - distance within its own) / the larger of the two."),
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
    if name in METRIC_DEFS:
        return METRIC_DEFS[name]
    if name.startswith(("tier_", "generator_", "archetype_")):
        kind, _, rest = name.partition("_")
        return TermDef(f"{rest.replace('_', ' ')} ({kind})", _PROVENANCE_EXPLAINER,
                       "Series carrying this corpus label.")
    return TermDef(name, f"No definition recorded for `{name}`.", "")


def describe_metric(name: str) -> TermDef | None:
    """Definition for a rendered TABLE COLUMN, or None if none is recorded.

    Deliberately returns `None` rather than `describe_term`'s
    "no definition recorded" placeholder: the legend under a table is built
    from the headers that table actually printed, and a column with no
    definition should be absent from the legend (visibly incomplete) rather
    than present with an apology in it. Matching is case-insensitive and
    ignores surrounding whitespace, because a header is a display string
    and one capitalization change should not silently empty the legend.
    """
    key = str(name).strip()
    if key in METRIC_DEFS:
        return METRIC_DEFS[key]
    lowered = {k.lower(): v for k, v in METRIC_DEFS.items()}
    return lowered.get(key.lower())


def pretty(name: str) -> str:
    """Short human label for an axis tick or table cell."""
    return describe_term(name).label
