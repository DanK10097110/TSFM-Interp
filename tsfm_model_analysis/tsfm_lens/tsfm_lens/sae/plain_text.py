"""Shared plain-language title machinery for causal-ablation profiles, used
by both `sae/concepts.py` (tight per-target concepts' `plain_name`) and
`sae/concept_families.py` (cross-model families' `title`).

Factored out here rather than left in `concept_families.py` (where it was
first written, ROADMAP.md sec 37 F1) to avoid an import cycle:
`concept_atlas.py` imports from `concepts.py`, and `concept_families.py`
imports from `concept_atlas.py`, so `concepts.py` importing
`concept_families.py` directly would close a cycle back to itself
(`concepts -> concept_families -> concept_atlas -> concepts`). This module
sits below all three (it only imports `response.py`'s `CHANNELS` and
`describe.py`'s vocabulary, neither of which imports `concepts.py` or
`concept_atlas.py`), so both callers import it instead of each other
(`CLAUDE.md` sec 11.52's "make the import lazy" lesson generalizes here to
"put the shared thing where neither needs the other").

SIGN CONVENTION (restated from `concept_families.py`'s own docstring, since
this is the one place both callers get it from): `response.py` records
`signed_effect` as the effect OF ABLATING (removing) a feature. A title
should state what the feature's PRESENCE does, so `directed_profile` negates
the raw profile before anything else in this module sees it. Both callers
title text they render from an already-negated vector; neither should negate
again.
"""
from __future__ import annotations

import numpy as np

from .describe import CHANNEL_GLOSS, CHANNEL_VERB
from .response import CHANNELS

__all__ = ["CLEAR_UNITS", "TITLE_NOUN", "FALLBACK_TITLE", "TITLE_SUFFIXES",
          "LEVEL_CARRIER_TITLE", "GENERATOR_PLAIN_LABELS",
          "directed_profile", "cleared_ranked", "direction_word", "compose_title",
          "plain_generator_label", "axis_channel_label"]

CLEAR_UNITS = 1.0  # same "cleared its own null" cut concepts.py/concept_atlas.py use.

_VERB_PLURAL = {"increases": "increase", "decreases": "decrease", "moves": "move"}

# Short, title-cased noun phrases for a profile's dominant (post-negation)
# channel + direction -- a SEPARATE vocabulary from `describe.py::
# CHANNEL_GLOSS` on purpose: a title is a 2-5 word label, not a clause that
# fits inside a generated sentence, and conflating the two jobs is what
# produced the atlas's own "strong raises horizon_shape_near" machine names
# this module exists to replace. `None` direction covers a channel whose
# statistic (mean ABSOLUTE deviation) carries no sign -- `CHANNEL_VERB` marks
# the same two channels undirected for the identical reason.
TITLE_NOUN = {
    ("trend", "up"): "Trend boosters", ("trend", "down"): "Trend dampeners",
    ("seasonal", "up"): "Seasonality amplifiers", ("seasonal", "down"): "Seasonality dampeners",
    ("spectral_centroid", "up"): "High-frequency shifters",
    ("spectral_centroid", "down"): "Smoothing features",
    ("level", "up"): "Level raisers", ("level", "down"): "Level lowerers",
    ("dispersion", "up"): "Volatility amplifiers", ("dispersion", "down"): "Volatility dampeners",
    ("horizon_shape_near", None): "Near-horizon shapers",
    ("horizon_shape_far", None): "Far-horizon shapers",
    ("mase", "up"): "Accuracy degraders", ("mase", "down"): "Accuracy improvers",
    ("flatness", "up"): "Flattening features", ("flatness", "down"): "De-flattening features",
}
FALLBACK_TITLE = "Mixed-effect features"
TITLE_SUFFIXES = ("II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")

# ROADMAP.md sec 37.7 P4's level-carrier guard, extended to families
# (orchestrator review of F1): a group whose members' effect is
# indistinguishable from a level shift (median `level_share` at/above
# `concepts.level_share_threshold`, and no member clears a level-removed
# shape channel) must not be titled from its raw, level-confounded channel
# profile -- exactly the failure mode `concept_atlas.py::_name_concepts`
# already guards tight atlas concepts against. This is the fallback title
# when the family's own `level` channel does not itself clear its null in a
# resolvable direction (`compose_title` still prefers "Level raisers"/"Level
# lowerers" when the `level` channel's OWN sign is resolvable -- level_share
# says how MUCH of the effect is level, not which way).
LEVEL_CARRIER_TITLE = "Level shifters"

# Plain-English labels for the generator/archetype names
# `sae/concept_profiles.json`'s per-part hypergeometric enrichment reports
# (`input_profile.enrichment[0]["label"]`, the bare `generator_*`/`archetype_*`
# name with its prefix stripped -- see `describe.py::FIELD_GLOSS` for the
# prefixed canonical list this mirrors). An unmapped name falls back to
# itself (`plain_generator_label`) rather than guessing a phrase (CLAUDE.md
# sec 8: degrade with a stated fallback).
GENERATOR_PLAIN_LABELS = {
    "parametric": "synthetic parametric series (trend, seasonality and noise, exact ground truth)",
    "random_parametric": "synthetic randomized-recipe series",
    "mixture": "real weather/ETT-derived mixtures",
    "block_bootstrap": "resampled real weather series",
    "sequential_par": "sequential-PAR real-derived series",
    "trend_dominant": "strongly trending series",
    "seasonal_dominant": "strongly seasonal series",
    "multi_seasonal_complex": "series with several overlapping seasonal cycles",
    "regime_switching": "series with regime changes",
    "ar_colored_noise": "noisy autocorrelated series",
    "anomaly_heavy": "series with frequent anomalies",
    "clean_low_noise": "clean, low-noise series",
    "noisy_chaotic": "noisy, chaotic series",
    "random_walk_drift": "random-walk-with-drift series",
    "intermittent_bursts": "intermittent/bursty series",
    "amplitude_modulated": "amplitude-modulated series",
    "nonsinusoidal_seasonal": "non-sinusoidal seasonal series",
}


def plain_generator_label(name: str) -> str:
    """Plain-English label for a bare generator/archetype name, falling back
    to `name` itself when unmapped."""
    return GENERATOR_PLAIN_LABELS.get(name, name)


def axis_channel_label(channel: str) -> str:
    """Short, plain axis/legend/chip label for one ablation channel (ROADMAP.md
    sec 37 R2), DERIVED from `describe.py::CHANNEL_GLOSS` -- never a second,
    independently hand-maintained vocabulary for the same nine channels
    (`CLAUDE.md` sec 11.53's "two hand-maintained lists" defect shape).
    `CHANNEL_GLOSS` gives a full clause ("the forecast's trend slope"); this
    strips the leading "the forecast's "/"the " so the result reads as a
    short axis tick or table header rather than a sentence fragment. Falls
    back to the raw key only for a channel `CHANNEL_GLOSS` does not cover
    (never expected in practice: every entry in `response.py::CHANNELS` has
    a `CHANNEL_GLOSS` entry, asserted at import time by `describe.py`)."""
    text = CHANNEL_GLOSS.get(channel, channel)
    for prefix in ("the forecast's ", "the "):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return (text[:1].upper() + text[1:]) if text else channel


def directed_profile(mean_profile_vec) -> np.ndarray:
    """The "feature does" convention (module docstring's SIGN CONVENTION):
    negate the raw (ablation-effect) mean profile."""
    return -np.asarray(mean_profile_vec, dtype=np.float64)


def cleared_ranked(directed_vec: np.ndarray) -> list:
    order = np.argsort(-np.abs(directed_vec))
    return [int(i) for i in order if abs(directed_vec[i]) >= CLEAR_UNITS]


def direction_word(channel: str, val: float):
    if channel in CHANNEL_VERB:
        return None
    return "up" if val > 0 else "down"


def compose_title(directed_vec: np.ndarray, cleared: list, used_titles: set) -> str:
    """The plain-language title for one profile (dominant cleared channel's
    noun phrase, from `TITLE_NOUN`), de-duplicated against every OTHER title
    already chosen in this run via a Roman-numeral suffix -- `used_titles` is
    mutated in place, so callers must process every profile through the same
    set in one deterministic order (never per-profile in isolation) for
    run-wide uniqueness to hold."""
    if not cleared:
        base = FALLBACK_TITLE
    else:
        ch = CHANNELS[cleared[0]]
        d = direction_word(ch, directed_vec[cleared[0]])
        base = TITLE_NOUN.get((ch, d), TITLE_NOUN.get((ch, None), FALLBACK_TITLE))
    title = base
    k = 0
    while title in used_titles:
        suffix = TITLE_SUFFIXES[min(k, len(TITLE_SUFFIXES) - 1)]
        title = f"{base} {suffix}"
        k += 1
    used_titles.add(title)
    return title
