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

from .describe import CHANNEL_VERB
from .response import CHANNELS

__all__ = ["CLEAR_UNITS", "TITLE_NOUN", "FALLBACK_TITLE", "TITLE_SUFFIXES",
          "directed_profile", "cleared_ranked", "direction_word", "compose_title"]

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
