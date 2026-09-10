"""Does the narrator say what the measurement actually says?

`compare.check_contrast_text` and `describe.check_text` answer a narrower
question than their acceptance rates suggest. They ask whether a sentence is
ADMISSIBLE -- that every concept in it is licensed, that no direction word
contradicts a sign, that no capability or ranking claim appears. A sentence
can clear all of that and still be nearly contentless ("removing this role
changes the forecast"), or name the licensed concept while describing the
wrong one of the two roles. Nothing in this repo measured that, because
there was never a case whose answer was known in advance.

This module builds those cases. A case is a simple synthetic series, a
PLANTED causal change to its forecast, and the sentence properties that
change entails -- and the channel effects are computed by
`response.battery_statistics`, the same function the real battery calls, not
by asserting the numbers the planting intended. So a failure here implicates
the whole chain (statistic -> evidence packet -> prompt -> sentence), which
is the point: `CLAUDE.md` sec 11.41's discipline is to validate an instrument
against a case whose answer you already know, and every existing check on
this narrator is a check against a case whose answer nobody knew.

🔴 Each case moves exactly ONE channel on purpose. A case where two channels
move admits a sentence that names either, so a scorer over it cannot
distinguish "described the effect" from "named a licensed word"; the
single-channel design is what makes `expected_sign` a decidable claim rather
than a preference.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

import numpy as np

from . import describe as _d
from . import response as _r

__all__ = ["SyntheticCase", "make_cases", "case_channels", "channel_null",
           "unmoved_concepts", "score_description", "score_all"]

HORIZON = 64
CONTEXT = 128
PERIOD = 16.0
_RNG = np.random.default_rng(0)


@dataclass
class SyntheticCase:
    """One planted causal change plus what any correct sentence must do."""

    name: str
    truth: str
    """The planted answer in one English clause, for the report table."""

    channel: str | None
    """The single channel the plant moves, or None for the no-effect case."""

    expected_sign: int
    """+1, -1, or 0. Scored against a direction word, never against a number."""

    baseline: np.ndarray
    steered: np.ndarray
    targets: np.ndarray
    contexts: np.ndarray
    periods: np.ndarray
    forbidden_concepts: tuple = _dc_field(default_factory=tuple)
    """Concepts a correct sentence must NOT name -- the fabrication half."""


def _series(n: int, kind: str) -> tuple:
    """`(contexts, baseline_forecast, targets)` for one simple family."""
    t_ctx = np.arange(CONTEXT, dtype=float)
    t_fut = np.arange(CONTEXT, CONTEXT + HORIZON, dtype=float)
    if kind == "seasonal":
        f = lambda t: np.sin(2 * np.pi * t / PERIOD)
    elif kind == "trend":
        f = lambda t: 0.02 * t
    else:
        f = lambda t: np.zeros_like(t)
    ctx = np.stack([f(t_ctx) + 0.01 * _RNG.standard_normal(CONTEXT) for _ in range(n)])
    base = np.stack([f(t_fut) for _ in range(n)])
    tgt = base + 0.01 * _RNG.standard_normal((n, HORIZON))
    return ctx, base, tgt


def make_cases(n_series: int = 24) -> list:
    """Six cases, each with exactly one channel planted (or none)."""
    cases = []
    h = np.arange(HORIZON, dtype=float)

    ctx, base, tgt = _series(n_series, "seasonal")
    per = np.full(n_series, PERIOD)

    cases.append(SyntheticCase(
        name="trend_up", truth="the forecast's trend slope goes UP",
        channel="trend", expected_sign=+1,
        baseline=base, steered=base + 0.05 * (h - h.mean()), targets=tgt, contexts=ctx,
        periods=per, forbidden_concepts=("channel:seasonal",)))

    cases.append(SyntheticCase(
        name="trend_down", truth="the forecast's trend slope goes DOWN",
        channel="trend", expected_sign=-1,
        baseline=base, steered=base - 0.05 * (h - h.mean()), targets=tgt, contexts=ctx,
        periods=per, forbidden_concepts=("channel:seasonal",)))

    cases.append(SyntheticCase(
        name="level_up", truth="the forecast's overall level goes UP",
        channel="level", expected_sign=+1,
        baseline=base, steered=base + 2.0, targets=tgt, contexts=ctx,
        periods=per, forbidden_concepts=("channel:seasonal",)))

    cases.append(SyntheticCase(
        name="seasonal_up", truth="the forecast's seasonal magnitude goes UP",
        channel="seasonal", expected_sign=+1,
        baseline=base, steered=base * 3.0, targets=tgt, contexts=ctx,
        periods=per, forbidden_concepts=("channel:trend",)))

    cases.append(SyntheticCase(
        name="seasonal_down", truth="the forecast's seasonal magnitude goes DOWN",
        channel="seasonal", expected_sign=-1,
        baseline=base, steered=base * 0.2, targets=tgt, contexts=ctx,
        periods=per, forbidden_concepts=("channel:trend",)))

    # 🔴 The sec 11.37 case, and the one worth reading first: an intervention
    # that does NOTHING. A narrator that always produces a confident effect
    # sentence scores perfectly on the five above and fails only here.
    cases.append(SyntheticCase(
        name="no_effect", truth="nothing moves; no channel clears its null",
        channel=None, expected_sign=0,
        baseline=base, steered=base.copy(), targets=tgt, contexts=ctx,
        periods=per, forbidden_concepts=("channel:trend", "channel:seasonal", "channel:level")))
    return cases


def _smooth_perturbation(shape: tuple, rng, scale: float) -> np.ndarray:
    """A random LOW-FREQUENCY perturbation, standing in for a decoder direction.

    🔴 White noise is the wrong null and using it changed the answer. Additive
    white noise moves the horizon-shape statistics enormously (they compare
    two curves point by point) while barely moving trend or level, so every
    synthetic case ranked a horizon channel first and the planted channel
    third. Measured against the real artifacts, that is not what a live null
    looks like: across 201 real ablation candidates the rank-1 channel is
    spread over all nine (`dispersion` 18%, `horizon_shape_near` 16%, ...,
    `trend` 8%), with no horizon dominance at all. A decoder direction
    perturbs a forecast smoothly, so the null must too, or the harness tests
    a regime the pipeline never enters.
    """
    n, h = shape
    t = np.linspace(0.0, 1.0, h)
    out = rng.standard_normal((n, 1)) * scale * 0.1
    out = out + rng.standard_normal((n, 1)) * scale * 0.1 * t[None, :]
    for k in (1.0, 2.0, 3.0):
        amp = rng.standard_normal((n, 1)) * scale * 0.05
        ph = rng.uniform(0, 2 * np.pi, (n, 1))
        out = out + amp * np.sin(2 * np.pi * k * t[None, :] + ph)
    return out


def channel_null(case: SyntheticCase, n_directions: int = 32,
                 seed: int = 0) -> dict:
    """`{channel: sd}` of each channel's effect under RANDOM perturbations.

    The live battery scores every channel against a random-direction null
    because the nine channels have wildly different natural scales -- a
    seasonal-band magnitude and a trend slope are not comparable numbers. A
    single flat constant here reproduced that mistake and cost the
    `trend_down` case its own planted channel at the threshold, so the null
    is measured per channel, the same way and for the same reason.
    """
    rng = np.random.default_rng(seed)
    scale = float(np.std(case.baseline)) or 1.0
    per: dict = {}
    for _ in range(n_directions):
        pert = case.baseline + _smooth_perturbation(case.baseline.shape, rng, scale)
        stats = _r.battery_statistics(pert, case.baseline, case.targets,
                                      case.contexts, case.periods)
        for ch, rec in stats.items():
            if rec.get("available") and rec.get("delta") is not None:
                per.setdefault(ch, []).append(float(np.mean(rec["delta"])))
    return {ch: (float(np.std(v)) or 1e-12) for ch, v in per.items()}


def case_channels(case: SyntheticCase, null: dict | None = None,
                  min_units: float = 2.0) -> dict:
    """`{channel: signed_null_units}` via the REAL battery statistic.

    A channel under `min_units` of its own null is dropped, exactly as a
    channel that does not clear its random-direction null is dropped live.
    """
    null = channel_null(case) if null is None else null
    stats = _r.battery_statistics(case.steered, case.baseline, case.targets,
                                  case.contexts, case.periods)
    out = {}
    for ch, rec in stats.items():
        if not rec.get("available") or rec.get("delta") is None:
            continue
        units = float(np.mean(rec["delta"])) / null.get(ch, 1e-12)
        if abs(units) >= min_units:
            out[ch] = units
    return out


def unmoved_concepts(case: SyntheticCase, channels: dict) -> tuple:
    """Concepts a correct sentence must not name, DERIVED from the measurement.

    Asserting a forbidden list by hand would be asserting what the plant was
    supposed to do; a channel that in fact moved is legitimate for the
    narrator to mention even when the plant did not intend it.
    """
    moved = {_d.channel_concept(ch) for ch in channels}
    return tuple(_d.channel_concept(c) for c in ("trend", "seasonal", "level")
                 if _d.channel_concept(c) not in moved)


def score_description(text: str, case: SyntheticCase, channels: dict) -> dict:
    """Compare one sentence against what the measurement actually says.

    🔴 The criterion is NOT "names the planted channel", and the reason is
    physical rather than a concession. A forecast statistic cannot be moved
    in isolation: steepening a ramp raises the forecast's spread, and scaling
    a seasonal component raises it too, so the planted channel is genuinely
    rank two or three of five-to-seven that genuinely moved. A one-sentence
    description cannot name seven channels, so requiring the planted one
    would score a faithful sentence as wrong.

    What IS decidable, and is what this scores:
      * `truthful`   -- every channel it names really moved, the direction it
                        gives is the measured direction, and it names at
                        least one moved channel. This is the pass criterion.
      * `names_top`  -- it names the STRONGEST measured channel. Informativeness
                        rather than correctness, reported beside it.
      * `names_planted` -- it names the planted one. Kept because it is what
                        the plant was for, and reported, never asserted.

    Scored through `describe.py`'s own vocabulary tables, so "named the
    concept" means here exactly what it means in the guard.
    """
    norm = _d._normalize(text or "")
    named = set()
    for term, concepts in _d.DOMAIN_TERMS.items():
        if _d._TERM_PATTERNS[term].search(norm):
            named.update(concepts)

    moved = {_d.channel_concept(ch): v for ch, v in channels.items()}
    unmoved = unmoved_concepts(case, channels)
    fabricated = sorted(c for c in unmoved if c in named)

    named_moved = {c: v for c, v in moved.items() if c in named}
    up = any(_d._UP_PATTERNS[t].search(norm) for t in _d.UP_VERBS)
    down = any(_d._DOWN_PATTERNS[t].search(norm) for t in _d.DOWN_VERBS)

    if not channels:
        # The no-effect case: any moved-channel claim or direction word is wrong.
        direction_ok = not (up or down)
        truthful = direction_ok and not named_moved and not fabricated
    elif not named_moved:
        direction_ok = False
        truthful = False
    else:
        signs = {1 if v > 0 else -1 for c, v in named_moved.items()
                 if not _is_unsigned_concept(c)}
        if len(signs) == 1:
            want = signs.pop()
            direction_ok = (up and not down) if want > 0 else (down and not up)
        else:
            # Mixed or unsigned-only: no single direction word is correct,
            # so the honest check is that it did not assert one.
            direction_ok = not (up or down)
        truthful = direction_ok and not fabricated

    top = max(channels, key=lambda k: abs(channels[k])) if channels else None
    return {
        "case": case.name,
        "truth": case.truth,
        "text": (text or "").strip(),
        "named_moved": sorted(named_moved),
        "fabricated_concepts": fabricated,
        "direction_ok": bool(direction_ok),
        "names_top": bool(top and _d.channel_concept(top) in named),
        "names_planted": bool(case.channel
                              and _d.channel_concept(case.channel) in named),
        "truthful": bool(truthful),
    }


def _is_unsigned_concept(concept: str) -> bool:
    """A horizon-shape concept is a SIZE; a direction word on it is a defect."""
    return concept in {_d.channel_concept(c) for c in _d.CHANNEL_VERB}


def score_all(rows: list) -> dict:
    """`{n, n_correct, by_case}` over `score_description` outputs."""
    return {"n": len(rows),
            "n_truthful": sum(1 for r in rows if r["truthful"]),
            "n_names_top": sum(1 for r in rows if r["names_top"]),
            "n_names_planted": sum(1 for r in rows if r["names_planted"]),
            "n_direction_ok": sum(1 for r in rows if r["direction_ok"]),
            "n_fabricating": sum(1 for r in rows if r["fabricated_concepts"])}
