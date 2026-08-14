"""Stage 0's alive-atom floor, after it stopped being a global constant.

ROADMAP.md §6.2.1's DECISION (2026-08-12) replaced `MIN_ALIVE = 500` with a
per-source floor of `20 x eff_dim`, because the old constant and
`MAX_DEAD_RATE = 0.30` were **jointly unsatisfiable** for a model whose alive
count saturates. The two bars pull in opposite directions: `alive >= 500` needs
a large dictionary (alive only ever grows with dict), while `dead <= 0.30` is
exactly `alive >= 0.70 * dict` and so needs a small one. Chronos-T5-Base at
`encoder.block.10` plateaus at ~576 alive atoms, which makes every dictionary
above 576/0.70 = 823 fail the rate bar permanently -- and the smallest
dictionary at which its alive count reaches 500 is 896. The passing window is
empty by construction, not by an unlucky choice of sizes.

The tests below pin the two properties that make that change defensible rather
than convenient, both stated as claims about *this repo's own measured
numbers*:

- the new floor does **not** by itself flip the failing case (Chronos still
  fails, on the untouched dead-rate bar), and
- it makes the crosscoder's bar **stricter**, not looser (563 > 500), while the
  committed winner still clears it.

Plus the arithmetic of the unsatisfiability argument itself, so that if anyone
later retunes `MAX_DEAD_RATE` or `ALIVE_PER_EFF_DIM` the test says plainly
whether the pair is still satisfiable for a saturating model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The layer pair every Stage 0 number on record was measured at, and the
# effective dimensionalities read from `internals/profile.json` for it.
EFF_TIMESFM = 28.15
EFF_CHRONOS = 13.93


def _harness():
    """Import the CLI script by path. Registering it in `sys.modules` before
    executing is required: `dataclasses` resolves `Row`'s annotations through
    `sys.modules[cls.__module__]`."""
    spec = importlib.util.spec_from_file_location(
        "run_crosscoder_stage0", ROOT / "run_crosscoder_stage0.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_crosscoder_stage0"] = module
    spec.loader.exec_module(module)
    return module


def test_floor_scales_per_source_and_keeps_an_absolute_guard():
    m = _harness()
    assert m.min_alive_for(EFF_CHRONOS) == 279
    assert m.min_alive_for(EFF_TIMESFM) == 563
    # A very low-dimensional layer must not drive the guard to nothing.
    assert m.min_alive_for(1.0) == m.MIN_ALIVE_FLOOR
    # No eff_dims at all falls back to the pre-decision constant rather than
    # silently admitting everything.
    assert m.min_alive_for(()) == m.MIN_ALIVE_LEGACY


def test_crosscoder_floor_is_the_strictest_source_not_the_loosest():
    """One dictionary serves every source, so its floor is the max. On this
    repo's own pair that makes the crosscoder's bar *stricter* than the
    constant it replaced -- the decision is not a loosening."""
    m = _harness()
    joint = m.min_alive_for((EFF_TIMESFM, EFF_CHRONOS))
    assert joint == m.min_alive_for(EFF_TIMESFM) == 563
    assert joint > m.MIN_ALIVE_LEGACY


def test_committed_winner_still_passes_under_the_stricter_joint_floor():
    m = _harness()
    joint = m.min_alive_for((EFF_TIMESFM, EFF_CHRONOS))
    v = m.verdict(0.020, 1254, [0.759, 0.756], 1280, joint)
    assert v["passes"] and v["alive_ok"] and v["min_alive"] == joint


def test_new_floor_does_not_rescue_the_failing_chronos_baseline():
    """The five-seed replicate mean at dict=640 (ROADMAP.md §6.2.1 finding
    (10)): dead 0.3447, alive 419. Under the new floor `alive_ok` flips True,
    but the row still fails -- on the dead-rate bar, which the decision left
    untouched. This is the property that distinguishes the change from picking
    whichever criterion makes the gate pass."""
    m = _harness()
    v = m.verdict(0.3447, 419, [0.877], 640, m.min_alive_for(EFF_CHRONOS))
    assert v["alive_ok"] is True
    assert v["dead_ok"] is False
    assert v["passes"] is False


# Chronos-T5-Base baseline at `encoder.block.10`, measured across the eight
# dictionary sizes swept by the `k`, `h2xh4` and `pinch` grids
# (ROADMAP.md §6.2.1 finding (7)). dict -> alive.
CHRONOS_ALIVE = {512: 386, 576: 355, 640: 467, 704: 405,
                 768: 473, 896: 529, 1024: 546, 1280: 576}
# TimesFM's, over the same sizes -- it tracks the dictionary instead of
# saturating, which is why the same criteria were never a problem for it.
TIMESFM_ALIVE = {512: 511, 640: 640, 768: 765, 896: 893, 1024: 1012, 1280: 1258}


def _passing_sizes(alive_by_dict: dict, floor: int, max_dead: float) -> list:
    """Dictionary sizes clearing both bars, straight from measured alive counts.

    `dead <= max_dead` is exactly `alive >= (1 - max_dead) * dict`, so both
    criteria are a statement about the same measured number.
    """
    return [d for d, a in alive_by_dict.items()
            if a >= floor and a >= (1.0 - max_dead) * d]


def test_legacy_pair_is_unsatisfiable_for_a_saturating_model():
    """The argument the decision rests on, as arithmetic over measured counts.

    A saturating model has two bars moving in opposite directions: the alive
    floor needs a *large* dictionary (alive only grows with dict), while the
    dead-rate bar needs a *small* one (it demands alive >= 0.70 * dict, and
    alive stops growing). Once alive plateaus at a ceiling C, every dictionary
    above C / 0.70 fails the rate bar permanently -- so a passing size must sit
    below that, and for Chronos the alive count never reaches 500 there.
    """
    m = _harness()
    ceiling = max(CHRONOS_ALIVE.values())                 # 576, plateaued
    largest_rate_feasible = ceiling / (1.0 - m.MAX_DEAD_RATE)
    smallest_clearing_500 = min(d for d, a in CHRONOS_ALIVE.items()
                                if a >= m.MIN_ALIVE_LEGACY)
    # The window [smallest size reaching the floor, largest size the rate bar
    # can still admit] is empty: 896 > 823.
    assert smallest_clearing_500 > largest_rate_feasible
    assert _passing_sizes(CHRONOS_ALIVE, m.MIN_ALIVE_LEGACY, m.MAX_DEAD_RATE) == []

    # Under the per-model floor the window is non-empty -- but only just, and
    # only at the two smallest sizes, which is why Stage 0 still needs the
    # replicate sweep rather than being declared passed here.
    assert _passing_sizes(CHRONOS_ALIVE, m.min_alive_for(EFF_CHRONOS),
                          m.MAX_DEAD_RATE) == [512, 640]


def test_timesfm_satisfies_both_bars_under_either_floor():
    """The same two constants were never in tension for the non-saturating
    side -- evidence the defect is an interaction with saturation, not a
    generally-too-strict gate."""
    m = _harness()
    assert _passing_sizes(TIMESFM_ALIVE, m.MIN_ALIVE_LEGACY, m.MAX_DEAD_RATE)
    assert _passing_sizes(TIMESFM_ALIVE, m.min_alive_for(EFF_TIMESFM), m.MAX_DEAD_RATE)


# The 5-seed pinch replicate (ROADMAP.md §6.2.1 findings (12)-(13)), as mean
# alive counts per dictionary size. dict -> (crosscoder, TimesFM, Chronos).
PINCH_ALIVE = {512: (473.4, 507.6, 389.8),
               576: (534.4, 570.6, 399.6),
               704: (655.2, 694.8, 442.4)}


def test_no_shared_dictionary_size_satisfies_all_three_artifacts():
    """Finding (13): the unsatisfiability survives the per-model floor, one
    level up. The exit criteria demand the crosscoder *and* both matched
    baselines pass at one shared size, but Chronos's rate bar caps that size
    near 569 while the crosscoder's alive floor needs ~606 -- opposite
    directions again, this time between models rather than within one.
    """
    m = _harness()
    cc_floor = m.min_alive_for((EFF_TIMESFM, EFF_CHRONOS))
    floors = (cc_floor, m.min_alive_for(EFF_TIMESFM), m.min_alive_for(EFF_CHRONOS))
    for dict_size, alive in PINCH_ALIVE.items():
        passing = [a >= f and a >= (1.0 - m.MAX_DEAD_RATE) * dict_size
                   for a, f in zip(alive, floors)]
        assert not all(passing), (dict_size, passing)

    # Which bar blocks changes ends of the range -- the property that makes
    # this a window problem rather than one artifact being weak everywhere.
    cc512, tf512, ch512 = PINCH_ALIVE[512]
    assert cc512 < cc_floor and tf512 < floors[1]          # alive floor, at 512
    assert ch512 >= (1.0 - m.MAX_DEAD_RATE) * 512          # ...where Chronos is fine
    cc704, tf704, ch704 = PINCH_ALIVE[704]
    assert cc704 >= cc_floor and tf704 >= floors[1]        # floor cleared, at 704
    assert ch704 < (1.0 - m.MAX_DEAD_RATE) * 704           # ...where Chronos now fails


def test_per_model_sizing_closes_the_window_the_shared_size_left_empty():
    """Finding (14), which the 2026-08-13 DECISION acted on: every artifact has
    a passing size individually, so the emptiness above is a property of the
    shared *constraint*, not of any artifact. This test asserted the evidence
    while the change was only a proposal; the change is now in force, and what
    it enables is pinned next to the mechanism in
    `test_stage0_baseline_sizing.py`. It stays here as the arithmetic the
    decision rests on, deliberately adjacent to the unsatisfiability proof it
    answers -- the pair is only readable together."""
    m = _harness()
    floors = {"cc": m.min_alive_for((EFF_TIMESFM, EFF_CHRONOS)),
              "tf": m.min_alive_for(EFF_TIMESFM),
              "ch": m.min_alive_for(EFF_CHRONOS)}
    ok = lambda a, d, f: a >= f and a >= (1.0 - m.MAX_DEAD_RATE) * d
    assert ok(PINCH_ALIVE[512][2], 512, floors["ch"])      # Chronos, small dict
    assert ok(PINCH_ALIVE[576][1], 576, floors["tf"])      # TimesFM, mid
    assert ok(PINCH_ALIVE[704][0], 704, floors["cc"])      # crosscoder, large


def test_verdict_records_the_bar_it_was_scored_against():
    """A stored result must stay interpretable after the constants move again
    (`CLAUDE.md` §11.24: a number's meaning can drift while its bytes do not)."""
    m = _harness()
    v = m.verdict(0.10, 300, [0.8], 640, 279)
    assert v["min_alive"] == 279
    assert m.verdict(0.10, 300, [0.8], 640, 563)["alive_ok"] is False
    assert v["alive_ok"] is True


def test_structurally_cannot_pass_tracks_the_active_floor():
    m = _harness()
    # A dictionary smaller than the floor could never have enough alive atoms.
    assert m.verdict(0.0, 200, [0.9], 200, 279)["structurally_cannot_pass"] is True
    assert m.verdict(0.0, 200, [0.9], 640, 279)["structurally_cannot_pass"] is False
