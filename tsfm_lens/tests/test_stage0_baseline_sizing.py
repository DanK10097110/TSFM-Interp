"""Per-model baseline dictionary sizing (ROADMAP.md §6.2.1's 2026-08-13 DECISION).

Stage 0's exit criteria require the crosscoder *and* both per-model `TopKSAE`
baselines to clear the same numeric bars "on the same rows at the same budget".
Until this change, "the same budget" was read as including one shared
dictionary size, and finding (13) showed that reading makes the gate
unsatisfiable: Chronos-T5-Base's dead-rate bar caps a shared dictionary near
569 atoms while the crosscoder's alive floor needs ~606, so the passing window
between them is empty -- not because any artifact is unhealthy, but because a
shared size is a shared bar on two models whose alive counts behave
differently (TimesFM's tracks the dictionary, Chronos's saturates near 576).

The decision keeps the budget matched on rows, `k`, epochs and AuxK settings,
and lets dictionary size vary per model. These tests pin the three properties
that make that a narrowing of scope rather than a loosening of the gate:

- an unset `baseline_dict_sizes` reproduces the old shared-size behaviour
  *exactly*, so the matched-size control is retained rather than replaced,
- a set one actually reaches the baseline that trains (not just the record),
  and the memo that makes the sweep affordable cannot serve a result trained
  at a different size, and
- the criterion now in force is satisfiable on this repo's own measured
  numbers, where the one it replaces provably was not.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EFF_TIMESFM = 28.15
EFF_CHRONOS = 13.93


def _harness():
    spec = importlib.util.spec_from_file_location(
        "run_crosscoder_stage0", ROOT / "run_crosscoder_stage0.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_crosscoder_stage0"] = module
    spec.loader.exec_module(module)
    return module


def test_unset_sizes_fall_back_to_the_crosscoders_own_size():
    """The matched-size control has to survive the change untouched -- it
    answers a different question (does joint training cost anything at
    identical budget?) and every recorded Stage 0 row was measured under it."""
    m = _harness()
    row = m.Row("plain", dict_size=704)
    assert row.baseline_dict_sizes == ()
    assert row.baseline_dict_size(0, row.dict_size) == 704
    assert row.baseline_dict_size(1, row.dict_size) == 704


def test_set_sizes_apply_per_model_and_coerce_to_ints():
    m = _harness()
    row = m.Row("gate", dict_size=896, baseline_dict_sizes=[576, 512])
    assert row.baseline_dict_sizes == (576, 512)
    assert row.baseline_dict_size(0, row.dict_size) == 576
    assert row.baseline_dict_size(1, row.dict_size) == 512
    # A short list falls back for the models it does not name, rather than
    # raising -- the sizes are per model and a two-model pair is not assumed.
    short = m.Row("partial", dict_size=896, baseline_dict_sizes=(576,))
    assert short.baseline_dict_size(1, short.dict_size) == 896


def test_gate_grid_carries_the_sizes_and_sits_above_the_pinch_range():
    """`pinch` measured 512/576/704 and found no shared size works; `gate`
    exists to test whether the crosscoder has a size that passes at *every*
    seed, so it must probe above 704, where the 4-of-5 result was recorded."""
    m = _harness()
    rows = m.build_grid("gate", (EFF_TIMESFM, EFF_CHRONOS), 1280, 60,
                        baseline_dicts=(576, 512))
    assert [r.dict_size for r in rows] == [896, 1024]
    assert all(r.dict_size > 704 for r in rows)
    assert all(r.baseline_dict_sizes == (576, 512) for r in rows)
    assert all(r.k == 48 and r.aux_k == 64 and r.k_is_swept for r in rows)


def test_unknown_grid_names_the_new_one():
    m = _harness()
    with pytest.raises(ValueError, match="gate"):
        m.build_grid("nonsense", (EFF_TIMESFM, EFF_CHRONOS), 1280, 60)


def test_params_file_can_commit_the_sizes():
    """Stage 0's fourth exit criterion is a config file, not a CLI line -- so
    a per-model sizing decision has to be expressible in one."""
    m = _harness()
    row = m.row_from_params({"train": {"dict_size": 896, "k": 48,
                                       "baseline_dict_sizes": [576, 512]}})
    assert row.baseline_dict_sizes == (576, 512)


# --- the memo, exercised through the real training path on tiny CPU data ---

def _tiny(n: int = 96, d: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def _row(m, label: str, dict_size: int, baseline_dict_sizes=()):
    return m.Row(label, dict_size=dict_size, k=2, epochs=1, resample_every=1,
                 baseline_dict_sizes=baseline_dict_sizes)


def test_memo_reuses_a_baseline_only_when_its_training_is_identical():
    """Per-model sizing decouples a baseline from the crosscoder's dictionary,
    so two `gate` rows differing only in `dict_size` train the identical
    baseline -- halving the sweep. The memo must key on what the baseline
    actually depends on, or it would serve a result from a different size."""
    m = _harness()
    xa, xb = _tiny(seed=1), _tiny(seed=2)
    names, device, eff = ("a", "b"), torch.device("cpu"), (EFF_TIMESFM, EFF_CHRONOS)
    cache: dict = {}

    first = m.run_row(_row(m, "d=8", 8, (4, 4)), xa, xb, names, device, 0, True, eff, cache)
    second = m.run_row(_row(m, "d=12", 12, (4, 4)), xa, xb, names, device, 0, True, eff, cache)
    assert first["baseline_sizing"] == "own"
    assert first["dict_size"] == 8 and second["dict_size"] == 12
    # Same baseline dictionary, same budget, same seed -> the same object.
    for name in names:
        assert second["baseline"][name] is first["baseline"][name]
        assert first["baseline"][name]["dict_size"] == 4

    # A different baseline size must miss the memo and train its own.
    third = m.run_row(_row(m, "d=12b", 12, (5, 5)), xa, xb, names, device, 0, True, eff, cache)
    for name in names:
        assert third["baseline"][name] is not first["baseline"][name]
        assert third["baseline"][name]["dict_size"] == 5


def test_matched_sizing_never_reuses_across_different_crosscoder_sizes():
    """The control's baselines follow the crosscoder's size, so rows that
    differ in `dict_size` differ in baseline too -- the memo must not collapse
    them, which is what would silently turn the control into the gate."""
    m = _harness()
    xa, xb = _tiny(seed=1), _tiny(seed=2)
    names, device, eff = ("a", "b"), torch.device("cpu"), (EFF_TIMESFM, EFF_CHRONOS)
    cache: dict = {}

    first = m.run_row(_row(m, "d=8", 8), xa, xb, names, device, 0, True, eff, cache)
    second = m.run_row(_row(m, "d=12", 12), xa, xb, names, device, 0, True, eff, cache)
    assert first["baseline_sizing"] == second["baseline_sizing"] == "matched"
    for name in names:
        assert first["baseline"][name]["dict_size"] == 8
        assert second["baseline"][name]["dict_size"] == 12


# --- the criterion's satisfiability, over this repo's own measured numbers ---

# The 5-seed pinch replicate (ROADMAP.md §6.2.1 findings (12)-(13)), as mean
# alive counts per dictionary size. dict -> (crosscoder, TimesFM, Chronos).
PINCH_ALIVE = {512: (473.4, 507.6, 389.8),
               576: (534.4, 570.6, 399.6),
               704: (655.2, 694.8, 442.4)}


def _ok(m, alive: float, dict_size: int, floor: int) -> bool:
    return alive >= floor and alive >= (1.0 - m.MAX_DEAD_RATE) * dict_size


def test_the_criterion_in_force_is_satisfiable_where_the_one_it_replaces_was_not():
    """The decision, as arithmetic. Under one shared size no assignment
    satisfies all three artifacts; under per-model sizes an assignment exists,
    and the sizes it picks are the ones the `gate` grid and its
    `--baseline-dict-sizes` default were chosen to be."""
    m = _harness()
    cc_floor = m.min_alive_for((EFF_TIMESFM, EFF_CHRONOS))
    tf_floor, ch_floor = m.min_alive_for(EFF_TIMESFM), m.min_alive_for(EFF_CHRONOS)

    shared_ok = [d for d, (cc, tf, ch) in PINCH_ALIVE.items()
                 if _ok(m, cc, d, cc_floor) and _ok(m, tf, d, tf_floor)
                 and _ok(m, ch, d, ch_floor)]
    assert shared_ok == [], "finding (13): no shared size satisfies all three"

    # Per-model, each artifact has a size that clears both bars, and the two
    # baseline sizes below are exactly what the gate run passes on the CLI.
    assert _ok(m, PINCH_ALIVE[576][1], 576, tf_floor)   # TimesFM baseline @576
    assert _ok(m, PINCH_ALIVE[512][2], 512, ch_floor)   # Chronos baseline @512
    assert _ok(m, PINCH_ALIVE[704][0], 704, cc_floor)   # crosscoder, above pinch


def test_per_model_sizing_does_not_relax_any_numeric_bar():
    """The change is to what is held constant across artifacts, not to the
    criteria. Every bar a baseline is scored against must be the same one it
    faced before -- otherwise this is a loosening wearing a scoping argument."""
    m = _harness()
    assert m.MAX_DEAD_RATE == 0.30 and m.MIN_FIDELITY == 0.70
    # Chronos at its own size still has to clear the untouched dead-rate bar:
    # its 640-atom five-seed mean (dead 0.3447, alive 419) still fails.
    assert m.verdict(0.3447, 419, [0.877], 640, m.min_alive_for(EFF_CHRONOS))["passes"] is False
    # And the sizes the gate hands the baselines are not chosen to be easy --
    # 512 is the only size in the whole recorded sweep where Chronos passes.
    assert m.verdict(0.2387, 390, [0.870], 512,
                     m.min_alive_for(EFF_CHRONOS))["passes"] is True
