"""Regression tests for tests/conftest.py's BLAS/OpenMP thread cap.

The cap exists because this suite was measured taking 18 of 32 cores at nice 0 on
a shared box, and because capping it is 1.8x FASTER (CLAUDE.md sec 9). Two things
can go wrong and neither raises: the variables can be set without the pools ever
reading them (a decorative cap), and a caller's own explicitly exported thread
variable can be half-overwritten into a state where OpenMP and MKL disagree. One
test below pins each, the second by planting exactly that mixed state and
asserting it does not happen.

The nice() path is deliberately tested only on its opt-out branch: lowering
niceness again requires privileges a test process does not have, so a test that
actually reniced would silently degrade every test that ran after it.
"""

from __future__ import annotations

import os

import pytest

import conftest


def test_the_cap_is_actually_in_force_not_merely_recorded():
    """A cap the header claims must be one the pools actually read.

    Keyed off conftest._CAP -- the policy the header reports -- rather than off
    the environment, so a conftest that reported a cap while setting nothing
    FAILS here instead of skipping. Skipping on an absent variable would make
    the inert case indistinguishable from the deliberately-uncapped one, which
    is the whole condition this test exists to detect.
    """
    if not conftest._CAP:
        pytest.skip("capping not in force for this run (disabled, or deferring to the environment)")

    assert os.environ.get("OMP_NUM_THREADS", "").strip() == str(conftest._CAP)

    torch = pytest.importorskip("torch")
    assert torch.get_num_threads() == conftest._CAP


def test_an_explicitly_set_thread_var_defers_wholesale(monkeypatch):
    """A caller who sets one variable keeps control of all of them.

    The load-bearing negative: filling in the remaining four around a pre-set
    OMP_NUM_THREADS would leave OpenMP at 7 and MKL at 4, which is a state
    nobody requested and which the header cannot make less surprising.
    """
    for var in conftest._THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("TSFM_TEST_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "7")

    monkeypatch.setattr(conftest, "_PRESET", conftest._preset_thread_vars())
    assert conftest._apply_thread_cap() == 0
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert [v for v in conftest._THREAD_VARS if v in os.environ] == ["OMP_NUM_THREADS"]


def test_an_untouched_environment_gets_every_variable_at_the_default(monkeypatch):
    for var in conftest._THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("TSFM_TEST_THREADS", raising=False)

    monkeypatch.setattr(conftest, "_PRESET", {})
    assert conftest._apply_thread_cap() == conftest._DEFAULT_THREADS
    assert {os.environ[v] for v in conftest._THREAD_VARS} == {str(conftest._DEFAULT_THREADS)}


@pytest.mark.parametrize(
    "raw,expected",
    [("0", 0), ("-1", 0), ("2", 2), ("banana", conftest._DEFAULT_THREADS), ("", conftest._DEFAULT_THREADS)],
)
def test_the_knob_disables_on_zero_and_falls_back_on_anything_unparseable(
    monkeypatch, raw, expected
):
    """An unreadable value must not silently mean 'uncapped' -- that is the failure mode."""
    for var in conftest._THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TSFM_TEST_THREADS", raw)

    monkeypatch.setattr(conftest, "_PRESET", {})
    assert conftest._apply_thread_cap() == expected


def test_renicing_is_opt_in(monkeypatch):
    """With the knob unset the process must not be reniced -- os.nice is one-way."""
    monkeypatch.delenv("TSFM_TEST_NICE", raising=False)
    before = os.nice(0)
    assert conftest._apply_nice() is None
    assert os.nice(0) == before


def test_the_header_names_which_policy_applied(monkeypatch):
    """A capped run states the cap; a deferring run names what it deferred to."""
    monkeypatch.setattr(conftest, "_PRESET", {})
    monkeypatch.setattr(conftest, "_CAP", 4)
    assert "thread cap: 4" in conftest._header_line()

    monkeypatch.setattr(conftest, "_CAP", 0)
    monkeypatch.setattr(conftest, "_PRESET", {"OMP_NUM_THREADS": "7"})
    assert "deferring to the environment" in conftest._header_line()
    assert "OMP_NUM_THREADS=7" in conftest._header_line()

    monkeypatch.setattr(conftest, "_PRESET", {})
    assert "disabled" in conftest._header_line()
