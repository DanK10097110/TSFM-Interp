"""Regression tests for tests/conftest.py's BLAS/OpenMP thread cap.

Compact by design: the full battery lives beside the tsfm_lens copy of the same
conftest. What this file adds that the other cannot is the drift guard -- the two
conftests are a deliberate duplicate (the packages install independently and
share no import path), and a duplicate nobody checks is exactly the shape of
trap CLAUDE.md sec 11.24 records, where two things that looked identical had
quietly stopped being so.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import conftest

_SIBLING = (
    Path(__file__).resolve().parents[2]
    / "tsfm_lens"
    / "tests"
    / "conftest.py"
)


def _code_half(text: str) -> str:
    """Everything after the module docstring -- the part that must not diverge."""
    return text.split('"""', 2)[2]


def test_the_cap_is_actually_in_force_not_merely_recorded():
    """A cap the header claims must be one the pools actually read.

    The check is that no pool runs MORE threads than the cap. Requiring equality
    failed on a 4-vCPU CI runner, where a second bundled libgomp (pip torch)
    chose the 2 physical cores: a pool below the cap honours it. An uncapped
    pool defaults to one thread per core, so on any box with more cores than
    the cap this still fails when the cap is not read; on a box with no more
    cores than the cap, capped and uncapped cannot be told apart, and the test
    says so by skipping."""
    if not conftest._CAP:
        pytest.skip("capping not in force for this run (disabled, or deferring to the environment)")

    assert os.environ.get("OMP_NUM_THREADS", "").strip() == str(conftest._CAP)

    np = pytest.importorskip("numpy")
    threadpoolctl = pytest.importorskip("threadpoolctl")
    np.zeros((2, 2)) @ np.zeros((2, 2))
    pools = threadpoolctl.threadpool_info()
    if not pools:
        pytest.skip("no BLAS pool reported its thread count on this build")
    if (os.cpu_count() or 1) <= conftest._CAP:
        pytest.skip(f"{os.cpu_count()} cores <= cap {conftest._CAP}: a capped and an uncapped pool look the same")
    assert all(p["num_threads"] <= conftest._CAP for p in pools), pools


def test_an_explicitly_set_thread_var_defers_wholesale(monkeypatch):
    """A caller who sets one variable keeps control of all of them, not a mixed state."""
    for var in conftest._THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("TSFM_TEST_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "7")

    monkeypatch.setattr(conftest, "_PRESET", conftest._preset_thread_vars())
    assert conftest._apply_thread_cap() == 0
    assert [v for v in conftest._THREAD_VARS if v in os.environ] == ["OMP_NUM_THREADS"]


def test_this_conftest_has_not_drifted_from_the_tsfm_lens_copy():
    """The two files' code halves are duplicated on purpose and must stay identical."""
    if not _SIBLING.exists():
        pytest.skip("tsfm_lens is not present alongside this package")

    mine = _code_half(Path(conftest.__file__).read_text(encoding="utf-8"))
    theirs = _code_half(_SIBLING.read_text(encoding="utf-8"))
    assert mine == theirs, (
        "tsfm_benchmark/tests/conftest.py and tsfm_lens/tests/"
        "conftest.py have diverged below their docstrings. They are a deliberate "
        "duplicate; change both, or give them different names and say why."
    )
