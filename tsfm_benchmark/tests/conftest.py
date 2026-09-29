"""Thread-count caps for this suite (CLAUDE.md sec 9's measured finding).

numpy, OpenBLAS and MKL each default to one thread per core and none of them
knows about the others, so this suite was measured peaking at 43 threads on a
32-core box shared by 30 users. Capping it to 4 takes that to 15.

Unlike the tsfm_lens suite -- where the same cap is also 1.78x FASTER, because
that suite is many small linear-algebra calls whose thread dispatch costs more
than the parallel work saves -- capping buys this suite no speed at all: 24.5s
either way, stable across repeats. It is here for the core footprint, not for
the clock, and saying so is the point. Do not quote a speedup for this package.

The BLAS backends read their environment variable exactly once, when the shared
library loads, which happens at the first `import numpy`. So the assignments
below must run at THIS module's import time and before any test module imports
numpy. pytest gives that ordering for free: conftest.py is imported during
collection, ahead of the test modules it applies to, and `import pytest` alone
pulls in neither numpy nor torch (checked, not assumed).

This file is a deliberate duplicate of tsfm_lens/tests/
conftest.py, not an oversight. The two packages install independently (CLAUDE.md
sec 3) and share no import path, so a common module would couple them for the
sake of five assignments. Keep the code below identical to that file's; only the
docstring differs, because only the measurements differ.

Knobs, both read from the environment:
  TSFM_TEST_THREADS=<n>  cap at n instead of the default 4; 0 disables the cap
                         entirely and restores the previous all-cores behavior.
  TSFM_TEST_NICE=<n>     additionally renice this process to n. Opt-in only,
                         because lowering nice again needs privileges this
                         process will not have -- an irreversible side effect
                         does not belong in a default.
"""

from __future__ import annotations

import os

_DEFAULT_THREADS = 4

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _int_env(name: str, default: int | None) -> int | None:
    """Read an integer environment variable, falling back on anything unparseable."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _preset_thread_vars() -> dict[str, str]:
    """Return whichever thread variables the caller had already exported."""
    return {v: os.environ[v] for v in _THREAD_VARS if os.environ.get(v, "").strip()}


def _apply_thread_cap() -> int:
    """Cap the BLAS/OpenMP thread pools, returning the cap actually applied (0 = none).

    Setting even one of these by hand is a deliberate act, so any pre-set variable
    hands threading policy back to the caller wholesale rather than per-variable.
    Filling in the others around it would leave the pools disagreeing -- an
    exported OMP_NUM_THREADS=7 would run OpenMP at 7 and MKL at 4 -- which is a
    state nobody asked for and which no header line makes less surprising.
    """
    if _PRESET:
        return 0
    n = _int_env("TSFM_TEST_THREADS", _DEFAULT_THREADS)
    if n is None or n <= 0:
        return 0
    for var in _THREAD_VARS:
        os.environ[var] = str(n)
    return n


def _apply_nice() -> int | None:
    """Renice this process when TSFM_TEST_NICE asks for it; report the new value."""
    n = _int_env("TSFM_TEST_NICE", None)
    if n is None:
        return None
    try:
        os.nice(n - os.nice(0))
        return os.nice(0)
    except (OSError, AttributeError):
        return None


_PRESET = _preset_thread_vars()
_CAP = _apply_thread_cap()
_NICE = _apply_nice()


def _header_line() -> str:
    """One line describing the caps actually in force, including any already-loaded pools."""
    if _CAP:
        line = f"thread cap: {_CAP} (TSFM_TEST_THREADS, default {_DEFAULT_THREADS})"
    elif _PRESET:
        preset = ", ".join(f"{k}={v}" for k, v in sorted(_PRESET.items()))
        line = f"thread cap: deferring to the environment ({preset})"
    else:
        requested = os.environ.get("TSFM_TEST_THREADS", "").strip()
        line = (
            f"thread cap: disabled (TSFM_TEST_THREADS={requested}) -- "
            "every BLAS pool will size to all cores"
        )

    try:
        import threadpoolctl

        pools = ", ".join(
            f"{d['internal_api']}={d['num_threads']}" for d in threadpoolctl.threadpool_info()
        )
        if pools:
            line += f"; loaded pools: {pools}"
    except Exception:
        pass

    if _NICE is not None:
        line += f"; niceness {_NICE}"
    return line


def pytest_report_header() -> list[str]:
    """State the caps in the standard header, so a capped run is never a silent one."""
    return [_header_line()]


def pytest_sessionstart(session) -> None:
    """Re-emit that line under -q, which suppresses the header the hook above writes to.

    The documented invocation in CLAUDE.md sec 9 passes -q, so without this the
    caps would be invisible in exactly the run everyone actually makes -- and a
    thread count that silently changes what a timing measurement means is the
    shape of trap CLAUDE.md sec 11.24 already cost this repo a session over.
    Guarded on the quiet flag so normal-verbosity runs print it once, not twice.
    """
    config = session.config
    if config.option.verbose >= 0:
        return
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(_header_line())
