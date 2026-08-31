"""Source pools can be required to carry series long enough to be usable.

`build_pipeline/sources.py::_filter_min_length`. This exists because a smoke
build of `configs/large_run.yaml` emitted a `block_bootstrap` family whose
every series ran 27-152 points against an analysis needing 576 -- that
generator reorders one real series in place and has no length parameter, so
its output length IS its source's, and a pooled Monash catalog spans yearly
series of ~14 points. The failure is silent at both ends: the builder admits
the samples, and the analysis loader drops them all as "shorter than
context+horizon", leaving a family the config promises and nothing measures.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_pipeline.sources import _filter_min_length  # noqa: E402


def _pool(*lengths):
    return [(f"ref{n}", np.zeros(n)) for n in lengths]


def test_absent_key_is_a_no_op_and_returns_the_pool_unchanged():
    """The load-bearing negative: no existing corpus may change.

    Every config written before this filter existed omits `min_length`, and
    those corpora are sealed and hash-pinned (invariant 1). So the untouched
    path must return the *same list*, not a filtered copy that happens to
    match.
    """
    pool = _pool(14, 100, 576, 5000)
    assert _filter_min_length(pool, {}) is pool
    assert _filter_min_length(pool, {"min_length": 0}) is pool
    assert _filter_min_length(pool, {"min_length": None}) is pool


def test_it_keeps_exactly_the_series_at_or_above_the_bound():
    kept = _filter_min_length(_pool(14, 575, 576, 577, 5000), {"min_length": 576})
    assert [len(a) for _, a in kept] == [576, 577, 5000], \
        "the bound is inclusive: a series of exactly context+horizon is usable"


def test_an_empty_result_raises_rather_than_returning_nothing():
    """A pool that cannot satisfy the task is a config error, not an empty family.

    Returning [] would surface much later and somewhere else -- as a generator
    failing on an empty pool, or as a family that silently never appears.
    """
    with pytest.raises(ValueError, match="min_length"):
        _filter_min_length(_pool(10, 20, 30), {"min_length": 576})


def test_the_error_names_the_three_ways_out():
    with pytest.raises(ValueError) as exc:
        _filter_min_length(_pool(10), {"min_length": 576})
    msg = str(exc.value)
    assert "min_length" in msg and "limit" in msg and "subset" in msg, msg
    assert "of 1 loaded" in msg, "it must say how big the pool was, not just that it emptied"


def test_it_reports_what_it_dropped(capsys):
    """A silent filter is the failure mode this replaces, not a fix for it."""
    _filter_min_length(_pool(14, 20, 576, 5000), {"min_length": 576})
    out = capsys.readouterr().out
    assert "dropped 2 of 4" in out, out


def test_it_is_silent_when_it_drops_nothing():
    """No news when there is no news -- otherwise the message stops being read."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _filter_min_length(_pool(576, 5000), {"min_length": 576})
    assert buf.getvalue() == ""
