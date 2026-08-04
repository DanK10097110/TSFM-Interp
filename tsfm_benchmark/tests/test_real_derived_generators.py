import numpy as np
import pytest

from tsfm_benchmark.build_pipeline.generators import block_bootstrap, sequential_par
from tsfm_benchmark.build_pipeline.schema import SourceRef

pytest.importorskip("tsbootstrap")
pytest.importorskip("sdv")
pytest.importorskip("pandas")


def _ref(corpus: str, item_id: str) -> SourceRef:
    return SourceRef(corpus=corpus, item_id=item_id, sha256="a" * 64, license="unknown")


def test_block_bootstrap_reproducible_and_preserves_length():
    rng = np.random.default_rng(0)
    arr = np.sin(np.linspace(0, 10, 120)) + rng.normal(0, 0.1, 120)
    source = (_ref("demo", "0"), arr)

    a = block_bootstrap(source, seed=1, block_length=24)
    b = block_bootstrap(source, seed=1, block_length=24)
    c = block_bootstrap(source, seed=2, block_length=24)

    assert a.values.shape == (120,)
    assert np.array_equal(a.values, b.values)
    assert not np.array_equal(a.values, c.values)
    assert a.provenance.source_refs == [source[0]]
    assert a.provenance.generator_params["effective_block_length"] == 24


def test_block_bootstrap_clamps_block_length_to_short_series():
    # Real Monash domains include yearly series as short as ~14 points --
    # a fixed block_length of 24 must not crash on them (see generators.py).
    rng = np.random.default_rng(0)
    short_arr = rng.normal(0, 1, 14)
    source = (_ref("demo", "short"), short_arr)

    sample = block_bootstrap(source, seed=0, block_length=24)

    assert sample.values.shape == (14,)
    assert sample.provenance.generator_params["effective_block_length"] < 24
    assert sample.provenance.generator_params["effective_block_length"] >= 2


def test_sequential_par_returns_requested_length_and_source_refs():
    rng = np.random.default_rng(0)
    cohort = [
        (_ref("demo", str(i)), np.sin(np.linspace(0, 6, 40)) + rng.normal(0, 0.1, 40))
        for i in range(4)
    ]

    sample = sequential_par(cohort, seed=0, epochs=2, length=40, cuda=False)

    assert sample.values.shape == (40,)
    assert np.isfinite(sample.values).all()
    assert len(sample.provenance.source_refs) == 4
    assert sample.provenance.generator_params["cohort_size"] == 4
    assert sample.provenance.generator_params["n_series_truncated"] == 0
    assert sample.ground_truth.notes.startswith("real-derived")


def test_sequential_par_truncates_long_series_before_fitting():
    # Real Monash domains include series tens of thousands of points long
    # (e.g. weather, solar) -- PAR's per-timestep RNN fit must not be handed
    # those unbounded, or cost scales with raw series length instead of
    # epochs (see generators.py's cost note, found via live calibration
    # 2026-08-03). A long synthetic series here stands in for that case.
    rng = np.random.default_rng(1)
    cohort = [
        (_ref("demo", "long"), rng.normal(0, 1, 5000)),
        (_ref("demo", "short"), np.sin(np.linspace(0, 6, 40)) + rng.normal(0, 0.1, 40)),
    ]

    sample = sequential_par(cohort, seed=0, epochs=2, cuda=False, max_train_length=64)

    assert sample.provenance.generator_params["n_series_truncated"] == 1
    assert sample.provenance.generator_params["max_train_length"] == 64
    # default sampled length is the mean of the (truncated) training lengths,
    # not inflated by the raw 5000-point series
    assert sample.values.shape[0] < 100
