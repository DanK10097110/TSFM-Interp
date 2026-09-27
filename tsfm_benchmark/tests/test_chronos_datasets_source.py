"""``kind: chronos_datasets`` (ROADMAP D1): a second real source for the
real-derived tier, `build_pipeline/sources.py::_load_chronos_datasets_config`.

Unlike the streaming, take-the-first-``limit``-rows path every other kind
here uses, this one loads a config non-streamed and draws a real seeded
sample of its rows -- streaming would always give "the first N in file
order" regardless of `seed`. That row selection, and the per-row window
offset applied to any series longer than `max_length`, are both mixed from
`seed` via sha256 (`_mix_seed`), so both must actually depend on it: a
regression that silently falls back to file order (or a fixed offset) would
build a corpus with none of the diversity `configs/large_run_v2.yaml`'s
rationale assumes, without erroring anywhere.

No network: a tiny local parquet fixture stands in for the Hub download by
monkeypatching `datasets.load_dataset` (the module attribute the function's
own `from datasets import load_dataset` resolves at call time) to route to
the local file while still asserting the (dataset_name, subset) it was
actually called with.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_pipeline.sources import _load_chronos_datasets_config, _mix_seed  # noqa: E402

import datasets as _datasets_pkg  # noqa: E402

# Captured once, at collection time, before any test monkeypatches
# `datasets.load_dataset` -- a test that calls `_load()` more than once (every
# seed-sensitivity test below does) would otherwise re-capture its own fake
# from the previous call as "the real one" and recurse into itself.
_REAL_LOAD_DATASET = _datasets_pkg.load_dataset

# id -> length. Deliberately spans below and above a 576 min_length, and one
# series (S5) far longer than any max_length used below, to exercise
# windowing.
_LENGTHS = {"S0": 200, "S1": 600, "S2": 650, "S3": 700, "S4": 750, "S5": 5000}


@pytest.fixture()
def fixture_path(tmp_path) -> str:
    rng = np.random.default_rng(0)
    rows = [{"id": sid, "target": list(rng.normal(size=n))} for sid, n in _LENGTHS.items()]
    path = tmp_path / "chronos_fixture.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return str(path)


def _patch_loader(monkeypatch, fixture_path, expect_dataset="test/chronos_fixture", expect_subset="fake_subset"):
    def fake_load_dataset(dataset_name, subset, split="train", streaming=False, trust_remote_code=False):
        assert dataset_name == expect_dataset
        assert subset == expect_subset
        assert streaming is False
        return _REAL_LOAD_DATASET("parquet", data_files=fixture_path, split=split)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)


def _load(monkeypatch, fixture_path, **kwargs):
    _patch_loader(monkeypatch, fixture_path)
    return _load_chronos_datasets_config(
        dataset_name="test/chronos_fixture",
        subset="fake_subset",
        field_name="target",
        split="train",
        license="unknown",
        **kwargs,
    )


def test_limit_caps_the_number_of_series_returned(monkeypatch, fixture_path):
    out = _load(monkeypatch, fixture_path, limit=3, seed=0, max_length=None)
    assert len(out) == 3


def test_limit_at_or_above_pool_size_returns_everything(monkeypatch, fixture_path):
    out = _load(monkeypatch, fixture_path, limit=1000, seed=0, max_length=None)
    assert len(out) == len(_LENGTHS)
    assert {int(ref.item_id.split(":")[0]) for ref, _ in out} == set(range(len(_LENGTHS)))


def test_seeded_row_selection_is_reproducible():
    """Same seed, same subset picked -- the load-bearing determinism guarantee."""
    a = np.random.default_rng(_mix_seed(7, "d", "s", "rows")).choice(6, size=3, replace=False)
    b = np.random.default_rng(_mix_seed(7, "d", "s", "rows")).choice(6, size=3, replace=False)
    assert list(a) == list(b)


def test_different_seeds_pick_different_subsets(monkeypatch, fixture_path):
    """The whole point of loading non-streamed: an actual seeded sample, not file order.

    With limit < pool size, two different seeds must be able to disagree on
    which rows are kept. A regression that ignores `seed` for row selection
    (e.g. always ``np.arange(n_keep)``) collapses every seed onto the same
    rows, which this asserts against directly.
    """
    seeds_results = []
    for seed in range(8):
        out = _load(monkeypatch, fixture_path, limit=3, seed=seed, max_length=None)
        seeds_results.append(tuple(sorted(ref.item_id.split(":")[0] for ref, _ in out)))
    assert len(set(seeds_results)) > 1, (
        f"every seed in range(8) picked the identical 3 rows out of 6: {seeds_results[0]} -- "
        "row selection is not actually reading `seed`"
    )


def test_min_length_filtering_composes_with_this_loader(monkeypatch, fixture_path):
    """`_filter_min_length` runs in `load_sources` on this loader's output the
    same as every other kind; called here directly to isolate the loader."""
    from build_pipeline.sources import _filter_min_length

    out = _load(monkeypatch, fixture_path, limit=1000, seed=0, max_length=None)
    kept = _filter_min_length(out, {"min_length": 576})
    assert all(len(arr) >= 576 for _, arr in kept)
    assert len(kept) == sum(1 for n in _LENGTHS.values() if n >= 576)


def test_series_longer_than_max_length_is_windowed_others_left_alone(monkeypatch, fixture_path):
    out = _load(monkeypatch, fixture_path, limit=1000, seed=0, max_length=576)
    orig_lengths = list(_LENGTHS.values())  # ordered S0..S5, matching row index
    assert len(out) == len(orig_lengths)
    for ref, arr in out:
        idx = int(ref.item_id.split(":")[0])
        assert len(arr) == min(orig_lengths[idx], 576)


def test_window_offset_is_reproducible_across_calls(monkeypatch, fixture_path):
    out_a = _load(monkeypatch, fixture_path, limit=1000, seed=3, max_length=576)
    out_b = _load(monkeypatch, fixture_path, limit=1000, seed=3, max_length=576)
    by_id_a = {ref.item_id: arr.tolist() for ref, arr in out_a}
    by_id_b = {ref.item_id: arr.tolist() for ref, arr in out_b}
    assert by_id_a == by_id_b


def test_window_offset_actually_moves_with_seed(monkeypatch, fixture_path):
    """A regression that always windows from offset 0 passes the reproducibility
    test above (offset 0 reproduces offset 0) but always samples the same
    calendar slice out of the one long series -- this is the test that catches it."""
    offsets = set()
    for seed in range(8):
        out = _load(monkeypatch, fixture_path, limit=1000, seed=seed, max_length=576)
        long_ref = next(ref for ref, arr in out if len(arr) == 576 and ":w" in ref.item_id)
        offsets.add(int(long_ref.item_id.rsplit("w", 1)[1]))
    assert len(offsets) > 1, f"the long series' window offset never moved across 8 seeds: {offsets}"
