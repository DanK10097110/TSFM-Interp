"""V3-A source options and split-overlap check (ROADMAP sec 41).

``windows_per_row`` / ``finite_windows`` on ``kind: chronos_datasets`` serve subsets with few, very
long rows holding scattered NaN (ercot, monash_kdd_cup_2018). Both are opt-in: the default call must
return exactly what it returned before, and a row with NaN must still be rejected unless
``finite_windows`` is set. The fixture has one NaN-free row, one row with a NaN placed so that only
some of its segments are clean (a decoy for "keep every window of a NaN row"), and one all-NaN row.

``overlap`` is the hash comparison ``mint_private.py`` runs: a planted shared hash must be found
under the right split path and none under a disjoint one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datasets as _datasets_pkg  # noqa: E402

from tsfm_benchmark.build_pipeline.overlap import discover_splits, manifest_hashes, overlap_counts  # noqa: E402
from tsfm_benchmark.build_pipeline.sources import _load_chronos_datasets_config  # noqa: E402

_REAL_LOAD_DATASET = _datasets_pkg.load_dataset
L = 100


@pytest.fixture()
def fixture_path(tmp_path) -> str:
    rng = np.random.default_rng(0)
    clean = rng.normal(size=450)
    holey = rng.normal(size=450)
    holey[150] = np.nan
    all_nan = np.full(450, np.nan)
    rows = [{"id": "clean", "target": clean.tolist()}, {"id": "holey", "target": holey.tolist()},
            {"id": "allnan", "target": all_nan.tolist()}]
    path = tmp_path / "fx.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return str(path)


def _load(monkeypatch, fixture_path, **kwargs):
    monkeypatch.setattr("datasets.load_dataset",
                        lambda name, subset, split="train", streaming=False, trust_remote_code=False:
                        _REAL_LOAD_DATASET("parquet", data_files=fixture_path, split=split))
    return _load_chronos_datasets_config(dataset_name="t/fx", subset="s", field_name="target", split="train",
                                         limit=10, seed=1, max_length=kwargs.pop("max_length", L), license="x", **kwargs)


def test_default_path_still_rejects_every_row_containing_nan(monkeypatch, fixture_path):
    out = _load(monkeypatch, fixture_path)
    assert len(out) == 1
    assert out[0][0].item_id.startswith("0:target")


def test_windows_per_row_cuts_non_overlapping_windows_and_ids_carry_the_offset(monkeypatch, fixture_path):
    out = _load(monkeypatch, fixture_path, windows_per_row=4)
    assert len(out) == 4
    offsets = sorted(int(ref.item_id.rsplit(":w", 1)[1]) for ref, _ in out)
    assert all(len(v) == L for _, v in out)
    assert all(b - a >= L for a, b in zip(offsets, offsets[1:])), offsets
    assert offsets[-1] + L <= 450
    raw = pd.read_parquet(fixture_path)["target"][0]
    for ref, v in out:
        off = int(ref.item_id.rsplit(":w", 1)[1])
        np.testing.assert_array_equal(v, np.asarray(raw[off:off + L]))


def test_windows_per_row_is_capped_by_how_many_windows_fit(monkeypatch, fixture_path):
    assert len(_load(monkeypatch, fixture_path, windows_per_row=50)) == 450 // L


def test_finite_windows_keeps_only_the_clean_windows_of_a_nan_row(monkeypatch, fixture_path):
    out = _load(monkeypatch, fixture_path, windows_per_row=4, finite_windows=True)
    by_row = {}
    for ref, v in out:
        assert np.isfinite(v).all()
        by_row.setdefault(int(ref.item_id.split(":")[0]), []).append(ref)
    assert set(by_row) == {0, 1}, "the all-NaN row must contribute nothing"
    assert len(by_row[0]) == 4
    assert len(by_row[1]) == 3, "segment 1 (offsets 112..124) always spans the NaN at 150 and must be skipped, not filled"
    for ref in by_row[1]:
        off = int(ref.item_id.rsplit(":w", 1)[1])
        assert not (off <= 150 < off + L)


def test_finite_windows_is_deterministic_and_seed_sensitive(monkeypatch, fixture_path):
    a = _load(monkeypatch, fixture_path, windows_per_row=4, finite_windows=True)
    b = _load(monkeypatch, fixture_path, windows_per_row=4, finite_windows=True)
    assert [r.item_id for r, _ in a] == [r.item_id for r, _ in b]
    monkeypatch.setattr("datasets.load_dataset",
                        lambda name, subset, split="train", streaming=False, trust_remote_code=False:
                        _REAL_LOAD_DATASET("parquet", data_files=fixture_path, split=split))
    c = _load_chronos_datasets_config(dataset_name="t/fx", subset="s", field_name="target", split="train",
                                      limit=10, seed=2, max_length=L, license="x", windows_per_row=4, finite_windows=True)
    assert [r.item_id for r, _ in a] != [r.item_id for r, _ in c]


def test_window_options_require_max_length(monkeypatch, fixture_path):
    monkeypatch.setattr("datasets.load_dataset",
                        lambda name, subset, split="train", streaming=False, trust_remote_code=False:
                        _REAL_LOAD_DATASET("parquet", data_files=fixture_path, split=split))
    with pytest.raises(ValueError, match="max_length"):
        _load_chronos_datasets_config(dataset_name="t/fx", subset="s", field_name="target", split="train",
                                      limit=10, seed=1, max_length=None, license="x", finite_windows=True)


def _split(root: Path, corpus: str, name: str, hashes: list[str]) -> Path:
    d = root / corpus / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"sample_hashes": hashes}))
    return d


def test_overlap_counts_finds_a_planted_shared_hash_only_in_the_right_split(tmp_path):
    target = _split(tmp_path, "benchmark_v3", "private_epoch1", ["h1", "h2", "h3"])
    _split(tmp_path, "benchmark_large", "private_epoch1", ["h3", "h9"])
    _split(tmp_path, "benchmark_large_v2", "public_dev", ["h7", "h8"])
    (tmp_path / "other_dir").mkdir()
    found = discover_splits(tmp_path)
    assert {p.parent.name for p in found} == {"benchmark_v3", "benchmark_large", "benchmark_large_v2"}
    counts = overlap_counts(manifest_hashes(target), [p for p in found if p != target])
    assert {Path(k).parent.name: v for k, v in counts.items()} == {"benchmark_large": 1, "benchmark_large_v2": 0}
