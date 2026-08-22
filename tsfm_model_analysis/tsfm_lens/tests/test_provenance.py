"""Regression tests for run provenance and the activation store's schema
version guard (`ROADMAP.md` sec 15 A7).
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config
from tsfm_lens.config import config_from_dict
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.manifest import load_manifest, save_manifest, verify_provenance
from tsfm_lens.pipeline import run_pipeline
from tsfm_lens.utils import run_provenance

_ABS_PATH_RE = re.compile(r"/home/[^\"'\s]+|[A-Za-z]:\\\\[^\"'\s]+")


def test_run_provenance_has_no_absolute_user_paths():
    prov = run_provenance()
    text = json.dumps(prov)
    assert not _ABS_PATH_RE.search(text), f"provenance leaks an absolute path: {text}"
    assert "packages" in prov and "torch" in prov["packages"]
    assert prov["packages"]["torch"], "torch is a hard dependency; its version must resolve"


def test_schema_version_mismatch_raises_a_specific_error():
    path = Path(tempfile.mkdtemp()) / "store.zarr"
    store = ActivationStore.create(path, n_series=4, n_windows=2, window=8, context_len=16)
    del store
    root = zarr.open_group(str(path), mode="a")
    root.attrs["schema_version"] = 999
    with pytest.raises(RuntimeError, match="schema_version=999"):
        ActivationStore(path, mode="r")


def test_fresh_store_round_trips_with_no_schema_error():
    path = Path(tempfile.mkdtemp()) / "store.zarr"
    ActivationStore.create(path, n_series=4, n_windows=2, window=8, context_len=16)
    reopened = ActivationStore(path, mode="r")  # must not raise
    assert reopened.root.attrs["schema_version"] == 1


def test_smoke_run_records_provenance_with_no_absolute_paths():
    out = tempfile.mkdtemp()
    cfg = config_from_dict(build_config(out))
    run_pipeline(cfg)
    manifest_path = cfg.run_dir() / "run_manifest.json"
    assert manifest_path.exists()
    text = manifest_path.read_text(encoding="utf-8")
    assert not _ABS_PATH_RE.search(text), f"run_manifest.json leaks an absolute path"
    manifest = json.loads(text)
    prov = manifest["provenance"]
    assert prov["packages"]["torch"]
    assert prov["corpus_digest"] is None, "smoke source has no sealed corpus to digest"

    align_path = cfg.run_dir() / "alignment" / "alignment_check.json"
    assert align_path.exists()
    html = (cfg.run_dir() / "report.html").read_text(encoding="utf-8")
    assert "Alignment &amp; provenance" in html
    assert "git" in html and "packages" in html.lower()


def test_store_summary_reads_shape_dtype_without_touching_activation_content():
    path = Path(tempfile.mkdtemp()) / "store.zarr"
    store = ActivationStore.create(path, n_series=4, n_windows=2, window=8, context_len=16)
    store.init_layer("m", "layer0", dim=6)
    store.set_layers({"m": ["layer0"]})
    summary = store.summary()
    assert summary["n_series"] == 4
    assert summary["models"]["m"]["n_layers"] == 1
    assert summary["models"]["m"]["shapes"]["layer0"]["shape"] == [4, 2, 6]
    assert summary["models"]["m"]["shapes"]["layer0"]["dtype"] == "float16"


def _built_run(tmp_out):
    cfg = config_from_dict(build_config(tmp_out))
    run_pipeline(cfg)
    return cfg.run_dir()


def test_verify_provenance_on_an_untouched_run_finds_no_diffs():
    run_dir = _built_run(tempfile.mkdtemp())
    result = verify_provenance(run_dir)
    assert result["diffs"] == []
    assert result["store_summary_changed"] is False
    assert "store_summary" in result["saved"]


def test_verify_provenance_detects_a_mutated_environment_field():
    run_dir = _built_run(tempfile.mkdtemp())
    manifest = load_manifest(run_dir)
    manifest["provenance"]["git_sha"] = "0" * 40
    manifest["provenance"]["packages"]["torch"] = "0.0.1-not-really-installed"
    save_manifest(run_dir, manifest)

    result = verify_provenance(run_dir)
    keys = {k for k, _, _ in result["diffs"]}
    assert "git_sha" in keys
    assert "packages" in keys
    # a genuinely unrelated field must not also fire (this isn't a
    # blanket "the whole dict differs" check)
    git_sha_diff = next(d for d in result["diffs"] if d[0] == "git_sha")
    assert git_sha_diff[1] == "0" * 40
    assert git_sha_diff[2] != "0" * 40


def test_verify_provenance_flags_a_store_that_no_longer_matches_its_summary():
    """The exact class of trap CLAUDE.md sec 11.15/11.25 named: a store that
    silently no longer looks like it did when its provenance was written.
    """
    run_dir = _built_run(tempfile.mkdtemp())
    manifest = load_manifest(run_dir)
    a_model = next(iter(manifest["provenance"]["store_summary"]["models"]))
    manifest["provenance"]["store_summary"]["models"][a_model]["n_layers"] = 999
    save_manifest(run_dir, manifest)

    result = verify_provenance(run_dir)
    assert result["store_summary_changed"] is True


def test_verify_provenance_raises_on_a_run_with_no_recorded_provenance():
    empty_run = Path(tempfile.mkdtemp())
    with pytest.raises(FileNotFoundError, match="no recorded provenance"):
        verify_provenance(empty_run)


if __name__ == "__main__":
    test_run_provenance_has_no_absolute_user_paths()
    test_schema_version_mismatch_raises_a_specific_error()
    test_fresh_store_round_trips_with_no_schema_error()
    test_smoke_run_records_provenance_with_no_absolute_paths()
    test_store_summary_reads_shape_dtype_without_touching_activation_content()
    test_verify_provenance_on_an_untouched_run_finds_no_diffs()
    test_verify_provenance_detects_a_mutated_environment_field()
    test_verify_provenance_flags_a_store_that_no_longer_matches_its_summary()
    test_verify_provenance_raises_on_a_run_with_no_recorded_provenance()
    print("provenance tests passed")
