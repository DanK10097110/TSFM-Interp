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


if __name__ == "__main__":
    test_run_provenance_has_no_absolute_user_paths()
    test_schema_version_mismatch_raises_a_specific_error()
    test_fresh_store_round_trips_with_no_schema_error()
    test_smoke_run_records_provenance_with_no_absolute_paths()
    print("provenance tests passed")
