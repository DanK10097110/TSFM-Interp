"""`data._sample_to_row` carries a sample's data `role` through (ROADMAP sec 41, V3-A).

Opt-in: a sample without a role (any corpus built before roles existed) yields a row with no `role`
key at all, so every existing row and everything keyed on its columns is unchanged. A sealed corpus
built with roles must reach the loader with them: loaded through `_load_corpus_rows` (the verified
`load_sealed` path), not only through a hand-built object.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.data import _load_corpus_rows, _sample_to_row

from tsfm_benchmark.build_pipeline.schema import GroundTruth, Provenance, TimeSeriesSample
from tsfm_benchmark.build_pipeline.seal import seal_corpus


def _sample(seed, role=None, generator="parametric"):
    return TimeSeriesSample(values=np.sin(np.arange(40) / 3.0) + seed, ground_truth=GroundTruth(),
                            provenance=Provenance(generator=generator, seed=seed, generator_params={"tier": "synthetic", "task_name": "t"}),
                            role=role)


def test_role_is_copied_when_present_and_absent_otherwise():
    assert _sample_to_row(_sample(0, role="external_real", generator="gifteval_window"))["role"] == "external_real"
    assert _sample_to_row(_sample(1, role="synthetic"))["role"] == "synthetic"
    assert "role" not in _sample_to_row(_sample(2))


def test_role_survives_the_verified_sealed_load_path(tmp_path):
    seal_corpus([_sample(0, role="synthetic"), _sample(1, role="real_derived", generator="mixture")], str(tmp_path), 0, "public")
    rows, _ = _load_corpus_rows(tmp_path, verify=True)
    assert [r["role"] for r in rows] == ["synthetic", "real_derived"]


def test_roleless_sealed_corpus_rows_are_unchanged(tmp_path):
    seal_corpus([_sample(0)], str(tmp_path), 0, "public")
    rows, _ = _load_corpus_rows(tmp_path, verify=True)
    assert "role" not in rows[0]
