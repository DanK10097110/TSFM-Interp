"""The ablation battery records and renders the rows each feature was really scored on.

`top_k_series` is a request. A feature that fires on fewer series gets fewer rows
(`top_firing_rows` never pads with silent rows) and the per-chunk cap truncates to
the strongest rows, both silently. The fixture is `test_profile_matched_null`'s
world: atom 0 fires on 3 series, every other atom on all 6.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_profile_matched_null import (  # noqa: E402,F401
    DICT, Z_ALL, _Cfg, _Data, _SAE, wired)
from tsfm_lens.report import sae_concepts as SC  # noqa: E402
from tsfm_lens.sae import response as R  # noqa: E402


def _run(adapter, top_k, batch_size=999, features=(0, 1, 2)):
    adapter.cfg.batch_size = batch_size
    acts = Z_ALL.max(axis=1).astype(np.float64)
    return R.feature_ablation_fingerprints(
        _Cfg, adapter, "blocks.0", _SAE(), _Data(), "cpu",
        candidates=[{"feature": f, "rules": []} for f in features],
        activations=acts, top_k_series=top_k, n_null_directions=4)


def _render(tmp_path, art):
    d = tmp_path / "sae" / "m"
    d.mkdir(parents=True)
    (d / "blocks0_ablation.json").write_text(json.dumps(art), encoding="utf-8")
    return SC.row_coverage_block(tmp_path)


def test_sparse_feature_with_fewer_firing_rows_is_counted_short(wired):
    out = _run(wired, top_k=5)
    n = {c["feature"]: c["n_rows_scored"] for c in out["candidates"]}
    assert n == {0: 3, 1: 5, 2: 5}
    cov = out["row_coverage"]
    assert (cov["top_k_requested"], cov["effective_k_cap"], cov["effective_k"]) == (5, 64, 5)
    assert (cov["n_candidates_short"], cov["n_candidates_zero_rows"],
            cov["n_candidates_full"], cov["cap_binds"]) == (1, 0, 2, False)


def test_a_feature_with_no_firing_row_is_counted_as_zero_rows(wired):
    acts = Z_ALL.max(axis=1).astype(np.float64)
    acts[:, 0] = 0.0
    out = R.feature_ablation_fingerprints(
        _Cfg, wired, "blocks.0", _SAE(), _Data(), "cpu",
        candidates=[{"feature": f, "rules": []} for f in (0, 1)],
        activations=acts, top_k_series=3, n_null_directions=4)
    assert out["candidates"][0]["n_rows_scored"] == 0
    assert out["row_coverage"]["n_candidates_zero_rows"] == 1
    assert out["row_coverage"]["n_candidates_full"] == 1


def test_chunk_cap_below_k_is_recorded_and_rendered_loudly(wired, tmp_path):
    out = _run(wired, top_k=4, batch_size=2)
    cov = out["row_coverage"]
    assert (cov["top_k_requested"], cov["effective_k_cap"], cov["effective_k"]) == (4, 2, 2)
    assert cov["cap_binds"] is True
    assert all(c["n_rows_scored"] == 2 for c in out["candidates"])
    html = _render(tmp_path, out)
    assert "Warning: the chunk cap is below the requested k" in html
    assert "<td>4</td><td>2</td><td>2</td>" in html


def test_no_warning_when_the_cap_does_not_bind(wired, tmp_path):
    html = _render(tmp_path, _run(wired, top_k=3))
    assert "Warning: the chunk cap" not in html
    assert "Rows each feature was actually ablated on" in html


def test_a_legacy_artifact_without_row_coverage_is_summarized_from_its_own_fields(tmp_path):
    art = {"withheld": False, "top_k_series": 32, "series_per_chunk_cap": 16,
           "candidates": [{"feature": 0, "n_top_series": 16}, {"feature": 1, "n_top_series": 0},
                          {"feature": 2, "n_top_series": 7}]}
    html = _render(tmp_path, art)
    assert "Warning: the chunk cap is below the requested k" in html
    cov = SC.row_coverage_rows(tmp_path)[0]
    assert (cov["n_candidates_full"], cov["n_candidates_short"],
            cov["n_candidates_zero_rows"]) == (1, 1, 1)
