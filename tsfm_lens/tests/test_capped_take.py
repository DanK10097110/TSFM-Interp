"""Regression tests for silent sample-cap tracking (`ROADMAP.md` sec 15
A16): several stages must fit one forward pass, so they clamp a configured
sample-size request to `adapter.cfg.batch_size` -- previously with no
warning and no record, making the *realized* n silently non-comparable
across runs or models (`CLAUDE.md` §5.3's own incident).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.meta_report import _batch_cap_warnings
from tsfm_lens.utils import capped_take


def test_no_reduction_is_silent_and_limited_by_is_none():
    out = capped_take(50, n_available=200, batch_size=64)
    assert out == {"n_requested": 50, "n_realized": 50, "limited_by": None}


def test_reduced_by_batch_size_names_it():
    out = capped_take(200, n_available=500, batch_size=32)
    assert out["n_realized"] == 32
    assert out["limited_by"] == ["batch_size"]


def test_reduced_by_corpus_size_names_it():
    out = capped_take(200, n_available=10, batch_size=64)
    assert out["n_realized"] == 10
    assert out["limited_by"] == ["n_available"]


def test_tied_limits_name_both():
    out = capped_take(200, n_available=32, batch_size=32)
    assert out["n_realized"] == 32
    assert out["limited_by"] == ["batch_size", "n_available"]


def test_exact_match_to_a_limit_is_not_flagged_as_reduced():
    # requested == n_available: not actually "reduced", even though the
    # value happens to equal one of the limits.
    out = capped_take(32, n_available=32, batch_size=64)
    assert out["limited_by"] is None


def test_lens_end_to_end_records_requested_and_realized():
    from tests.test_smoke import build_config
    from tsfm_lens.config import config_from_dict
    from tsfm_lens.pipeline import run_pipeline
    out = tempfile.mkdtemp()
    cfg_dict = build_config(out)
    cfg_dict["lens"]["max_series"] = 999  # exceeds every mock's batch_size (64)
    cfg = config_from_dict(cfg_dict)
    run_pipeline(cfg)
    import json
    lens = json.loads((cfg.run_dir() / "lens" / "lens.json").read_text(encoding="utf-8"))
    for model, payload in lens.items():
        assert payload["n_requested_skip"] == 999
        assert payload["n_series_skip"] <= 64
        assert payload["limited_by_skip"] is not None


def test_l3_patching_end_to_end_records_cap_fields():
    from tests.test_smoke import build_config
    from tsfm_lens.config import config_from_dict
    from tsfm_lens.pipeline import run_pipeline
    out = tempfile.mkdtemp()
    cfg_dict = build_config(out)
    cfg_dict["l3"]["patching"]["max_series"] = 999
    cfg = config_from_dict(cfg_dict)
    run_pipeline(cfg)
    import json
    patching = json.loads((cfg.run_dir() / "l3" / "patching.json").read_text(encoding="utf-8"))
    for model, payload in patching.items():
        assert payload["n_requested"] == 999
        assert payload["limited_by"] is not None


def test_batch_cap_warnings_fires_on_differing_realized_n():
    runs = [
        {"label": "run1", "lens_n_realized": {"TimesFM": 32}, "lens_limited_by": {"TimesFM": ["batch_size"]}},
        {"label": "run2", "lens_n_realized": {"TimesFM": 64}, "lens_limited_by": {"TimesFM": None}},
    ]
    warnings = _batch_cap_warnings(runs)
    assert any("TimesFM" in w and "not directly comparable" in w for w in warnings)


def test_batch_cap_warnings_silent_when_realized_n_matches():
    runs = [
        {"label": "run1", "lens_n_realized": {"TimesFM": 32}, "lens_limited_by": {"TimesFM": None}},
        {"label": "run2", "lens_n_realized": {"TimesFM": 32}, "lens_limited_by": {"TimesFM": None}},
    ]
    assert _batch_cap_warnings(runs) == []


if __name__ == "__main__":
    test_no_reduction_is_silent_and_limited_by_is_none()
    test_reduced_by_batch_size_names_it()
    test_reduced_by_corpus_size_names_it()
    test_tied_limits_name_both()
    test_exact_match_to_a_limit_is_not_flagged_as_reduced()
    test_lens_end_to_end_records_requested_and_realized()
    test_l3_patching_end_to_end_records_cap_fields()
    test_batch_cap_warnings_fires_on_differing_realized_n()
    test_batch_cap_warnings_silent_when_realized_n_matches()
    print("capped_take tests passed")
