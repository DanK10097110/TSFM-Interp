"""`l3/patching.json`'s `rel_depth`/`depth_axis` must describe the PATCHED
layer subset, not the full sensitivity layer list (ROADMAP.md sec 18 F1's
real-checkpoint acceptance test, run against `configs/medium_run_chronos_
base.yaml`, found this as a live regression: `run_l3` computed one
`depth_axis_for_run(...)` call per model from `_sensitivity`'s full layer
list, then reused it verbatim for `patching.json`, whose own `layers` come
from `_patching`'s coarser `layers[::l3.patching.layer_stride]` subsample.
Whenever `layer_stride > 1` the two lists have different lengths, so
`patching.json["rel_depth"]` silently carried more entries than
`patching.json["layers"]` -- Plotly.js zips mismatched-length x/y arrays
index-for-index rather than raising, so every patched layer past the first
got plotted at the wrong depth in `report.py`'s L3 patching-curve and
window/horizon-heatmap figures.

The mock smoke config (`configs/smoke.yaml`, `tests/test_smoke.py`) never
caught this because it sets `l3.patching.layer_stride: 1`, where the
sensitivity and patching layer lists coincide by construction -- this test
exists specifically to exercise `layer_stride > 1` on fast mock adapters.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config
from tsfm_lens.config import config_from_dict
from tsfm_lens.pipeline import run_pipeline
from tsfm_lens.utils import load_json


def test_patching_json_rel_depth_matches_the_patched_layer_subset_length():
    d = build_config(tempfile.mkdtemp())
    # Sensitivity runs over every captured layer; patching subsamples every
    # other one -- the two lists must differ in length for this test to
    # actually exercise the bug.
    d["l3"]["patching"]["layer_stride"] = 2
    cfg = config_from_dict(d)
    run_pipeline(cfg, stages=["extract", "l3"])

    meta = load_json(cfg.run_dir() / "l3" / "meta.json")
    patching = load_json(cfg.run_dir() / "l3" / "patching.json")

    for model, info in patching.items():
        n_patched = len(info["layers"])
        n_sensitivity = len(meta["layers"][model])
        assert n_sensitivity > n_patched, (
            "test precondition failed -- layer_stride=2 should make the "
            "sensitivity layer list strictly longer than the patched subset "
            f"for {model!r} ({n_sensitivity} vs {n_patched})")
        assert len(info["rel_depth"]) == n_patched, (
            f"{model!r}: patching.json's rel_depth has {len(info['rel_depth'])} "
            f"entries but the patched layer subset has {n_patched} -- these "
            f"must match 1:1 or every downstream plot zips them against the "
            f"wrong depth coordinate")
        # And it must be the PATCHED-subset axis, not sensitivity's, reused --
        # not just the right length by coincidence.
        assert info["rel_depth"] != meta["rel_depth"][model][:n_patched], (
            f"{model!r}: patching rel_depth looks like a truncated slice of "
            f"the full sensitivity axis rather than its own recomputed axis "
            f"over the patched subset")


def test_patching_json_rel_depth_equals_sensitivity_axis_when_stride_is_one():
    """Regression guard the other direction: when the two layer lists
    genuinely coincide (the smoke config's own `layer_stride: 1`), the two
    axes must still agree exactly -- confirms the fix doesn't invent a
    spurious difference where none exists.
    """
    d = build_config(tempfile.mkdtemp())
    assert d["l3"]["patching"]["layer_stride"] == 1
    cfg = config_from_dict(d)
    run_pipeline(cfg, stages=["extract", "l3"])

    meta = load_json(cfg.run_dir() / "l3" / "meta.json")
    patching = load_json(cfg.run_dir() / "l3" / "patching.json")

    for model, info in patching.items():
        assert info["layers"] == meta["layers"][model]
        assert info["rel_depth"] == meta["rel_depth"][model]
