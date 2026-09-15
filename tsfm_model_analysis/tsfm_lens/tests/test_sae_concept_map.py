"""Tests for `report/sae_concept_map.py` (ROADMAP.md sec 32.14, Item M).

Synthetic with planted answers throughout (sec 2.4) -- a fixture read off a
real run would make these pass for whatever that run happened to contain.
Each test builds a minimal `run_dir` with exactly `sae/concepts.json` plus
each target's own `sae/<model>/<layer>_ablation.json`, matching
`build_concept_map_data`'s own two-artifact read.

Every load-bearing assertion here was confirmed to discriminate (sec 11.53's
postscript): `test_reducer_fit_once_across_run_not_per_target` was run
against a deliberately-reverted per-target-fit version of
`build_concept_map_data` (fitting `StandardScaler`+`PCA` inside the target
loop instead of once over the concatenated matrix) and failed as expected
(ratio collapsed from ~10x to ~1x); restored before committing.
`test_well_separated_concepts_are_not_one_blob` was likewise confirmed to
fail when the fixture's three concepts were moved on top of each other.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.response import CHANNELS  # noqa: E402
from tsfm_lens.sae.train import sanitize  # noqa: E402
from tsfm_lens.utils import save_json  # noqa: E402
from tsfm_lens.report.sae_concept_map import (  # noqa: E402
    build_concept_map_data, concept_map_block)


def _target_rec(model, layer, concepts, withheld=False, non_modular=False):
    return {
        "model": model, "layer": layer, "withheld": withheld,
        "non_modular": non_modular, "channel_columns": list(CHANNELS),
        "concepts": [] if (withheld or non_modular) else concepts,
    }


def _concept(concept_id, features, name="concept"):
    return {"concept": concept_id, "features": list(features), "name": name}


def _candidate(feature, channel_values: dict):
    """One ablation candidate; `channels[ch]["null_p95"]` pinned at 1.0 so
    `signed_effect` IS the null-unit value directly, matching
    `test_sae_concept_report.py`'s own `_candidate` convention."""
    channels = {ch: {"null_p95": 1.0, "signed_effect": float(v)}
               for ch, v in channel_values.items()}
    return {"feature": feature, "scorable": True,
            "n_channels_clearing": sum(1 for v in channel_values.values() if abs(v) >= 1.0),
            "channels": channels}


def _write_concepts(run_dir, targets):
    save_json(Path(run_dir) / "sae" / "concepts.json",
              {"schema_version": 1, "space": "ablation", "targets": targets})


def _write_ablation(run_dir, model, layer, candidates):
    path = (Path(run_dir) / "sae" / sanitize(model)
            / f"{sanitize(layer)}_ablation.json")
    save_json(path, {"candidates": candidates})


def _row(channel, value, rest=0.0):
    return {ch: (value if ch == channel else rest) for ch in CHANNELS}


# ---------------------------------------------------------------------------
# 1. Degradation: too few points.
# ---------------------------------------------------------------------------

def test_returns_none_when_concepts_json_absent(tmp_path):
    (tmp_path / "sae").mkdir()
    assert build_concept_map_data(tmp_path) is None


def test_returns_none_with_fewer_than_two_points(tmp_path):
    _write_concepts(tmp_path, {
        "Alpha/layer.0": _target_rec("Alpha", "layer.0",
                                     [_concept(0, [0], "solo")])})
    _write_ablation(tmp_path, "Alpha", "layer.0",
                    [_candidate(0, _row("trend", 5.0))])
    assert build_concept_map_data(tmp_path) is None


# ---------------------------------------------------------------------------
# 2. Withheld / non-modular targets contribute no rows.
# ---------------------------------------------------------------------------

def test_withheld_and_non_modular_targets_contribute_no_points(tmp_path):
    good_concepts = [_concept(0, [0, 1], "a"), _concept(1, [2, 3], "b")]
    _write_concepts(tmp_path, {
        "Alpha/layer.0": _target_rec("Alpha", "layer.0", [], withheld=True),
        "Beta/layer.0": _target_rec("Beta", "layer.0", [], non_modular=True),
        "Gamma/layer.0": _target_rec("Gamma", "layer.0", good_concepts),
    })
    _write_ablation(tmp_path, "Gamma", "layer.0", [
        _candidate(0, _row("trend", 5.0)), _candidate(1, _row("trend", 5.2)),
        _candidate(2, _row("trend", -5.0)), _candidate(3, _row("trend", -5.2)),
    ])
    # Alpha/Beta have no ablation file at all -- if the withheld/non_modular
    # skip were broken, loading it would raise or silently return zero rows
    # anyway; write nothing for them so a regression here would surface as
    # a crash, not a silent pass.
    data = build_concept_map_data(tmp_path)
    assert data is not None
    assert set(data["points"]["target"].unique()) == {"Gamma/layer.0"}


# ---------------------------------------------------------------------------
# 3. Basic per-point fields.
# ---------------------------------------------------------------------------

def test_points_carry_norm_top_channel_and_concept_name(tmp_path):
    vec_values = {"trend": 3.0, "seasonal": -4.0}  # norm = 5.0, top = seasonal
    _write_concepts(tmp_path, {
        "Alpha/layer.0": _target_rec(
            "Alpha", "layer.0",
            [_concept(0, [0, 1], "riser"), _concept(1, [2, 3], "faller")]),
    })
    row_a = {**{ch: 0.0 for ch in CHANNELS}, **vec_values}
    _write_ablation(tmp_path, "Alpha", "layer.0", [
        _candidate(0, row_a), _candidate(1, row_a),
        _candidate(2, _row("trend", -3.0)), _candidate(3, _row("trend", -3.1)),
    ])
    data = build_concept_map_data(tmp_path)
    assert data is not None
    pts = data["points"]
    row0 = pts[pts["feature"] == 0].iloc[0]
    assert abs(float(row0["norm"]) - 5.0) < 1e-9
    assert row0["top_channel"] == "seasonal"
    assert row0["concept_name"] == "riser"
    assert len(data["explained_variance"]) == 2
    assert all(0.0 <= v <= 1.0 for v in data["explained_variance"])
    assert len(data["pc_labels"]) == 2


def test_centroids_equal_mean_of_member_projections(tmp_path):
    _write_concepts(tmp_path, {
        "Alpha/layer.0": _target_rec("Alpha", "layer.0",
                                     [_concept(0, [0, 1, 2], "c0")]),
    })
    _write_ablation(tmp_path, "Alpha", "layer.0", [
        _candidate(0, _row("trend", 1.0)),
        _candidate(1, _row("trend", 2.0)),
        _candidate(2, _row("trend", 3.0)),
    ])
    data = build_concept_map_data(tmp_path)
    pts, cents = data["points"], data["centroids"]
    expect_pc1 = pts["pc1"].mean()
    expect_pc2 = pts["pc2"].mean()
    got = cents[(cents["target"] == "Alpha/layer.0") & (cents["concept"] == 0)].iloc[0]
    assert abs(float(got["pc1"]) - expect_pc1) < 1e-9
    assert abs(float(got["pc2"]) - expect_pc2) < 1e-9


# ---------------------------------------------------------------------------
# 4. Load-bearing negative (ROADMAP.md sec 32.14's acceptance criterion):
#    genuinely well-separated concepts must NOT render as one blob.
# ---------------------------------------------------------------------------

def test_well_separated_concepts_are_not_one_blob(tmp_path):
    """Three concepts planted far apart along two channels; centroid-to-
    centroid distances must dwarf each concept's own internal spread. A
    reducer that collapsed everything to one blob (e.g. a broken scaling
    step, or a PCA fit on the wrong axis) would fail this ratio check even
    though it would still 'render a figure' -- the acceptance criterion is
    quantitative, not merely that the code runs (sec 11.48's own lesson:
    a control that explains everything is as wrong as one that explains
    nothing)."""
    jitter = [-0.05, 0.0, 0.05, 0.02]
    concepts = [_concept(0, [0, 1, 2, 3], "c0"),
                _concept(1, [4, 5, 6, 7], "c1"),
                _concept(2, [8, 9, 10, 11], "c2")]
    _write_concepts(tmp_path, {"Alpha/layer.0": _target_rec("Alpha", "layer.0", concepts)})
    cands = []
    for i, j in enumerate(jitter):
        cands.append(_candidate(i, _row("trend", 10.0 + j)))
    for i, j in enumerate(jitter):
        cands.append(_candidate(4 + i, _row("trend", -10.0 + j)))
    for i, j in enumerate(jitter):
        cands.append(_candidate(8 + i, _row("seasonal", 10.0 + j)))
    _write_ablation(tmp_path, "Alpha", "layer.0", cands)

    data = build_concept_map_data(tmp_path)
    pts, cents = data["points"], data["centroids"]

    def spread(cid):
        g = pts[pts["concept"] == cid]
        c = cents[cents["concept"] == cid].iloc[0]
        d = np.hypot(g["pc1"] - c["pc1"], g["pc2"] - c["pc2"])
        return float(d.max())

    max_spread = max(spread(0), spread(1), spread(2))
    pairs = [(0, 1), (0, 2), (1, 2)]
    for a, b in pairs:
        ca = cents[cents["concept"] == a].iloc[0]
        cb = cents[cents["concept"] == b].iloc[0]
        dist = float(np.hypot(ca["pc1"] - cb["pc1"], ca["pc2"] - cb["pc2"]))
        assert dist > 5.0 * max_spread, (
            f"concepts {a} and {b} are not visually separated: "
            f"centroid distance {dist} vs max internal spread {max_spread}")


# ---------------------------------------------------------------------------
# 5. M2: the reducer is fit ONCE across the whole run, not once per target.
# ---------------------------------------------------------------------------

def test_reducer_fit_once_across_run_not_per_target(tmp_path):
    """Two targets separated along the SAME channel by very different
    absolute scales (5 units vs 50 units). A single run-wide `StandardScaler`
    normalizes by the GLOBAL std of that channel, so the small-scale
    target's own separation is suppressed far more than the large-scale
    target's -- the two targets' own within-target centroid distances (in
    projected space) must differ by roughly the same ~10x ratio as their raw
    scales. A per-target fit (the bug this guards against) would normalize
    each target independently and make the two ratios equal instead
    (confirmed by reverting to a per-target fit during development -- see
    module docstring)."""
    concepts = [_concept(0, [0, 1, 2, 3], "hi"), _concept(1, [4, 5, 6, 7], "lo")]
    _write_concepts(tmp_path, {
        "Alpha/layer.0": _target_rec("Alpha", "layer.0", concepts),
        "Beta/layer.0": _target_rec("Beta", "layer.0", concepts),
    })
    jitter = [-0.1, 0.0, 0.1, 0.05]
    small_scale, large_scale = 2.5, 25.0

    def cands(scale):
        out = []
        for i, j in enumerate(jitter):
            out.append(_candidate(i, _row("trend", scale + j)))
        for i, j in enumerate(jitter):
            out.append(_candidate(4 + i, _row("trend", -scale + j)))
        return out

    _write_ablation(tmp_path, "Alpha", "layer.0", cands(small_scale))
    _write_ablation(tmp_path, "Beta", "layer.0", cands(large_scale))

    data = build_concept_map_data(tmp_path)
    pts, cents = data["points"], data["centroids"]

    def centroid_dist(target):
        c = cents[cents["target"] == target]
        c0 = c[c["concept"] == 0].iloc[0]
        c1 = c[c["concept"] == 1].iloc[0]
        return float(np.hypot(c0["pc1"] - c1["pc1"], c0["pc2"] - c1["pc2"]))

    dist_alpha = centroid_dist("Alpha/layer.0")
    dist_beta = centroid_dist("Beta/layer.0")
    assert dist_alpha > 1e-9
    ratio = dist_beta / dist_alpha
    assert ratio > 5.0, (
        f"expected Beta's (50-unit-scale) separation to read far larger "
        f"than Alpha's (5-unit-scale) under a single run-wide fit; got "
        f"ratio {ratio} (dist_alpha={dist_alpha}, dist_beta={dist_beta})")


# ---------------------------------------------------------------------------
# 6. End-to-end render via concept_map_block.
# ---------------------------------------------------------------------------

def test_concept_map_block_renders_figure_and_variance_caption(tmp_path):
    concepts = [_concept(0, [0, 1, 2], "c0"), _concept(1, [3, 4, 5], "c1")]
    _write_concepts(tmp_path, {"Alpha/layer.0": _target_rec("Alpha", "layer.0", concepts)})
    _write_ablation(tmp_path, "Alpha", "layer.0", [
        _candidate(0, _row("trend", 5.0)), _candidate(1, _row("trend", 5.2)),
        _candidate(2, _row("trend", 4.8)), _candidate(3, _row("trend", -5.0)),
        _candidate(4, _row("trend", -5.1)), _candidate(5, _row("trend", -4.9)),
    ])
    cfg = SimpleNamespace(run=SimpleNamespace(seed=0))
    out = concept_map_block(tmp_path, cfg)
    assert "Concept map" in out
    assert "PC1" in out or "trend" in out  # axis label names top loadings
    assert "%" in out  # explained-variance figure in the caption
    assert "UMAP" in out  # limitations note states the PCA-not-UMAP rule


def test_concept_map_block_empty_when_no_data(tmp_path):
    (tmp_path / "sae").mkdir()
    cfg = SimpleNamespace(run=SimpleNamespace(seed=0))
    assert concept_map_block(tmp_path, cfg) == ""
