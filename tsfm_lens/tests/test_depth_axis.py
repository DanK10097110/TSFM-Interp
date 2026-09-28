"""Tests for the shared depth axis (ROADMAP.md sec 18 F1).

Three things are worth pinning, and only the first is about arithmetic. The
`index` axis must stay bit-identical to `utils.relative_depths`, or every
number recorded before this module existed silently moves. The `block` axis
must place an encoder-only capture surface in the bottom half of the range,
which is the whole point of building it. And `align_on_axis` must refuse to
report statistics over a range one model never reached -- the specific
fabrication `np.interp`'s edge clamping used to produce silently.

Everything runs on mock adapters, so there is no checkpoint, no GPU and no
download; the encoder-decoder case uses `mock_encdec`, whose decoder blocks
genuinely run and genuinely sit outside the capture regex.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis import depth_axis as da
from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.models import build_adapter
from tsfm_lens.utils import relative_depths


def _adapter(name: str, adapter: str, stride: int = 1):
    cfg = ModelConfig(name=name, adapter=adapter, capture_layer_stride=stride)
    data = DataConfig(context_len=64, horizon=16)
    return build_adapter(cfg, data, torch.device("cpu"), torch.float32)


def test_index_axis_matches_relative_depths_exactly():
    """The legacy axis is preserved bit-for-bit, including the n=1 convention."""
    for n in (1, 2, 3, 6, 20):
        adapter = _adapter("m", "mock_patch")
        layers = [f"blocks.{i}" for i in range(n)]
        got = da.depth_axis(adapter, layers, "index").coords
        assert np.array_equal(got, relative_depths(n))


def test_block_axis_is_stride_invariant():
    """A block keeps its coordinate whether or not its neighbours were captured.

    This is the defect that makes the legacy axis unusable across runs: under
    `index`, block 4 of 6 sits at 0.8 when every block is captured and at 1.0
    when stride 2 drops the odd ones.
    """
    full = _adapter("a", "mock_patch", stride=1)
    strided = _adapter("b", "mock_patch", stride=2)

    coords_full = da.depth_axis(full, full.layer_names(), "block").coords
    coords_strided = da.depth_axis(strided, strided.layer_names(), "block").coords

    kept = [full.layer_names().index(n) for n in strided.layer_names()]
    assert np.allclose(coords_strided, coords_full[kept])

    idx_full = da.depth_axis(full, full.layer_names(), "index").coords
    idx_strided = da.depth_axis(strided, strided.layer_names(), "index").coords
    assert not np.allclose(idx_strided, idx_full[kept])


def test_encoder_only_capture_surface_occupies_the_lower_half():
    """The acceptance criterion: an encoder-decoder model must stop mid-axis."""
    enc_dec = _adapter("chronos_like", "mock_encdec")
    coords = da.depth_axis(enc_dec, enc_dec.layer_names(), "block").coords
    assert coords[-1] == pytest.approx(3 / 7)

    full = _adapter("timesfm_like", "mock_patch")
    assert da.depth_axis(full, full.layer_names(), "block").coords[-1] == 1.0


def test_total_stack_size_separates_stride_loss_from_architectural_loss():
    """The two losses have different fixes, so collapsing them hides which applies."""
    enc_dec = _adapter("chronos_like", "mock_encdec", stride=2)
    stack = da.total_stack_size(enc_dec)
    assert stack["n_blocks_total"] == 8
    assert stack["n_blocks_matched"] == 4
    assert stack["n_blocks_captured"] == 2
    assert stack["n_blocks_uncaptured_in_captured_surface"] == 2
    assert stack["n_blocks_outside_captured_surface"] == 4
    assert stack["uncaptured_surfaces"] == {"decoder": 4}

    plain = da.total_stack_size(_adapter("m", "mock_patch"))
    assert plain["n_blocks_outside_captured_surface"] == 0
    assert plain["uncaptured_surfaces"] == {}


def test_uncaptured_decoder_blocks_really_run():
    """The mock declares a surface it actually computes, not a fiction.

    `total_stack_size` would report the same numbers either way, so the
    declaration is only honest if removing the blocks changes the forecast.
    """
    enc_dec = _adapter("chronos_like", "mock_encdec")
    enc_dec.ensure_loaded()
    x = np.random.default_rng(0).normal(size=(2, 64)).astype(np.float32)
    with_decoder = enc_dec.predict(x, 16, [0.5])["point"]
    enc_dec.module.decoder_blocks = torch.nn.ModuleList()
    without = enc_dec.predict(x, 16, [0.5])["point"]
    assert not np.allclose(with_decoder, without)


def test_compute_axis_falls_back_loudly_and_prefers_the_forecast_denominator():
    """Absent per-block FLOPs the axis degrades to `block` and says so."""
    adapter = _adapter("m", "mock_patch")
    layers = adapter.layer_names()

    degraded = da.depth_axis(adapter, layers, "compute", budget={})
    assert degraded.axis == "block"
    assert degraded.fallback_from == "compute"
    assert degraded.degraded
    assert "compute_unavailable_because" in degraded.detail

    cumulative = {n: float(i + 1) for i, n in enumerate(layers)}
    budget = {"forward": {"flops": 6.0, "blocks": {"cumulative": cumulative}},
              "predict": {"flops": 12.0}}
    axis = da.depth_axis(adapter, layers, "compute", budget=budget)
    assert axis.fallback_from is None
    assert axis.detail["denominator_is_forecast"]
    assert axis.coords[-1] == pytest.approx(0.5)

    no_forecast = {"forward": {"flops": 6.0, "blocks": {"cumulative": cumulative}}}
    capture_only = da.depth_axis(adapter, layers, "compute", budget=no_forecast)
    assert capture_only.coords[-1] == pytest.approx(1.0)
    assert not capture_only.detail["denominator_is_forecast"]


def test_functional_axis_needs_one_value_per_layer():
    """A wrong-length property is a caller error, not a silently truncated axis."""
    adapter = _adapter("m", "mock_patch")
    layers = adapter.layer_names()
    out = da.depth_axis(adapter, layers, "functional", functional_values=np.array([0.1]))
    assert out.axis == "block" and out.fallback_from == "functional"

    values = np.linspace(1.0, 0.0, len(layers))
    ok = da.depth_axis(adapter, layers, "functional", functional_values=values)
    assert ok.axis == "functional"
    assert np.array_equal(ok.coords, values)


def test_align_on_axis_covers_only_the_shared_range():
    """The fabrication this exists to prevent, stated as a number.

    Model B spans [0, 0.5]; interpolating it onto [0, 1] would clamp its last
    value flat across the top half and then average agreement over it. The
    grid must stop at 0.5 and the unmatched range must be reported.
    """
    ca, va = np.linspace(0.0, 1.0, 11), np.linspace(0.0, 10.0, 11)
    cb, vb = np.linspace(0.0, 0.5, 6), np.linspace(0.0, 5.0, 6)

    out = da.align_on_axis(ca, va, cb, vb, n_grid=11)
    assert out["overlap"] == (0.0, 0.5)
    assert out["grid"].max() == pytest.approx(0.5)
    assert out["unmatched"]["a"] == [(0.5, 1.0)]
    assert out["unmatched"]["b"] == []
    assert out["overlap_fraction"] == pytest.approx(0.5)
    assert np.allclose(out["a"], out["b"])

    naive = np.interp(np.linspace(0.0, 1.0, 11), cb, vb)
    assert np.allclose(naive[6:], 5.0)


def test_align_on_axis_reports_disjoint_ranges_instead_of_raising():
    """Two models sharing no depth range is a finding about the run, not a crash."""
    out = da.align_on_axis([0.0, 0.1], [1.0, 2.0], [0.8, 0.9], [3.0, 4.0])
    assert out["n_grid"] == 0
    assert out["grid"].size == 0
    assert out["overlap_fraction"] == 0.0
    assert "share no depth range" in out["note"]


def test_align_on_axis_handles_decreasing_functional_coordinates():
    """`np.interp` needs increasing x; a functional axis commonly decreases."""
    ca = np.array([1.0, 0.7, 0.3, 0.0])
    va = np.array([0.0, 1.0, 2.0, 3.0])
    out = da.align_on_axis(ca, va, ca, va, n_grid=5)
    assert np.all(np.diff(out["a"]) <= 0)
    assert np.allclose(out["a"], out["b"])


def test_align_on_axis_interpolates_2d_values_column_wise():
    """Per-series curves interpolate on the same grid without reshaping."""
    ca = np.linspace(0.0, 1.0, 5)
    va = np.stack([np.linspace(0.0, 1.0, 5), np.linspace(0.0, 2.0, 5)], axis=1)
    out = da.align_on_axis(ca, va, ca, va, n_grid=3)
    assert out["a"].shape == (3, 2)
    assert out["a"][-1, 1] == pytest.approx(2.0)


def test_unknown_axis_and_empty_layers_fail_loudly():
    adapter = _adapter("m", "mock_patch")
    with pytest.raises(ValueError):
        da.depth_axis(adapter, adapter.layer_names(), "depth")
    with pytest.raises(ValueError):
        da.depth_axis(adapter, [], "block")
    with pytest.raises(ValueError):
        da.depth_axis(adapter, ["blocks.99"], "block")


def test_depth_axis_config_default_is_block_and_index_is_selectable():
    from tsfm_lens.config import AlignmentConfig
    assert AlignmentConfig().depth_axis == "block"
    assert AlignmentConfig(depth_axis="index").depth_axis == "index"


def test_block_axis_from_persisted_meta_matches_the_live_adapter():
    """`report`/`internals` have no live model -- persisted metadata must
    reproduce exactly what the live adapter path computes (ROADMAP.md sec
    18 F1's remaining wiring: `ActivationStore.stack_meta`).
    """
    enc_dec = _adapter("chronos_like", "mock_encdec")
    layers = enc_dec.layer_names()
    from_adapter = da.depth_axis(enc_dec, layers, "block")
    from_meta = da.depth_axis(None, layers, "block",
                              all_layer_names=enc_dec.all_layer_names(),
                              uncaptured_surfaces=da.adapter_uncaptured_surfaces(enc_dec))
    assert np.array_equal(from_adapter.coords, from_meta.coords)
    assert from_meta.coords[-1] == pytest.approx(3 / 7)
    assert from_meta.fallback_from is None


def test_block_axis_falls_back_to_index_with_neither_adapter_nor_meta():
    """A store predating `stack_meta` (or a model that never reported it)
    must degrade loudly to `index`, not crash trying to call a method on
    `None`.
    """
    n = 6
    layers = [f"blocks.{i}" for i in range(n)]
    out = da.depth_axis(None, layers, "block")
    assert out.fallback_from == "block"
    assert out.axis == "index"
    assert np.array_equal(out.coords, relative_depths(n))


def test_compute_and_functional_fallback_to_block_forward_persisted_meta():
    """The existing compute/functional -> block fallbacks must not silently
    drop back to `index` just because the caller supplied metadata instead
    of a live adapter -- a store-metadata-only caller (e.g. a report
    section) asking for `compute` with no budget should still land on the
    real `block` axis, not `index`.
    """
    enc_dec = _adapter("chronos_like", "mock_encdec")
    layers = enc_dec.layer_names()
    meta = {"all_layer_names": enc_dec.all_layer_names(),
           "uncaptured_surfaces": da.adapter_uncaptured_surfaces(enc_dec)}
    out = da.depth_axis(None, layers, "compute", budget={},
                        all_layer_names=meta["all_layer_names"],
                        uncaptured_surfaces=meta["uncaptured_surfaces"])
    assert out.fallback_from == "compute"
    assert out.axis == "block"
    assert out.coords[-1] == pytest.approx(3 / 7)


def test_depth_axis_for_run_prefers_store_meta_over_adapter_and_degrades_without_one():
    """`depth_axis_for_run` is the one call site every stage should use; pin
    its two contracts: prefer persisted metadata (works with no model
    reload), and degrade to `index` when `store` is `None` or has nothing
    for this model.
    """
    class _FakeStore:
        def __init__(self, meta):
            self._meta = meta

        def stack_meta(self, model):
            return self._meta.get(model, {})

    enc_dec = _adapter("chronos_like", "mock_encdec")
    layers = enc_dec.layer_names()
    meta = {"m": {"all_layers": enc_dec.all_layer_names(),
                  "uncaptured_surfaces": da.adapter_uncaptured_surfaces(enc_dec)}}
    store = _FakeStore(meta)

    with_store = da.depth_axis_for_run("block", store, "m", layers)
    assert with_store.coords[-1] == pytest.approx(3 / 7)
    assert with_store.fallback_from is None

    no_store = da.depth_axis_for_run("block", None, "m", layers)
    assert no_store.axis == "index"
    assert no_store.fallback_from == "block"

    empty_store = _FakeStore({})
    no_meta = da.depth_axis_for_run("block", empty_store, "other_model", layers)
    assert no_meta.axis == "index"
    assert no_meta.fallback_from == "block"

    # A live adapter still works as the fallback when the store has nothing
    # for this model.
    via_adapter = da.depth_axis_for_run("block", empty_store, "m", layers, adapter=enc_dec)
    assert via_adapter.coords[-1] == pytest.approx(3 / 7)
    assert via_adapter.fallback_from is None


def test_activation_store_stack_meta_round_trip(tmp_path):
    """`ActivationStore.set_stack_meta`/`.stack_meta` persist per-model stack
    layout, and a store that predates this metadata (or lacks a model)
    degrades to `{}` rather than raising.
    """
    from tsfm_lens.extraction.store import ActivationStore

    path = tmp_path / "activations.zarr"
    store = ActivationStore.create(path, n_series=4, n_windows=2, window=32, context_len=64)
    meta = {"model_a": {"all_layers": ["blocks.0", "blocks.1"],
                        "uncaptured_surfaces": {"decoder": 2}},
           "model_b": {"all_layers": ["blocks.0"], "uncaptured_surfaces": {}}}
    store.set_stack_meta(meta)

    reopened = ActivationStore(path, mode="r")
    assert reopened.stack_meta("model_a") == meta["model_a"]
    assert reopened.stack_meta("model_b") == meta["model_b"]
    assert reopened.stack_meta("never_extracted") == {}

    fresh = ActivationStore.create(tmp_path.parent / "no_meta.zarr", n_series=1, n_windows=1,
                                   window=32, context_len=32)
    assert fresh.stack_meta("anything") == {}
