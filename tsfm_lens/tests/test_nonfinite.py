"""Regression tests for activation-store finiteness guards (`ROADMAP.md`
sec 15 A19): a single inf/NaN layer should be caught loudly at write time
(partial) or refused outright (wholly non-finite), and `store.load` must
raise, naming the layer, the first time any consumer actually reads a
poisoned array -- rather than silently producing a plausible-looking
downstream number.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.extraction.store import ActivationStore


def _store(tmp="", n=8, w=3, d=4):
    path = Path(tempfile.mkdtemp()) / "activations.zarr"
    store = ActivationStore.create(path, n_series=n, n_windows=w, window=32, context_len=96)
    store.init_layer("m", "layer0", d, "float32")
    return store, n, w, d


def test_finite_batch_records_zero_and_does_not_warn():
    store, n, w, d = _store()
    aligned = np.random.default_rng(0).normal(size=(n, w, d)).astype(np.float32)
    store.write_batch("m", "layer0", 0, aligned)
    result = store.finalize_layer("m", "layer0")
    assert result == {"count": 0, "total": n * w * d, "fraction": 0.0}


def test_partial_nonfinite_is_recorded_not_raised():
    store, n, w, d = _store()
    aligned = np.random.default_rng(0).normal(size=(n, w, d)).astype(np.float32)
    aligned[0, 0, 0] = np.inf
    aligned[1, 0, 0] = np.nan
    store.write_batch("m", "layer0", 0, aligned)
    result = store.finalize_layer("m", "layer0")
    assert result["count"] == 2
    assert 0 < result["fraction"] < 1.0
    assert store.root.attrs["nonfinite"]["m/layer0"]["count"] == 2


def test_wholly_nonfinite_layer_raises():
    store, n, w, d = _store()
    aligned = np.full((n, w, d), np.inf, dtype=np.float32)
    store.write_batch("m", "layer0", 0, aligned)
    with pytest.raises(RuntimeError, match="wholly non-finite"):
        store.finalize_layer("m", "layer0")


def test_counts_accumulate_across_batches():
    store, n, w, d = _store(n=16)
    a1 = np.random.default_rng(0).normal(size=(8, w, d)).astype(np.float32)
    a1[0, 0, 0] = np.inf
    a2 = np.random.default_rng(1).normal(size=(8, w, d)).astype(np.float32)
    a2[0, 0, 0] = np.nan
    store.write_batch("m", "layer0", 0, a1)
    store.write_batch("m", "layer0", 8, a2)
    result = store.finalize_layer("m", "layer0")
    assert result["count"] == 2
    assert result["total"] == 16 * w * d


def test_load_raises_on_poisoned_layer_naming_it():
    store, n, w, d = _store()
    aligned = np.random.default_rng(0).normal(size=(n, w, d)).astype(np.float32)
    aligned[2, 1, 0] = np.inf
    store.write_batch("m", "layer0", 0, aligned)
    store.finalize_layer("m", "layer0")
    with pytest.raises(ValueError, match="m.*layer0"):
        store.load("m", "layer0", level="window")


def test_load_check_finite_false_bypasses():
    store, n, w, d = _store()
    aligned = np.random.default_rng(0).normal(size=(n, w, d)).astype(np.float32)
    aligned[2, 1, 0] = np.inf
    store.write_batch("m", "layer0", 0, aligned)
    store.finalize_layer("m", "layer0")
    out = store.load("m", "layer0", level="window", check_finite=False)
    assert not np.isfinite(out).all()  # the poison is still there, just not raised on


def test_load_clean_layer_does_not_raise():
    store, n, w, d = _store()
    aligned = np.random.default_rng(0).normal(size=(n, w, d)).astype(np.float32)
    store.write_batch("m", "layer0", 0, aligned)
    store.finalize_layer("m", "layer0")
    out = store.load("m", "layer0", level="window")
    assert out.shape == (n, w, d)


def test_extraction_store_dtype_float32_actually_preserves_precision():
    """The real bug this item found: `store_dtype` used to only change the
    zarr array's declared dtype while the values written into it had
    already been irreversibly cast to float16 first. A value that
    overflows float16 (~65504 max) but fits float32 must survive when
    `store_dtype: float32` is configured, and must NOT survive (becomes inf)
    under the float16 default -- proving the *values*, not just the
    array's dtype label, actually differ."""
    from tests.test_smoke import build_config
    from tsfm_lens.config import config_from_dict
    from tsfm_lens.pipeline import run_pipeline
    import tsfm_lens.extraction.extract as extract_mod

    big_value = 1.0e5  # overflows float16 (~65504 max), fits float32 easily
    orig_align = extract_mod.align

    def _spiked_align(hidden, pool):
        out = orig_align(hidden, pool)
        out = out.clone()
        out[0, 0, 0] = big_value
        return out

    for dtype, should_overflow in (("float16", True), ("float32", False)):
        extract_mod.align = _spiked_align
        try:
            out = tempfile.mkdtemp()
            cfg_dict = build_config(out)
            cfg_dict["extraction"] = {"store_dtype": dtype}
            cfg_dict["confirm"]["enabled"] = False
            cfg = config_from_dict(cfg_dict)
            run_pipeline(cfg, stages=["extract"])
        finally:
            extract_mod.align = orig_align

        from tsfm_lens.extraction.store import ActivationStore
        store = ActivationStore(cfg.run_dir() / "activations.zarr", mode="r")
        nonfinite = dict(store.root.attrs.get("nonfinite", {}))
        any_overflowed = any(v["count"] > 0 for v in nonfinite.values())
        assert any_overflowed == should_overflow, (dtype, nonfinite)


if __name__ == "__main__":
    test_finite_batch_records_zero_and_does_not_warn()
    test_partial_nonfinite_is_recorded_not_raised()
    test_wholly_nonfinite_layer_raises()
    test_counts_accumulate_across_batches()
    test_load_raises_on_poisoned_layer_naming_it()
    test_load_check_finite_false_bypasses()
    test_load_clean_layer_does_not_raise()
    test_extraction_store_dtype_float32_actually_preserves_precision()
    print("nonfinite tests passed")
