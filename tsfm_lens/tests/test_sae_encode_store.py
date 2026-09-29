"""The encode-store seam (ROADMAP.md sec 6.2.1 Stage 3d): persisting a
trained baseline SAE's encoded features back into the store under
`sae`/`sae_pooled/{model}/{layer}`, and reading them back via
`store.load(..., space="sae")`. Two layers of test: the store's own
round-trip (this file's first half, no torch training involved) and
`train.py::encode_and_persist_features` actually encoding with a real
`TopKSAE` (the second half).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.sae.models import TopKSAE
from tsfm_lens.sae.train import encode_and_persist_features


def _store(n=10, w=3, d=4):
    path = Path(tempfile.mkdtemp()) / "activations.zarr"
    store = ActivationStore.create(path, n_series=n, n_windows=w, window=32, context_len=96)
    store.init_layer("m", "layer0", d, "float32")
    aligned = np.random.default_rng(0).normal(size=(n, w, d)).astype(np.float32)
    store.write_batch("m", "layer0", 0, aligned)
    store.finalize_layer("m", "layer0")
    return store, n, w, d


def test_has_sae_features_false_before_anything_written():
    store, *_ = _store()
    assert store.has_sae_features("m", "layer0") is False


def test_load_space_sae_raises_actionably_when_absent():
    store, *_ = _store()
    with pytest.raises(KeyError, match="has_sae_features"):
        store.load("m", "layer0", level="series", space="sae")


def test_load_space_act_is_unaffected_by_the_new_parameter():
    store, n, w, d = _store()
    series = store.load("m", "layer0", level="series")
    window = store.load("m", "layer0", level="window")
    assert series.shape == (n, d)
    assert window.shape == (n, w, d)


def test_init_and_write_sae_batch_round_trips_at_both_granularities():
    store, n, w, d = _store()
    n_features = 6
    store.init_sae_layer("m", "layer0", n_features)
    assert store.has_sae_features("m", "layer0") is True

    features = np.random.default_rng(1).normal(size=(n, w, n_features)).astype(np.float32)
    store.write_sae_batch("m", "layer0", 0, features.astype(np.float16))

    window_back = store.load("m", "layer0", level="window", space="sae")
    series_back = store.load("m", "layer0", level="series", space="sae")
    assert window_back.shape == (n, w, n_features)
    assert series_back.shape == (n, n_features)
    np.testing.assert_allclose(window_back, features.astype(np.float16).astype(np.float32),
                               atol=1e-3)
    np.testing.assert_allclose(series_back, window_back.mean(axis=1), atol=1e-3)


def test_write_sae_batch_supports_row_subset_writes():
    """`encode_and_persist_features` writes in series batches, not one shot --
    a batch starting mid-array must land at the right offset, and rows never
    written stay at zarr's zero-fill default rather than aliasing another
    batch's data."""
    store, n, w, d = _store(n=10)
    n_features = 3
    store.init_sae_layer("m", "layer0", n_features)
    first = np.full((4, w, n_features), 1.0, dtype=np.float16)
    second = np.full((6, w, n_features), 2.0, dtype=np.float16)
    store.write_sae_batch("m", "layer0", 0, first)
    store.write_sae_batch("m", "layer0", 4, second)

    out = store.load("m", "layer0", level="window", space="sae")
    assert np.all(out[:4] == 1.0)
    assert np.all(out[4:] == 2.0)


def test_sae_space_does_not_alias_act_space():
    store, n, w, d = _store()
    n_features = 4
    store.init_sae_layer("m", "layer0", n_features)
    store.write_sae_batch("m", "layer0", 0,
                          np.zeros((n, w, n_features), dtype=np.float16))
    act = store.load("m", "layer0", level="window", space="act")
    sae = store.load("m", "layer0", level="window", space="sae")
    assert act.shape[-1] == d
    assert sae.shape[-1] == n_features
    assert not np.allclose(act[..., :min(d, n_features)], sae[..., :min(d, n_features)])


def test_encode_and_persist_features_matches_direct_encode():
    store, n, w, d = _store(n=12, w=3, d=5)
    torch.manual_seed(0)
    sae = TopKSAE(d, dict_size=8, k=3)
    device = torch.device("cpu")

    encode_and_persist_features(store, "m", "layer0", sae, device, series_batch=5)

    assert store.has_sae_features("m", "layer0") is True
    persisted = store.load("m", "layer0", level="window", space="sae")
    assert persisted.shape == (n, w, 8)

    raw = store.load("m", "layer0", level="window", space="act")
    with torch.no_grad():
        expected = sae.encode(torch.from_numpy(raw.reshape(-1, d).astype(np.float32))).numpy()
    expected = expected.reshape(n, w, 8)
    # float16 round-trip tolerance, not bit-exact
    np.testing.assert_allclose(persisted.astype(np.float32), expected, atol=1e-2, rtol=1e-2)


def test_encode_and_persist_features_batches_do_not_lose_or_duplicate_rows():
    """series_batch smaller than n_series exercises the batch-slicing path
    the single-batch case above doesn't."""
    store, n, w, d = _store(n=17, w=2, d=4)
    torch.manual_seed(1)
    sae = TopKSAE(d, dict_size=6, k=2)
    device = torch.device("cpu")
    encode_and_persist_features(store, "m", "layer0", sae, device, series_batch=5)

    persisted = store.load("m", "layer0", level="window", space="sae")
    raw = store.load("m", "layer0", level="window", space="act")
    with torch.no_grad():
        expected = sae.encode(
            torch.from_numpy(raw.reshape(-1, d).astype(np.float32))).numpy().reshape(n, w, 6)
    np.testing.assert_allclose(persisted.astype(np.float32), expected, atol=1e-2, rtol=1e-2)
