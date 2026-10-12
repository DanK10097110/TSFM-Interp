"""ROADMAP.md sec 42 decision 4 and the opt-in knobs of task B.

`sae.forecast_preservation_chunked` lets `forecast_preservation_max_series` exceed the model's batch
size by scoring explicit batch-size chunks; off, the request is clamped as before. The planted
answer is a stub forecaster whose forecast is an exact function of the patched tokens, so the
chunked score must equal the single-batch score on the same rows, and the decoy is a forward that
would exceed the batch size (the stub raises on it, as `token_patch` does on a chunked batch).
Every new config field must leave the fingerprints of existing configs unchanged.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens.config import load_config  # noqa: E402
from tsfm_lens.manifest import fingerprint_stage, resolve_config_keys  # noqa: E402
from tsfm_lens.pipeline import _stages  # noqa: E402
from tsfm_lens.sae import eval as E  # noqa: E402

N, T, D_IN, H = 10, 2, 3, 4

PRE_CHANGE_FINGERPRINTS = {
    "known_answer": {'corpus': '2d2bf22c410839b7', 'extract': '0ffdfe2233b18c4f', 'budget': '88b093a6a8f7ac57', 'frontend': '1cf6a10e23b7368a', 'layer_screen': '394d92e8c66b1d0e', 'l0': 'a61ed1f12c11eb4a', 'internals': '31a18964cc40ee6b', 'lens': 'b096efeb45fb388c', 'l1': '603f901e0101bf7a', 'l2': '3f541c74d766ebb5', 'l3': '62885e04897e0eb7', 'attention': '9b435c76be0c0c23', 'cluster': 'd11a3835c323cdd4', 'sae': '46b4fb722ff6e0f0', 'concepts': '02973c8311bbfc8c', 'exemplars': '4108c4b9b4fb14bd', 'register': '5745122ab2b37170', 'confirm': '6f9769f4de374b47', 'report': '6df0ba9473d0cb01'},
    "panel7_v3": {'corpus': 'fc504e3f1e624193', 'extract': '4bf94dfcfcd05e58', 'budget': '16132a4c14a848de', 'frontend': 'bf9c5aadfda5ec84', 'layer_screen': '137926713fdb92da', 'l0': '712a644460af6946', 'internals': '6a37c51d9a02cfee', 'lens': '44bf7b19982516db', 'l1': '654a0a40a70cb139', 'l2': 'f42f43b70adfbbf0', 'l3': 'ace48ed6934155b5', 'attention': 'f0a36f5921ee7d3d', 'cluster': '3b48b7f9c3151270', 'sae': 'b6a9804fbfe1a7b0', 'concepts': 'f891d700a55fdb8f', 'exemplars': '716571e0a32329ba', 'register': 'a33edca0a2ded9dd', 'confirm': 'ee88d17ef62f25df', 'report': '3f384ca093578d0f'},
}


def _fingerprints(cfg) -> dict:
    cur = {}
    for s in _stages():
        cur[s.name] = fingerprint_stage(resolve_config_keys(cfg, s.config_keys),
                                        {d: cur[d] for d in s.deps})
    return cur


@pytest.mark.parametrize("name", sorted(PRE_CHANGE_FINGERPRINTS))
def test_every_new_field_at_its_default_leaves_every_stage_fingerprint_unchanged(name):
    """`confirm.path` is cleared first: panel7_v3's pins were taken while its private split
    was still unminted (path ""), and setting the path after `register` is the documented
    workflow, not a change this test is about."""
    cfg = load_config(str(ROOT / "configs" / f"{name}.yaml"))
    cfg.confirm.path = ""
    assert _fingerprints(cfg) == PRE_CHANGE_FINGERPRINTS[name]


def test_opting_in_moves_only_the_stages_that_read_the_field():
    base = load_config(str(ROOT / "configs" / "known_answer.yaml"))
    ref = _fingerprints(base)
    powered = load_config(str(ROOT / "configs" / "known_answer.yaml"))
    powered.sae.ablation_powered = True
    powered.sae.causal_verdict = "powered"
    moved = {k for k, v in _fingerprints(powered).items() if v != ref[k]}
    assert "concepts" in moved and "sae" not in moved and "extract" not in moved
    chunked = load_config(str(ROOT / "configs" / "known_answer.yaml"))
    chunked.sae.forecast_preservation_chunked = True
    moved = {k for k, v in _fingerprints(chunked).items() if v != ref[k]}
    assert "sae" in moved and "extract" not in moved


# --------------------------------------------------------------------------- chunked forecast preservation

class _Sae:
    def __call__(self, x):
        return x * 0.5, x


class _Data:
    n = N
    meta = pd.DataFrame({"family": ["a", "b"] * (N // 2)})

    def contexts(self):
        c = np.ones((N, 8), dtype=np.float32) * (1.0 + np.arange(N)[:, None])
        c[:, 0] = np.arange(N)
        return c

    def targets(self):
        return np.ones((N, H), dtype=np.float32) * 3.0


class _Adapter:
    name = "stub"

    def __init__(self, batch_size):
        self.cfg = SimpleNamespace(batch_size=batch_size)
        self.module = object()
        self.holder = {}
        self.batches = []

    def ensure_loaded(self):
        pass

    def token_slice(self, live_len):
        return slice(0, live_len)

    def predict(self, contexts, horizon, quantiles):
        n = len(contexts)
        if n > self.cfg.batch_size:
            raise AssertionError(f"a forward of {n} series exceeds batch_size {self.cfg.batch_size}")
        self.batches.append(n)
        base = np.asarray(contexts)[:, 1:2] * np.ones((1, horizon))
        repl = self.holder.get("repl")
        if repl is None:
            return {"point": base.astype(np.float32)}
        shift = np.asarray(repl).reshape(n, -1).sum(axis=1)[:, None]
        return {"point": (base + shift).astype(np.float32)}


@pytest.fixture
def wired(monkeypatch):
    holder = {}

    def _capture(adapter, contexts, layers, **_kw):
        n = len(contexts)
        tok = np.zeros((n, T, D_IN), dtype=np.float32)
        tok[:, :, 0] = np.asarray(contexts)[:, 1:2]
        return {layers[0]: torch.as_tensor(tok)}

    @contextmanager
    def _patch(module, layer, slicer, replacement):
        holder["repl"] = replacement
        try:
            yield
        finally:
            holder["repl"] = None

    monkeypatch.setattr(E, "capture_raw_tokens", _capture)
    monkeypatch.setattr(E, "token_patch", _patch)
    return holder


def _cfg(request, chunked):
    return SimpleNamespace(
        run=SimpleNamespace(seed=0), data=SimpleNamespace(horizon=H),
        l0=SimpleNamespace(quantiles=[0.5]),
        sae=SimpleNamespace(forecast_preservation_max_series=request,
                            forecast_preservation_chunked=chunked))


def _run(wired, batch_size, request, chunked):
    adapter = _Adapter(batch_size)
    orig = adapter.predict

    def predict(contexts, horizon, quantiles):
        wired_repl = wired.get("repl")
        adapter.holder["repl"] = wired_repl
        return orig(contexts, horizon, quantiles)

    adapter.predict = predict
    out = E.forecast_preservation(_cfg(request, chunked), adapter, "blocks.0", _Sae(), None,
                                  _Data(), "cpu", granularity="token")
    return out, adapter


def test_unchunked_request_above_batch_size_is_clamped_and_says_so(wired):
    out, adapter = _run(wired, batch_size=4, request=10, chunked=False)
    assert out["n_series"] == 4 and out["n_realized"] == 4 and out["n_requested"] == 10
    assert out["limited_by"] == ["batch_size"]
    assert max(adapter.batches) == 4


def test_chunked_request_above_batch_size_scores_every_series_in_batch_sized_forwards(wired):
    out, adapter = _run(wired, batch_size=4, request=10, chunked=True)
    assert out["n_series"] == 10 and out["n_realized"] == 10 and out["limited_by"] is None
    assert sorted(adapter.batches) == [2, 2, 4, 4, 4, 4]
    assert max(adapter.batches) <= 4


def test_chunked_score_equals_the_single_batch_score_on_the_same_rows(wired):
    chunk, _ = _run(wired, batch_size=4, request=10, chunked=True)
    whole, adapter = _run(wired, batch_size=64, request=10, chunked=True)
    assert whole["rows"] == chunk["rows"]
    assert adapter.batches == [10, 10]
    for key in ("mase_clean", "mase_reconstructed", "mase_delta"):
        assert chunk[key] == pytest.approx(whole[key], abs=1e-6)
    assert chunk["mase_delta"] != 0.0


def test_default_path_is_unchanged_when_the_request_fits_one_batch(wired):
    off, _ = _run(wired, batch_size=64, request=10, chunked=False)
    on, _ = _run(wired, batch_size=64, request=10, chunked=True)
    assert off == on
