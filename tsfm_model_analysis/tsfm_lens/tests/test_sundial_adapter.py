"""Mechanism tests for `models/sundial_adapter.py` (ROADMAP.md §9, Phase 4;
spec S1 / ROADMAP §32.7's revin fix).

Matches this repo's established precedent that real-checkpoint adapters
(chronos/chronos_bolt/timesfm/chronos2) carry no pytest coverage requiring
live weights -- see `test_adapter_conformance.py`'s own docstring and
`--check-alignment`'s role as the real-weights validation instead. These
tests instead pin down the adapter's pure, checkpoint-independent
mechanisms directly: the patch/front-pad geometry formula (mirrored from
`SundialPatchEmbedding.forward`'s own left-padding math, verified against
the loaded module's source in ROADMAP.md §9's Findings, not just assumed),
the sample-to-quantile prediction reduction (identical in shape to
`ChronosAdapter.predict`'s handling of its own sampled decoder),
`mlp_info`'s existence-checked (not blindly trusted) module-name lookup, and
(spec S1) the manual per-series `revin` normalization: `predict()`'s
scale-equivariance against a nonlinear mock inner model, `forward()`
(capture) seeing the identical normalized tensor `predict()` used, and the
checkpoint's own `sd -> 1` floor rule on a near-constant context. One
skippable real-checkpoint test runs when `thuml/sundial-base-128m` is
already in the local HF cache and CUDA is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.models.sundial_adapter import SundialAdapter

_HAS_CUDA = torch.cuda.is_available()
try:
    from huggingface_hub import try_to_load_from_cache
    _SUNDIAL_CACHED = try_to_load_from_cache(
        "thuml/sundial-base-128m", "config.json") is not None
except Exception:
    _SUNDIAL_CACHED = False


def _bare_adapter(context_len: int = 512, horizon: int = 64, patch: int = 16) -> SundialAdapter:
    """Construct a SundialAdapter without calling load() (no network/weights);
    set only the attributes load() would have set from the checkpoint config,
    mirroring how other pure-logic tests in this repo probe adapter geometry
    without a real checkpoint."""
    mcfg = ModelConfig(name="sundial_test", adapter="sundial")
    data_cfg = DataConfig(context_len=context_len, horizon=horizon)
    adapter = SundialAdapter(mcfg, data_cfg, torch.device("cpu"), torch.float32)
    adapter._patch = patch
    adapter._max_horizon = 720
    return adapter


def test_geometry_matches_patch_embedding_padding_formula_on_a_multiple():
    """context_len a multiple of patch: no front padding, one span per patch,
    contiguous and non-overlapping."""
    adapter = _bare_adapter(context_len=512, patch=16)
    n_tokens, front_pad = adapter._geometry()
    assert n_tokens == 32
    assert front_pad == 0
    spans = adapter.token_time_spans()
    assert spans.shape == (32, 2)
    assert np.all(spans[:, 1] - spans[:, 0] == 16)
    assert spans[0, 0] == 0 and spans[-1, 1] == 512
    assert np.all(np.diff(spans[:, 0]) == 16)


def test_geometry_front_pads_when_context_len_is_not_a_multiple_of_patch():
    """A context_len that is not a multiple of the patch length needs left
    padding -- mirrors `SundialPatchEmbedding.forward`'s own
    `(patch - (length % patch)) % patch` formula exactly, and the leading
    span must clip to 0 rather than go negative."""
    adapter = _bare_adapter(context_len=500, patch=16)  # 500 % 16 == 4
    n_tokens, front_pad = adapter._geometry()
    assert front_pad == (16 - (500 % 16)) % 16 == 12
    assert n_tokens == (500 + 12) // 16 == 32
    spans = adapter.token_time_spans()
    assert spans.shape == (n_tokens, 2)
    assert spans[0, 0] == 0.0  # clipped, not negative
    assert spans[-1, 1] == 500.0
    assert np.all(spans[:, 1] > spans[:, 0])
    assert np.all(np.diff(spans[:, 0]) >= 0)


def test_predict_reduces_samples_to_point_median_and_quantiles():
    """Exercises the sample -> {point, quantiles} contract in isolation, with
    a stubbed `_inner` standing in for the real flow-matching head -- the
    same reduction shape as `ChronosAdapter.predict`'s handling of its own
    sampled decoder (median for point, `torch.quantile` over the sample
    axis for quantiles)."""
    adapter = _bare_adapter(context_len=64, horizon=4, patch=16)
    # odd num_samples so torch.Tensor.median() and torch.quantile(0.5) pick
    # the same middle element exactly -- with an even count they legitimately
    # differ (median takes the lower of the two middle values, quantile
    # interpolates/averages them), which is a property of that pair of torch
    # ops, not a bug in this reduction; odd keeps the test's equality check
    # meaningful rather than tolerance-fudged.
    batch, num_samples, horizon = 3, 21, 4
    rng = np.random.default_rng(0)
    samples = torch.from_numpy(
        rng.normal(loc=10.0, scale=1.0, size=(batch, num_samples, horizon)).astype(np.float32))

    class _FakeOutput:
        def __init__(self, logits):
            self.logits = logits

    class _FakeInner:
        def __call__(self, **kwargs):
            assert kwargs["use_cache"] is False
            assert kwargs["max_output_length"] == horizon
            return _FakeOutput(samples)

    adapter._inner = _FakeInner()
    contexts = rng.normal(size=(batch, 64)).astype(np.float32)
    out = adapter.predict(contexts, horizon, [0.1, 0.5, 0.9])

    # `predict()` now denormalizes the (fake) inner model's sampled output by
    # the SAME per-row mu/sd `prepare()` computed for this `contexts` batch
    # (spec S1) -- the raw `samples` fixture above is in NORMALIZED units, so
    # the expected point/quantiles must be transformed back the same way
    # before comparison, not compared to `samples` directly.
    mu = contexts.mean(axis=1, keepdims=True).astype(np.float64)
    sd = contexts.std(axis=1, keepdims=True).astype(np.float64)
    sd = np.where(sd > 1e-2, sd, 1.0)
    expected_point = samples.median(dim=1).values.numpy() * sd + mu

    assert out["point"].shape == (batch, horizon)
    assert out["quantiles"].shape == (batch, horizon, 3)
    np.testing.assert_allclose(out["point"], expected_point, rtol=1e-5)
    # the 0.5 quantile column should equal the point (median) exactly
    np.testing.assert_allclose(out["quantiles"][:, :, 1], out["point"])
    # quantiles must be non-decreasing across the requested levels
    assert np.all(out["quantiles"][:, :, 0] <= out["quantiles"][:, :, 1])
    assert np.all(out["quantiles"][:, :, 1] <= out["quantiles"][:, :, 2])


def test_mlp_info_finds_ffn_layer_when_present_and_degrades_to_none_when_absent():
    """`mlp_info` builds `{block}.ffn_layer` directly (the shared `_scan_mlp`
    helper's fixed name set does not include Sundial's `ffn_layer` leaf --
    a naming-convention gap, not a bug) and checks it exists rather than
    assuming so -- verified both ways against a tiny synthetic module tree,
    matching invariant 8's "unsupported capability degrades with a log,
    never raises" for the absent case."""
    class _Block(nn.Module):
        def __init__(self, with_ffn: bool):
            super().__init__()
            if with_ffn:
                self.ffn_layer = nn.Linear(4, 4)

    class _Root(nn.Module):
        def __init__(self, with_ffn: bool):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([_Block(with_ffn) for _ in range(2)])

    adapter = _bare_adapter(context_len=64, horizon=4, patch=16)
    adapter._inner = _Root(with_ffn=True)
    adapter._loaded = True  # skip ensure_loaded()'s real load() -- no network
    adapter.cfg.layer_regex = r"model\.layers\.\d+$"
    mapping = adapter.mlp_info()
    assert mapping == {"model.layers.0": "model.layers.0.ffn_layer",
                       "model.layers.1": "model.layers.1.ffn_layer"}

    adapter_no_ffn = _bare_adapter(context_len=64, horizon=4, patch=16)
    adapter_no_ffn._inner = _Root(with_ffn=False)
    adapter_no_ffn._loaded = True
    adapter_no_ffn.cfg.layer_regex = r"model\.layers\.\d+$"
    assert adapter_no_ffn.mlp_info() is None


class _CaptureModel:
    """Stands in for `SundialModel` (the `.model` backbone `forward()`
    calls): records the exact tensor it was fed so the test can compare it,
    bit for bit, against what `predict()` fed its own `_inner` call for the
    same input."""

    def __init__(self):
        self.seen = None

    def __call__(self, input_ids, use_cache):
        self.seen = input_ids.detach().clone()
        return None


class _NonlinearFakeInner:
    """A deterministic, NONLINEAR-in-its-input stand-in for Sundial's
    flow-matching head, so that feeding it a scale-dependent (i.e.
    unnormalized) tensor at different scale factors provably produces
    non-proportional output -- the exact failure mode a missing/incorrect
    `revin` normalization would cause, and the plant below exercises.
    """

    def __init__(self, horizon: int):
        self.horizon = horizon
        self.model = _CaptureModel()
        self.seen_predict_input = None

    def __call__(self, input_ids, max_output_length, num_samples, use_cache, return_dict):
        assert use_cache is False
        assert max_output_length == self.horizon
        self.seen_predict_input = input_ids.detach().clone()
        base = torch.tanh(input_ids.mean(dim=1)) + 0.3 * input_ids[:, 0] ** 2  # [B], nonlinear
        logits = base[:, None, None].expand(
            input_ids.shape[0], num_samples, self.horizon).clone()

        class _Out:
            pass
        out = _Out()
        out.logits = logits
        return out


def test_predict_is_scale_equivariant_through_manual_revin_normalization():
    """Spec S1's core claim: `predict(k * contexts)` equals `k * predict(contexts)`
    for k spanning six orders of magnitude, because `prepare()`'s per-series
    `(x - mu) / sd` normalization is scale-invariant (feeding the NONLINEAR
    fake inner model above the identical normalized tensor regardless of k),
    and `predict()` denormalizes the sampled output by the same k-scaled
    `mu`/`sd` afterward. `loc=5.0, scale=20.0` keeps `sd * k` above the
    checkpoint's own `1e-2` floor even at `k=1e-3` (`sd approx 20 * 1e-3 =
    0.02 > 0.01`), so the floor does not asymmetrically kick in at only one
    tested scale and break the equivariance the checkpoint's own rule is
    supposed to have at ordinary amplitudes.
    """
    horizon = 4
    adapter = _bare_adapter(context_len=32, horizon=horizon, patch=16)
    inner = _NonlinearFakeInner(horizon)
    adapter._inner = inner

    rng = np.random.default_rng(1)
    base_contexts = rng.normal(loc=5.0, scale=20.0, size=(3, 32)).astype(np.float32)

    points = {}
    normalized_inputs = {}
    for k in (1e-3, 1.0, 1e3):
        out = adapter.predict(base_contexts * k, horizon, [0.5])
        points[k] = out["point"]
        normalized_inputs[k] = inner.seen_predict_input.clone()
        # forward() (capture) must see the IDENTICAL normalized tensor
        # predict() just fed the inner model for this same scaled input.
        adapter.forward(adapter.prepare(base_contexts * k))
        np.testing.assert_allclose(inner.model.seen.numpy(),
                                   normalized_inputs[k].numpy(), atol=1e-6)

    # The manual normalization is exactly scale-invariant: (kx - k*mu) / (k*sd)
    # == (x - mu) / sd, so the (nonlinear) inner model sees the same input at
    # every scale, independent of k.
    np.testing.assert_allclose(normalized_inputs[1e-3].numpy(),
                               normalized_inputs[1.0].numpy(), atol=1e-5)
    np.testing.assert_allclose(normalized_inputs[1e3].numpy(),
                               normalized_inputs[1.0].numpy(), atol=1e-5)

    # predict(k * x) == k * predict(x) for every tested k.
    for k in (1e-3, 1e3):
        np.testing.assert_allclose(points[k], points[1.0] * k, rtol=1e-4, atol=1e-6)


def test_prepare_floors_sd_to_one_on_a_near_constant_context_with_no_nan_or_inf():
    """Decoy: a context whose std is below the checkpoint's own `1e-2` floor
    (constructed as visually near-constant, not exactly constant, so the
    test cannot pass merely by special-casing an exact-zero std) must
    normalize to `(x - mu) / 1`, never divide by a near-zero std, and must
    stay fully finite."""
    adapter = _bare_adapter(context_len=16, horizon=4, patch=16)
    contexts = np.full((2, 16), 7.0, dtype=np.float32)
    contexts[1, -1] += 1e-6  # near-, not exactly, constant -- the decoy
    prepared = adapter.prepare(contexts)

    assert torch.all(torch.isfinite(prepared.normalized))
    assert torch.all(torch.isfinite(prepared.sd))
    np.testing.assert_allclose(prepared.sd.numpy().ravel(), [1.0, 1.0], atol=1e-6)
    expected = contexts - contexts.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(prepared.normalized.numpy(), expected, atol=1e-4)


@pytest.mark.skipif(not _HAS_CUDA, reason="Sundial's default config targets CUDA")
@pytest.mark.skipif(not _SUNDIAL_CACHED,
                    reason="thuml/sundial-base-128m not in the local HF cache")
def test_sundial_real_checkpoint_scale_equivariance():
    """The real-checkpoint counterpart of the mock equivariance test above:
    loads actual `thuml/sundial-base-128m` weights and checks that
    `predict()` is close to scale-equivariant at 0.001x/1000x -- the
    frontend-stage probe that originally caught this adapter defect
    (ROADMAP.md §32.7, FINDINGS PM-10), run directly rather than through the
    full pipeline. Skipped, not failed, when the checkpoint is not already
    cached or no CUDA device is present, matching this repo's established
    real-weights-test convention (`test_lag_llama_adapter.py`).
    """
    from tsfm_lens.models import build_adapter

    mcfg = ModelConfig(name="Sundial", adapter="sundial",
                       checkpoint="thuml/sundial-base-128m",
                       batch_size=8, capture_layer_stride=1)
    dcfg = DataConfig(source="smoke", context_len=64, horizon=8)
    adapter = build_adapter(mcfg, dcfg, torch.device("cuda"), torch.bfloat16)
    adapter.ensure_loaded()

    rng = np.random.default_rng(2)
    contexts = rng.normal(loc=50.0, scale=15.0, size=(4, 64)).astype(np.float32)
    torch.manual_seed(0)
    base = adapter.predict(contexts, 8, [0.5])["point"]
    residuals = []
    for k in (1e-3, 1e3):
        torch.manual_seed(0)
        scaled = adapter.predict(contexts * k, 8, [0.5])["point"] / k
        scale = np.abs(np.diff(contexts, axis=1)).mean(axis=1)
        residuals.append(np.abs(scaled - base).mean(axis=1) / scale)
    worst = float(np.max(np.concatenate(residuals)))
    # Pre-fix this was measured at 38.4318 context-scale units (FINDINGS
    # PM-10); comparable models in `runs/full_report_run_4model` sit at
    # <= 0.0086. A generous 1.0-unit ceiling catches a regression back
    # toward the old, unnormalized behavior without being sensitive to the
    # sampled flow-matching head's own draw-to-draw noise.
    assert worst < 1.0, f"worst scale-equivariance residual {worst} -- revin regression?"


if __name__ == "__main__":
    test_geometry_matches_patch_embedding_padding_formula_on_a_multiple()
    test_geometry_front_pads_when_context_len_is_not_a_multiple_of_patch()
    test_predict_reduces_samples_to_point_median_and_quantiles()
    test_mlp_info_finds_ffn_layer_when_present_and_degrades_to_none_when_absent()
    test_predict_is_scale_equivariant_through_manual_revin_normalization()
    test_prepare_floors_sd_to_one_on_a_near_constant_context_with_no_nan_or_inf()
    print("sundial adapter mechanism tests passed")
