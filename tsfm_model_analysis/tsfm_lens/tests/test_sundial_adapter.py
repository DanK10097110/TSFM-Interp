"""Mechanism tests for `models/sundial_adapter.py` (ROADMAP.md §9, Phase 4).

Matches this repo's established precedent that real-checkpoint adapters
(chronos/chronos_bolt/timesfm/chronos2) carry no pytest coverage requiring
live weights -- see `test_adapter_conformance.py`'s own docstring and
`--check-alignment`'s role as the real-weights validation instead. These
tests instead pin down the adapter's pure, checkpoint-independent
mechanisms directly: the patch/front-pad geometry formula (mirrored from
`SundialPatchEmbedding.forward`'s own left-padding math, verified against
the loaded module's source in ROADMAP.md §9's Findings, not just assumed),
the sample-to-quantile prediction reduction (identical in shape to
`ChronosAdapter.predict`'s handling of its own sampled decoder), and
`mlp_info`'s existence-checked (not blindly trusted) module-name lookup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.models.sundial_adapter import SundialAdapter


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

    assert out["point"].shape == (batch, horizon)
    assert out["quantiles"].shape == (batch, horizon, 3)
    np.testing.assert_allclose(out["point"], samples.median(dim=1).values.numpy())
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


if __name__ == "__main__":
    test_geometry_matches_patch_embedding_padding_formula_on_a_multiple()
    test_geometry_front_pads_when_context_len_is_not_a_multiple_of_patch()
    test_predict_reduces_samples_to_point_median_and_quantiles()
    test_mlp_info_finds_ffn_layer_when_present_and_degrades_to_none_when_absent()
    print("sundial adapter mechanism tests passed")
