"""Tests for `ROADMAP.md` sec 16 E3(b)'s zero-code `GenericHFAdapter`.

Every seam this adapter has is a *probe* -- it guesses which forward kwarg
carries the series, which modules are the block stack, where the forecast
comes out, and whether the tokens are time-localized at all. So the thing
worth testing is not that a probe succeeds on a happy path but that each one
**refuses correctly**: a wrong guess here does not raise later, it produces a
plausible number on a fiction, which is the failure mode `CLAUDE.md` sec 2.5
exists to prevent.

The nets below are local to this file and deliberately synthetic. Two reasons,
both load-bearing: a real checkpoint would need a network round trip inside a
unit test, and -- more importantly -- a synthetic net's *correct answer is
known by construction*, so "the probe resolved something" and "the probe
resolved the right thing" are distinguishable assertions. `_instantiate` is
the single production seam these override; every strategy under test runs
unmodified.
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
from tsfm_lens.models.generic_hf_adapter import GenericHFAdapter, _skeleton

_CONTEXT = 64
_PATCH = 8
_DIM = 12


class _Cfg:
    def __init__(self, hidden_size=_DIM):
        self.hidden_size = hidden_size


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return x + torch.tanh(self.lin(x))


class _PatchNet(nn.Module):
    """A patch-tokenized net whose token *i* provably reads `[8i, 8i+8)` only.

    Non-overlapping patches through an elementwise-only block stack means
    there is no path by which one token's activation can depend on another
    token's timesteps -- so the span map discovery must recover is exact, not
    approximately right.
    """

    def __init__(self, n_blocks=4, extra=True):
        super().__init__()
        self.config = _Cfg()
        self.embed = nn.Linear(_PATCH, _DIM)
        self.layers = nn.ModuleList([_Block(_DIM) for _ in range(n_blocks)])
        # A decoy stack whose name ends with the real stack's name. Anchoring
        # is what keeps `^layers\.\d+$` from swallowing it (`CLAUDE.md` 11.30).
        self.extra_layers = nn.ModuleList([_Block(_DIM) for _ in range(2)]) if extra else None
        self.head = nn.Linear(_DIM, 4)

    def forward(self, past_values, **kw):
        b = past_values.shape[0]
        x = self.embed(past_values.reshape(b, -1, _PATCH))
        for blk in self.layers:
            x = blk(x)
        if self.extra_layers is not None:
            y = x
            for blk in self.extra_layers:
                y = blk(y)
        return {"last_hidden_state": x, "prediction_outputs": self.head(x[:, -1])}


class _CovariateNet(_PatchNet):
    def forward(self, past_values, past_time_features, **kw):
        return super().forward(past_values)


class _NoKnownInputNet(_PatchNet):
    def forward(self, pixel_values, **kw):
        return super().forward(pixel_values)


class _ShortHeadNet(_PatchNet):
    def forward(self, past_values, **kw):
        out = super().forward(past_values)
        return {"prediction_outputs": out["prediction_outputs"][:, :2]}


class _RevinNet(_PatchNet):
    """A `forward()` that declares `revin: bool = False` -- mirroring the
    thuml lineage's own signature (`Sundial`, `Timer`; spec S1 / ROADMAP
    §32.7 item 2) that `_resolve_revin` exists to detect and correct for.
    Scales its forecast output by 1000x whenever `revin` is left at its
    (checkpoint) default so the two cases are trivially distinguishable in a
    test, without needing real normalization math to tell them apart."""

    def forward(self, past_values, revin=False, **kw):
        out = super().forward(past_values)
        out["prediction_outputs"] = out["prediction_outputs"] * (1.0 if revin else 1000.0)
        return out


class _Gen(dict):
    def __init__(self, sequences):
        super().__init__(sequences=sequences)
        self.sequences = sequences


class _GenerateNet(_PatchNet):
    """A sampled forecast head, so real (non-degenerate) quantiles are available."""

    def generate(self, past_values, **kw):
        torch.manual_seed(0)
        b = past_values.shape[0]
        base = past_values[:, -1:].expand(b, 4)
        return _Gen(base[:, None, :] + torch.randn(b, 7, 4))


class _DiffuseNet(nn.Module):
    """Every token reads every timestep: there is no correct span map to find."""

    def __init__(self, n_tokens=4):
        super().__init__()
        torch.manual_seed(3)
        self.config = _Cfg()
        self.n_tokens = n_tokens
        self.mix = nn.Linear(_CONTEXT, n_tokens * _DIM)
        self.layers = nn.ModuleList([_Block(_DIM) for _ in range(3)])

    def forward(self, past_values, **kw):
        x = self.mix(past_values).reshape(past_values.shape[0], self.n_tokens, _DIM)
        for blk in self.layers:
            x = blk(x)
        return {"last_hidden_state": x}


def _adapter(net_cls, name="gen", **kw):
    class _Probe(GenericHFAdapter):
        def _instantiate(self):
            return net_cls(**kw)

    a = _Probe(ModelConfig(name=name, adapter="generic_hf", checkpoint="local/test",
                           batch_size=16),
               DataConfig(context_len=_CONTEXT, horizon=4),
               torch.device("cpu"), torch.float32)
    return a


def test_input_kwarg_is_resolved_from_the_forward_signature():
    a = _adapter(_PatchNet)
    a.ensure_loaded()
    assert a.describe_strategies()["input_kwarg"] == "past_values"


def test_a_required_covariate_is_refused_rather_than_fabricated():
    """The single most dangerous silent failure this adapter could have.

    Zeros for `past_time_features` runs fine and returns a forecast for a
    series with no calendar structure -- a wrong number that looks like a
    right one. It must be an error at load, naming the parameter.
    """
    a = _adapter(_CovariateNet)
    with pytest.raises(ValueError, match="past_time_features"):
        a.ensure_loaded()


def test_an_unrecognized_input_signature_names_what_it_looked_for():
    a = _adapter(_NoKnownInputNet)
    with pytest.raises(ValueError) as exc:
        a.ensure_loaded()
    assert "past_values" in str(exc.value) and "pixel_values" in str(exc.value)


def test_layer_regex_discovery_finds_the_block_stack_and_is_anchored():
    """`^layers\\.\\d+$`, not `layers\\.\\d+$` -- the decoy stack is named
    `extra_layers.N`, and an unanchored regex matches it too (sec 11.30)."""
    a = _adapter(_PatchNet)
    a.ensure_loaded()
    regex = a.describe_strategies()["layer_regex"]
    assert regex.startswith("^") and regex.endswith("$")
    names = a.all_layer_names()
    assert names == ["layers.0", "layers.1", "layers.2", "layers.3"]
    assert not any(n.startswith("extra_") for n in names)


def test_the_block_wins_over_its_own_residual_width_component():
    """The bug this probe actually had on first run, and the expensive kind.

    `_Block.lin` is a Linear at residual width inside every block, so
    `layers.#.lin` has the same member count and the same `[B, T, D]` shape as
    `layers.#` -- shape probing alone cannot tell them apart, and the first
    version picked the component. Capturing a block's internal projection
    while calling it the residual stream produces a full, plausible set of
    CKA/stitching/patching numbers on the wrong tensor.
    """
    a = _adapter(_PatchNet)
    a.ensure_loaded()
    assert a.all_layer_names() == ["layers.0", "layers.1", "layers.2", "layers.3"]
    assert not any(n.endswith(".lin") for n in a.all_layer_names())


def test_the_larger_uniform_group_wins_over_a_smaller_one():
    a = _adapter(_PatchNet, n_blocks=6)
    a.ensure_loaded()
    assert len(a.all_layer_names()) == 6


def test_no_repeated_block_stack_is_an_error_not_an_empty_capture():
    a = _adapter(_PatchNet, n_blocks=1, extra=False)
    with pytest.raises(ValueError, match="no repeated module group"):
        a.ensure_loaded()


def test_discovered_spans_are_exactly_the_constructed_patch_boundaries():
    """Construction-exact, not close: token i reads [8i, 8i+8) and nothing else."""
    a = _adapter(_PatchNet)
    a.ensure_loaded()
    spans = a.token_time_spans()
    expected = np.stack([np.arange(0, _CONTEXT, _PATCH),
                         np.arange(_PATCH, _CONTEXT + _PATCH, _PATCH)], axis=1)
    assert np.array_equal(spans, expected.astype(np.float64))


def test_spans_are_measured_once_and_cached():
    a = _adapter(_PatchNet)
    a.ensure_loaded()
    first = a.token_time_spans()
    assert a.token_time_spans() is first


def test_time_localization_reports_nothing_before_it_has_measured():
    """A refusal signal that reports before it has measured anything would be a
    guess wearing a measurement's clothes -- so it is None on an unloaded
    adapter, and populated once loading has run discovery."""
    a = _adapter(_PatchNet)
    assert a.time_localization() is None
    a.ensure_loaded()
    assert a.time_localization()["localized"] is True


def test_spans_are_warmed_at_load_so_discovery_never_runs_inside_a_hook():
    """The recursion this eager warm-up exists to prevent, pinned.

    `token_slice` -- and so every `token_patch` hook -- asks for
    `token_time_spans`, and lazy discovery answers by running its own forward
    pass, which re-enters the hook that asked. The lens stage hits this for
    real whenever `keep_models_loaded: false` has released and reloaded the
    model. Warming at load is what makes the spans already present by the time
    any hook can ask, so this asserts the cache is populated by
    `ensure_loaded` alone, and that `token_slice` needs no forward pass.
    """
    a = _adapter(_PatchNet)
    a.ensure_loaded()
    assert a._spans is not None
    calls = {"n": 0}
    original = a.forward
    a.forward = lambda prepared: (calls.__setitem__("n", calls["n"] + 1),
                                  original(prepared))[1]
    assert a.token_slice(a._spans.shape[0]) == slice(0, a._spans.shape[0])
    assert calls["n"] == 0


def test_a_refusal_at_warm_up_still_leaves_the_model_loadable():
    """Warming must not turn E3(c)'s refusal into a load failure: a
    non-time-localized model is explicitly still valid for behavioral (L0)
    comparison, and only pooled analyses are off limits."""
    a = _adapter(_DiffuseNet)
    a.ensure_loaded()
    assert a._loaded is True
    assert a.time_localization()["localized"] is False
    with pytest.raises(ValueError, match="not time-localized"):
        a.token_time_spans()


def test_a_non_localized_model_refuses_spans_and_says_why():
    a = _adapter(_DiffuseNet)
    a.ensure_loaded()
    with pytest.raises(ValueError) as exc:
        a.token_time_spans()
    msg = str(exc.value)
    assert "not time-localized" in msg and "contrast" in msg and "L0" in msg
    assert a.time_localization()["localized"] is False


def test_postprocess_passes_through_during_discovery_and_asserts_after():
    """Discovery's own captures arrive before any span exists, so the base
    class's token-count assertion cannot apply to them -- but it must come back
    the moment it can, or a real misalignment would pass silently."""
    a = _adapter(_PatchNet)
    a.ensure_loaded()
    hidden = torch.zeros(2, 99, _DIM)
    with pytest.raises(ValueError):
        a.postprocess_tokens("layers.0", hidden)
    a._spans = None                       # the state discovery itself runs in
    assert a.postprocess_tokens("layers.0", hidden).shape[1] == 99


def test_a_point_only_head_gives_degenerate_quantiles_and_records_that():
    """Silence here would make L0's calibration section report a perfect
    reliability curve for a model that never produced a quantile."""
    a = _adapter(_PatchNet)
    a.ensure_loaded()
    out = a.predict(np.zeros((3, _CONTEXT), dtype=np.float32), 4, [0.1, 0.5, 0.9])
    assert out["point"].shape == (3, 4)
    assert out["quantiles"].shape == (3, 4, 3)
    assert np.array_equal(out["quantiles"][:, :, 0], out["quantiles"][:, :, 2])
    assert a.describe_strategies()["quantiles"] == "degenerate_point"


def test_revin_is_enabled_when_forward_declares_it_off_by_default():
    """Spec S1 item 2, generalized from `sundial_adapter.py`'s hand-written
    fix: a checkpoint whose OWN `forward()` signature declares `revin`
    defaulting to `False` (the thuml lineage's convention) gets `revin=True`
    passed explicitly on every `prepare()`-built call -- matching what that
    checkpoint's own `.generate()` already defaults to, read off the
    signature rather than invented."""
    a = _adapter(_RevinNet)
    a.ensure_loaded()
    assert a.describe_strategies()["revin_enabled"] is True
    prepared = a.prepare(np.ones((2, _CONTEXT), dtype=np.float32))
    assert prepared["revin"] is True
    # end to end: forward-field predict() must read the NORMALIZED-path
    # branch (1x), not the raw-scale default branch (1000x).
    out = a.predict(np.ones((2, _CONTEXT), dtype=np.float32), 4, [0.5])
    assert np.max(np.abs(out["point"])) < 100.0  # would be >= 1000x otherwise


def test_revin_stays_off_when_forward_has_no_such_parameter():
    """A checkpoint with no `revin` parameter at all (every existing
    generic_hf test fixture except `_RevinNet` above) must be completely
    unaffected -- no `revin` key is added to `prepare()`'s output."""
    a = _adapter(_PatchNet)
    a.ensure_loaded()
    assert a.describe_strategies()["revin_enabled"] is False
    prepared = a.prepare(np.ones((2, _CONTEXT), dtype=np.float32))
    assert "revin" not in prepared


def test_a_sampled_head_is_preferred_and_gives_real_quantiles():
    a = _adapter(_GenerateNet)
    a.ensure_loaded()
    out = a.predict(np.zeros((3, _CONTEXT), dtype=np.float32), 4, [0.1, 0.5, 0.9])
    assert a.describe_strategies()["forecast"] == "generate"
    assert a.describe_strategies()["quantiles"] == "sampled"
    assert (out["quantiles"][:, :, 0] < out["quantiles"][:, :, 2]).all()


def test_a_head_shorter_than_the_horizon_refuses_rather_than_pads():
    a = _adapter(_ShortHeadNet)
    a.ensure_loaded()
    with pytest.raises(ValueError, match="steps but the run asks for"):
        a.predict(np.zeros((2, _CONTEXT), dtype=np.float32), 4, [0.5])


def test_skeleton_only_collapses_whole_numeric_segments():
    """`layer2.weight` is one module, not a member of a `layer#` family --
    collapsing every digit anywhere would merge unrelated stacks."""
    assert _skeleton("model.layers.11.attn") == "model.layers.#.attn"
    assert _skeleton("model.layer2.attn") == "model.layer2.attn"
