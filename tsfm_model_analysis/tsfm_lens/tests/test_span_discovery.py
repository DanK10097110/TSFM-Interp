"""Tests for `ROADMAP.md` sec 16 E3's empirical token->time span discovery.

The mocks are the right test surface here for a reason that does not hold for
most of this repo's modules: their true spans are known **exactly by
construction** (`MockPatchAdapter` tokenizes in fixed 32-step patches, so
token *i* covers `[32i, 32i+32)` and nothing else), so "did the measurement
recover the right answer" is a real assertion rather than a self-consistency
check. Every other adapter's declared spans are themselves an assertion; the
real-checkpoint acceptance criterion (five hand-written adapters at mean
IoU >= 0.9) is a separate, live-weights run recorded in sec 16 E3's Findings,
not something a unit test can stand in for.

The refusal path gets its own deliberately non-localized net rather than a
production mock, so `CLAUDE.md` sec 2.1 holds: no already-recorded mock number
changes because a test needed a new shape.
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
from tsfm_lens.extraction.span_discovery import (SpanDiscovery, compare_declared,
                                                  discover_spans)
from tsfm_lens.models.mock import (MockPatchAdapter, MockStepAdapter,
                                    MockWaveAdapter)

_CONTEXT = 128


def _adapter(cls, name: str):
    a = cls(ModelConfig(name=name, adapter="mock_patch", batch_size=64),
            DataConfig(context_len=_CONTEXT, horizon=16),
            torch.device("cpu"), torch.float32)
    a.ensure_loaded()
    return a


class _DiffuseNet(nn.Module):
    """A net with no time-localized tokens at all: every token reads every timestep.

    The point of the linear whole-sequence mixer is that it is not a
    *degraded* version of a localized model -- it has no correct span map to
    recover, which is precisely the class of architecture (spectral
    tokenizers, latent-query attention) `CLAUDE.md` sec 12 puts outside the
    pooling premise and E3 routes to L0-only.
    """

    def __init__(self, context_len: int, n_tokens: int, dim: int, seed: int = 3):
        super().__init__()
        torch.manual_seed(seed)
        self.n_tokens, self.dim = n_tokens, dim
        self.mix = nn.Linear(context_len, n_tokens * dim)
        self.blocks = nn.ModuleList([nn.Identity()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mix(x).view(x.shape[0], self.n_tokens, self.dim)
        for block in self.blocks:
            h = block(h)
        return h.mean(dim=(1, 2))


class _DiffuseAdapter(MockPatchAdapter):
    """Declares patch spans it does not actually honor -- the case worth catching."""

    def load(self) -> None:
        self._net = _DiffuseNet(self.data_cfg.context_len,
                                self.data_cfg.context_len // self.patch, self.dim)


@pytest.mark.parametrize("cls,name", [(MockPatchAdapter, "patchy"),
                                       (MockStepAdapter, "steppy"),
                                       (MockWaveAdapter, "wavy")])
def test_discovered_spans_exactly_recover_the_known_true_spans(cls, name):
    """The core claim: the measurement reproduces an answer known to be right."""
    a = _adapter(cls, name)
    found = discover_spans(a, stride=1)
    declared = a.token_time_spans()
    assert found.n_tokens == len(declared)
    assert np.array_equal(found.spans, declared), (
        f"{name}: discovered spans differ from the mock's construction-exact spans")
    assert found.flagged_tokens == []


@pytest.mark.parametrize("cls,name", [(MockPatchAdapter, "patchy"),
                                       (MockStepAdapter, "steppy"),
                                       (MockWaveAdapter, "wavy")])
def test_compare_declared_reports_a_perfect_iou_for_a_correct_adapter(cls, name):
    a = _adapter(cls, name)
    cmp = compare_declared(a, discover_spans(a, stride=1))
    assert cmp["available"] is True
    assert cmp["mean_iou"] == pytest.approx(1.0)
    assert cmp["worst_iou"] == pytest.approx(1.0)
    assert cmp["n_declared"] == cmp["n_discovered"]


def test_localized_mocks_clear_the_refusal_gate():
    for cls, name in [(MockPatchAdapter, "patchy"), (MockStepAdapter, "steppy"),
                      (MockWaveAdapter, "wavy")]:
        found = discover_spans(_adapter(cls, name), stride=1)
        assert found.is_time_localized(), f"{name} wrongly refused"


def test_the_gate_reads_contrast_because_diffuseness_scales_with_token_count():
    """The finding from E3's live acceptance run, pinned so it cannot regress.

    Chronos-T5-Small scored diffuseness 0.794 with a **bit-exact** span map,
    purely because 512 one-timestep tokens spread a fixed per-token leakage
    512 ways; a 0.5 cut on diffuseness would have refused three of five
    adapters whose maps had just been proven exact. The mocks reproduce the
    mechanism at small scale: `MockStepAdapter` (128 tokens) and
    `MockPatchAdapter` (4 tokens) are *both* exactly localized, yet their
    contrast differs by more than an order of magnitude while their
    diffuseness barely moves -- so contrast is the statistic with headroom
    and diffuseness is the one that would misfire.
    """
    patch = discover_spans(_adapter(MockPatchAdapter, "patchy"), stride=1)
    step = discover_spans(_adapter(MockStepAdapter, "steppy"), stride=1)
    assert step.n_tokens > patch.n_tokens * 8
    assert step.contrast > patch.contrast * 8, (
        "contrast should grow with token count for an equally-exact map")
    assert abs(step.diffuseness - patch.diffuseness) < 0.2, (
        "diffuseness barely separates two maps that are both exact")
    assert patch.is_time_localized() and step.is_time_localized()


def test_the_refusal_path_fires_on_a_non_time_localized_model():
    a = _adapter(_DiffuseAdapter, "diffuse")
    found = discover_spans(a, stride=1)
    assert not found.is_time_localized()
    assert found.diffuseness > 0.9
    # Every token is flagged, not merely a low aggregate score: the structural
    # check and the concentration score must agree, or one of them is measuring
    # something other than time-localization.
    assert len(found.flagged_tokens) == found.n_tokens


def test_a_wrong_declaration_is_caught_as_a_low_iou_not_a_silent_pass():
    """E3(d): the cross-check has to actually disagree when the declaration is wrong."""
    a = _adapter(_DiffuseAdapter, "diffuse")
    cmp = compare_declared(a, discover_spans(a, stride=1))
    assert cmp["available"] is True
    assert cmp["mean_iou"] < 0.9, (
        "an adapter declaring spans it does not honor passed the E3(d) cross-check")


def test_amplitude_agreement_is_perfect_when_the_map_is_amplitude_independent():
    found = discover_spans(_adapter(MockPatchAdapter, "patchy"), stride=1)
    assert found.per_amplitude_agreement == pytest.approx(1.0)
    assert len(found.per_amplitude_spans) == len(found.amplitudes)


def test_a_coarser_stride_still_recovers_spans_when_it_divides_the_token_width():
    """Stride is a cost knob, and the failure it can cause is worth pinning.

    A stride that divides the patch width leaves every token still selected by
    some probe, so the map is unchanged. This is the guarantee that makes the
    knob safe to raise on a long context.
    """
    a = _adapter(MockPatchAdapter, "patchy")
    fine = discover_spans(a, stride=1)
    coarse = discover_spans(a, stride=8)
    assert np.array_equal(coarse.spans, fine.spans)
    assert coarse.flagged_tokens == []


def test_a_stride_coarser_than_the_token_width_flags_rather_than_inventing_spans():
    """The opposite case: unprobed tokens must surface, not be quietly widened.

    `MockStepAdapter` has one timestep per token, so a stride of 4 leaves
    three of every four tokens unselected by any probe. The right behavior is
    a zero-width span and a flag -- a plausible-looking interpolated span
    would be exactly the silent fiction `CLAUDE.md` sec 2.5 forbids.
    """
    a = _adapter(MockStepAdapter, "steppy")
    found = discover_spans(a, stride=4)
    assert len(found.flagged_tokens) >= found.n_tokens * 0.7
    for tok in found.flagged_tokens:
        assert found.spans[tok][1] - found.spans[tok][0] == 0


def test_span_discovery_serializes_without_numpy_types():
    """`to_dict` feeds `save_json`; a stray np.float32 raises there, not here."""
    import json
    found = discover_spans(_adapter(MockPatchAdapter, "patchy"), stride=8)
    json.dumps(found.to_dict())


def test_is_time_localized_threshold_is_a_caller_policy_not_baked_in():
    """Pinned deliberately: a threshold frozen into the artifact would make a
    later recalibration silently rewrite what old artifacts meant."""
    found = SpanDiscovery(spans=np.zeros((2, 2)), diffuseness=0.6,
                          per_amplitude_agreement=1.0, flagged_tokens=[],
                          n_tokens=2, amplitudes=[0.25], stride=1,
                          concentration=0.4, contrast=5.0)
    assert found.is_time_localized(min_contrast=4.0)
    assert not found.is_time_localized(min_contrast=6.0)


def test_high_diffuseness_alone_does_not_refuse_a_well_localized_map():
    """The exact live-run case: exact spans, high diffuseness, must still pass."""
    found = SpanDiscovery(spans=np.zeros((512, 2)), diffuseness=0.794,
                          per_amplitude_agreement=1.0, flagged_tokens=[],
                          n_tokens=512, amplitudes=[0.25], stride=1,
                          concentration=0.207, contrast=133.55)
    assert found.is_time_localized(), (
        "a 512-token model with a bit-exact map was refused -- the E3 "
        "acceptance-run regression")


class _LagFeatureNet(nn.Module):
    """Sharply time-localized tokens that each read a *set of lags*, not an interval.

    Token *i* reads exactly the timesteps congruent to *i* mod `n_tokens`.
    Every probed timestep therefore moves exactly one token -- the response is
    as peaked as a patch tokenizer's, and peak:pedestal contrast cannot tell
    the two apart -- but no token's owned timesteps form a contiguous run, so
    `[min, max]` describes a range each token mostly does not touch. This is
    ROADMAP.md sec 19 G2's Lag-Llama shape, built deliberately as the decoy
    `CLAUDE.md` sec 11.34 says a probe test needs.
    """

    def __init__(self, context_len: int, n_tokens: int, dim: int, seed: int = 5):
        super().__init__()
        torch.manual_seed(seed)
        self.n_tokens, self.dim = n_tokens, dim
        self.per = context_len // n_tokens
        self.proj = nn.Linear(self.per, dim)
        self.blocks = nn.ModuleList([nn.Identity()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lags = x.view(x.shape[0], self.per, self.n_tokens).transpose(1, 2)
        h = self.proj(lags)
        for block in self.blocks:
            h = block(h)
        return h.mean(dim=(1, 2))


class _LagFeatureAdapter(MockPatchAdapter):
    def load(self) -> None:
        self._net = _LagFeatureNet(self.data_cfg.context_len,
                                   self.data_cfg.context_len // self.patch, self.dim)


def _lag_discovery():
    return discover_spans(_adapter(_LagFeatureAdapter, "laggy"), stride=1)


def test_a_sharp_but_noncontiguous_model_is_refused_by_the_contiguity_gate():
    found = _lag_discovery()
    assert found.contiguity < 0.95
    assert not found.is_contiguous()
    reason = found.refusal_reason()
    assert reason is not None
    assert "not to contiguous intervals" in reason
    assert "pooling premise does not hold" in reason


def test_contrast_alone_would_have_admitted_the_noncontiguous_model():
    """The load-bearing negative: the second gate is not redundant with the first.

    If contrast could already see this case there would be no reason for
    `is_contiguous` to exist, and the honest thing would be to say so rather
    than ship a gate that never fires.
    """
    found = _lag_discovery()
    assert found.is_time_localized(), (
        "the decoy is supposed to be sharply localized -- if it is not, this "
        "test is no longer exercising the gap it was built for")
    assert found.contrast > 4.0


def test_the_diffuse_model_is_still_refused_on_contrast_not_contiguity():
    """Each gate must name the failure that actually occurred (sec 11.33's cost)."""
    a = _adapter(_DiffuseAdapter, "diffuse")
    reason = discover_spans(a, stride=1).refusal_reason()
    assert reason is not None
    assert "not time-localized" in reason
    assert "contiguous intervals" not in reason


@pytest.mark.parametrize("cls,name", [(MockPatchAdapter, "patchy"),
                                       (MockStepAdapter, "steppy"),
                                       (MockWaveAdapter, "wavy")])
def test_the_new_gate_is_a_no_op_for_every_existing_localized_mock(cls, name):
    found = discover_spans(_adapter(cls, name), stride=1)
    assert found.contiguity == pytest.approx(1.0)
    assert found.noncontiguous_tokens == []
    assert found.refusal_reason() is None, f"{name} newly refused"


def test_empty_and_noncontiguous_tokens_are_counted_separately():
    """A stripped special and a lag-feature token are not the same event.

    Pooled into one `flagged_tokens` number they are indistinguishable, which
    is exactly how a model with the second problem passes as a model with a
    few of the first.
    """
    coarse = discover_spans(_adapter(MockStepAdapter, "steppy"), stride=4)
    assert coarse.empty_tokens, "a stride coarser than the token width must leave gaps"
    assert coarse.noncontiguous_tokens == []
    assert coarse.contiguity == pytest.approx(1.0)
    assert sorted(coarse.empty_tokens + coarse.noncontiguous_tokens) == coarse.flagged_tokens


def test_contiguity_excludes_empty_tokens_from_its_denominator():
    coarse = discover_spans(_adapter(MockStepAdapter, "steppy"), stride=4)
    assert coarse.empty_tokens
    # Counting empties as failures would drag this well below 1.0 and refuse a
    # model for a probe stride the caller chose.
    assert coarse.contiguity == pytest.approx(1.0)
    assert coarse.refusal_reason() is None


def test_new_fields_serialize_without_numpy_types():
    import json
    json.dumps(_lag_discovery().to_dict())
