"""Adapter capability tiers (`ROADMAP.md` sec 19 G1).

The gate these cover is unusual in one way worth stating: it decides what a
run is *allowed to attempt*, so its failure mode is a false refusal --
silently deleting analyses a model could actually have supported, with a
stated-looking reason (`CLAUDE.md` sec 11.33/11.35). Several tests below are
therefore negatives: that tier 3 drops nothing, that a real capability
failure is not relabelled as a tier verdict, and that the dropped-stage list
does not depend on which stages an invocation happened to select.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import PipelineConfig, load_config
from tsfm_lens.models import ADAPTERS
from tsfm_lens.models.base import (TIER_NAMES, CapabilityUnavailable, ModelAdapter,
                                   NotTimeLocalized)
from tsfm_lens.pipeline import _STAGE_MIN_TIER, _apply_tiers, stage_names
from tsfm_lens.report.report import _tier_skip_reason
from tsfm_lens.utils import save_json

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# Pinned against `CLAUDE.md` sec 6.2's support matrix, which is prose and
# therefore cannot drift with the code on its own. A tier here changing
# means either an adapter gained/lost a capability or the derivation rule
# moved; both should be a deliberate edit, not a surprise.
EXPECTED_TIERS = {
    "chronos": 3, "chronos2": 3, "chronos_bolt": 2, "generic_hf": 2,
    "mock_blackbox": 0, "mock_encdec": 3, "mock_patch": 3, "mock_step": 3,
    "mock_planted": 3, "mock_wave": 3, "sundial": 2, "timesfm": 3,
}


def test_registered_adapter_tiers_match_the_documented_support_matrix():
    assert {n: a.capability_tier() for n, a in ADAPTERS.items()} == EXPECTED_TIERS


def test_every_pipeline_stage_declares_a_minimum_tier():
    # A stage missing from the map falls back to 1, which would quietly
    # exclude it from every tier-0 run without anyone deciding that.
    assert set(_STAGE_MIN_TIER) == set(stage_names())


def test_tier_names_cover_every_tier_the_derivation_can_return():
    assert set(TIER_NAMES) == {0, 1, 2, 3}


def _blackbox_adapter():
    cfg = load_config(CONFIGS / "smoke_blackbox.yaml")
    from tsfm_lens.models import build_adapter
    import torch
    return build_adapter(cfg.models[0], cfg.data, torch.device("cpu"), torch.float32)


_TIER1_CALLS = {
    "module": lambda a: a.module,
    "prepare": lambda a: a.prepare(np.zeros((1, 8), dtype=np.float32)),
    "forward": lambda a: a.forward(None),
    "token_time_spans": lambda a: a.token_time_spans(),
}


@pytest.mark.parametrize("capability", sorted(_TIER1_CALLS))
def test_tier0_adapter_refuses_each_tier1_capability_with_a_typed_error(capability):
    a = _blackbox_adapter()
    with pytest.raises(CapabilityUnavailable) as exc:
        _TIER1_CALLS[capability](a)
    rec = exc.value.as_record()
    assert rec["capability"] == capability
    assert rec["tier_actual"] == 0 and rec["tier_required"] == 1
    # The message must name the model, not just the method: a run with two
    # black boxes otherwise gives an error that cannot be attributed.
    assert a.name in str(exc.value)


def test_tier0_adapter_still_forecasts():
    a = _blackbox_adapter()
    a.ensure_loaded()  # must not touch `module`
    ctx = np.tile(np.sin(np.linspace(0, 8, 128)), (3, 1)).astype(np.float32)
    out = a.predict(ctx, horizon=32, quantiles=[0.1, 0.5, 0.9])
    assert out["point"].shape == (3, 32)
    assert out["quantiles"].shape == (3, 32, 3)
    assert np.isfinite(out["point"]).all()


def test_capability_unavailable_is_not_confusable_with_the_routing_refusal():
    # `NotTimeLocalized` subclasses ValueError and means "this model HAS
    # internals and its token->time map is unusable" -- a measured verdict.
    # Tier refusal means "the adapter never claimed to expose them". Sharing
    # a base class would let one be caught as the other, turning a missing
    # implementation into a finding about the model.
    assert issubclass(CapabilityUnavailable, NotImplementedError)
    assert not issubclass(CapabilityUnavailable, NotTimeLocalized)
    assert not issubclass(CapabilityUnavailable, ValueError)


class _BrokenTier1(ModelAdapter):
    """Tier 1+ by declaration, but its `module` genuinely fails."""
    default_layer_regex = r"^blocks\.\d+$"

    def load(self): pass
    def _release(self): pass
    def predict(self, contexts, horizon, quantiles): raise NotImplementedError

    @property
    def module(self): raise RuntimeError("checkpoint is corrupt")
    def prepare(self, contexts): return contexts
    def forward(self, prepared): return None
    def token_time_spans(self): return np.zeros((1, 2))


def test_a_real_module_failure_is_not_excused_by_the_tier_guard():
    # ensure_loaded skips `.eval()` for tier 0 only. Guarding by catching
    # CapabilityUnavailable instead would also swallow this, which is the
    # silent-degradation failure `CLAUDE.md` sec 2.5 forbids.
    import torch
    from tsfm_lens.config import DataConfig, ModelConfig
    a = _BrokenTier1(ModelConfig(name="broken", adapter="x"), DataConfig(),
                     torch.device("cpu"), torch.float32)
    assert a.capability_tier() >= 1
    with pytest.raises(RuntimeError, match="corrupt"):
        a.ensure_loaded()


class _Ctx:
    def __init__(self, hub):
        self.hub = hub


class _Hub:
    def __init__(self, adapters):
        self._a = adapters

    def get(self, name):
        return self._a[name]


def _tiers_for(cfg, tier_by_model, selected):
    class _Stub:
        def __init__(self, t): self._t = t
        def tier_report(self): return {"tier": self._t, "name": TIER_NAMES[self._t],
                                       "adapter": "stub", "missing": {}}
    hub = _Hub({n: _Stub(t) for n, t in tier_by_model.items()})
    return _apply_tiers(cfg, _Ctx(hub), selected)


def _smoke_cfg() -> PipelineConfig:
    return load_config(CONFIGS / "smoke.yaml")


def test_tier3_run_drops_nothing():
    cfg = _smoke_cfg()
    selected = set(stage_names())
    rec = _tiers_for(cfg, {"patchy": 3, "steppy": 3}, selected)
    assert rec["run_tier"] == 3
    assert rec["dropped_stages"] == []
    assert selected == set(stage_names())


def test_tier0_run_keeps_exactly_the_forecast_only_stages():
    cfg = _smoke_cfg()
    selected = set(stage_names())
    rec = _tiers_for(cfg, {"patchy": 0, "steppy": 3}, selected)
    assert rec["run_tier"] == 0
    # Only stages enabled in this config can be dropped; the survivors are
    # everything at min-tier 0 that was selected.
    assert set(rec["dropped_stages"]).isdisjoint({"l0", "budget", "report"})
    assert {"extract", "l1", "l2", "l3", "lens", "internals"} <= set(rec["dropped_stages"])
    assert "l0" in selected and "report" in selected


def test_a_non_single_pass_adapter_loses_patching_but_keeps_geometry():
    cfg = _smoke_cfg()
    selected = set(stage_names())
    rec = _tiers_for(cfg, {"patchy": 1, "steppy": 3}, selected)
    assert rec["run_tier"] == 1
    assert set(rec["dropped_stages"]) == {"lens", "l3"}
    assert {"extract", "l1", "l2", "internals"} <= selected


def test_dropped_stage_list_does_not_depend_on_which_stages_were_selected():
    # The bug this pins: computing `dropped` from `selected` makes a
    # `--stages report` rerun rewrite tiers.json to claim nothing was
    # dropped, erasing the reason the report prints next to each skip.
    cfg = _smoke_cfg()
    full = _tiers_for(cfg, {"patchy": 0, "steppy": 0}, set(stage_names()))
    report_only = _tiers_for(cfg, {"patchy": 0, "steppy": 0}, {"report"})
    assert full["dropped_stages"] == report_only["dropped_stages"]
    assert full["dropped_stages"]


def test_report_skip_reason_names_the_tier_only_when_the_tier_dropped_it(tmp_path):
    assert _tier_skip_reason(tmp_path, "l1") == ""
    save_json(tmp_path / "tiers.json",
              {"models": {"bb": {"tier": 0, "name": "black box"}},
               "run_tier": 0, "dropped_stages": ["l1", "cluster"]})
    reason = _tier_skip_reason(tmp_path, "l1")
    assert "capability tier" in reason and "tier 0" in reason and "bb" in reason
    # The report's config attribute for L4 is `clustering`; the stage is
    # `cluster`. An unmapped name would silently report "artifacts missing".
    assert "cluster" in _tier_skip_reason(tmp_path, "clustering")
    assert _tier_skip_reason(tmp_path, "l2") == ""


def test_conformance_accepts_a_tier0_adapter_on_its_own_terms():
    from tsfm_lens.models.conformance import check_adapter_conformance, make_test_adapter
    rep = check_adapter_conformance(make_test_adapter("mock_blackbox"), window=32)
    assert rep["tier"] == 0 and rep["tier_name"] == "black box"
    assert sorted(rep["refused_capabilities"]) == [
        "forward", "module", "prepare", "token_time_spans"]


def test_conformance_rejects_an_adapter_whose_tier_and_implementation_disagree():
    # The check that makes the derived tier load-bearing rather than
    # decorative: an adapter can only claim tier 0 by genuinely not
    # implementing the tier-1 surface.
    from tsfm_lens.models.conformance import check_adapter_conformance
    from tsfm_lens.models.mock import MockBlackBoxAdapter
    from tsfm_lens.config import DataConfig, ModelConfig
    import torch

    class _Liar(MockBlackBoxAdapter):
        @classmethod
        def capability_tier(cls):
            return 0

        def token_time_spans(self):
            return np.array([[0.0, 1.0]])

    a = _Liar(ModelConfig(name="liar", adapter="mock_blackbox"),
              DataConfig(context_len=128, horizon=8), torch.device("cpu"), torch.float32)
    a.ensure_loaded()
    with pytest.raises(AssertionError, match="declares tier 0"):
        check_adapter_conformance(a, window=32)


def test_conformance_rejects_an_untyped_refusal():
    from tsfm_lens.models.conformance import check_adapter_conformance
    from tsfm_lens.models.mock import MockBlackBoxAdapter
    from tsfm_lens.config import DataConfig, ModelConfig
    import torch

    class _WrongError(MockBlackBoxAdapter):
        def token_time_spans(self):
            raise AttributeError("no spans here")

    a = _WrongError(ModelConfig(name="wrong", adapter="mock_blackbox"),
                    DataConfig(context_len=128, horizon=8), torch.device("cpu"), torch.float32)
    a.ensure_loaded()
    with pytest.raises(AssertionError, match="must raise CapabilityUnavailable"):
        check_adapter_conformance(a, window=32)


def _fairness_html(cfg, run_dir):
    from tsfm_lens.report.report import _sec_fairness
    return _sec_fairness(cfg, run_dir)


def test_fairness_card_states_the_tier0_eligibility_rather_than_pending(tmp_path):
    # A tier-0 run has no routing.json at all -- `extract` was dropped, so
    # nothing ever measured a token→time map. "not yet measured" would read
    # as a pending measurement rather than a settled decision. The contrast
    # against the no-tiers.json case is what makes this test mean something:
    # the same builder must still say "not yet measured" when the reason
    # genuinely is that nothing has run.
    cfg = _smoke_cfg()
    pending = _fairness_html(cfg, tmp_path)
    assert "not yet measured" in pending and "L0 only (tier 0)" not in pending

    save_json(tmp_path / "tiers.json",
              {"models": {m.name: {"tier": 0, "name": "black box", "adapter": "x",
                                   "missing": {}} for m in cfg.models},
               "run_tier": 0, "dropped_stages": ["extract", "l1"]})
    decided = _fairness_html(cfg, tmp_path)
    assert "L0 only (tier 0)" in decided
    assert "Capability tier" in decided
    # The dropped stages must be named, not merely counted -- a reader
    # otherwise cannot tell which analyses this run gave up.
    assert "extract" in decided and "l1" in decided
