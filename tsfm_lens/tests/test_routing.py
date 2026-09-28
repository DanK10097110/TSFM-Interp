"""E3(c)'s routing half: a model with no token->time map runs L0 only.

The refusal itself (measuring an impulse response and finding it diffuse)
is `tests/test_generic_hf_adapter.py`'s subject. This file is about what the
*pipeline* does with that verdict -- which until now was nothing, so a
`NotTimeLocalized` escaped mid-extraction and killed the run rather than
producing the restricted-but-honest report `CLAUDE.md` sec 12 promises.

Everything here is a mock adapter raising the real exception type; no
checkpoint and no discovery pass is involved, because the decision under
test is the routing, not the measurement.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import config_from_dict
from tsfm_lens.models import ADAPTERS
from tsfm_lens.models.base import NotTimeLocalized
from tsfm_lens.models.mock import MockPatchAdapter
from tsfm_lens.pipeline import Context, resolve_routing, run_pipeline
from tsfm_lens.utils import load_json


class _DiffuseMock(MockPatchAdapter):
    """A model whose measured impulse response has no contiguous span map."""

    measures_own_spans = True

    def token_time_spans(self):
        raise NotTimeLocalized(
            "measured impulse response is not time-localized",
            model=self.name, checkpoint="mock://diffuse",
            contrast=1.07, min_contrast=4.0, diffuseness=0.31)


class _BrokenMock(MockPatchAdapter):
    """A model whose span resolution fails for a reason that is a real bug."""

    measures_own_spans = True

    def token_time_spans(self):
        raise RuntimeError("hooks returned nothing -- this is a defect, not a verdict")


class _LagMock(MockPatchAdapter):
    """Sharply localized, but each token reads disjoint lags rather than an interval.

    The second refusal gate (ROADMAP.md sec 19 G2). Note the numbers: this
    model's contrast is *healthy* and above its floor -- the refusal is on
    contiguity alone, which is why every surface that renders a refusal has
    to say which gate fired rather than assuming contrast.
    """

    measures_own_spans = True

    def token_time_spans(self):
        raise NotTimeLocalized(
            "tokens are time-localized but not to contiguous intervals; "
            "the pooling premise does not hold",
            model=self.name, checkpoint="mock://laggy",
            contrast=41.2, min_contrast=4.0, diffuseness=0.88,
            contiguity=0.0, min_contiguity=0.95)


@pytest.fixture(autouse=True)
def _register():
    ADAPTERS["mock_diffuse"] = _DiffuseMock
    ADAPTERS["mock_broken"] = _BrokenMock
    ADAPTERS["mock_lag"] = _LagMock
    yield
    ADAPTERS.pop("mock_diffuse", None)
    ADAPTERS.pop("mock_broken", None)
    ADAPTERS.pop("mock_lag", None)


def _cfg(out_dir: str, adapter_b: str = "mock_step") -> dict:
    return {
        "run": {"name": "routing", "out_dir": out_dir, "device": "cpu",
                "dtype": "float32", "seed": 0},
        "data": {"source": "smoke", "context_len": 128, "horizon": 32,
                 "smoke_series_per_family": 8},
        "alignment": {"window": 32, "sanity_check": False},
        "models": [
            {"name": "patchy", "adapter": "mock_patch", "batch_size": 32},
            {"name": "other", "adapter": adapter_b, "batch_size": 32},
        ],
        "stats": {"enabled": False},
        "budget": {"enabled": False},
        "layer_screen": {"enabled": False},
        "internals": {"enabled": False},
        "lens": {"enabled": False},
        "l3": {"enabled": False},
        "attention": {"enabled": False},
        "clustering": {"enabled": False},
        "exemplars": {"enabled": False},
        "confirm": {"enabled": False},
        "sae": {"enabled": False},
        "report": {"title": "Routing", "verbose": False},
    }


def test_a_time_localized_pair_is_routed_to_everything():
    with tempfile.TemporaryDirectory() as d:
        cfg = config_from_dict(_cfg(d))
        routing = resolve_routing(cfg, Context(cfg).hub)
    assert {r["eligible"] for r in routing.values()} == {"full"}
    # Hand-written adapters declare their spans rather than measuring them,
    # so "full" here is an absence of refusal, not a measurement -- the
    # report must be able to say which, so the record says which. They are
    # also never probed at all: `measures_own_spans` is False, and probing
    # would load a checkpoint to learn nothing.
    assert routing["patchy"]["measured"] is False


def test_a_diffuse_model_is_routed_to_l0_with_the_number_that_decided_it():
    with tempfile.TemporaryDirectory() as d:
        cfg = config_from_dict(_cfg(d, adapter_b="mock_diffuse"))
        routing = resolve_routing(cfg, Context(cfg).hub)
    assert routing["other"]["eligible"] == "l0_only"
    assert routing["other"]["contrast"] == 1.07
    assert routing["other"]["min_contrast"] == 4.0
    assert "not time-localized" in routing["other"]["reason"]
    assert routing["patchy"]["eligible"] == "full"


def test_a_real_defect_is_not_relabelled_as_a_diffuse_model():
    """The load-bearing negative: catching ValueError here would turn any
    bug in span resolution into a considered-looking "this model has no
    time structure" verdict (CLAUDE.md sec 11.33's false-refusal cost)."""
    with tempfile.TemporaryDirectory() as d:
        cfg = config_from_dict(_cfg(d, adapter_b="mock_broken"))
        with pytest.raises(RuntimeError, match="this is a defect"):
            resolve_routing(cfg, Context(cfg).hub)


def test_the_pipeline_completes_and_says_so_instead_of_crashing():
    with tempfile.TemporaryDirectory() as d:
        cfg = config_from_dict(_cfg(d, adapter_b="mock_diffuse"))
        run_pipeline(cfg)
        run_dir = cfg.run_dir()

        recorded = load_json(run_dir / "routing.json")
        assert recorded["other"]["eligible"] == "l0_only"

        # L0 ran; nothing that needs an aligned window axis did.
        assert (run_dir / "l0" / "summary.json").exists()
        for restricted in ("l1", "l2", "l3", "lens", "internals", "clustering"):
            assert not (run_dir / restricted).exists(), restricted

        # And the report says why, in the body -- not only in the log, which
        # is invisible by the time anyone opens the HTML (invariant 8).
        html = (run_dir / "report.html").read_text(encoding="utf-8")
        assert "restricted to L0" in html
        assert "fairness-restricted" in html
        assert "1.07" in html


def test_the_decision_survives_a_report_only_rerun():
    """A `--stages report` rerun must not silently re-widen the run: the
    routing artifact is the record, and the banner has to render from it
    without reloading a checkpoint to rediscover the verdict."""
    with tempfile.TemporaryDirectory() as d:
        cfg = config_from_dict(_cfg(d, adapter_b="mock_diffuse"))
        run_pipeline(cfg)
        run_pipeline(cfg, stages=["report"], force={"report"})
        html = (cfg.run_dir() / "report.html").read_text(encoding="utf-8")
        assert "restricted to L0" in html


def test_a_declared_spans_adapter_is_never_probed():
    """The preflight must not load a checkpoint to learn nothing.

    Every hand-written adapter declares its spans, so it cannot refuse
    itself; probing one anyway would also hold every model resident at once,
    which extraction deliberately avoids for VRAM.
    """
    calls = []

    class _Watched(MockPatchAdapter):
        def token_time_spans(self):
            calls.append(self.name)
            return super().token_time_spans()

    ADAPTERS["mock_watched"] = _Watched
    try:
        with tempfile.TemporaryDirectory() as d:
            cfg = config_from_dict(_cfg(d, adapter_b="mock_watched"))
            routing = resolve_routing(cfg, Context(cfg).hub)
    finally:
        ADAPTERS.pop("mock_watched", None)
    assert calls == []
    assert routing["other"] == {"eligible": "full", "measured": False}


def test_a_noncontiguous_model_is_routed_out_and_the_report_names_the_right_gate():
    """A refusal that quotes contrast for a contiguity failure shows a healthy
    number beside a refusal, which reads as a broken gate rather than as the
    model's actual property (CLAUDE.md sec 11.33/sec 11.35's shape)."""
    from tsfm_lens.report.report import _eligibility_cell, _refused_on_contiguity
    with tempfile.TemporaryDirectory() as d:
        cfg = config_from_dict(_cfg(d, adapter_b="mock_lag"))
        routing = resolve_routing(cfg, Context(cfg).hub)
    rec = routing["other"]
    assert rec["eligible"] == "l0_only"
    assert rec["contiguity"] == 0.0 and rec["min_contiguity"] == 0.95
    assert rec["contrast"] == 41.2, "the decoy must clear the contrast floor"
    assert _refused_on_contiguity(rec) is True
    assert "contiguity" in _eligibility_cell(rec)
    assert "contrast" not in _eligibility_cell(rec)


def test_a_diffuse_refusal_still_renders_as_a_contrast_refusal():
    from tsfm_lens.report.report import _eligibility_cell, _refused_on_contiguity
    with tempfile.TemporaryDirectory() as d:
        cfg = config_from_dict(_cfg(d, adapter_b="mock_diffuse"))
        routing = resolve_routing(cfg, Context(cfg).hub)
    rec = routing["other"]
    assert _refused_on_contiguity(rec) is False
    assert "contrast" in _eligibility_cell(rec)
