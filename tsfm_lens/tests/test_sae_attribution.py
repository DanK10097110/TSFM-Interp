"""ROADMAP.md sec 42, decision 1: the attribution-ranked candidate rule.

The fixture is a linear model with a known answer. Four tokens of width four go
through an identity block, are mean-pooled and read out by `w`; the SAE is the
identity dictionary, so atom j is coordinate j. The forecast moves only through
atoms 0 (weight 3) and 3 (weight 1). Atom 1 is the DECOY: it has by far the
largest activation variance and zero readout weight, so a variance rule picks it
first and a forecast-effect rule must not.

Because the model is linear, the first-order score equals the real ablation
effect exactly, which gives a closed form to test against.

What is asserted:
  1. the ranking is {0, 3}, in that order, and the decoy has score exactly 0 and
     is not listed; the variance rule ranks the decoy first (the confusable case);
  2. the score equals the closed form `|w_f| * mean_s mean_t a[s,t,f]`;
  3. the first-order prediction matches the real ablation (`linearization_check`);
  4. a forecast that is not differentiable, and one that never reads the patched
     layer, each return a stated `unavailable` reason instead of an empty ranking;
  5. `select_candidates` records every nominating rule, keeps legacy output
     byte-identical, and appends attribution nominations after the legacy ones;
  6. the new config fields leave every stage fingerprint of a default config
     unchanged;
  7. `predict_differentiable` of the planted mock equals `predict`'s point forecast.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens.config import DataConfig, ModelConfig, PipelineConfig  # noqa: E402
from tsfm_lens.manifest import resolve_config_keys  # noqa: E402
from tsfm_lens.models import build_adapter  # noqa: E402
from tsfm_lens.models.base import ModelAdapter  # noqa: E402
from tsfm_lens.pipeline import _stages  # noqa: E402
from tsfm_lens.sae.attribution import (attribution_candidates, attribution_scores,  # noqa: E402
                                       linearization_check, rank_atoms)
from tsfm_lens.sae.models import TopKSAE  # noqa: E402
from tsfm_lens.sae.response import select_candidates  # noqa: E402

D, T, H, N = 4, 4, 3, 40
W_READ = np.array([3.0, 0.0, 0.0, 1.0])
SCALE = np.array([2.0, 10.0, 1.0, 2.0])


class _Net(nn.Module):
    cancelling = False

    def __init__(self, reads_layer: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(D, D, bias=False)])
        with torch.no_grad():
            self.blocks[0].weight.copy_(torch.eye(D))
        self.readout = nn.Linear(D, H, bias=False)
        with torch.no_grad():
            self.readout.weight.copy_(torch.from_numpy(np.tile(W_READ, (H, 1))).float())
        if self.cancelling:
            with torch.no_grad():
                self.readout.weight.zero_()
                self.readout.weight[0, 0], self.readout.weight[1, 0] = 1.0, -1.0
        self.reads_layer = reads_layer

    def forward(self, x):
        h = self.blocks[0](x.view(x.shape[0], T, D))
        if not self.reads_layer:
            return self.readout(torch.zeros(x.shape[0], D))
        return self.readout(h.mean(dim=1))


class _Adapter(ModelAdapter):
    default_layer_regex = r"^blocks\.\d+$"
    differentiable = True
    reads_layer = True

    def load(self):
        self._net = _Net(self.reads_layer)

    @property
    def module(self):
        return self._net

    def prepare(self, contexts):
        return torch.from_numpy(np.ascontiguousarray(contexts)).float()

    def forward(self, prepared):
        self._net(prepared)

    def token_time_spans(self):
        starts = np.arange(T) * D
        return np.stack([starts, starts + D], axis=1).astype(np.float64)

    def predict(self, contexts, horizon, quantiles):
        with torch.no_grad():
            return {"point": self._net(self.prepare(contexts)).numpy()[:, :horizon]}

    def predict_differentiable(self, contexts, horizon):
        if not self.differentiable:
            return super().predict_differentiable(contexts, horizon)
        return self._net(self.prepare(contexts))[:, :horizon]


class _CancellingNet(_Net):
    cancelling = True


class _Cancelling(_Adapter):
    def load(self):
        self._net = _CancellingNet(True)


class _Opaque(_Adapter):
    differentiable = False


class _Unread(_Adapter):
    reads_layer = False


class _Data:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n = N
        self._x = (np.abs(rng.normal(size=(N, T, D))) * SCALE + 0.5).reshape(N, T * D).astype(np.float32)
        self._y = rng.normal(size=(N, H)).astype(np.float32)
        self.meta = pd.DataFrame({"family": ["a", "b"] * (N // 2)})

    def contexts(self):
        return self._x

    def targets(self):
        return self._y


def _cfg():
    return SimpleNamespace(data=SimpleNamespace(horizon=H), l0=SimpleNamespace(quantiles=[0.5]),
                           run=SimpleNamespace(seed=0))


def _sae():
    sae = TopKSAE(D, D, k=D)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(D))
        sae.W_dec.copy_(torch.eye(D))
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    return sae


def _adapter(cls=_Adapter):
    cfg = SimpleNamespace(name="toy", batch_size=16, adapter="toy", random_init=False)
    a = cls(cfg, SimpleNamespace(horizon=H), torch.device("cpu"), torch.float32)
    a.ensure_loaded()
    return a


@pytest.fixture
def world():
    return _cfg(), _adapter(), _sae(), _Data()


def _scores(world):
    cfg, adapter, sae, data = world
    return attribution_scores(cfg, adapter, "blocks.0", sae, data, torch.device("cpu"),
                              np.arange(N), seed=0)


def test_ranking_finds_the_planted_atoms_and_not_the_high_variance_decoy(world):
    cfg, adapter, sae, data = world
    alive = np.ones(D, dtype=bool)
    scores = _scores(world)["forecast_mean"]
    ranked = rank_atoms(scores, alive, n=4)
    assert [r["feature"] for r in ranked] == [0, 3]
    assert scores[1] == 0.0 and scores[2] == 0.0
    var = data.contexts().reshape(-1, D).var(axis=0)
    assert int(np.argmax(var)) == 1


def test_score_equals_the_closed_form_first_order_effect(world):
    cfg, adapter, sae, data = world
    scores = _scores(world)["forecast_mean"]
    a = data.contexts().reshape(N, T, D)
    expected = np.abs(W_READ) * a.mean(axis=1).mean(axis=0)
    np.testing.assert_allclose(scores, expected, rtol=1e-5)


def test_mase_ranking_is_computed_beside_the_forecast_ranking(world):
    scores = _scores(world)
    assert scores["mase"].shape == scores["forecast_mean"].shape
    assert scores["mase"][1] == 0.0 and scores["mase"][0] > 0.0


def test_first_order_prediction_matches_the_real_ablation(world):
    cfg, adapter, sae, data = world
    out = linearization_check(cfg, adapter, "blocks.0", sae, data, torch.device("cpu"),
                              np.arange(N), [0, 3], seed=0)
    assert [a["feature"] for a in out["atoms"]] == [0, 3]
    for atom in out["atoms"]:
        assert atom["pearson"] == pytest.approx(1.0, abs=1e-4)
        assert atom["mean_predicted"] == pytest.approx(atom["mean_real"], rel=1e-4)


def test_candidates_block_lists_the_planted_atoms_with_status_ok(world):
    cfg, adapter, sae, data = world
    out = attribution_candidates(cfg, adapter, "blocks.0", sae, data, torch.device("cpu"),
                                 np.ones(D, dtype=bool), n_attribution=2, max_series=N, seed=0)
    assert out["status"] == "ok"
    assert [x["feature"] for x in out["top_forecast_mean"]] == [0, 3]
    assert out["error_term"].startswith("absent in both arms")


def test_non_differentiable_forecast_is_unavailable_with_a_reason():
    cfg, sae, data = _cfg(), _sae(), _Data()
    out = attribution_candidates(cfg, _adapter(_Opaque), "blocks.0", sae, data, torch.device("cpu"),
                                 np.ones(D, dtype=bool), n_attribution=2, max_series=N, seed=0)
    assert out["status"] == "unavailable"
    assert "predict_differentiable" in out["reason"]


def test_forecast_that_never_reads_the_patched_layer_is_unavailable_not_empty():
    cfg, sae, data = _cfg(), _sae(), _Data()
    out = attribution_candidates(cfg, _adapter(_Unread), "blocks.0", sae, data, torch.device("cpu"),
                                 np.ones(D, dtype=bool), n_attribution=2, max_series=N, seed=0)
    assert out["status"] == "unavailable"
    assert "exactly 0" in out["reason"]


def test_horizon_l1_sees_an_effect_the_horizon_mean_cancels():
    cfg, sae, data = _cfg(), _sae(), _Data()
    scores = attribution_scores(cfg, _adapter(_Cancelling), "blocks.0", sae, data,
                                torch.device("cpu"), np.arange(N), seed=0, vector_steps=3)
    assert scores["forecast_mean"][0] == pytest.approx(0.0, abs=1e-6)
    assert scores["forecast_horizon_l1"][0] > 1.0
    assert scores["forecast_horizon_l1"][1] == 0.0


def test_dead_atoms_are_never_ranked(world):
    scores = _scores(world)["forecast_mean"]
    alive = np.array([False, True, True, True])
    assert [r["feature"] for r in rank_atoms(scores, alive, n=4)] == [3]


def _alive(n=12):
    return np.ones(n, dtype=bool)


def test_select_candidates_records_every_rule_and_appends_attribution_last():
    variance = np.zeros(12)
    variance[5] = 9.0
    attr = [{"feature": 5, "score": 2.0}, {"feature": 7, "score": 1.0}, {"feature": 8, "score": 0.5}]
    out = select_candidates(_alive(), 1, activation_variance=variance, seed=0,
                            rules=("variance", "random", "attribution"),
                            attribution_ranked=attr, n_attribution=2)
    by = {c["feature"]: c["rules"] for c in out}
    assert by[5][:1] == ["variance"] and "attribution" in by[5]
    assert by[7] == ["attribution"] and 8 not in by
    assert out[-1]["feature"] in (5, 7)
    assert [c["feature"] for c in out][-1] == 7


def test_default_rules_are_byte_identical_to_the_legacy_call():
    variance = np.arange(12, dtype=float)
    legacy = select_candidates(_alive(), 3, activation_variance=variance, seed=4)
    explicit = select_candidates(
        _alive(), 3, activation_variance=variance, seed=4,
        rules=("structural", "variance", "provenance", "random"),
        attribution_ranked=[{"feature": 0, "score": 1.0}], n_attribution=5)
    assert legacy == explicit


def test_attribution_mase_nominates_only_when_configured():
    attr = [{"feature": 1, "score": 1.0}]
    mase = [{"feature": 2, "score": 1.0}]
    off = select_candidates(_alive(), 1, seed=0, rules=("attribution",), attribution_ranked=attr,
                            attribution_mase_ranked=mase)
    on = select_candidates(_alive(), 1, seed=0, rules=("attribution", "attribution_mase"),
                           attribution_ranked=attr, attribution_mase_ranked=mase)
    assert [c["feature"] for c in off] == [1]
    assert sorted(c["feature"] for c in on) == [1, 2]


def test_unknown_candidate_rule_is_refused_at_config_validation():
    cfg = PipelineConfig()
    cfg.sae.candidate_rules = ("variance", "attributon")
    with pytest.raises(ValueError, match="candidate_rules"):
        cfg.validate()


def test_new_fields_leave_every_default_stage_fingerprint_unchanged():
    base = PipelineConfig()
    tuned = PipelineConfig()
    tuned.sae.candidate_rules = ("variance", "random", "attribution")
    tuned.sae.n_attribution = 7
    for stage in _stages():
        before = resolve_config_keys(base, stage.config_keys)
        after = resolve_config_keys(tuned, stage.config_keys)
        if stage.name == "concepts":
            assert before != after
        else:
            assert before == after, stage.name
    keys = resolve_config_keys(base, [k for s in _stages() if s.name == "concepts"
                                      for k in s.config_keys])
    assert not any("candidate_rules" in k or "n_attribution" in k for k in keys)


def test_planted_mock_differentiable_forecast_equals_predict():
    cfg = ModelConfig(name="A", adapter="mock_planted", batch_size=8,
                      kwargs={"plant_set": "A", "dose": 1.0, "construction_seed": 0})
    a = build_adapter(cfg, DataConfig(source="smoke", context_len=256, horizon=32,
                                      smoke_series_per_family=4),
                      torch.device("cpu"), torch.float32)
    a.ensure_loaded()
    x = np.random.default_rng(0).normal(size=(4, 256)).astype(np.float32)
    point = a.predict_differentiable(x, 32)
    assert point.requires_grad or point.grad_fn is not None or torch.is_grad_enabled()
    np.testing.assert_allclose(point.detach().numpy(), a.predict(x, 32, [0.5])["point"], atol=1e-6)


def test_attribution_is_appended_to_a_frozen_stage2_list_without_moving_it():
    from tsfm_lens.sae.ablation_run import _add_attribution
    stage2 = [{"feature": 4, "rules": ["variance"]}, {"feature": 9, "rules": ["random"]}]
    top = {"attribution": [{"feature": 9, "score": 3.0}, {"feature": 1, "score": 2.0}],
           "attribution_mase": [{"feature": 2, "score": 1.0}]}
    out = _add_attribution(stage2, top, np.ones(12, dtype=bool), ("attribution",), 5)
    assert [c["feature"] for c in out] == [4, 9, 1]
    assert out[1]["rules"] == ["random", "attribution"]
    assert out[0]["rules"] == ["variance"]
