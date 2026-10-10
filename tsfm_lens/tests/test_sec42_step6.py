"""ROADMAP.md sec 42 step 6 (revised in 42.4): the horizon-L1 candidate rule, the completeness
ranking, admission re-evaluation without retraining, and the narrowed target set.

Planted answers, each with its decoy:
  * `attribution_horizon_l1` nominates only when the rule is configured, and the decoy is a
    DIFFERENT horizon-mean ranking passed in beside it that must not nominate;
  * an effect the horizon mean cancels (the `_Cancelling` linear world of test_sae_attribution)
    makes the horizon-L1 ranking {0} while the horizon-mean ranking is empty, and the block must
    stay `ok` (it used to be `unavailable` whenever the mean ranking was empty);
  * `battery_ranking` reads `top_forecast_horizon_l1`, not `top_forecast_mean`, and returns `None`
    for an `unavailable` block, an absent artifact, or an empty ranking (the fallback case);
  * `reevaluate_admission`: a target with fidelity 0.6 fails the 0.85 gate it trained under and
    passes the 0.5 floor; the forecast bar is decided by the freshly measured held-out token
    dMASE on the held-out series only; the legacy keys of a `meta.json` entry survive `attach`
    byte for byte;
  * `select_targets`: admitted targets plus the two best layer_screen layers per model, where
    the decoy is a high-scoring layer that layer_screen did not select.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_sae_attribution import (D, N, _adapter, _Cancelling, _cfg, _Data, _sae)  # noqa: E402
from tests.test_sec42_admission_chunking import (_Adapter as _FpAdapter, _Data as _FpData,  # noqa: E402
                                                 _Sae as _FpSae, wired)  # noqa: F401
from tsfm_lens.config import PipelineConfig  # noqa: E402
from tsfm_lens.sae import eval as E  # noqa: E402
from tsfm_lens.sae.ablation_run import _add_attribution, attribution_rankings  # noqa: E402
from tsfm_lens.sae.attribution import attribution_candidates  # noqa: E402
from tsfm_lens.sae.completeness import battery_ranking  # noqa: E402
from tsfm_lens.sae.response import select_candidates  # noqa: E402
from tsfm_lens.sae.train import split_series  # noqa: E402
from tsfm_lens.sae.ablation_run import ablation_path  # noqa: E402

import build_sec42_run_dir as B  # noqa: E402
import run_admission_reeval as R  # noqa: E402

LEGACY = ("structural", "variance", "provenance", "random")


def _alive(n=12):
    return np.ones(n, dtype=bool)


def test_horizon_l1_nominates_only_when_configured_and_ignores_the_mean_ranking_decoy():
    l1 = [{"feature": 7, "score": 2.0}, {"feature": 8, "score": 1.0}]
    mean_decoy = [{"feature": 3, "score": 9.0}]
    off = select_candidates(_alive(), 1, seed=0, rules=LEGACY, attribution_ranked=mean_decoy,
                            attribution_horizon_l1_ranked=l1)
    on = select_candidates(_alive(), 1, seed=0, rules=LEGACY + ("attribution_horizon_l1",),
                           attribution_ranked=mean_decoy, attribution_horizon_l1_ranked=l1,
                           n_attribution=2)
    assert 7 not in [c["feature"] for c in off] and 3 not in [c["feature"] for c in off]
    by = {c["feature"]: c["rules"] for c in on}
    assert by[7] == ["attribution_horizon_l1"] and by[8] == ["attribution_horizon_l1"]
    assert 3 not in by
    assert [c["feature"] for c in on][-2:] == [7, 8]


def test_horizon_l1_rule_leaves_the_legacy_candidates_in_place():
    variance = np.arange(12, dtype=float)
    legacy = select_candidates(_alive(), 3, activation_variance=variance, seed=4)
    with_l1 = select_candidates(_alive(), 3, activation_variance=variance, seed=4,
                                rules=LEGACY + ("attribution_horizon_l1",),
                                attribution_horizon_l1_ranked=[{"feature": 0, "score": 1.0}],
                                n_attribution=5)
    assert with_l1[:len(legacy)] == legacy


def test_add_attribution_appends_the_horizon_l1_nominations_to_a_frozen_list():
    stage2 = [{"feature": 4, "rules": ["variance"]}, {"feature": 9, "rules": ["random"]}]
    top = {"attribution": [{"feature": 2, "score": 1.0}],
           "attribution_horizon_l1": [{"feature": 9, "score": 3.0}, {"feature": 1, "score": 2.0}]}
    out = _add_attribution([dict(c, rules=list(c["rules"])) for c in stage2], top,
                           np.ones(12, dtype=bool), ("attribution_horizon_l1",), 5)
    assert [c["feature"] for c in out] == [4, 9, 1]
    assert out[1]["rules"] == ["random", "attribution_horizon_l1"]
    unnamed = _add_attribution([dict(c, rules=list(c["rules"])) for c in stage2], top,
                               np.ones(12, dtype=bool), ("attribution",), 5)
    assert [c["feature"] for c in unnamed] == [4, 9, 2]


def test_a_horizon_mean_cancelling_effect_keeps_the_block_ok_and_ranks_the_atom():
    cfg, sae, data = _cfg(), _sae(), _Data()
    adapter = _adapter(_Cancelling)
    kw = dict(n_attribution=4, max_series=N, seed=0)
    block = attribution_candidates(cfg, adapter, "blocks.0", sae, data, torch.device("cpu"),
                                   np.ones(D, dtype=bool), vector_steps=3, **kw)
    assert block["status"] == "ok"
    assert block["top_forecast_mean"] == []
    assert [x["feature"] for x in block["top_forecast_horizon_l1"]] == [0]
    assert attribution_rankings(block)["attribution_horizon_l1"] == block["top_forecast_horizon_l1"]
    no_vec = attribution_candidates(cfg, adapter, "blocks.0", sae, data, torch.device("cpu"),
                                    np.ones(D, dtype=bool), vector_steps=0, **kw)
    assert no_vec["status"] == "unavailable"
    assert attribution_rankings(no_vec) == {}


def test_attribution_rankings_carries_horizon_l1_only_when_it_was_computed():
    mean_block = {"status": "ok", "top_forecast_mean": [{"feature": 1, "score": 1.0}],
                  "top_mase": [{"feature": 2, "score": 1.0}]}
    assert set(attribution_rankings(mean_block)) == {"attribution", "attribution_mase"}
    assert set(attribution_rankings({**mean_block, "top_forecast_horizon_l1": []})) == {
        "attribution", "attribution_mase", "attribution_horizon_l1"}
    assert attribution_rankings(None) == {}


def test_config_refuses_the_horizon_l1_rule_without_vector_steps():
    cfg = PipelineConfig()
    cfg.sae.candidate_rules = LEGACY + ("attribution_horizon_l1",)
    with pytest.raises(ValueError, match="attribution_vector_steps"):
        cfg.validate()
    cfg.sae.attribution_vector_steps = 8
    cfg.validate()


def test_sec42_config_is_the_registered_design():
    from tsfm_lens.config import load_config
    cfg = load_config(str(ROOT / "configs" / "panel7_v3_sec42.yaml"))
    assert cfg.run.name == "panel7_v3_sec42"
    assert tuple(cfg.sae.candidate_rules) == LEGACY + ("attribution_horizon_l1",)
    assert (cfg.sae.n_attribution, cfg.sae.attribution_vector_steps) == (20, 8)
    assert (cfg.sae.top_k_series if hasattr(cfg.sae, "top_k_series") else None) in (None, 8)
    assert (cfg.concepts.top_k_series, cfg.concepts.n_null_directions) == (8, 16)
    assert cfg.sae.causal_verdict == "p95_count" and cfg.sae.ablation_null == "profile_matched_cov"
    assert cfg.sae.min_fidelity_gate == 0.5
    assert cfg.sae.forecast_preservation_max_series == 128 and cfg.sae.forecast_preservation_chunked
    assert cfg.concepts.completeness is True
    assert cfg.confirm.enabled is False and cfg.confirm.path == "" and cfg.confirm.external_path == ""


# --------------------------------------------------------------------------- completeness ranking

def _write_artifact(run, model, layer, doc):
    path = ablation_path(run, model, layer)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_battery_ranking_reads_the_horizon_l1_ranking_in_score_order(tmp_path):
    block = {"status": "ok",
             "top_forecast_mean": [{"feature": 11, "score": 9.0}],
             "top_forecast_horizon_l1": [{"feature": 5, "score": 3.0}, {"feature": 2, "score": 1.0}]}
    _write_artifact(tmp_path, "M", "blocks.3", {"attribution": block})
    got = battery_ranking(tmp_path, "M", "blocks.3")
    assert got == {"source": "attribution_horizon_l1", "features": [5, 2]}


def test_battery_ranking_falls_back_when_there_is_no_usable_ranking(tmp_path):
    _write_artifact(tmp_path, "M", "blocks.3", {"attribution": {
        "status": "unavailable", "reason": "sampled head"}})
    _write_artifact(tmp_path, "M", "blocks.4", {"candidates": []})
    _write_artifact(tmp_path, "M", "blocks.5", {"attribution": {
        "status": "ok", "top_forecast_mean": [{"feature": 1, "score": 1.0}],
        "top_forecast_horizon_l1": []}})
    assert battery_ranking(tmp_path, "M", "blocks.3") is None
    assert battery_ranking(tmp_path, "M", "blocks.4") is None
    assert battery_ranking(tmp_path, "M", "blocks.5") is None
    assert battery_ranking(tmp_path, "M", "blocks.6") is None


# --------------------------------------------------------------------------- admission re-evaluation

def _fp_cfg(min_fidelity, max_dm, request=10, holdout=0.5):
    return SimpleNamespace(
        run=SimpleNamespace(seed=0), data=SimpleNamespace(horizon=4),
        l0=SimpleNamespace(quantiles=[0.5]),
        sae=SimpleNamespace(forecast_preservation_max_series=request,
                            forecast_preservation_chunked=True, holdout_frac=holdout,
                            min_fidelity_gate=min_fidelity, max_abs_delta_mase=max_dm))


def _store():
    return SimpleNamespace(root=SimpleNamespace(attrs={"n_series": _FpData.n}))


def _entry(fid=0.6, dead_ok=True):
    return {"reconstruction_fidelity": 0.9, "reconstruction_fidelity_heldout": fid,
            "dead_rate_gate": {"passed": dead_ok},
            "admission": {"passed": False, "reason": "failed: reconstruction fidelity"}}


def _reeval(wired, cfg, entry, batch=4):
    adapter = _FpAdapter(batch)
    orig = adapter.predict

    def predict(contexts, horizon, quantiles):
        adapter.holder["repl"] = wired.get("repl")
        return orig(contexts, horizon, quantiles)

    adapter.predict = predict
    return E.reevaluate_admission(cfg, adapter, "blocks.0", _FpSae(), _store(), _FpData(), "cpu", entry)


def test_reevaluation_admits_a_target_the_training_gate_refused_and_measures_only_heldout_series(wired):
    probe = _reeval(wired, _fp_cfg(0.5, 1e9), _entry())
    dm = probe["forecast_preservation_token_heldout"]["mase_delta"]
    assert dm != 0.0
    out = _reeval(wired, _fp_cfg(0.5, abs(dm) + 1.0), _entry())
    assert out["passed"] is True and out["legacy_passed"] is False
    heldout = set(split_series(_FpData.n, 0.5, 0 + 23, families=_FpData.meta["family"].to_numpy())["heldout"])
    fp = out["forecast_preservation_token_heldout"]
    assert set(fp["rows"]) <= heldout and fp["n_realized"] == len(heldout) == 5
    assert fp["split"] == "heldout" and fp["limited_by"] == ["n_available"]
    assert out["rule"]["min_fidelity_gate"] == 0.5 and out["rule"]["forecast_preservation_chunked"] is True


def test_reevaluation_refuses_on_each_bar_independently(wired):
    probe = _reeval(wired, _fp_cfg(0.5, 1e9), _entry())
    dm = abs(probe["forecast_preservation_token_heldout"]["mase_delta"])
    fid = _reeval(wired, _fp_cfg(0.85, dm + 1.0), _entry())
    assert fid["passed"] is False and fid["failed_checks"] == ["reconstruction fidelity"]
    fp = _reeval(wired, _fp_cfg(0.5, dm / 2.0), _entry())
    assert fp["passed"] is False and fp["failed_checks"] == ["forecast preservation"]
    dead = _reeval(wired, _fp_cfg(0.5, dm + 1.0), _entry(dead_ok=False))
    assert dead["passed"] is False and dead["failed_checks"] == ["dead-feature rate"]


def test_reevaluation_without_a_heldout_split_is_undecided_not_passed(wired):
    out = _reeval(wired, _fp_cfg(0.5, 1.0, holdout=0.0), _entry())
    assert out["passed"] is None and out["forecast_preservation_token_heldout"] is None
    assert out["unmeasured_checks"] == ["forecast preservation"]


def test_attach_keeps_every_legacy_key_byte_identical_and_appends_the_new_ones(wired):
    entry = {**_entry(), "dict_size": 6144, "train_mse_history": [0.5, 0.25]}
    legacy_bytes = json.dumps(entry)
    result = _reeval(wired, _fp_cfg(0.5, 1e9), entry)
    out = R.attach(entry, result)
    assert json.dumps({k: out[k] for k in entry}) == legacy_bytes
    assert list(out)[:len(entry)] == list(entry)
    assert list(out)[len(entry):] == [R.NEW_FP_KEY, R.NEW_KEY]
    assert "forecast_preservation_token_heldout" not in out[R.NEW_KEY]
    assert json.dumps(entry) == legacy_bytes


# --------------------------------------------------------------------------- target set

def _meta(passed):
    return {k: {"admission_sec42": {"passed": v, "reason": "r"}} for k, v in passed.items()}


def test_select_targets_is_admitted_plus_the_two_best_screen_layers_per_model():
    meta = _meta({"A/l0": True, "A/l1": False, "A/l2": False, "A/l3": False, "B/l0": False, "B/l1": False})
    screen = {"A": {"layers": ["l0", "l1", "l2", "l3", "l4"], "score_per_layer": [0.1, 5.0, 3.0, 1.0, 99.0],
                    "selected": ["l1", "l2", "l3", "l4"]},
              "B": {"layers": ["l0", "l1"], "score_per_layer": [1.0, 2.0], "selected": ["l0"]}}
    sel = B.select_targets(meta, screen)
    assert sel["targets"] == ["A/l0", "A/l1", "A/l2", "B/l0"]
    assert "A/l3" not in sel["reasons"] and "B/l1" not in sel["reasons"]
    assert any("pick 1 of 2" in r for r in sel["reasons"]["A/l1"])
    assert any("pick 2 of 2" in r for r in sel["reasons"]["A/l2"])
    assert sel["reasons"]["A/l0"] == ["admitted under the sec 42 gate"]
    assert any("not admitted" in r for r in sel["reasons"]["B/l0"])


def test_select_targets_refuses_a_meta_that_was_not_reevaluated():
    with pytest.raises(ValueError, match="admission_sec42"):
        B.select_targets({"A/l0": {}}, {})


def test_build_hardlinks_the_bulk_and_withholds_the_conclusions(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "activations.zarr" / "g").mkdir(parents=True)
    (src / "activations.zarr" / "g" / "0.0").write_bytes(b"chunk")
    (src / "sae" / "M").mkdir(parents=True)
    (src / "sae" / "M" / "layer.pt").write_bytes(b"primary")
    (src / "sae" / "M" / "layer@r1.pt").write_bytes(b"replicate")
    (src / "sae" / "M" / "layer_ablation.json").write_text("{}", encoding="utf-8")
    (src / "sae" / "meta.json").write_text("{}", encoding="utf-8")
    (src / "sae" / "concept_stage.json").write_text("{}", encoding="utf-8")
    (src / "l0").mkdir()
    (src / "l0" / "x.json").write_text("{}", encoding="utf-8")
    (src / "report").mkdir()
    (src / "hypotheses.json").write_text("{}", encoding="utf-8")
    (src / "run_manifest.json").write_text("{}", encoding="utf-8")
    B.build(src, dst)
    same = lambda a, b: (src / a).stat().st_ino == (dst / b).stat().st_ino  # noqa: E731
    assert same("activations.zarr/g/0.0", "activations.zarr/g/0.0")
    assert same("sae/M/layer.pt", "sae/M/layer.pt")
    assert not (dst / "sae" / "M" / "layer@r1.pt").exists()
    assert not (dst / "sae" / "M" / "layer_ablation.json").exists()
    assert not (dst / "sae" / "concept_stage.json").exists()
    assert not (dst / "report").exists() and not (dst / "hypotheses.json").exists()
    assert (dst / "sae" / "meta.json").stat().st_ino != (src / "sae" / "meta.json").stat().st_ino
    assert (dst / "l0" / "x.json").stat().st_ino != (src / "l0" / "x.json").stat().st_ino
    with pytest.raises(SystemExit):
        B.build(src, dst)


def test_select_writes_the_narrowed_meta_beside_the_full_one(tmp_path):
    run = tmp_path / "run"
    (run / "sae").mkdir(parents=True)
    (run / "layer_screen").mkdir()
    meta = _meta({"A/l0": True, "A/l1": False})
    (run / "sae" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run / "layer_screen" / "selection.json").write_text(json.dumps(
        {"A": {"layers": ["l0", "l1"], "score_per_layer": [1.0, 2.0], "selected": ["l0"]}}), encoding="utf-8")
    rec = B.select(run, per_model=1)
    assert list(json.loads((run / "sae" / "meta.json").read_text(encoding="utf-8"))) == ["A/l0"]
    assert json.loads((run / "sae" / "meta_full.json").read_text(encoding="utf-8")) == meta
    assert rec["not_selected"] == ["A/l1"] and rec["n_trained"] == 2 and rec["n_selected"] == 1
    with pytest.raises(SystemExit):
        B.select(run)


def test_run_completeness_uses_the_battery_ranking_and_names_the_fallback_elsewhere(tmp_path, monkeypatch):
    from tsfm_lens.sae import completeness as C
    from tsfm_lens.sae import ground_truth as G
    from tsfm_lens.sae import response as RS
    from tsfm_lens.sae import train as T
    run = tmp_path / "run"
    (run / "sae").mkdir(parents=True)
    (run / "sae" / "meta.json").write_text(json.dumps({"M/blocks.1": {}, "M/blocks.2": {}}), encoding="utf-8")
    for layer in ("blocks.1", "blocks.2"):
        ckpt = run / "sae" / "M" / f"{layer.replace('.', '_')}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"x")
    _write_artifact(run, "M", "blocks.1", {"attribution": {
        "status": "ok", "top_forecast_mean": [{"feature": 9, "score": 1.0}],
        "top_forecast_horizon_l1": [{"feature": 4, "score": 2.0}, {"feature": 6, "score": 1.0}]}})
    seen = {}
    monkeypatch.setattr(T, "load_sae_checkpoint", lambda path: SimpleNamespace(to=lambda d: "sae"))
    monkeypatch.setattr(T, "load_all_windows", lambda *a, **k: None)
    monkeypatch.setattr(RS, "alive_feature_mask", lambda *a, **k: np.ones(12, dtype=bool))
    monkeypatch.setattr(G, "encode_series_level", lambda *a, **k: np.zeros((3, 12)))

    def fake_target(cfg, adapter, layer, sae, data, device, **kw):
        seen[layer] = (kw["ranking"], kw["ranking_source"])
        return {"withheld": False}

    monkeypatch.setattr(C, "completeness_target", fake_target)
    hub = SimpleNamespace(get=lambda m: None, release=lambda m: None)
    cfg = SimpleNamespace(run=SimpleNamespace(keep_models_loaded=True), run_dir=lambda: run,
                          concepts=SimpleNamespace(completeness_set_sizes=None, completeness_n_random=2,
                                                   completeness_max_series=4, completeness_n_boot=10))
    C.run_completeness(cfg, run, hub, SimpleNamespace(), SimpleNamespace(n=3), "cpu")
    assert seen["blocks.1"] == ([4, 6], "attribution_horizon_l1")
    assert seen["blocks.2"] == (None, None)
