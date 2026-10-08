"""ROADMAP.md sec 41.1 V3 controls: transfer negative controls, window sensitivity,
per-data-role breakdown. Planted known answers with a decoy for each, plus one
mock-pipeline integration (twin run -> concepts stage -> report).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_transfer_fdr import _make_cfg, _make_store  # noqa: E402
from tsfm_lens.analysis.tier_breakdown import (_l0_by_role, _transfer_by_role,  # noqa: E402
                                               dominant_role, series_roles)
from tsfm_lens.analysis.window_sensitivity import (coarsen_windows,  # noqa: E402
                                                   _pooled_at_factor,
                                                   preflight_problem as window_problem,
                                                   run_transfer_window_sensitivity)
from tsfm_lens.config import ModelConfig, config_from_dict  # noqa: E402
from tsfm_lens.data import _assemble  # noqa: E402
from tsfm_lens.config import DataConfig  # noqa: E402
from tsfm_lens.extraction.alignment import align, pooling_matrix  # noqa: E402
from tsfm_lens.extraction.store import ActivationStore, save_meta  # noqa: E402
from tsfm_lens.sae.control_transfer import (input_feature_matrix,  # noqa: E402
                                            preflight_problem as control_problem,
                                            run_control_transfer)
from tsfm_lens.sae.models import TopKSAE  # noqa: E402
from tsfm_lens.sae.train import save_sae  # noqa: E402
from tsfm_lens.sae.transfer import run_atlas_transfer  # noqa: E402
from tsfm_lens.utils import load_json, save_json  # noqa: E402


# ---------------------------------------------------------------------------
# Exact averaging of adjacent windows == re-pooling from tokens
# ---------------------------------------------------------------------------

def _tiling_spans(width: int, ctx: int) -> np.ndarray:
    starts = np.arange(ctx // width) * width
    return np.stack([starts, starts + width], axis=1).astype(np.float64)


@pytest.mark.parametrize("token_width,ctx,w,factor", [(32, 128, 32, 2), (48, 192, 32, 3),
                                                       (16, 192, 32, 2), (96, 480, 32, 3)])
def test_adjacent_window_average_equals_repooling_from_tokens(token_width, ctx, w, factor):
    spans = _tiling_spans(token_width, ctx)
    hidden = torch.from_numpy(np.random.default_rng(0).normal(size=(3, len(spans), 5))
                              .astype(np.float32))
    fine = align(hidden, pooling_matrix(spans, ctx, w)).numpy()
    coarse_direct = align(hidden, pooling_matrix(spans, ctx, w * factor)).numpy()
    np.testing.assert_allclose(coarsen_windows(fine, factor), coarse_direct, atol=1e-5)


def test_exactness_fails_when_spans_overlap_decoy():
    """The identity needs token spans that tile the context. Overlapping spans
    (stride 16, width 32) give window row sums != width, so averaging is only
    approximate; the test documents the condition by showing the gap."""
    ctx, w = 128, 32
    starts = np.arange(0, ctx - 16, 16)
    spans = np.stack([starts, starts + 32], axis=1).astype(np.float64)
    hidden = torch.from_numpy(np.random.default_rng(1).normal(size=(2, len(spans), 4))
                              .astype(np.float32))
    fine = align(hidden, pooling_matrix(spans, ctx, w)).numpy()
    direct = align(hidden, pooling_matrix(spans, ctx, 64)).numpy()
    assert np.abs(coarsen_windows(fine, 2) - direct).max() > 1e-3


def test_mock_adapter_spans_are_exact():
    from tsfm_lens.models.mock import MockPatchAdapter
    data_cfg = DataConfig(source="smoke", context_len=128, horizon=32)
    adapter = MockPatchAdapter(ModelConfig(name="m", adapter="mock_patch"), data_cfg,
                               torch.device("cpu"), torch.float32)
    spans = adapter.token_time_spans()
    hidden = torch.from_numpy(np.random.default_rng(2).normal(size=(2, len(spans), 6))
                              .astype(np.float32))
    fine = align(hidden, pooling_matrix(spans, 128, 32)).numpy()
    np.testing.assert_allclose(coarsen_windows(fine, 2),
                               align(hidden, pooling_matrix(spans, 128, 64)).numpy(), atol=1e-5)


def test_coarsen_drops_trailing_windows_like_pooling_matrix():
    a = np.arange(5 * 2, dtype=np.float32).reshape(1, 5, 2)
    out = coarsen_windows(a, 2)
    assert out.shape == (1, 2, 2)
    np.testing.assert_allclose(out[0, 1], a[0, 2:4].mean(axis=0))
    with pytest.raises(ValueError):
        coarsen_windows(a, 6)


def test_window_preflight():
    cfg = config_from_dict({"data": {"context_len": 128}, "alignment": {"window": 32}})
    cfg.concepts.window_sensitivity = [64]
    assert window_problem(cfg) is None
    cfg.concepts.window_sensitivity = [48]
    assert "multiple" in window_problem(cfg)
    cfg.concepts.window_sensitivity = [32]
    assert "larger" in window_problem(cfg)
    cfg.concepts.window_sensitivity = [256]
    assert "exceeds" in window_problem(cfg)


def _identity_sae(tmp_path: Path, d: int) -> TopKSAE:
    sae = TopKSAE(d, d, d)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(d))
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    return sae


def test_pooled_features_at_coarser_window_are_encode_of_the_average(tmp_path):
    """Planted: encoder = ReLU(identity). Windows (+1, -1) pair up: native pooled
    feature is mean(relu) = 0.5, the pair-average has relu(0) = 0. A coarsening that
    ignored `factor` would return 0.5 at window 64."""
    store = ActivationStore.create(tmp_path / "a.zarr", n_series=3, n_windows=4, window=32,
                                   context_len=128)
    store.init_layer("m", "L0", 2)
    acts = np.zeros((3, 4, 2), dtype=np.float32)
    acts[:, 0::2, 0], acts[:, 1::2, 0] = 1.0, -1.0
    store.write_batch("m", "L0", 0, acts)
    sae = _identity_sae(tmp_path, 2)
    native = _pooled_at_factor(store, "m", "L0", sae, 1, torch.device("cpu"))
    coarse = _pooled_at_factor(store, "m", "L0", sae, 2, torch.device("cpu"))
    assert native[0, 0] == pytest.approx(0.5)
    assert coarse[0, 0] == pytest.approx(0.0)


def test_transfer_window_sensitivity_native_control_reproduces_stored(tmp_path):
    """Two models, features planted so concept 0 transfers; the native-window
    re-encode through the stored checkpoints must reproduce the stored
    atlas_transfer.json verdicts (agreement 1.0), and each coarser window is
    reported with its own counts."""
    models = ("m1", "m2")
    d, n_per, n_strata, n_win = 4, 30, 4, 4
    n = n_per * n_strata
    rng = np.random.default_rng(3)
    strata = np.repeat([f"s{i}" for i in range(n_strata)], n_per)
    special = rng.choice(n, size=8, replace=False)
    store = ActivationStore.create(tmp_path / "activations.zarr", n_series=n, n_windows=n_win,
                                   window=32, context_len=128)
    sae_meta = {}
    for m in models:
        store.init_layer(m, "L0", d)
        acts = rng.normal(0, 0.1, size=(n, n_win, d)).astype(np.float32)
        acts[special, :, 0] += 5.0
        store.write_batch(m, "L0", 0, acts)
        sae = _identity_sae(tmp_path, d)
        save_sae(sae, tmp_path / "sae" / m / "L0.pt")
        from tsfm_lens.sae.train import encode_and_persist_features
        encode_and_persist_features(store, m, "L0", sae, torch.device("cpu"))
        sae_meta[f"{m}/L0"] = {"features_persisted": True, "dict_size": d}
    save_meta(tmp_path, pd.DataFrame({"archetype": strata, "family": strata}))
    save_json(tmp_path / "sae" / "meta.json", sae_meta)
    cfg = _make_cfg(tmp_path, models, top_k=8)
    atlas = {"rows": [{"model": m, "layer": "L0", "feature": 0, "concept": 0} for m in models],
             "concepts": [{"concept": 0}]}
    run_atlas_transfer(tmp_path, atlas, cfg)
    store_r = ActivationStore(tmp_path / "activations.zarr", mode="r")
    out = run_transfer_window_sensitivity(cfg, store_r, tmp_path, atlas, torch.device("cpu"), [2])
    assert out["windows"]["32"]["agreement_with_stored_reciprocal"] == 1.0
    assert out["windows"]["32"]["n_reciprocal"] == out["stored_native"]["n_reciprocal"] == 2
    assert set(out["windows"]) == {"32", "64"}
    assert out["windows"]["64"]["n_tests"] == 2


# ---------------------------------------------------------------------------
# Control destinations
# ---------------------------------------------------------------------------

def _control_run(root: Path, name: str, n_feat: int, shared_idx, n: int, seed: int,
                 random_init=True, twin_of_checkpoint="ckpt", meta_like=None):
    ctl = root / name
    store = ActivationStore.create(ctl / "activations.zarr", n_series=n, n_windows=4, window=32,
                                   context_len=128)
    rng = np.random.default_rng(seed)
    pooled = rng.normal(0, 0.1, size=(n, n_feat))
    if shared_idx is not None:
        pooled[shared_idx, 0] += 5.0
    store.init_sae_layer(name, "L0", n_feat)
    store.write_sae_batch(name, "L0", 0, np.repeat(pooled[:, None, :], 4, axis=1))
    save_meta(ctl, meta_like)
    save_json(ctl / "sae" / "meta.json", {f"{name}/L0": {"features_persisted": True}})
    import yaml
    (ctl / "config_resolved.yaml").write_text(yaml.safe_dump(
        {"models": [{"name": name, "adapter": "mock", "checkpoint": twin_of_checkpoint,
                     "random_init": random_init}]}), encoding="utf-8")
    return ctl


def _main_run(tmp_path, k_signal=8):
    run_dir, special_shared, by_model = _make_store(tmp_path, ("m1", "m2"), n_feat=6, seed=42,
                                                    k_signal=k_signal)
    from tsfm_lens.extraction.store import load_meta
    m0 = load_meta(run_dir)
    save_meta(run_dir, m0.assign(series_id=[f"s{i}" for i in range(len(m0))]))
    cfg = _make_cfg(run_dir, ("m1", "m2"), top_k=8)
    cfg.models[0].checkpoint = "ckpt"
    save_json(run_dir / "sae" / "meta.json", {"m1/L0": {"features_persisted": True},
                                              "m2/L0": {"features_persisted": True}})
    atlas = {"rows": [{"model": m, "layer": "L0", "feature": f, "concept": f}
                      for m in ("m1", "m2") for f in (0, 1)],
             "concepts": [{"concept": 0}, {"concept": 1}]}
    run_atlas_transfer(run_dir, atlas, cfg)
    from tsfm_lens.extraction.store import load_meta
    return run_dir, cfg, atlas, special_shared, load_meta(run_dir)


def test_negative_control_noise_twin_fails_and_leaky_twin_passes_decoy(tmp_path):
    """A twin whose dictionary has NO feature on the planted series must pass nothing
    (the negative control working); a decoy twin that leaks the shared concept's series
    into its feature 0 must pass concept 0 and not concept 1, so the module can tell
    the two apart rather than reporting 0 for everything."""
    run_dir, cfg, atlas, shared, meta = _main_run(tmp_path / "main")
    n = len(meta)
    noise = _control_run(tmp_path, "twin_noise", 6, None, n, 1, meta_like=meta)
    leaky = _control_run(tmp_path, "twin_leaky", 6, shared, n, 2, meta_like=meta)
    cfg.concepts.negative_control_runs = {"twin_noise": str(noise), "twin_leaky": str(leaky)}
    assert control_problem(cfg) is None
    out = run_control_transfer(run_dir, atlas, cfg, data=None)
    assert out["controls"]["twin_noise"]["n_reciprocal"] == 0
    assert out["controls"]["twin_noise"]["n_tests"] == 4
    assert out["controls"]["twin_noise"]["twin_of"] == "m1"
    leaky_tests = {(t["src_model"], t["concept"]): t for t in out["tests"]["twin_leaky"]}
    assert leaky_tests[("m1", 0)]["reciprocal"] and leaky_tests[("m2", 0)]["reciprocal"]
    assert not leaky_tests[("m1", 1)]["reciprocal"]
    assert out["controls"]["twin_leaky"]["excluding_twin_parent_as_source"]["n_tests"] == 2
    assert out["real_destinations"]["m1"]["n_reciprocal"] == 1
    saved = load_json(run_dir / "sae" / "control_transfer.json")
    assert saved["controls"]["twin_leaky"]["n_reciprocal"] == 2


def test_control_refusals(tmp_path):
    run_dir, cfg, atlas, shared, meta = _main_run(tmp_path / "main")
    n = len(meta)
    trained = _control_run(tmp_path, "trained", 6, None, n, 1, random_init=False, meta_like=meta)
    cfg.concepts.negative_control_runs = {"trained": str(trained)}
    with pytest.raises(ValueError, match="random_init"):
        run_control_transfer(run_dir, atlas, cfg, data=None)
    other = meta.assign(series_id=[f"x{i}" for i in range(n)])
    mism = _control_run(tmp_path, "mism", 6, None, n, 1, meta_like=other)
    cfg.concepts.negative_control_runs = {"mism": str(mism)}
    with pytest.raises(ValueError, match="series"):
        run_control_transfer(run_dir, atlas, cfg, data=None)
    cfg.concepts.negative_control_runs = {"gone": str(tmp_path / "nope")}
    assert "run its extract and sae stages" in control_problem(cfg)
    cfg.concepts.negative_control_runs = {"m1": str(trained)}
    assert "collides" in control_problem(cfg)


def test_input_feature_matrix_shape_and_degenerate_rows():
    rng = np.random.default_rng(0)
    ctx = rng.normal(size=(5, 128))
    ctx[0] = 3.0
    X, names = input_feature_matrix(ctx, 32)
    assert X.shape == (5, len(names)) and np.isfinite(X).all()
    assert np.allclose(X[0], 0.0, atol=1e-9) or np.isfinite(X[0]).all()
    trend = np.arange(128.0)[None, :] + rng.normal(0, 0.01, size=(1, 128))
    Xt, _ = input_feature_matrix(np.vstack([trend, ctx[1:2]]), 32)
    j = names.index("trend_r2")
    assert Xt[0, j] > 0.99 > Xt[1, j]


def test_input_feature_control_separates_planted_input_structure(tmp_path):
    """Planted: the shared concept's top series are exactly the series with a strong
    trend in their raw context, so the input-feature destination must pass concept 0.
    Decoy: concept 1's series (m1-only special set) are unrelated to any input
    statistic and must not pass."""
    run_dir, cfg, atlas, shared, meta = _main_run(tmp_path / "main")
    n = len(meta)
    rng = np.random.default_rng(5)
    ctx = rng.normal(size=(n, 128))
    ctx[shared] += np.linspace(0, 40, 128)[None, :]

    class _Data:
        def contexts(self):
            return ctx

    cfg.concepts.input_feature_control = True
    out = run_control_transfer(run_dir, atlas, cfg, data=_Data())
    tests = {(t["src_model"], t["concept"]): t for t in out["tests"]["input_features"]}
    assert tests[("m1", 0)]["reciprocal"] and tests[("m2", 0)]["reciprocal"]
    assert not tests[("m1", 1)]["reciprocal"] and not tests[("m2", 1)]["reciprocal"]
    assert out["controls"]["input_features"]["n_features"]["input_features/context_stats"] > 10


# ---------------------------------------------------------------------------
# Roles and the per-tier breakdown
# ---------------------------------------------------------------------------

def _rows(roles):
    return [{"values": np.sin(np.arange(60) / 3.0) + i, "sample_id": f"s{i}", "family": "f",
             **({"role": r} if r else {})} for i, r in enumerate(roles)]


def test_meta_role_column_only_when_rows_carry_roles():
    cfg = DataConfig(source="x", context_len=40, horizon=10)
    with_roles = _assemble(_rows(["synthetic", "real_derived", None]), cfg)
    assert with_roles.meta["role"].tolist() == ["synthetic", "real_derived", "unknown"]
    without = _assemble(_rows([None, None, None]), cfg)
    assert "role" not in without.meta.columns
    assert list(without.meta.columns) == ["series_id", "family", "tier", "archetype", "generator"]


def test_dominant_role_threshold_and_mixed():
    roles = np.array(["synthetic"] * 6 + ["real_derived"] * 4 + ["external_real"] * 2)
    assert dominant_role(np.array([0, 1, 2, 3]), roles) == ("synthetic", 1.0)
    assert dominant_role(np.array([0, 1, 6, 7]), roles)[0] == "synthetic"  # 0.5 meets the bar
    assert dominant_role(np.array([0, 1, 6, 7, 10]), roles)[0] == "mixed"
    assert dominant_role(np.array([], dtype=int), roles) == (None, 0.0)


def test_series_roles_falls_back_to_tier():
    meta = pd.DataFrame({"tier": ["a", "b"]})
    assert series_roles(meta)[1] == "tier"
    assert list(series_roles(meta)[0]) == ["a", "b"]
    meta["role"] = ["x", "y"]
    assert series_roles(meta)[1] == "role" and list(series_roles(meta)[0]) == ["x", "y"]


def test_l0_by_role_reads_metrics_and_does_not_pool(tmp_path):
    meta = pd.DataFrame({"series_id": ["a", "b", "c", "d"],
                         "role": ["synthetic", "synthetic", "real_derived", "real_derived"]})
    (tmp_path / "l0").mkdir()
    pd.DataFrame({"model": "M", "series_id": ["a", "b", "c", "d"], "mase": [1.0, 3.0, 10.0, 30.0],
                  "mase_reliable": [True, True, True, False], "smape": 0.1, "pinball": 0.2}
                 ).to_parquet(tmp_path / "l0" / "metrics.parquet")
    out = _l0_by_role(tmp_path, meta, meta["role"].to_numpy())
    by = {r["role"]: r for r in out["rows"]}
    assert by["synthetic"]["mase"] == pytest.approx(2.0)
    assert by["real_derived"]["mase"] == pytest.approx(10.0)  # unreliable series excluded
    assert by["real_derived"]["n_series"] == 2 and by["real_derived"]["n_mase_reliable"] == 1


def test_transfer_by_role_uses_source_top_series_not_corpus_majority(tmp_path):
    """Planted: 75% of the corpus is synthetic, but concept 1's top series are all
    real_derived, concept 0's all synthetic. A labelling by corpus majority would call
    both synthetic."""
    run_dir, shared, by_model = _make_store(tmp_path, ("m1", "m2"), n_feat=5, seed=7, k_signal=8)
    n = 120
    roles = np.array(["synthetic"] * n, dtype=object)
    roles[by_model["m1"]] = "real_derived"
    save_json(run_dir / "sae" / "meta.json", {"m1/L0": {"features_persisted": True},
                                              "m2/L0": {"features_persisted": True}})
    cfg = _make_cfg(run_dir, ("m1", "m2"), top_k=8)
    atlas = {"rows": [{"model": "m1", "layer": "L0", "feature": f, "concept": f}
                      for f in (0, 1)] +
                     [{"model": "m2", "layer": "L0", "feature": f, "concept": f} for f in (0, 1)],
             "concepts": [{"concept": 0}, {"concept": 1}]}
    save_json(run_dir / "sae" / "concept_atlas.json", atlas)
    run_atlas_transfer(run_dir, atlas, cfg)
    out = _transfer_by_role(run_dir, cfg, roles)
    groups = out["by_source_role"]
    assert groups["synthetic"]["n_tests"] == 3 and groups["real_derived"]["n_tests"] == 1
    assert groups["synthetic"]["n_reciprocal"] == 2
    assert groups["real_derived"]["n_reciprocal"] == 0


# ---------------------------------------------------------------------------
# Pipeline integration on mocks, legacy byte-identity, report
# ---------------------------------------------------------------------------

def test_new_fields_leave_every_fingerprint_unchanged():
    from tests.test_ablation_null_default import OLD_FINGERPRINTS  # noqa: F401
    from tsfm_lens.manifest import fingerprint_stage, resolve_config_keys
    from tsfm_lens.pipeline import _stages
    from tests.test_smoke import build_config
    base = config_from_dict(build_config("x"))
    new = config_from_dict(build_config("x"))
    new.concepts.negative_control_runs = {"t": "d"}
    new.concepts.window_sensitivity = [64]
    new.concepts.input_feature_control = True
    fp = {s.name: fingerprint_stage(resolve_config_keys(c, s.config_keys), {})
          for s in _stages() for c in (base,)}
    fp2 = {s.name: fingerprint_stage(resolve_config_keys(new, s.config_keys), {})
           for s in _stages()}
    changed = sorted(k for k in fp if fp[k] != fp2[k])
    assert changed == ["concepts"]
    new.concepts.negative_control_runs = None
    new.concepts.window_sensitivity = None
    new.concepts.input_feature_control = False
    new.concepts.tier_breakdown = True
    assert {s.name: fingerprint_stage(resolve_config_keys(new, s.config_keys), {})
            for s in _stages()} == fp


@pytest.fixture(scope="module")
def pipeline_runs():
    from tests.test_concept_stage import _cfg
    from tsfm_lens.pipeline import run_pipeline
    out = tempfile.mkdtemp()
    main = _cfg(out, "v3_main")
    run_pipeline(main, stages=["extract"])
    store = ActivationStore(main.run_dir() / "activations.zarr", mode="r")
    main.sae.targets = [{"model": "patchy", "layer": store.layers("patchy")[-1]},
                        {"model": "steppy", "layer": store.layers("steppy")[-1]}]
    twin = _cfg(out, "v3_twin")
    twin.models = [ModelConfig(name="patchy_random", adapter="mock_patch", batch_size=64,
                               random_init=True)]
    run_pipeline(twin, stages=["extract"])
    twin.sae.targets = [{"model": "patchy_random", "layer": store.layers("patchy")[-1]}]
    run_pipeline(twin, stages=["sae"])
    main.concepts.negative_control_runs = {"patchy_random": str(twin.run_dir())}
    main.concepts.input_feature_control = True
    main.concepts.window_sensitivity = [64]
    main.concepts.tier_breakdown = True
    main.concepts.atlas_min_cosine = 0.2
    main.concepts.atlas_min_members = 2
    run_pipeline(main, stages=["sae", "concepts"])
    return main, twin


def test_twin_run_is_excluded_from_main_run(pipeline_runs):
    main, twin = pipeline_runs
    rd = main.run_dir()
    assert "patchy_random" not in ActivationStore(rd / "activations.zarr", mode="r").models()
    assert "patchy_random" not in json.dumps(load_json(rd / "sae" / "meta.json"))
    assert "patchy_random" not in json.dumps(load_json(rd / "sae" / "atlas_transfer.json"))
    assert not (twin.run_dir() / "l1").exists()
    assert (twin.run_dir() / "sae" / "meta.json").exists()


def test_stage_record_and_artifacts(pipeline_runs):
    main, _ = pipeline_runs
    rd = main.run_dir()
    rec = load_json(rd / "sae" / "concept_stage.json")
    assert rec["control_transfer"]["status"] == "ran"
    assert set(rec["control_transfer"]["controls"]) == {"patchy_random", "input_features"}
    ct = load_json(rd / "sae" / "control_transfer.json")
    n_real = sum(r["n_tests"] for r in ct["real_destinations"].values())
    assert n_real == load_json(rd / "sae" / "concept_stage.json")["atlas_transfer"]["n_tests"]
    ws = load_json(rd / "sae" / "window_sensitivity.json")
    assert ws["windows"] == [32, 64]
    assert ws["transfer"]["windows"]["32"]["agreement_with_stored_reciprocal"] == 1.0
    assert set(ws["l1"]["windows"]) == {"32", "64"}
    assert ws["l1"]["windows"]["32"]["patchy__steppy"]["peak_cka"] > 0


def test_report_renders_controls_section(pipeline_runs):
    from tsfm_lens.pipeline import run_pipeline
    main, _ = pipeline_runs
    run_pipeline(main, stages=["report"], allow_stale=True)
    html = (main.run_dir() / "report.html").read_text(encoding="utf-8")
    assert "Transfer negative controls" in html
    assert "Window-size sensitivity" in html
    assert "Breakdown by data role" in html
    assert html.count("patchy_random") >= 1
    findings = load_json(main.run_dir() / "report" / "findings.json")["findings"]
    assert any("Transfer negative controls" in f["text"] for f in findings)
