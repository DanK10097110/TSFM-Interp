"""Silent-NaN gaps closed after the L0 / skip-lens guards (FINDINGS PM-16).

Planted answers. A constant depth profile has no rank order, so its Spearman
rho is undefined and must be recorded as `None` with a reason -- never 0.0,
which reads as "no agreement". A forecast NaN on a known set of series must be
recorded once at write time (store attrs), counted (not skipped quietly) in the
ablation battery and the shared-input agreement, and dropped-and-counted by the
readers that would otherwise let one NaN poison every head or correlation.
Legacy artifacts (no NaN anywhere) must not gain any new key.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_agreement import _FakeStore, _N, _corpus, _preds
from tests.test_sae_ablation_fingerprint import (DICT, HORIZON, N_SERIES, ROW_GAIN, _Adapter,
                                                 _run, wired)  # noqa: F401
from tsfm_lens.analysis.agreement import agreement_reliability
from tsfm_lens.analysis.attention import drop_nonfinite_clean
from tsfm_lens.analysis.l3_perturbation import _agreement_with_ci, _fingerprint_agreement
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.sae import response as R
from tsfm_lens.sae.concept_profiles import _pair_agreement
from tsfm_lens.sae.shared_input_agreement import nonfinite_row_record


NAMES = ["c0", "c1"]


def _profiles(constant_second: bool):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 2))
    b = a + rng.normal(scale=0.1, size=(6, 2))
    if constant_second:
        a[:, 1] = 1.0
    return a, b


def test_l3_undefined_rho_is_none_with_a_reason_not_zero():
    a, b = _profiles(constant_second=True)
    out = _fingerprint_agreement(a, b, NAMES)
    assert out["per_corruption"]["c1"] is None
    assert "constant" in out["undefined"]["c1"]
    assert out["per_corruption"]["c0"] > 0.5
    assert out["most_divergent"] == "c0"


def test_l3_defined_rho_adds_no_key_and_keeps_legacy_shape():
    a, b = _profiles(constant_second=False)
    out = _fingerprint_agreement(a, b, NAMES)
    assert "undefined" not in out
    assert list(out) == ["per_corruption", "overall", "most_divergent",
                         "overlap_fraction", "unmatched"]
    assert all(isinstance(v, float) for v in out["per_corruption"].values())


def test_l3_bootstrap_ci_of_an_undefined_rho_is_none_and_the_draws_are_counted():
    a, b = _profiles(constant_second=True)
    rng = np.random.default_rng(1)
    psa = a[None] + rng.normal(scale=0.01, size=(10, 6, 2))
    psb = b[None] + rng.normal(scale=0.01, size=(10, 6, 2))
    psa[:, :, 1] = 1.0
    cfg = SimpleNamespace(stats=SimpleNamespace(enabled=True, n_boot=40, n_boot_heavy=40, ci=0.95),
                          run=SimpleNamespace(seed=0))
    out = _agreement_with_ci(cfg, psa, psb, NAMES)
    c1 = out["per_corruption"]["c1"]
    assert c1 == {"value": None, "lo": None, "hi": None}
    assert out["n_undefined_resamples"]["c1"] == 40
    assert out["per_corruption"]["c0"]["lo"] is not None
    assert "c1" in out["undefined"]


def test_report_renders_an_undefined_value_as_not_defined_never_zero():
    from tsfm_lens.report.report import _ci_str, _undefined_agreement_note
    assert _ci_str({"value": None, "lo": None, "hi": None}) == "not defined"
    note = _undefined_agreement_note([{"a": "A", "b": "B", "agreement": {
        "undefined": {"c1": "a depth profile is constant"}}}])
    assert "Not defined" in note and "never as zero" in note and "c1" in note
    assert _undefined_agreement_note([{"a": "A", "b": "B", "agreement": {}}]) == ""


def test_hypotheses_do_not_register_an_undefined_l3_claim(tmp_path):
    from tsfm_lens.analysis.hypotheses import _l3_entries
    (tmp_path / "l3").mkdir()
    (tmp_path / "l3" / "meta.json").write_text(json.dumps({"agreement": {"per_corruption": {
        "c0": {"value": 0.4, "lo": 0.1, "hi": 0.7},
        "c1": {"value": None, "lo": None, "hi": None}}}}), encoding="utf-8")
    assert [e["corruption"] for e in _l3_entries(tmp_path)] == ["c0"]


def test_concept_profile_undefined_rho_is_none_with_a_reason():
    s_a, s_b = np.ones(20), np.arange(20, dtype=float)
    strata = np.array(["x"] * 20)
    by_stratum = {"x": np.arange(20)}
    res = _pair_agreement(s_a, s_b, np.arange(3), np.arange(3), strata, by_stratum, 0, 20)
    assert res["rho"] is None and res["p_uncond"] is None
    assert "not defined" in res["rho_undefined_reason"]
    ok = _pair_agreement(s_b, s_b[::-1].copy(), np.arange(3), np.arange(3), strata,
                         by_stratum, 0, 20)
    assert ok["rho"] is not None and "rho_undefined_reason" not in ok


class _NanAdapter(_Adapter):
    """The planted stub, but row 4's forecast is NaN in every pass."""

    def predict(self, contexts, horizon, quantiles):
        out = super().predict(contexts, horizon, quantiles)
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        out["point"][rows == 4] = np.nan
        return out


def test_battery_counts_the_nonfinite_rows_it_skips(wired):
    out = _run(_NanAdapter(wired._holder))
    real, sham = out["candidates"]
    assert real["n_nonfinite_forecast_rows"] == 1
    assert real["channels"]["level"]["n_nonfinite_rows_skipped"] == 1
    assert real["shape_channels"]["flatness"]["n_nonfinite_rows_skipped"] == 1
    assert "n_nonfinite_forecast_rows" not in sham
    assert "n_nonfinite_rows_skipped" not in sham["channels"]["level"]
    assert out["n_nonfinite_forecast_rows_total"] == 1
    assert out["n_candidates_with_nonfinite_rows"] == 1
    assert real["channels"]["level"]["effect"] > 0 and np.isfinite(real["channels"]["level"]["effect"])


def test_battery_without_nonfinite_rows_gains_no_key(wired):
    out = _run(wired)
    assert "n_nonfinite_forecast_rows_total" not in out
    for c in out["candidates"]:
        assert "n_nonfinite_forecast_rows" not in c
        assert not any("n_nonfinite_rows_skipped" in v for v in c["channels"].values())


def test_level_share_ignores_nonfinite_rows_and_does_not_blame_a_zero_denominator():
    steered = np.ones((3, 4))
    baseline = np.zeros((3, 4))
    steered[1] = np.nan
    v, reason = R.level_share(steered, baseline)
    assert v == pytest.approx(1.0) and reason == ""
    v, reason = R.level_share(np.full((2, 4), np.nan), np.zeros((2, 4)))
    assert v is None and "non-finite" in reason


def _stats(level):
    return {"level": {"available": True, "delta": np.asarray(level, dtype=float)}}


def test_shared_input_agreement_counts_nonfinite_rows_per_side_and_null_draws():
    clean = _stats([1.0, 2.0, 3.0])
    bad = _stats([1.0, np.nan, 3.0])
    rec = nonfinite_row_record(clean, bad, [(clean, None), (bad, None)], [(clean, None)],
                               [(bad, None)], [])
    assert rec == {"src_rows": 0, "dst_rows": 1, "src_matched_null_draws": 1,
                   "dst_matched_null_draws": 0, "src_own_null_draws": 1,
                   "dst_own_null_draws": 0}
    assert nonfinite_row_record(clean, clean, [(clean, None)], [(clean, None)], [], []) == {}


def _store(tmp_path, n=8):
    return ActivationStore.create(tmp_path / "a.zarr", n_series=n, n_windows=2, window=32,
                                  context_len=64)


def test_write_predictions_records_nonfinite_series_once_and_leaves_values_alone(tmp_path):
    store = _store(tmp_path)
    point = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    quants = np.zeros((8, 4, 3), dtype=np.float32)
    point[2, 1] = np.nan
    quants[5, 0, 2] = np.inf
    store.write_predictions("m", point, quants)
    rec = dict(store.root["pred/m"].attrs["nonfinite_predictions"])
    assert rec == {"n_series": 2, "n_total": 8, "rows": [2, 5]}
    got = store.load_predictions("m")
    assert np.isnan(got["point"][2, 1]) and np.isinf(got["quantiles"][5, 0, 2])
    assert store.nonfinite_prediction_rows("m").tolist() == [2, 5]
    store.write_predictions("m", np.nan_to_num(point), np.nan_to_num(quants, posinf=0.0))
    assert "nonfinite_predictions" not in store.root["pred/m"].attrs
    assert store.nonfinite_prediction_rows("m").size == 0


def test_write_predictions_of_finite_forecasts_writes_no_attrs(tmp_path):
    store = _store(tmp_path)
    store.write_predictions("m", np.zeros((8, 4), np.float32), np.zeros((8, 4, 3), np.float32))
    assert dict(store.root["pred/m"].attrs) == {}


def test_nonfinite_prediction_rows_answers_for_a_store_without_the_record(tmp_path):
    store = _store(tmp_path)
    point = np.zeros((8, 4), np.float32)
    point[3] = np.nan
    store.write_predictions("m", point, np.zeros((8, 4, 3), np.float32))
    del store.root["pred/m"].attrs["nonfinite_predictions"]
    assert store.nonfinite_prediction_rows("m").tolist() == [3]


def test_agreement_drops_and_counts_series_with_a_nonfinite_stored_forecast():
    ctx, tgt = _corpus()
    rng = np.random.default_rng(3)
    pa = tgt + rng.normal(0, 0.3, tgt.shape)
    pb = tgt + rng.normal(0, 0.3, tgt.shape)
    w = np.full(_N, 0.5)
    clean = agreement_reliability(_FakeStore({"a": _preds(pa, w), "b": _preds(pb, w)}),
                                  "a", "b", ctx, tgt, n_boot=50)
    assert "n_series_nonfinite_dropped" not in clean
    pa_bad = pa.copy()
    pa_bad[[4, 9]] = np.nan
    out = agreement_reliability(_FakeStore({"a": _preds(pa_bad, w), "b": _preds(pb, w)}),
                                "a", "b", ctx, tgt, n_boot=50)
    assert out["n_series_nonfinite_dropped"] == 2 and out["n_series"] == _N - 2
    assert np.isfinite(out["mase"]["a"])


def test_attention_helper_drops_series_with_a_nonfinite_clean_forecast():
    f = np.ones((5, 3))
    f[1] = np.nan
    fams = np.array(["a", "b", "b", "c", "c"])
    fc, ctx, tgt, sc, fm, fl, n = drop_nonfinite_clean(
        f, np.arange(5)[:, None], np.arange(5)[:, None], np.arange(5.0), fams, ["a", "b", "c"])
    assert n == 1 and len(fc) == 4 and fl == ["a", "b", "c"]
    fc, ctx, tgt, sc, fm, fl, n = drop_nonfinite_clean(
        np.where((fams == "a")[:, None], np.nan, 1.0), np.arange(5)[:, None],
        np.arange(5)[:, None], np.arange(5.0), fams, ["a", "b", "c"])
    assert n == 1 and fl == ["b", "c"]
    same = drop_nonfinite_clean(np.ones((5, 3)), 0, 1, 2, fams, ["a"])
    assert same[-1] == 0 and same[1] == 0


@pytest.fixture(scope="module")
def run_with_nan_forecasts(tmp_path_factory):
    from tests.test_smoke import build_config
    from tsfm_lens.analysis import l0_behavioral as l0
    from tsfm_lens.config import config_from_dict
    from tsfm_lens.pipeline import run_pipeline

    real = l0._predict_all

    def poisoned(adapter, contexts, horizon, quantiles):
        point, quants = real(adapter, contexts, horizon, quantiles)
        if adapter.name == "patchy":
            point, quants = point.copy(), quants.copy()
            point[::5] = np.nan
            quants[::5] = np.nan
        return point, quants

    l0._predict_all = poisoned
    try:
        cfg = config_from_dict(build_config(str(tmp_path_factory.mktemp("nanrun"))))
        run_pipeline(cfg, stages=["extract", "l0", "l3", "attention", "report"])
    finally:
        l0._predict_all = real
    return cfg


def test_pipeline_records_the_nonfinite_series_at_write_time(run_with_nan_forecasts):
    cfg = run_with_nan_forecasts
    store = ActivationStore(cfg.run_dir() / "activations.zarr", mode="r")
    rec = store.root["pred/patchy"].attrs["nonfinite_predictions"]
    assert rec["rows"] == list(range(0, rec["n_total"], 5))
    assert "nonfinite_predictions" not in store.root["pred/steppy"].attrs
    assert np.isnan(store.load_predictions("patchy")["point"][0]).all()


def test_pipeline_l3_drops_and_lists_the_nonfinite_behavior_series(run_with_nan_forecasts):
    cfg = run_with_nan_forecasts
    meta = json.loads((cfg.run_dir() / "l3" / "meta.json").read_text(encoding="utf-8"))
    assert len(meta["behavior_nonfinite_dropped"]["patchy"]) > 0
    assert "steppy" not in meta["behavior_nonfinite_dropped"]
    assert all(np.isfinite(v["value"]) for v in meta["behavior_ci"]["patchy"].values())
    html = (cfg.run_dir() / "report.html").read_text(encoding="utf-8")
    assert "behavior_nonfinite_dropped" in html


def test_pipeline_attention_ablation_survives_and_counts_the_dropped_series(run_with_nan_forecasts):
    cfg = run_with_nan_forecasts
    meta = json.loads((cfg.run_dir() / "attention" / "meta.json").read_text(encoding="utf-8"))
    abl = meta["patchy"]["ablation"]
    assert abl.get("status") != "error", abl
    assert abl["n_series_nonfinite_dropped"] > 0
    arrays = np.load(cfg.run_dir() / "attention" / "arrays.npz")
    assert np.isfinite(arrays["head_delta_patchy"]).all()
    assert "n_series_nonfinite_dropped" not in meta["steppy"]["ablation"]


def test_pipeline_without_nan_adds_no_new_keys():
    from tests.test_smoke import build_config
    from tsfm_lens.config import config_from_dict
    from tsfm_lens.pipeline import run_pipeline
    cfg = config_from_dict(build_config(tempfile.mkdtemp()))
    run_pipeline(cfg, stages=["extract", "l0", "l3"])
    store = ActivationStore(cfg.run_dir() / "activations.zarr", mode="r")
    for m in ("patchy", "steppy"):
        assert dict(store.root[f"pred/{m}"].attrs) == {}
    meta = json.loads((cfg.run_dir() / "l3" / "meta.json").read_text(encoding="utf-8"))
    assert "behavior_nonfinite_dropped" not in meta
    assert "undefined" not in meta["agreement"]
    assert "n_undefined_resamples" not in meta["agreement"]


def test_rendered_report_shows_an_undefined_agreement_as_not_defined(run_with_nan_forecasts):
    """Rendered-HTML check: an undefined per-corruption rho (and undefined
    pooled rho) reads 'not defined', and produces neither a zero bar label nor
    a failed section."""
    from tsfm_lens.pipeline import run_pipeline
    cfg = run_with_nan_forecasts
    path = cfg.run_dir() / "l3" / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    names = meta["corruptions"]

    def _undefine(ag):
        ag["per_corruption"][names[0]] = {"value": None, "lo": None, "hi": None}
        ag["overall"] = {"value": None, "lo": None, "hi": None}
        ag["most_divergent"] = None
        ag["undefined"] = {names[0]: "a depth profile is constant, so its rank "
                                      "correlation is undefined"}
        ag["n_undefined_resamples"] = {names[0]: 7}

    _undefine(meta["agreement"])
    for entry in meta["pairwise"]:
        _undefine(entry["agreement"])
    path.write_text(json.dumps(meta), encoding="utf-8")
    run_pipeline(cfg, stages=["report"], force=["report"])
    html = (cfg.run_dir() / "report.html").read_text(encoding="utf-8")
    assert "Not defined (shown as a gap, never as zero agreement)" in html
    assert "a depth profile is constant" in html
    assert "section failed" not in html.lower()
    findings = json.loads((cfg.run_dir() / "report" / "findings.json").read_text(encoding="utf-8"))
    findings = findings["findings"] if isinstance(findings, dict) else findings
    assert not any("fingerprint agreement" in f["text"] and "nan" in f["text"].lower()
                   for f in findings)


def _ablation_artifact(tmp_path, nonfinite: bool):
    d = tmp_path / "sae" / "M"
    d.mkdir(parents=True)
    cands = [{"feature": 0, "n_top_series": 3, "n_rows_scored": 3, "scorable": True},
             {"feature": 1, "n_top_series": 3, "n_rows_scored": 3, "scorable": True}]
    if nonfinite:
        cands[0]["n_nonfinite_forecast_rows"] = 2
    (d / "blocks_0_ablation.json").write_text(json.dumps({
        "top_k_series": 3, "series_per_chunk_cap": 3, "withheld": False,
        "candidates": cands}), encoding="utf-8")


def test_ablation_report_section_totals_the_nonfinite_rows_only_when_present(tmp_path):
    from tsfm_lens.report import sae_concepts
    _ablation_artifact(tmp_path, nonfinite=True)
    html = sae_concepts.row_coverage_block(tmp_path)
    assert "Non-finite forecasts: 2 (feature, series) row(s) across 1 target(s)" in html
    clean = tmp_path / "clean"
    clean.mkdir()
    _ablation_artifact(clean, nonfinite=False)
    assert "Non-finite forecasts" not in sae_concepts.row_coverage_block(clean)


def test_shared_input_section_totals_the_nonfinite_rows_only_when_present():
    from tsfm_lens.report.model_comparison import _shared_input_pair_table
    base = {"pair_verdict_counts": {"A->B": {"same causal effect": 1}},
            "params": {"verdicts": ["same causal effect"]}}
    assert "Non-finite forecasts" not in _shared_input_pair_table(base)
    html = _shared_input_pair_table({**base, "n_nonfinite_rows_total": 5,
                                     "n_tests_with_nonfinite_rows": 2})
    assert "Non-finite forecasts: 5 shared-series row(s) across 2 test(s)" in html


def test_error_fingerprint_excludes_and_counts_series_with_a_nonfinite_stored_forecast():
    from tests.test_error_fingerprint import _FakeStore as _EStore
    from tests.test_error_fingerprint import _corpus as _ecorpus
    from tsfm_lens.analysis.error_fingerprint import error_fingerprint
    contexts, targets = _ecorpus()
    rng = np.random.default_rng(2)
    pa = targets + rng.normal(0, 1, targets.shape)
    pb = targets + rng.normal(0, 1, targets.shape)
    clean = error_fingerprint(_EStore({"a": pa, "b": pb}), "a", "b", contexts, targets, n_boot=50)
    assert "n_excluded_nonfinite_forecast" not in clean
    pa_bad = pa.copy()
    pa_bad[[3, 50, 51]] = np.nan
    out = error_fingerprint(_EStore({"a": pa_bad, "b": pb}), "a", "b", contexts, targets,
                            n_boot=50)
    assert out["n_excluded_nonfinite_forecast"] == 3
    assert out["n_series"] == clean["n_series"] - 3
    assert np.isfinite(out["shape"]["residual_corr"])


def test_spec_curve_recompute_drops_series_with_a_nonfinite_stored_forecast(tmp_path):
    from tests.test_spec_curve import (_FakeData, _cfg_two_models, _planted_l0_corpus)
    from tsfm_lens.analysis import spec_curve as sc
    contexts, targets, meta, families, pa, qa, pb, qb = _planted_l0_corpus()
    pa = pa.copy()
    pa[[1, 70]] = np.nan
    store = ActivationStore(tmp_path / "a.zarr", mode="w")
    store.write_predictions("A", pa, qa)
    store.write_predictions("B", pb, qb)
    data = _FakeData(contexts, targets, meta, families)
    cfg = _cfg_two_models(n_boot=100)
    out = sc._l0_recompute(cfg, store, data, ["A", "B"], scale="mean_abs_diff",
                           n_boot_mult=1, row_seed=None)
    rows = [r for r in out["overall"] if not r["model"].startswith("__")]
    assert rows and all(np.isfinite(r["mase"]) for r in rows)
    assert out["mase_reliability"]["n_total"] == 2 * (len(contexts) - 2)
    assert all(np.isfinite(r["mase_lo"]) and np.isfinite(r["mase_hi"]) for r in out["per_family"])
