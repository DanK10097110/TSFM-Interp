"""ROADMAP.md sec 42 task C: forecast completeness (`sae/completeness.py`).

Every fixture plants a known answer. The SAE is exactly linear: `W_dec` is the
first `K` coordinate axes of a `D`-dimensional residual, `encode = relu(x W^T)`,
and the clean activation is `z W_dec + error` with the error in the last
coordinates, so the error term is known exactly. The stub forecast is a linear
readout of the patched activation, so which atoms and which error coordinates the
forecast depends on is a choice of the fixture:

  features-only  readout = coordinate 0: only atom 0 moves the forecast; the error
                 term cannot (top-1 share 1, error Delta 0).
  error-only     readout = an error coordinate: no atom moves the forecast (top-N
                 Delta 0, share 0).
  decoy          atom 1 fires 4x harder than atom 0 and decodes to a coordinate the
                 readout never reads: the prominence fallback ranks it FIRST, so
                 top-1 explains nothing while top-5 (which contains atom 0) does.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae import completeness as C

D, K, T, N_SERIES, HORIZON = 32, 24, 2, 12, 4
REAL, DECOY = 0, 1


def _planted(seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.0, 0.5, size=(N_SERIES, T, K))
    z[:, :, REAL] = rng.uniform(1.0, 3.0, size=(N_SERIES, T))
    z[:, :, DECOY] = rng.uniform(8.0, 10.0, size=(N_SERIES, T))
    err = rng.uniform(1.0, 2.0, size=(N_SERIES, T, D - K))
    tok = np.concatenate([z, err], axis=-1).astype(np.float32)
    return z, err, tok


Z, ERR, TOK = _planted()


class _SAE:
    d_in, dict_size = D, K
    W_dec = torch.eye(D)[:K]

    def encode(self, x):
        return torch.relu(x @ self.W_dec.T)

    def decode(self, f):
        return f @ self.W_dec

    def __call__(self, x):
        f = self.encode(x)
        return self.decode(f), f


class _Cfg:
    class run:
        seed = 0
    class data:
        horizon = HORIZON
    class l0:
        quantiles = [0.5]


class _Data:
    n = N_SERIES
    families = np.array(["f"] * N_SERIES)

    def contexts(self):
        rng = np.random.default_rng(1)
        c = rng.normal(size=(N_SERIES, 16)).astype(np.float32) * np.linspace(1, 5, N_SERIES)[:, None]
        c[:, 0] = np.arange(N_SERIES)
        return c


class _Adapter:
    name = "stub"

    def __init__(self, holder, readout, noise=None):
        self.module = object()
        self.cfg = type("C", (), {"batch_size": 5})()
        self._holder, self._u, self._noise = holder, readout, noise

    def ensure_loaded(self):
        pass

    def token_slice(self, live_len):
        return slice(0, live_len)

    def predict(self, contexts, horizon, quantiles):
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        repl = self._holder.get("repl")
        tokens = TOK[rows] if repl is None else np.asarray(repl)
        val = tokens.mean(axis=1) @ self._u
        point = val[:, None] * np.ones((1, horizon))
        if self._noise == "torch":
            point = point + torch.randn(point.shape).numpy()
        elif self._noise == "unseeded":
            point = point + np.random.normal(size=point.shape)
        return {"point": point.astype(np.float32)}


def _readout(coord: int) -> np.ndarray:
    u = np.zeros(D, dtype=np.float32)
    u[coord] = 1.0
    return u


@pytest.fixture
def wired(monkeypatch):
    holder: dict = {}

    def _capture(adapter, contexts, layers, **_kw):
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        return {layers[0]: torch.as_tensor(TOK[rows])}

    @contextmanager
    def _patch(module, layer, slicer, replacement):
        holder["repl"] = replacement
        try:
            yield
        finally:
            holder["repl"] = None

    monkeypatch.setattr(C, "capture_raw_tokens", _capture)
    monkeypatch.setattr(C, "token_patch", _patch)
    monkeypatch.setattr(C, "reach_probe", lambda *a, **k: {
        "reachable": True, "reason": "", "self_patch_delta": 0.0})
    return holder


def _pooled():
    return Z.mean(axis=1)


def _run(holder, readout, noise=None, **kw):
    kw.setdefault("ranking", None)
    return C.completeness_target(
        _Cfg, _Adapter(holder, readout, noise), "blocks.0", _SAE(), _Data(), "cpu",
        pooled=_pooled(), alive_mask=np.ones(K, dtype=bool), n_random=8, n_boot=200,
        max_series=N_SERIES, **kw)


def _scale():
    return C.context_scale(_Data().contexts())


def test_features_only_forecast_is_all_in_the_feature(wired):
    out = _run(wired, _readout(REAL), ranking=[REAL] + list(range(2, K)), ranking_source="planted")
    assert out["withheld"] is False
    r = out["per_set_size"]["1"]
    assert r["delta_error"] == pytest.approx(0.0, abs=1e-7)
    assert r["share_explained"] == pytest.approx(1.0)
    assert r["share_explained_ci"][0] == pytest.approx(1.0)
    assert r["share_explained_median_series"] == pytest.approx(1.0)
    expected = float((Z[:, :, REAL].mean(axis=1) / _scale()).mean())
    assert r["delta_features"] == pytest.approx(expected, rel=1e-5)
    assert r["delta_random"] == pytest.approx(0.0, abs=1e-7)
    assert r["excess_over_random"] > 0 and r["excess_over_random_ci"][0] > 0
    assert r["p_vs_random_draws"] == pytest.approx(1.0 / 9.0)
    assert r["frac_series_features_exceed_random"] == 1.0


def test_error_only_forecast_is_all_in_the_error_term(wired):
    out = _run(wired, _readout(K), ranking=[REAL] + list(range(2, K)), ranking_source="planted")
    for n in ("1", "5", "20"):
        r = out["per_set_size"][n]
        assert r["delta_features"] == pytest.approx(0.0, abs=1e-7)
        assert r["share_explained"] == pytest.approx(0.0, abs=1e-6)
        assert r["delta_error"] > 0
    expected = float((ERR[:, :, 0].mean(axis=1) / _scale()).mean())
    assert out["per_set_size"]["1"]["delta_error"] == pytest.approx(expected, rel=1e-5)


def test_decoy_outranks_the_real_atom_under_the_fallback_and_explains_nothing(wired):
    out = _run(wired, _readout(REAL) + _readout(K))
    assert out["ranking_source"] == C.FALLBACK_RANKING
    assert out["ranking_head"][0] == DECOY
    assert REAL in out["ranking_head"][:5]
    top1, top5 = out["per_set_size"]["1"], out["per_set_size"]["5"]
    assert top1["delta_features"] == pytest.approx(0.0, abs=1e-7)
    assert top1["share_explained"] == pytest.approx(0.0, abs=1e-6)
    assert top5["delta_features"] > 0
    assert 0.0 < top5["share_explained"] < 1.0
    assert top5["delta_error"] == pytest.approx(top1["delta_error"])


def test_share_is_undefined_not_zero_when_nothing_moves_the_forecast(wired):
    out = _run(wired, _readout(DECOY + 1))
    assert out["per_set_size"]["1"]["share_explained"] is None


def test_fallback_ranking_is_activation_times_decoder_norm():
    pooled = np.array([[1.0, 4.0, 2.0, 9.0]])
    norm = np.array([5.0, 1.0, 1.0, 1.0])
    alive = np.array([True, True, True, False])
    assert C.fallback_ranking(pooled, norm, alive) == [0, 1, 2]


def test_caller_ranking_is_used_and_named(wired):
    out = _run(wired, _readout(REAL), ranking=[REAL, DECOY], ranking_source="attribution_v1")
    assert out["ranking_source"] == "attribution_v1"
    assert out["ranking_head"] == [REAL, DECOY]
    assert out["per_set_size"]["1"]["share_explained"] == pytest.approx(1.0)


def test_chunked_forward_covers_every_series(wired):
    out = _run(wired, _readout(REAL), ranking=[REAL])
    assert out["per_set_size"]["1"]["n_series"] == N_SERIES
    assert len(out["per_series"]["delta_features"]["1"]) == N_SERIES


def test_sampled_model_is_compared_under_one_seed(wired):
    out = _run(wired, _readout(REAL), noise="torch", ranking=[REAL])
    assert out["withheld"] is False
    expected = float((Z[:, :, REAL].mean(axis=1) / _scale()).mean())
    assert out["per_set_size"]["1"]["delta_features"] == pytest.approx(expected, rel=1e-5)
    assert out["controls"]["repeat_max_abs"] == 0.0


def test_unseedable_nondeterminism_withholds_the_target(wired):
    out = _run(wired, _readout(REAL), noise="unseeded", ranking=[REAL])
    assert out["withheld"] is True
    assert "differ" in out["reason"]
    assert "per_set_size" not in out


def test_broken_self_patch_withholds_the_target(wired, monkeypatch):
    @contextmanager
    def _leaky(module, layer, slicer, replacement):
        wired["repl"] = replacement + 1e-3
        try:
            yield
        finally:
            wired["repl"] = None

    monkeypatch.setattr(C, "token_patch", _leaky)
    out = _run(wired, _readout(REAL), ranking=[REAL])
    assert out["withheld"] is True
    assert out["controls"]["self_patch_max_abs"] > 0
    assert "own clean tokens" in out["reason"]


def test_unreachable_target_is_withheld_with_the_probes_reason(wired, monkeypatch):
    monkeypatch.setattr(C, "reach_probe", lambda *a, **k: {"reachable": False, "reason": "dead"})
    out = _run(wired, _readout(REAL), ranking=[REAL])
    assert out["withheld"] is True and out["reason"] == "dead"


def test_ranking_shorter_than_set_size_is_recorded(wired):
    out = _run(wired, _readout(REAL), ranking=[REAL, 2])
    assert out["per_set_size"]["20"]["matching"]["effective_set_size"] == 2


def test_duplicate_ranking_is_refused(wired):
    with pytest.raises(ValueError, match="repeats"):
        _run(wired, _readout(REAL), ranking=[REAL, REAL])


def test_share_is_a_ratio_of_means():
    df, de = np.array([1.0, 1.0, 1.0]), np.array([3.0, 3.0, 3.0])
    r = C.summarize_set_size(df, de, np.array([0.5, 0.5, 0.5]), np.array([0.5]), 50, 0)
    assert r["share_explained"] == pytest.approx(0.25)
    assert r["excess_over_random"] == pytest.approx(0.5)


def test_median_series_share_resists_one_dominant_series():
    df, de = np.array([1.0, 1.0, 1.0, 100.0]), np.ones(4)
    r = C.summarize_set_size(df, de, np.zeros(4), np.zeros(2), 50, 0)
    assert r["share_explained_median_series"] == pytest.approx(0.5)
    assert r["share_explained"] > 0.9
    assert r["share_explained_median_series_ci"][0] <= 0.5 <= r["share_explained_median_series_ci"][1]


def test_bootstrap_resamples_series_at_full_size():
    n = 400
    df = np.tile([0.0, 1.0], n // 2)
    r = C.summarize_set_size(df, np.ones(n), np.zeros(n), np.zeros(5), 2000, 0)
    width = r["delta_features_ci"][1] - r["delta_features_ci"][0]
    assert width == pytest.approx(3.92 * 0.5 / np.sqrt(n), rel=0.15)


def test_nonfinite_series_are_dropped_and_counted():
    df = np.array([1.0, np.nan, 1.0, 1.0])
    r = C.summarize_set_size(df, np.ones(4), np.zeros(4), np.zeros(3), 50, 0)
    assert r["n_series"] == 3 and r["n_nonfinite_dropped"] == 1


def test_scaled_delta_divides_by_context_scale_and_flags_nonfinite():
    p = np.array([[2.0, 2.0], [np.nan, 1.0]])
    b = np.zeros((2, 2))
    d = C.scaled_delta(p, b, np.array([2.0, 1.0]))
    assert d[0] == pytest.approx(1.0) and np.isnan(d[1])


def test_context_scale_never_zero():
    s = C.context_scale(np.array([[3.0, 3.0, 3.0], [0.0, 0.0, 0.0], [1.0, -1.0, 1.0]]))
    assert s[0] == 3.0 and s[1] == 1.0 and s[2] > 0


def test_default_concepts_fingerprint_unchanged_and_enabling_changes_it():
    from tsfm_lens.config import PipelineConfig
    from tsfm_lens.manifest import resolve_config_keys
    a = PipelineConfig()
    base = resolve_config_keys(a, ("concepts",))
    assert not any("completeness" in k for k in base["concepts"])
    a.concepts.completeness = True
    assert resolve_config_keys(a, ("concepts",)) != base


# ---------------------------------------------------------------------------
# Stage wiring and report panel, on real mock adapters (CPU).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stage_run():
    import tempfile
    from tests.test_concept_stage import _cfg
    from tsfm_lens.extraction.store import ActivationStore
    from tsfm_lens.pipeline import run_pipeline
    out = tempfile.mkdtemp()
    cfg = _cfg(out, name="completeness_stage")
    run_pipeline(cfg, stages=["extract"])
    store = ActivationStore(cfg.run_dir() / "activations.zarr", mode="r")
    cfg.sae.targets = [{"model": "patchy", "layer": store.layers("patchy")[-1]},
                       {"model": "steppy", "layer": store.layers("steppy")[-1]}]
    run_pipeline(cfg, stages=["sae", "concepts"])
    return cfg.run_dir(), cfg


def test_default_stage_writes_no_completeness_artifact_or_block(stage_run):
    from tsfm_lens.utils import load_json
    rd, _ = stage_run
    assert not (rd / "sae" / "completeness.json").exists()
    assert "completeness" not in load_json(rd / "sae" / "concept_stage.json")


def test_enabled_stage_writes_artifact_report_panel_and_drops_it_when_disabled(stage_run):
    import shutil
    import tempfile
    from tsfm_lens.pipeline import run_pipeline
    from tsfm_lens.utils import load_json
    from tests.test_concept_stage import _cfg
    rd, cfg = stage_run
    work = Path(tempfile.mkdtemp()) / rd.name
    shutil.copytree(rd, work)
    cfg2 = _cfg(str(work.parent), name=work.name)
    cfg2.sae.targets = cfg.sae.targets
    cfg2.concepts.completeness = True
    cfg2.concepts.completeness_max_series = 12
    cfg2.concepts.completeness_n_random = 4
    cfg2.concepts.completeness_n_boot = 50
    run_pipeline(cfg2, stages=["concepts"], force={"concepts"}, allow_stale=True)
    art = load_json(work / "sae" / "completeness.json")
    assert art["evidence_class"] == "descriptive"
    assert set(art["targets"]) == {t["model"] + "/" + t["layer"] for t in cfg.sae.targets}
    record = load_json(work / "sae" / "concept_stage.json")
    assert record["completeness"]["status"] == "ran"
    measured = [t for t in art["targets"].values() if not t.get("withheld")]
    assert measured and all(t["ranking_source"] == C.FALLBACK_RANKING for t in measured)
    assert all(t["controls"]["self_patch_max_abs"] == 0.0 for t in measured)
    for t in measured:
        assert set(t["per_set_size"]) == {"1", "5", "20"}

    run_pipeline(cfg2, stages=["report"], force={"report"}, allow_stale=True)
    html = (work / "report.html").read_text(encoding="utf-8")
    assert "Forecast completeness" in html
    assert html.count("How much of the forecast change do a few SAE features carry") == 1
    assert "Evidence class: descriptive" in html
    assert "fallback (prominence)" in html

    cfg2.concepts.completeness = False
    run_pipeline(cfg2, stages=["concepts"], force={"concepts"}, allow_stale=True)
    assert not (work / "sae" / "completeness.json").exists()
    assert "completeness" not in load_json(work / "sae" / "concept_stage.json")
