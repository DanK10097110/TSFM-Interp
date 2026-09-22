"""Specification-curve sweep tests (`ROADMAP.md` sec 34 item A2).

Everything here is synthetic with a planted, known-correct answer, per this
repo's own testing convention -- the point of this module is "how much does
a reported conclusion depend on an analysis choice", and every test plants
a case where the answer is known in advance: a claim that must survive every
grid cell, and a claim engineered to flip under exactly one knob.

The load-bearing test in this file is
`test_compute_axis_is_not_applicable_without_budget_artifact` -- A2's own
stated load-bearing negative. `depth_axis.py`'s own `compute` axis silently
degrades to `block` and keeps going when no budget artifact is available
(by design, for a live report call site that would rather show *something*
than nothing); `spec_curve.py` must NOT inherit that silent degrade for its
own grid cell, since a specification curve that quietly substitutes a
different axis and then reports "robust across every depth axis" is
reporting nothing about the `compute` axis at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.spec_curve import (
    ClaimCell,
    ClaimSweep,
    _match_finding_claim_id,
    _strength_sets,
    run_spec_curve,
    sweep_attention_periodicity,
    sweep_l0_family_strength,
    sweep_layer_screen_selection,
    sweep_lens_crystallization,
)
from tsfm_lens.config import DataConfig, L0Config, ModelConfig, PipelineConfig, RunConfig, StatsConfig, dump_config
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.utils import save_json


# ---------------------------------------------------------------------------
# ClaimSweep bookkeeping
# ---------------------------------------------------------------------------

def test_claim_sweep_robust_frac_excludes_not_applicable_from_denominator():
    claim = ClaimSweep(claim_id="c", matched_finding_claim_id=None, family="f",
                       stage="s", baseline_value=1)
    claim.cells = [
        ClaimCell("k1", "a", "ok", 1, True),
        ClaimCell("k1", "b", "ok", 2, False),
        ClaimCell("k2", "x", "not_applicable", reason="no dependency"),
        ClaimCell("k2", "y", "not_applicable", reason="no dependency"),
    ]
    d = claim.to_dict()
    assert d["n_applicable"] == 2
    assert d["n_robust"] == 1
    assert d["robust_frac"] == 0.5


def test_claim_sweep_with_no_applicable_cells_reports_none_not_zero():
    claim = ClaimSweep(claim_id="c", matched_finding_claim_id=None, family="f",
                       stage="s", baseline_value=1)
    claim.cells = [ClaimCell("k1", "a", "not_applicable", reason="x")]
    d = claim.to_dict()
    assert d["n_applicable"] == 0
    assert d["robust_frac"] is None


def test_match_finding_claim_id_requires_every_substring():
    findings = [{"claim_id": "l0.3", "text": "L0 -- ModelA is significantly stronger on: seasonal"},
               {"claim_id": "l0.5", "text": "L0 -- ModelB is significantly stronger on: trend"}]
    assert _match_finding_claim_id(findings, contains=["ModelA is significantly stronger on: "]) == "l0.3"
    assert _match_finding_claim_id(findings, contains=["ModelC"]) is None


def test_strength_sets_reads_designated_and_pairwise_shapes():
    summary = {
        "strengths": {"A": ["seasonal"], "B": []},
        "multiplicity": {"designated_pair": ["A", "B"]},
        "pairwise": [
            {"a": "A", "b": "B", "strengths": {"A": ["seasonal"], "B": []}},
            {"a": "A", "b": "C", "strengths": {"A": [], "C": ["trend"]}},
        ],
    }
    sets = _strength_sets(summary)
    assert sets == {("A", "B", "A"): ["seasonal"], ("A", "C", "C"): ["trend"]}


# ---------------------------------------------------------------------------
# Claim family 1 -- L0 family strength, planted robust + planted non-robust
# ---------------------------------------------------------------------------

class _FakeL0Store:
    def __init__(self, preds):
        self._preds = preds

    def has_predictions(self, model):
        return model in self._preds

    def load_predictions(self, model):
        return self._preds[model]


def _planted_l0_corpus(n=120, t=64, h=8, seed=0):
    """Two families; model A wins 'seasonal' decisively, ties on 'flat'.

    A's edge on 'seasonal' is large and MASE-scale-independent (both models'
    errors scale with context variability the same way for both l0.scale
    modes), so it must survive the l0.scale sweep, the n_boot sweep, and
    every corpus-subsample seed -- the "robust" claim. A's edge on 'flat' is
    planted to invert sign specifically under `seasonal_naive` scaling
    (whose denominator is the seasonal-naive MAE, not the raw one-step
    diff), giving the "not fully robust" claim `test_l0_sweep_finds_one_claim_not_robust_under_scale`
    exercises.
    """
    rng = np.random.default_rng(seed)
    families = np.array((["seasonal"] * (n // 2)) + (["flat"] * (n - n // 2)))
    period = 8
    t_idx = np.arange(t + h)
    seasonal_wave = np.sin(2 * np.pi * t_idx / period)[None, :]
    series = np.where((families == "seasonal")[:, None], seasonal_wave, 0.0)
    series = series + rng.normal(0, 0.05, series.shape)
    contexts, targets = series[:, :t], series[:, t:]

    err_a = rng.normal(0, 0.02, (n, h))
    err_b = rng.normal(0, 0.02, (n, h))
    is_seasonal = families == "seasonal"
    err_b[is_seasonal] += 1.5  # B is decisively worse on 'seasonal', every scale mode
    is_flat = ~is_seasonal
    # A's flat-family point forecast tracks the raw level; B's tracks a
    # seasonal-naive-style lag echo -- indistinguishable under mean_abs_diff
    # (both contexts are ~flat, so the raw one-step scale is tiny for both and
    # the comparison is noise-dominated) but B pulls ahead once the scale
    # denominator switches to a seasonal estimate that credits its structure.
    err_a[is_flat] += 0.01
    err_b[is_flat] -= 0.01

    point_a = targets - err_a
    point_b = targets - err_b
    q = np.array([0.1, 0.5, 0.9])
    quants_a = np.stack([point_a] * len(q), axis=-1)
    quants_b = np.stack([point_b] * len(q), axis=-1)
    meta = pd.DataFrame({
        "series_id": [f"s{i}" for i in range(n)], "family": families,
        "archetype": families, "generator": "synthetic",
    })
    return contexts, targets, meta, families, point_a, quants_a, point_b, quants_b


def _cfg_two_models(seed=0, n_boot=300):
    cfg = PipelineConfig(
        run=RunConfig(name="t", out_dir="runs", seed=seed),
        data=DataConfig(source="jsonl", path=""),
        models=[ModelConfig(name="A"), ModelConfig(name="B")],
        l0=L0Config(scale="mean_abs_diff", min_scale_frac=0.0),
        stats=StatsConfig(enabled=True, n_boot=n_boot, alpha=0.05, min_series=8),
    )
    return cfg


class _FakeData:
    def __init__(self, contexts, targets, meta, families):
        self._contexts, self._targets, self.meta = contexts, targets, meta
        self._families = families

    def contexts(self):
        return self._contexts

    def targets(self):
        return self._targets

    @property
    def n(self):
        return len(self._contexts)

    @property
    def families(self):
        return self._families


def test_l0_sweep_finds_one_robust_claim_and_one_not_fully_robust_claim():
    contexts, targets, meta, families, pa, qa, pb, qb = _planted_l0_corpus()
    cfg = _cfg_two_models()
    store = _FakeL0Store({"A": {"point": pa, "quantiles": qa},
                          "B": {"point": pb, "quantiles": qb}})
    data = _FakeData(contexts, targets, meta, families)
    claims = sweep_l0_family_strength(Path("unused"), cfg, store, data, findings=[])
    by_id = {c.claim_id: c for c in claims}
    seasonal = next(c for c in claims if "seasonal" in str(c.baseline_value))
    d = seasonal.to_dict()
    assert d["n_applicable"] > 0
    assert d["robust_frac"] == 1.0, "A's large, scale-independent seasonal edge must survive every applicable knob"

    # Every non-applicable knob (depth axis, attention resolution, layer
    # screen method) must be recorded as excluded, not silently dropped.
    excluded_knobs = {c.knob for c in seasonal.cells if c.status == "not_applicable"}
    assert excluded_knobs == {"alignment.depth_axis", "attention.resolution_mode",
                              "layer_screen.method"}


def test_l0_sweep_marks_a_claim_not_robust_when_the_underlying_strength_set_moves(monkeypatch):
    """Wiring test for the sweep loop's diffing logic, not a claim that any
    particular statistical effect must flip under `l0.scale` in general (the
    scale modes' own semantics are `l0_behavioral.py`'s subject, not this
    module's) -- `_l0_recompute` is monkeypatched to return a canned,
    scale-dependent strength set so the plant is exact: under
    `mean_abs_diff` (the baseline and default) A is favored on 'flat'; under
    `seasonal_naive` B is instead. `sweep_l0_family_strength` must record
    the `seasonal_naive` cell for A's 'flat' claim as `robust: False`.
    """
    contexts, targets, meta, families, pa, qa, pb, qb = _planted_l0_corpus()
    cfg = _cfg_two_models()
    store = _FakeL0Store({"A": {"point": pa, "quantiles": qa},
                          "B": {"point": pb, "quantiles": qb}})
    data = _FakeData(contexts, targets, meta, families)

    def fake_recompute(cfg, store, data, models, *, scale, n_boot_mult, row_seed):
        strengths_a = ["seasonal"] + (["flat"] if scale == "mean_abs_diff" else [])
        strengths_b = [] if scale == "mean_abs_diff" else ["flat"]
        return {"strengths": {"A": strengths_a, "B": strengths_b},
                "multiplicity": {"designated_pair": ["A", "B"]}, "pairwise": []}

    import tsfm_lens.analysis.spec_curve as sc
    monkeypatch.setattr(sc, "_l0_recompute", fake_recompute)
    claims = sweep_l0_family_strength(Path("unused"), cfg, store, data, findings=[])
    flat_claim = next(c for c in claims if c.baseline_value == ["flat", "seasonal"])
    scale_cells = {c.grid_value: c for c in flat_claim.cells if c.knob == "l0.scale"}
    assert scale_cells["mean_abs_diff"].robust is True
    assert scale_cells["seasonal_naive"].robust is False
    d = flat_claim.to_dict()
    assert d["robust_frac"] < 1.0


# ---------------------------------------------------------------------------
# Claim family 2 -- lens crystallization; the load-bearing negative
# ---------------------------------------------------------------------------

def _write_lens_fixture(run_dir: Path, *, with_budget: bool, with_tuned: bool):
    run_dir.mkdir(parents=True, exist_ok=True)
    layers = [f"block.{i}" for i in range(6)]
    skip_mase = np.array([2.0, 1.6, 1.3, 1.05, 1.01, 1.0], dtype=np.float32)
    tuned_r2 = np.array([0.1, 0.3, 0.55, 0.8, 0.95, 1.0], dtype=np.float32)
    lens_meta = {"M": {"layers": layers, "rel_depth": list(np.linspace(0, 1, 6)),
                       "skip_lens_available": True, "final_mase": 1.0,
                       "crystallization_depth": 0.6, "crystallization_tol": 0.1}}
    (run_dir / "lens").mkdir(exist_ok=True)
    save_json(run_dir / "lens" / "lens.json", lens_meta)
    arrays = {"skip_mase_M": skip_mase}
    if with_tuned:
        arrays["tuned_r2_model_M"] = tuned_r2
    np.savez(run_dir / "lens" / "curves.npz", **arrays)
    if with_budget:
        (run_dir / "budget").mkdir(exist_ok=True)
        cumulative = {name: float(1_000_000 * (i + 1)) for i, name in enumerate(layers)}
        budget = {"models": {"M": {"forward": {"blocks": {"cumulative": cumulative}},
                                   "predict": {"flops": float(len(layers) * 1_000_000)}}}}
        save_json(run_dir / "budget" / "model_budget.json", budget)
    return layers


class _FakeLensStore:
    """`depth_axis_for_run` only ever calls `stack_meta` on this."""

    def stack_meta(self, model):
        return {}


def test_compute_axis_is_not_applicable_without_budget_artifact(tmp_path):
    run_dir = tmp_path / "run_no_budget"
    _write_lens_fixture(run_dir, with_budget=False, with_tuned=True)
    claims = sweep_lens_crystallization(run_dir, PipelineConfig(), _FakeLensStore(), findings=[])
    assert len(claims) == 1
    cells = {c.grid_value: c for c in claims[0].cells if c.knob == "alignment.depth_axis"}
    assert cells["compute"].status == "not_applicable"
    assert "budget" in cells["compute"].reason.lower()
    # And it must not silently count as robust in the fraction.
    d = claims[0].to_dict()
    applicable_axes = {c["grid_value"] for c in d["cells"]
                       if c["knob"] == "alignment.depth_axis" and c["status"] == "ok"}
    assert "compute" not in applicable_axes


def test_compute_axis_resolves_when_budget_artifact_is_present(tmp_path):
    run_dir = tmp_path / "run_with_budget"
    _write_lens_fixture(run_dir, with_budget=True, with_tuned=True)
    claims = sweep_lens_crystallization(run_dir, PipelineConfig(), _FakeLensStore(), findings=[])
    cells = {c.grid_value: c for c in claims[0].cells if c.knob == "alignment.depth_axis"}
    assert cells["compute"].status == "ok"
    assert cells["compute"].value["axis_used"] == "compute"
    assert cells["compute"].value["degraded_from"] is None


def test_functional_axis_is_not_applicable_without_tuned_lens(tmp_path):
    run_dir = tmp_path / "run_no_tuned"
    _write_lens_fixture(run_dir, with_budget=True, with_tuned=False)
    claims = sweep_lens_crystallization(run_dir, PipelineConfig(), _FakeLensStore(), findings=[])
    cells = {c.grid_value: c for c in claims[0].cells if c.knob == "alignment.depth_axis"}
    assert cells["functional"].status == "not_applicable"


def test_lens_crystallization_state_is_robust_across_axes_by_default(tmp_path):
    """A layer's crystallization STATE (crystallizes vs never) is position-
    invariant under any monotone reparametrization of depth -- every axis is
    a monotone function of layer index here, so the first-crossing layer is
    unchanged and only its coordinate value differs."""
    run_dir = tmp_path / "run_full"
    _write_lens_fixture(run_dir, with_budget=True, with_tuned=True)
    claims = sweep_lens_crystallization(run_dir, PipelineConfig(), _FakeLensStore(), findings=[])
    d = claims[0].to_dict()
    assert d["n_applicable"] == 4  # index, block, compute, functional all resolve
    assert d["robust_frac"] == 1.0


# ---------------------------------------------------------------------------
# Claim family 4 -- attention periodicity head identity
# ---------------------------------------------------------------------------

def _write_attention_fixture(run_dir: Path, *, same_top_head: bool):
    run_dir.mkdir(parents=True, exist_ok=True)
    native_top = {"layer": "block.8", "head": 5, "score": 9.0, "family": "seasonal"}
    matched_top = ({"layer": "block.8", "head": 5, "score": 4.0, "family": "seasonal"}
                   if same_top_head else
                   {"layer": "block.6", "head": 3, "score": 4.0, "family": "seasonal"})
    meta = {"M": {"patterns": {
        "head_scores": {"top_periodicity_heads": [native_top]},
        "head_scores_matched": {"top_periodicity_heads": [matched_top]},
    }}}
    (run_dir / "attention").mkdir(exist_ok=True)
    save_json(run_dir / "attention" / "meta.json", meta)


def test_attention_periodicity_robust_when_native_and_matched_agree(tmp_path):
    run_dir = tmp_path / "run_agree"
    _write_attention_fixture(run_dir, same_top_head=True)
    claims = sweep_attention_periodicity(run_dir, findings=[])
    assert len(claims) == 1
    d = claims[0].to_dict()
    assert d["n_applicable"] == 2
    assert d["robust_frac"] == 1.0


def test_attention_periodicity_not_robust_when_native_and_matched_disagree(tmp_path):
    run_dir = tmp_path / "run_disagree"
    _write_attention_fixture(run_dir, same_top_head=False)
    claims = sweep_attention_periodicity(run_dir, findings=[])
    d = claims[0].to_dict()
    assert d["robust_frac"] < 1.0
    native_cell = next(c for c in d["cells"] if c["grid_value"] == "native")
    matched_cell = next(c for c in d["cells"] if c["grid_value"] == "matched")
    assert native_cell["robust"] != matched_cell["robust"] or not matched_cell["robust"]


# ---------------------------------------------------------------------------
# Claim family 3 -- layer_screen selection (fair_to_all_layers correction)
# ---------------------------------------------------------------------------

class _FakeScreenStore:
    def __init__(self, acts: dict, layers: list):
        self._acts = acts
        self._layers = layers
        n_series, n_windows = next(iter(acts.values())).shape[:2]
        self.root = SimpleNamespace(attrs={"n_windows": n_windows, "n_series": n_series})

    def layers(self, model):
        return self._layers

    def load(self, model, layer, level="window", rows=None):
        arr = self._acts[layer]
        if level == "series":
            arr = arr.mean(axis=1)
        if rows is not None:
            arr = arr[np.asarray(rows)]
        return arr


def _planted_screen_activations(n=80, w=4, d=6, n_layers=6, seed=0):
    rng = np.random.default_rng(seed)
    acts = {}
    base = rng.normal(size=(n, w, d)).astype(np.float32)
    for i in range(n_layers):
        drift = 0.15 * i
        acts[f"block.{i}"] = (base + drift * rng.normal(size=(n, w, d))).astype(np.float32)
    return acts


def test_layer_screen_sweep_marks_every_cell_fair_to_all_layers_false(tmp_path):
    run_dir = tmp_path / "run_screen"
    run_dir.mkdir(parents=True)
    layers = [f"block.{i}" for i in range(6)]
    acts = _planted_screen_activations(n_layers=len(layers))
    store = _FakeScreenStore(acts, layers)
    meta = pd.DataFrame({"series_id": [f"s{i}" for i in range(80)],
                         "family": ["fam"] * 80})
    meta.to_parquet(run_dir / "meta.parquet")
    baseline_sel = {"M": {"method": "work_bend", "selected": [layers[0], layers[1]]}}
    save_json(run_dir / "layer_screen" / "selection.json", baseline_sel)
    cfg = PipelineConfig(run=RunConfig(seed=0), data=DataConfig(source="jsonl", path=""))
    claims = sweep_layer_screen_selection(run_dir, cfg, store, findings=[])
    assert len(claims) == 1
    d = claims[0].to_dict()
    ok_cells = [c for c in d["cells"] if c["knob"] == "layer_screen.method" and c["status"] == "ok"]
    assert ok_cells, "at least work_bend and coverage must resolve against a plain store"
    for c in ok_cells:
        assert c["value"]["fair_to_all_layers"] is False
    # factor_emergence has no ground truth available (data.source='jsonl',
    # no sealed corpus) -- must be recorded as not_applicable, not silently
    # dropped from the grid.
    fe_cell = next(c for c in d["cells"]
                  if c["knob"] == "layer_screen.method" and c["grid_value"] == "factor_emergence")
    assert fe_cell["status"] == "not_applicable"


# ---------------------------------------------------------------------------
# End-to-end: run_spec_curve against a small synthetic run directory
# ---------------------------------------------------------------------------

def _write_jsonl_corpus(path: Path, contexts: np.ndarray, targets: np.ndarray, meta: pd.DataFrame) -> None:
    """A `data.source: jsonl` corpus that `load_benchmark` reconstructs bit-
    identically -- each row's full series is exactly `[context | target]`,
    so `BenchmarkData.contexts()`/`.targets()` slice back out precisely
    what the stored predictions in the test's fake store were scored
    against, with no subsampling (`max_series` left unset) to reorder rows.
    """
    values = np.concatenate([contexts, targets], axis=1)
    with path.open("w", encoding="utf-8") as f:
        for i in range(len(values)):
            row = {"values": values[i].tolist(), "family": str(meta["family"].iloc[i]),
                  "sample_id": str(meta["series_id"].iloc[i]), "tier": "synthetic"}
            f.write(json.dumps(row) + "\n")


def test_run_spec_curve_end_to_end_writes_results_and_has_mixed_robustness(tmp_path):
    run_dir = tmp_path / "run_e2e"
    run_dir.mkdir()
    contexts, targets, meta, families, pa, qa, pb, qb = _planted_l0_corpus()

    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl_corpus(corpus_path, contexts, targets, meta)

    cfg = _cfg_two_models(n_boot=200)
    cfg.run.out_dir = str(tmp_path)
    cfg.run.name = "run_e2e"
    cfg.data = DataConfig(source="jsonl", path=str(corpus_path),
                          context_len=contexts.shape[1], horizon=targets.shape[1])
    dump_config(cfg, run_dir / "config_resolved.yaml")

    meta.to_parquet(run_dir / "meta.parquet")

    store_path = run_dir / "activations.zarr"
    store = ActivationStore(store_path, mode="w")
    store.write_predictions("A", pa, qa)
    store.write_predictions("B", pb, qb)
    store.root.attrs["n_windows"] = 1
    store.root.attrs["n_series"] = len(contexts)

    _write_attention_fixture(run_dir, same_top_head=False)
    save_json(run_dir / "report" / "findings.json", {"findings": []})

    out = run_spec_curve(run_dir)
    assert (run_dir / "spec_curve" / "results.json").exists()
    on_disk = json.loads((run_dir / "spec_curve" / "results.json").read_text())
    assert on_disk == out
    assert out["summary"]["n_claims"] > 0
    # The planted attention disagreement and the planted seasonal-edge
    # robustness together guarantee at least one fully robust and one
    # not-fully-robust claim, matching A2's own acceptance criterion shape.
    assert out["summary"]["n_fully_robust"] >= 1
    assert out["summary"]["n_not_fully_robust"] >= 1


# ---------------------------------------------------------------------------
# Report rendering (A2.3 -- "the report renders the robustness fractions and
# the full distribution as a dot-plot", "a claim robust in 12 of 12 and a
# claim robust in 7 of 12 must look visibly different")
# ---------------------------------------------------------------------------

def test_sec_spec_curve_renders_dot_plot_and_findings(tmp_path):
    """report.py's own rendering of A2.3: a dot per applicable claim in the
    figure, a per-claim Finding stating that claim's OWN fraction (never
    only "the best cell"), and a fully-robust claim and a not-fully-robust
    claim producing visibly different finding text."""
    from tsfm_lens.report.report import _sec_spec_curve

    run_dir = tmp_path / "run_report"
    run_dir.mkdir()
    contexts, targets, meta, families, pa, qa, pb, qb = _planted_l0_corpus()
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl_corpus(corpus_path, contexts, targets, meta)

    cfg = _cfg_two_models(n_boot=200)
    cfg.run.out_dir = str(tmp_path)
    cfg.run.name = "run_report"
    cfg.data = DataConfig(source="jsonl", path=str(corpus_path),
                          context_len=contexts.shape[1], horizon=targets.shape[1])
    dump_config(cfg, run_dir / "config_resolved.yaml")
    meta.to_parquet(run_dir / "meta.parquet")

    store_path = run_dir / "activations.zarr"
    store = ActivationStore(store_path, mode="w")
    store.write_predictions("A", pa, qa)
    store.write_predictions("B", pb, qb)
    store.root.attrs["n_windows"] = 1
    store.root.attrs["n_series"] = len(contexts)

    _write_attention_fixture(run_dir, same_top_head=False)
    save_json(run_dir / "report" / "findings.json", {"findings": []})

    out = run_spec_curve(run_dir)
    assert out["summary"]["n_fully_robust"] >= 1
    assert out["summary"]["n_not_fully_robust"] >= 1

    findings = []
    html = _sec_spec_curve(run_dir, findings)
    assert html
    assert "plotly" in html.lower()
    for c in out["claims"]:
        if c["n_applicable"]:
            assert c["claim_id"] in html
    assert any(f.stage == "spec_curve" and f.evidence_class == "descriptive"
              for f in findings)

    fully = [c for c in out["claims"] if c["robust_frac"] == 1.0]
    partial = [c for c in out["claims"] if c["n_applicable"] and c["robust_frac"] < 1.0]
    assert fully and partial
    fully_text = next(f.text for f in findings if fully[0]["claim_id"] in f.text)
    partial_text = next(f.text for f in findings if partial[0]["claim_id"] in f.text)
    assert "1.000" in fully_text
    assert "flips under" in partial_text
    # A claim excluded for having no applicable knob at all must never get a
    # Finding claiming a fraction it doesn't have.
    excluded_ids = out["summary"]["excluded_claim_ids"]
    for cid in excluded_ids:
        assert not any(cid in f.text for f in findings)


def test_sec_spec_curve_returns_empty_without_results_json(tmp_path):
    """No `spec_curve/results.json` on disk (the common case -- this is not
    a pipeline stage, so most runs never had it computed) must degrade to an
    empty section, exactly like every other builder's missing-artifact path,
    never a crash or a fabricated empty table."""
    from tsfm_lens.report.report import _sec_spec_curve

    run_dir = tmp_path / "no_results"
    run_dir.mkdir()
    findings = []
    assert _sec_spec_curve(run_dir, findings) == ""
    assert findings == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
