"""ROADMAP.md sec 32.7 Item H — record it, render it, control it.

Synthetic with planted answers (CLAUDE.md sec 2.4): a real run's own numbers
would make these pass for whatever that run happened to contain, which is
the one thing a reduction/renderer test must not do.

Covers the four pieces this item wires together:
  * `derived.ablation_panel_summary` (H1) — the per-panel reduction.
  * `derived.flatness_population` (H2's shared-numbers fix) — factored out
    of `report.py::_sae_flatness_block` so a block-level summary and a
    per-panel clause read IDENTICAL numbers.
  * `report.py::_flat_clause` (H2) — the per-panel sentence, gated on both
    `flat` and a non-empty population.
  * `report.py::_sae_flatness_block` (H3/H4) — the archetype x model table
    and the raw-vs-reconstruction confound control, and that it is actually
    CALLED from `_sec_sae` (the wiring gap this item closes: the functions
    existed, unreachably, before this session).
  * `sae_features.py::ablation_cell` — that a flat panel's clause reaches
    the rendered `data-meta` attribute when a population is passed, and
    that passing no population changes nothing (existing call sites that
    do not yet thread one degrade exactly as before).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inspect

import pandas as pd

from tsfm_lens.sae.train import sanitize
from tsfm_lens.utils import save_json
from tsfm_lens.report import derived
from tsfm_lens.report.report import _flat_clause, _sae_flatness_block, _sec_sae
from tsfm_lens.report.sae_features import ablation_cell


# ---------------------------------------------------------------------------
# H1: ablation_panel_summary
# ---------------------------------------------------------------------------

def _flat_forecast_entry():
    # Noise-like context (no autocorrelation to extrapolate) -> a
    # near-constant reconstruction is the MMSE-optimal response.
    return {"context": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0], "target": [0.01, -0.01],
            "with_feature": [0.001, -0.001], "unpatched": [0.002, -0.002],
            "series_id": "s1"}


def _sharp_forecast_entry():
    # Smooth (high lag-1) context and a forecast that tracks its own scale.
    return {"context": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "target": [7.5, 9.0],
            "with_feature": [7.0, 9.0], "unpatched": [7.2, 8.8],
            "series_id": "s2"}


def test_flat_panel_scores_below_threshold_and_smooth_one_does_not():
    flat = derived.ablation_panel_summary(_flat_forecast_entry())
    sharp = derived.ablation_panel_summary(_sharp_forecast_entry())
    assert flat["flat"] is True
    assert sharp["flat"] is False
    # A monotone ramp is positively autocorrelated; a strictly-alternating
    # context is negatively autocorrelated -- the two entries above are
    # planted to land on opposite sides of the 0.2 noise-like cutoff
    # `_sae_flatness_block`/`flatness_population` gate on.
    assert sharp["context_lag1"] > 0.2
    assert flat["context_lag1"] < 0.2
    assert flat["raw_sd_ratio"] is not None and flat["raw_sd_ratio"] < 0.10


def test_panel_summary_holds_the_adaptivity_contract():
    # No model name, no architecture family, no cfg.models[i] index anywhere
    # in the function's actual CODE -- it only ever sees one forecast dict.
    # (The docstring names `cfg.models[i]` in prose, explaining what is NOT
    # done, so the check runs against the parsed body, not raw source text.)
    import ast
    sig = inspect.signature(derived.ablation_panel_summary)
    assert "cfg" not in sig.parameters
    tree = ast.parse(inspect.getsource(derived.ablation_panel_summary))
    fn = tree.body[0]
    body_without_docstring = fn.body[1:] if (
        fn.body and isinstance(fn.body[0], ast.Expr)
        and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    ) else fn.body
    body_src = "\n".join(ast.unparse(n) for n in body_without_docstring)
    for banned in ("cfg.models[", "TimesFM", "Chronos", "Sundial"):
        assert banned not in body_src


def test_panel_summary_degrades_when_context_or_forecast_missing():
    out = derived.ablation_panel_summary({"context": [], "with_feature": None})
    assert out["flat"] is False
    assert out["forecast_sd_ratio"] is None
    assert out["mase"] is None


# ---------------------------------------------------------------------------
# H2 (shared numbers): flatness_population
# ---------------------------------------------------------------------------

def _population_df():
    return pd.DataFrame([
        # Noise-like (lag1 < 0.2), flat, low MASE.
        {"context_lag1": 0.05, "flat": True, "mase": 0.5},
        {"context_lag1": 0.10, "flat": True, "mase": 0.6},
        {"context_lag1": 0.15, "flat": False, "mase": 0.9},
        # Not noise-like.
        {"context_lag1": 0.9, "flat": False, "mase": 1.2},
        {"context_lag1": 0.7, "flat": False, "mase": 1.0},
    ])


def test_flatness_population_matches_hand_computed_shares():
    pop = derived.flatness_population(_population_df())
    # 2 of 3 noise-like (lag1 < 0.2) rows are flat.
    assert pop["noise_like_n"] == 3
    assert abs(pop["noise_like_flat_share"] - (2 / 3)) < 1e-9
    # Flat median mase (0.55) < non-flat median (1.0).
    assert pop["flat_scores_better_mase"] is True


def test_flatness_population_is_none_for_empty_frame():
    assert derived.flatness_population(pd.DataFrame()) is None


def test_flatness_population_false_when_flat_panels_score_worse():
    df = pd.DataFrame([
        {"context_lag1": 0.05, "flat": True, "mase": 5.0},
        {"context_lag1": 0.9, "flat": False, "mase": 0.5},
    ])
    pop = derived.flatness_population(df)
    assert pop["flat_scores_better_mase"] is False


# ---------------------------------------------------------------------------
# H2: _flat_clause
# ---------------------------------------------------------------------------

def test_flat_clause_fires_only_on_a_flat_panel_with_a_population():
    pop = {"noise_like_flat_share": 0.786, "noise_like_n": 40,
           "flat_scores_better_mase": True}
    flat_panel = {"flat": True, "context_lag1": 0.08}
    nonflat_panel = {"flat": False, "context_lag1": 0.08}
    assert _flat_clause(flat_panel, pop) != ""
    assert _flat_clause(nonflat_panel, pop) == ""
    assert _flat_clause(flat_panel, None) == ""
    assert _flat_clause(flat_panel, {}) == ""


def test_flat_clause_quotes_the_panels_own_lag1_and_population_numbers():
    pop = {"noise_like_flat_share": 0.786, "noise_like_n": 40,
           "flat_scores_better_mase": True}
    clause = _flat_clause({"flat": True, "context_lag1": 0.0821}, pop)
    assert "0.08" in clause
    assert "78.6%" in clause
    assert "better on MASE" in clause


def test_flat_clause_omits_the_better_mase_claim_when_not_measured_true():
    pop = {"noise_like_flat_share": 0.5, "noise_like_n": 10,
           "flat_scores_better_mase": False}
    clause = _flat_clause({"flat": True, "context_lag1": 0.1}, pop)
    assert "better on MASE" not in clause


def test_flat_clause_needs_a_measured_lag1():
    pop = {"noise_like_flat_share": 0.5, "noise_like_n": 10,
           "flat_scores_better_mase": True}
    assert _flat_clause({"flat": True, "context_lag1": None}, pop) == ""


# ---------------------------------------------------------------------------
# H3/H4: _sae_flatness_block, and that _sec_sae actually calls it
# ---------------------------------------------------------------------------

def _write_ablation(run_dir, model, layer, entries):
    path = (Path(run_dir) / "sae" / sanitize(model)
            / f"{sanitize(layer)}_ablation.json")
    candidates = [{"feature": i, "forecasts": [e]} for i, e in enumerate(entries)]
    save_json(path, {"candidates": candidates})


def test_sae_flatness_block_renders_a_table_and_a_finding(tmp_path):
    save_json(Path(tmp_path) / "sae" / "meta.json", {"Alpha/layer0": {}})
    _write_ablation(tmp_path, "Alpha", "layer0",
                    [_flat_forecast_entry(), _sharp_forecast_entry()])
    findings = []
    out = _sae_flatness_block(cfg=None, run_dir=Path(tmp_path), findings=findings)
    assert "rendered ablation" in out
    assert "Flat share by archetype and model" not in out  # no archetype lookup on cfg=None
    assert "Does the SAE flatten the forecast" in out
    assert len(findings) == 1
    assert findings[0].stage == "sae"


def test_sae_flatness_block_empty_when_no_ablation_artifacts(tmp_path):
    save_json(Path(tmp_path) / "sae" / "meta.json", {})
    out = _sae_flatness_block(cfg=None, run_dir=Path(tmp_path), findings=[])
    assert out == ""


def test_sae_flatness_block_reuses_a_precomputed_df_without_rereading_disk(tmp_path, monkeypatch):
    save_json(Path(tmp_path) / "sae" / "meta.json", {"Alpha/layer0": {}})
    _write_ablation(tmp_path, "Alpha", "layer0", [_flat_forecast_entry()])
    df = derived.ablation_panel_table(Path(tmp_path))

    def _boom(*a, **kw):
        raise AssertionError("ablation_panel_table should not be called when df is given")

    monkeypatch.setattr(derived, "ablation_panel_table", _boom)
    out = _sae_flatness_block(cfg=None, run_dir=Path(tmp_path), findings=[], df=df)
    assert "rendered ablation" in out


def test_sec_sae_source_calls_the_flatness_block_and_threads_population():
    # ROADMAP.md sec 32.7 Item H's actual defect: `_sae_flatness_block` and
    # `_flat_clause` were both fully implemented and never called from
    # anywhere else in the file. Pin that the wiring exists, the way this
    # repo's other source-inspection tests pin an adaptivity contract.
    src = inspect.getsource(_sec_sae)
    assert "_sae_flatness_block(" in src
    assert "flatness_population" in src
    assert "population=flatness_population" in src


# ---------------------------------------------------------------------------
# ablation_cell: the per-panel clause reaching the rendered meta
# ---------------------------------------------------------------------------

def _ablation_entry_with(entries):
    return {"feature": 3, "scorable": True, "reason": "",
            "n_channels_clearing": 1, "channels": {}, "forecasts": entries}


def test_ablation_cell_appends_the_flat_clause_when_population_given():
    pop = {"noise_like_flat_share": 0.786, "noise_like_n": 40,
           "flat_scores_better_mase": True}
    svg = ablation_cell(_ablation_entry_with([_flat_forecast_entry()]),
                        population=pop)
    assert "lag-1 autocorrelation" in svg
    assert "78.6%" in svg


def test_ablation_cell_unchanged_when_no_population_passed():
    svg_no_pop = ablation_cell(_ablation_entry_with([_flat_forecast_entry()]))
    assert "lag-1 autocorrelation" not in svg_no_pop


def test_ablation_cell_no_clause_on_a_non_flat_panel_even_with_population():
    pop = {"noise_like_flat_share": 0.786, "noise_like_n": 40,
           "flat_scores_better_mase": True}
    svg = ablation_cell(_ablation_entry_with([_sharp_forecast_entry()]),
                        population=pop)
    assert "lag-1 autocorrelation" not in svg
