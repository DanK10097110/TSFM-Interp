"""The seasonality-circuit report section (ROADMAP.md sec 20 H8 Stage 4):
renders whatever Stage 1/2/3 artifacts a run directory happens to hold under
`seasonality_circuit/`. Not a pipeline stage -- `run_seasonality_circuit.py`
(study driver, dev branch) writes these files standalone, mirroring
`run_layer_screen_bakeoff.py`'s own precedent -- so this section must
degrade to "" (logged, CLAUDE.md sec 2.5)
rather than reading a config `enabled` flag that does not exist for this
analysis, and must degrade per model and per stage within a model.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.report import _sec_seasonality_circuit
from tsfm_lens.utils import save_json

_COLORS = {"ModelA": "#2E6E8E", "ModelB": "#9A5B88"}


def _stage2_fixture(cleared: bool = True):
    return {
        "minimal_set_search_result": {
            "trace": [
                {"set_size": 1, "restoration": 0.4},
                {"set_size": 2, "restoration": 0.82},
            ],
            "null_floor_mean": 0.2,
            "null_draws_restoration_mean": [0.15, 0.18, 0.22, 0.25],
            "selected_set": [{"layer": "encoder.block.4", "head": 3},
                              {"layer": "encoder.block.6", "head": 1}],
            "best_prefix_size": 2,
            "sufficiency_restoration": 0.82,
            "necessity_restoration": 0.15,
            "gap_vs_null": {"mean": 0.62 if cleared else 0.02,
                            "lo": 0.5 if cleared else -0.05,
                            "hi": 0.74 if cleared else 0.09,
                            "p": 0.002 if cleared else 0.41},
            "cleared_null": cleared,
        }
    }


def _stage3_fixture(conserves: bool = True):
    return {
        "per_src": [
            {"src": {"layer": "encoder.block.4", "head": 3},
             "effect_total": 0.30, "effect_direct": 0.30 if conserves else 0.10,
             "sum_paths": 0.30 if conserves else 0.30, "conservation_gap": 0.0 if conserves else 0.20,
             "degenerate": False},
            {"src": {"layer": "encoder.block.6", "head": 1},
             "effect_total": 0.12, "effect_direct": 0.12, "sum_paths": 0.12,
             "conservation_gap": 0.0, "degenerate": True},
        ],
        "mean_abs_conservation_gap": 0.0 if conserves else 0.30,
        "mean_abs_effect_total": 0.21,
    }


def test_missing_directory_degrades_to_empty_string():
    run_dir = Path(tempfile.mkdtemp())
    findings = []
    assert _sec_seasonality_circuit(run_dir, _COLORS, findings) == ""
    assert findings == []


def test_directory_exists_but_empty_degrades_to_empty_string():
    run_dir = Path(tempfile.mkdtemp())
    (run_dir / "seasonality_circuit").mkdir()
    findings = []
    assert _sec_seasonality_circuit(run_dir, _COLORS, findings) == ""


def test_full_fixture_renders_both_stages_and_findings():
    run_dir = Path(tempfile.mkdtemp())
    circuit_dir = run_dir / "seasonality_circuit"
    circuit_dir.mkdir()
    save_json(circuit_dir / "ModelA_stage2_minimal_set.json", _stage2_fixture(cleared=True))
    save_json(circuit_dir / "ModelA_stage3_path_patch.json", _stage3_fixture(conserves=False))

    findings = []
    html = _sec_seasonality_circuit(run_dir, _COLORS, findings)
    assert html
    assert "<h4>ModelA</h4>" in html
    assert "encoder.block.4" in html
    assert "conservation_gap" in html
    # two findings: one from stage 2 (gap vs null), one from stage 3 (conservation)
    assert len(findings) == 2
    assert all(f.stage == "seasonality_circuit" for f in findings)
    assert all(f.evidence_class == "causal_within_model" for f in findings)
    assert "clears" in findings[0].text
    assert "does NOT conserve" in findings[1].text


def test_stage3_only_still_renders_without_stage2():
    """A model with only Stage 3's conservation table (no Stage 1/2 trace)
    must still render that table, not be skipped outright."""
    run_dir = Path(tempfile.mkdtemp())
    circuit_dir = run_dir / "seasonality_circuit"
    circuit_dir.mkdir()
    save_json(circuit_dir / "ModelB_stage3_path_patch.json", _stage3_fixture(conserves=True))

    findings = []
    html = _sec_seasonality_circuit(run_dir, _COLORS, findings)
    assert html
    assert "<h4>ModelB</h4>" in html
    assert len(findings) == 1
    assert findings[0].stage == "seasonality_circuit"
    assert "the decomposition conserves" in findings[0].text


def test_stage2_only_still_renders_without_stage3():
    run_dir = Path(tempfile.mkdtemp())
    circuit_dir = run_dir / "seasonality_circuit"
    circuit_dir.mkdir()
    save_json(circuit_dir / "ModelA_stage2_minimal_set.json", _stage2_fixture(cleared=False))

    findings = []
    html = _sec_seasonality_circuit(run_dir, _COLORS, findings)
    assert html
    assert "<h4>ModelA</h4>" in html
    assert "conservation_gap" not in html
    assert len(findings) == 1
    assert "does not clear" in findings[0].text


def test_multiple_models_each_get_their_own_subsection():
    run_dir = Path(tempfile.mkdtemp())
    circuit_dir = run_dir / "seasonality_circuit"
    circuit_dir.mkdir()
    save_json(circuit_dir / "ModelA_stage2_minimal_set.json", _stage2_fixture(cleared=True))
    save_json(circuit_dir / "ModelA_stage3_path_patch.json", _stage3_fixture(conserves=True))
    save_json(circuit_dir / "ModelB_stage2_minimal_set.json", _stage2_fixture(cleared=False))
    save_json(circuit_dir / "ModelB_stage3_path_patch.json", _stage3_fixture(conserves=False))

    findings = []
    html = _sec_seasonality_circuit(run_dir, _COLORS, findings)
    assert "<h4>ModelA</h4>" in html
    assert "<h4>ModelB</h4>" in html
    assert len(findings) == 4
