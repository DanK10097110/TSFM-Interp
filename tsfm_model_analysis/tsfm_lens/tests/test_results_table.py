"""Tests for `report/results_table.py` (`ROADMAP.md` §34 item A4).

The module's whole contract is "pure reduction, never fabricate a value it
doesn't have" -- so most of these tests are built around the same shape
`CLAUDE.md` §11.36/§11.37 warn about repeatedly: a missing or unrecognized
input must read as an explicit absence (NaN + a named WARNING), never as a
plausible-looking default that a truncated artifact never actually had.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.results_table import (
    RESULTS_COLUMNS,
    _base_row,
    _extract_confirm_rows,
    _num,
    build_results_rows,
    build_results_table,
    export_results_table,
)


def _finding(claim_id: str, stage: str, **kw) -> dict:
    base = {
        "claim_id": claim_id, "stage": stage, "evidence_class": "descriptive",
        "text": "", "plain": "", "registered": False, "cleared_noise_floor": None,
        "value": None, "ci": None, "caveat": "",
    }
    base.update(kw)
    return base


def _write_findings(run_dir: Path, findings: list) -> None:
    (run_dir / "report").mkdir(parents=True, exist_ok=True)
    (run_dir / "report" / "findings.json").write_text(
        json.dumps({"findings": findings}), encoding="utf-8")


# ---------------------------------------------------------------------------
# T-A4.1 -- one row per Finding, count equal
# ---------------------------------------------------------------------------

def test_every_finding_gets_at_least_one_row_from_an_unhandled_stage(tmp_path):
    findings = [_finding("l1.1", "l1"), _finding("l1.2", "l1"), _finding("attn.1", "attention")]
    _write_findings(tmp_path, findings)
    rows = build_results_rows(tmp_path)
    assert len(rows) == 3
    assert [r["claim_id"] for r in rows] == ["l1.1", "l1.2", "attn.1"]


def test_base_row_has_exactly_the_a4_columns():
    row = _base_row(_finding("x.1", "l2"))
    assert set(row) == set(RESULTS_COLUMNS)
    for col in ("subject_kind", "subject", "stratum_kind", "stratum", "statistic",
                "ci_method", "p", "p_adjusted", "n", "reference_value",
                "reference_kind", "floor_units", "verdict", "rule_text",
                "replicated", "mde", "spec_robustness"):
        assert row[col] is None or (isinstance(row[col], float) and np.isnan(row[col]))
    assert row["artifact_path"] == "report/findings.json"


def test_finding_value_and_ci_pass_through_when_populated():
    row = _base_row(_finding("x.1", "l2", value=0.4132, ci=[0.38, 0.45]))
    assert row["value"] == pytest.approx(0.4132)
    assert row["ci_lo"] == pytest.approx(0.38)
    assert row["ci_hi"] == pytest.approx(0.45)


def test_no_findings_file_at_all_yields_an_empty_table_not_a_crash(tmp_path):
    df = build_results_table(tmp_path)
    assert list(df.columns) == RESULTS_COLUMNS
    assert len(df) == 0


# ---------------------------------------------------------------------------
# Schema-drift failure mode: an undeclared stage warns once, grouped, and
# never crashes or drops a finding.
# ---------------------------------------------------------------------------

def test_unhandled_stage_logs_one_grouped_warning_naming_it_and_the_count(tmp_path, caplog):
    # Both "brand_new_stage" and "l0" are undeclared in `_STAGE_EXTRACTORS`
    # ("confirm" is the only declared one), so both must appear in the SAME
    # grouped warning rather than one warning per stage or per finding.
    findings = [_finding("newstage.1", "brand_new_stage"),
                _finding("newstage.2", "brand_new_stage"),
                _finding("l0.1", "l0")]
    _write_findings(tmp_path, findings)
    with caplog.at_level("WARNING"):
        rows = build_results_rows(tmp_path)
    assert len(rows) == 3
    warnings = [r for r in caplog.records if "no declared extractor" in r.message]
    assert len(warnings) == 1
    assert "brand_new_stage" in warnings[0].message
    assert "l0" in warnings[0].message
    assert "3 finding" in warnings[0].message


# ---------------------------------------------------------------------------
# The `confirm` extractor -- the one declared stage extractor this item ships.
# ---------------------------------------------------------------------------

def _confirmation(**kw) -> dict:
    base = {"alpha": 0.05, "tests": [], "n_private_series": 300}
    base.update(kw)
    return base


def test_confirm_family_hypotheses_expand_one_row_per_family(tmp_path):
    conf = _confirmation(tests=[
        {"family": "seasonal", "dev_favored": "chronos", "mean": 0.1234567890123,
         "lo": 0.05, "hi": 0.2, "p": 0.004, "p_holm": 0.012, "n": 288,
         "resample_unit": "series", "status": "tested", "confirmed": True,
         "mde_private": {"mde": 0.03}},
        {"family": "trend", "dev_favored": "timesfm", "mean": -0.02, "lo": -0.1,
         "hi": 0.05, "p": 0.4, "p_holm": 0.6, "n": 150, "resample_unit": "series",
         "status": "tested", "confirmed": False, "mde_private": {"mde": None,
         "reason": "unsatisfiable_correction"}},
    ])
    (tmp_path / "confirm").mkdir()
    (tmp_path / "confirm" / "confirmation.json").write_text(json.dumps(conf), encoding="utf-8")
    finding = _finding("confirm.1", "confirm", evidence_class="behavioral",
                       text="CONFIRM — 1/2 dev family hypotheses confirmed on the "
                            "private benchmark (paired bootstrap, Holm α=0.05).",
                       registered=True)
    rows = _extract_confirm_rows(tmp_path, finding)
    assert rows is not None and len(rows) == 2
    assert {r["claim_id"] for r in rows} == {"confirm.1"}
    seasonal = next(r for r in rows if r["stratum"] == "seasonal")
    assert seasonal["value"] == pytest.approx(0.1234567890123)
    assert seasonal["p_adjusted"] == pytest.approx(0.012)
    assert seasonal["verdict"] == "confirmed"
    assert seasonal["replicated"] is True
    assert seasonal["mde"] == pytest.approx(0.03)
    assert seasonal["reference_value"] == 0.0
    assert seasonal["reference_kind"] == "null"
    trend = next(r for r in rows if r["stratum"] == "trend")
    assert trend["verdict"] == "not confirmed"
    assert trend["replicated"] is False
    assert np.isnan(trend["mde"])


def test_confirm_untestable_family_replicated_is_nan_not_false(tmp_path):
    """§11.37: untestable is a third state, not a failure to replicate."""
    conf = _confirmation(tests=[
        {"family": "rare", "dev_favored": "chronos", "mean": None, "lo": None,
         "hi": None, "p": None, "p_holm": None, "n": 4, "resample_unit": "series",
         "status": "untestable", "confirmed": False, "mde_private": None},
    ])
    (tmp_path / "confirm").mkdir()
    (tmp_path / "confirm" / "confirmation.json").write_text(json.dumps(conf), encoding="utf-8")
    finding = _finding("confirm.1", "confirm",
                       text="CONFIRM — 0/1 dev family hypotheses confirmed on the "
                            "private benchmark (paired bootstrap, Holm α=0.05).")
    rows = _extract_confirm_rows(tmp_path, finding)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "untestable"
    assert np.isnan(rows[0]["replicated"])


def test_confirm_l3_replication_one_row_per_corruption(tmp_path):
    # Fixture shape verified against a real `confirmation.json`
    # (`runs/full_report_run_4model`): `private` is keyed "value" (matching
    # `report.py::_ci_str`'s default key), not "mean" -- and carries no "n"
    # at all in real data (only "lo"/"hi", sometimes "resample_unit").
    conf = _confirmation(l3_replication={
        "model_a": "timesfm", "model_b": "chronos", "overall": {"value": 0.5},
        "tests": [
            {"corruption": "noise", "dev_rho": -1.0,
             "private": {"value": -0.778, "lo": -0.965, "hi": -0.583},
             "replicates": False},
            {"corruption": "detrend", "dev_rho": 0.6,
             "private": {"value": 0.62, "lo": 0.4, "hi": 0.8},
             "replicates": True},
        ]})
    (tmp_path / "confirm").mkdir()
    (tmp_path / "confirm" / "confirmation.json").write_text(json.dumps(conf), encoding="utf-8")
    finding = _finding("confirm.2", "confirm", evidence_class="causal_within_model",
                       text='CONFIRM — L3 fingerprint agreement: 1/2 registered '
                            'per-corruption values replicate on private data '
                            '(overall ρ=0.500).')
    rows = _extract_confirm_rows(tmp_path, finding)
    assert len(rows) == 2
    noise = next(r for r in rows if r["stratum"] == "noise")
    assert noise["subject"] == "timesfm__chronos"
    assert noise["value"] == pytest.approx(-0.778)
    assert noise["verdict"] == "does not replicate"
    assert noise["replicated"] is False
    assert noise["reference_value"] == pytest.approx(-1.0)
    assert noise["reference_kind"] == "baseline"
    assert np.isnan(noise["n"])  # genuinely absent from this real artifact's schema
    detrend = next(r for r in rows if r["stratum"] == "detrend")
    assert detrend["value"] == pytest.approx(0.62)
    assert detrend["replicated"] is True


def test_confirm_cka_replication_single_row(tmp_path):
    # Same "value"-keyed shape, verified against the same real artifact.
    conf = _confirmation(cka_replication={
        "status": "tested", "dev_cka": 0.38, "layer_a": "stacked_xf.4",
        "layer_b": "encoder.block.6",
        "private": {"value": 0.3797, "lo": 0.3678, "hi": 0.3961,
                    "resample_unit": "series"},
        "replicates": True})
    (tmp_path / "confirm").mkdir()
    (tmp_path / "confirm" / "confirmation.json").write_text(json.dumps(conf), encoding="utf-8")
    finding = _finding("confirm.3", "confirm", evidence_class="geometric",
                       text='CONFIRM — peak-CKA layer pair replicates on private '
                            'data (0.380 [0.368, 0.396]).')
    rows = _extract_confirm_rows(tmp_path, finding)
    assert len(rows) == 1
    row = rows[0]
    assert row["stratum"] == "stacked_xf.4__encoder.block.6"
    assert row["value"] == pytest.approx(0.3797)
    assert row["reference_value"] == pytest.approx(0.38)
    assert row["verdict"] == "replicates"


def test_confirm_extractor_declines_when_artifact_missing_falls_back_to_base(tmp_path):
    # No confirmation.json written at all.
    finding = _finding("confirm.1", "confirm",
                       text="CONFIRM — 1/2 dev family hypotheses confirmed...")
    assert _extract_confirm_rows(tmp_path, finding) is None


def test_confirm_extractor_declines_on_unrecognized_text(tmp_path):
    (tmp_path / "confirm").mkdir()
    (tmp_path / "confirm" / "confirmation.json").write_text(
        json.dumps(_confirmation()), encoding="utf-8")
    finding = _finding("confirm.9", "confirm", text="some future confirm finding shape")
    assert _extract_confirm_rows(tmp_path, finding) is None


def test_confirm_stage_end_to_end_via_build_results_rows_falls_back_cleanly(tmp_path):
    """A confirm finding with no confirmation.json on disk must still yield
    exactly one base row -- the extractor IS declared for this stage, so this
    must not trip the 'no declared extractor' warning path either.
    """
    findings = [_finding("confirm.1", "confirm", text="CONFIRM — nothing to see")]
    _write_findings(tmp_path, findings)
    rows = build_results_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["claim_id"] == "confirm.1"


# ---------------------------------------------------------------------------
# T-A4.2 -- load-bearing negative: a truncated artifact -> NaN + WARNING
# naming the key, never a plausible default (`.get(key, 0.0)`'s anti-pattern).
# ---------------------------------------------------------------------------

def test_num_missing_key_is_nan_with_a_named_warning(caplog):
    with caplog.at_level("WARNING"):
        result = _num({"mean": 0.5}, "lo", context="a test row")
    assert np.isnan(result)
    assert any("lo" in r.message and "a test row" in r.message for r in caplog.records)


def test_num_non_numeric_key_is_nan_with_a_named_warning(caplog):
    with caplog.at_level("WARNING"):
        result = _num({"n": "not-a-number"}, "n")
    assert np.isnan(result)
    assert any("'n'" in r.message for r in caplog.records)


def test_num_present_numeric_key_passes_through_and_warns_nothing(caplog):
    with caplog.at_level("WARNING"):
        result = _num({"mean": 0.123456789}, "mean")
    assert result == pytest.approx(0.123456789)
    assert not caplog.records


def test_truncated_confirm_artifact_produces_nan_not_a_get_default(tmp_path, caplog):
    """The load-bearing negative: delete one key from a real-shaped test row
    and require NaN, never 0.0. A version using `d.get(key, 0.0)` instead of
    `_num` would read this test's `value` assertion as 0.0 == 0.0 and pass
    silently -- confirmed by hand: temporarily changing `_num`'s missing-key
    branch to `return 0.0` makes `np.isnan(seasonal["value"])` below fail
    (0.0 is not NaN), exactly the discrimination this test exists to force.
    """
    conf = _confirmation(tests=[
        {"family": "seasonal", "dev_favored": "chronos",
         # "mean" deliberately removed -- a truncated artifact.
         "lo": 0.05, "hi": 0.2, "p": 0.004, "p_holm": 0.012, "n": 288,
         "resample_unit": "series", "status": "tested", "confirmed": True,
         "mde_private": {"mde": 0.03}},
    ])
    (tmp_path / "confirm").mkdir()
    (tmp_path / "confirm" / "confirmation.json").write_text(json.dumps(conf), encoding="utf-8")
    finding = _finding("confirm.1", "confirm",
                       text="CONFIRM — 1/1 dev family hypotheses confirmed...")
    with caplog.at_level("WARNING"):
        rows = _extract_confirm_rows(tmp_path, finding)
    assert len(rows) == 1
    assert np.isnan(rows[0]["value"])
    assert any("mean" in r.message and "seasonal" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Table/export shape: dtypes, precision, row-count accounting.
# ---------------------------------------------------------------------------

def test_row_count_equals_finding_count_plus_declared_strata_expansion(tmp_path):
    conf = _confirmation(tests=[
        {"family": "a", "mean": 0.1, "lo": 0.0, "hi": 0.2, "p": 0.01, "p_holm": 0.02,
         "n": 100, "resample_unit": "series", "status": "tested", "confirmed": True},
        {"family": "b", "mean": 0.2, "lo": 0.0, "hi": 0.3, "p": 0.01, "p_holm": 0.02,
         "n": 100, "resample_unit": "series", "status": "tested", "confirmed": True},
        {"family": "c", "mean": 0.3, "lo": 0.0, "hi": 0.4, "p": 0.01, "p_holm": 0.02,
         "n": 100, "resample_unit": "series", "status": "tested", "confirmed": True},
    ])
    (tmp_path / "confirm").mkdir()
    (tmp_path / "confirm" / "confirmation.json").write_text(json.dumps(conf), encoding="utf-8")
    findings = [
        _finding("l0.1", "l0"),
        _finding("l1.1", "l1"),
        _finding("confirm.1", "confirm",
                 text="CONFIRM — 3/3 dev family hypotheses confirmed..."),
    ]
    _write_findings(tmp_path, findings)
    rows = build_results_rows(tmp_path)
    # 1 (l0) + 1 (l1) + 3 (one per family, expanded from the single confirm finding)
    assert len(rows) == 1 + 1 + 3


def test_dataframe_has_a4_column_order_and_no_object_dtype_numeric_columns(tmp_path):
    conf = _confirmation(tests=[
        {"family": "a", "mean": 0.1, "lo": 0.0, "hi": 0.2, "p": 0.01, "p_holm": 0.02,
         "n": 100, "resample_unit": "series", "status": "tested", "confirmed": True},
    ])
    (tmp_path / "confirm").mkdir()
    (tmp_path / "confirm" / "confirmation.json").write_text(json.dumps(conf), encoding="utf-8")
    findings = [_finding("l0.1", "l0", value=1.79),
                _finding("confirm.1", "confirm",
                         text="CONFIRM — 1/1 dev family hypotheses confirmed...")]
    _write_findings(tmp_path, findings)
    df = build_results_table(tmp_path)
    assert list(df.columns) == RESULTS_COLUMNS
    for col in ("value", "ci_lo", "ci_hi", "p", "p_adjusted", "n",
                "reference_value", "mde"):
        assert df[col].dtype != object, f"{col} is object-dtype, expected numeric"


def test_export_round_trips_full_precision_through_csv_and_parquet(tmp_path):
    precise = 0.123456789012345
    findings = [_finding("l0.1", "l0", value=precise)]
    _write_findings(tmp_path, findings)
    paths = export_results_table(tmp_path)
    assert paths["csv"].exists() and paths["parquet"].exists()

    back_csv = pd.read_csv(paths["csv"])
    assert back_csv.loc[0, "value"] == pytest.approx(precise, abs=0, rel=1e-15)
    assert back_csv["value"].dtype != object

    back_parquet = pd.read_parquet(paths["parquet"])
    assert back_parquet.loc[0, "value"] == precise  # exact bit-for-bit round trip


def test_export_creates_report_dir_if_absent(tmp_path):
    findings = [_finding("l0.1", "l0")]
    _write_findings(tmp_path, findings)
    paths = export_results_table(tmp_path)
    assert paths["csv"].parent == tmp_path / "report"


# ---------------------------------------------------------------------------
# CLI wiring: `run.py --export-results <run_dir>` needs no --config.
# ---------------------------------------------------------------------------

def test_export_results_cli_flag_needs_no_config_and_writes_both_files(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "a_run"
    findings = [_finding("l0.1", "l0", value=1.5)]
    _write_findings(run_dir, findings)
    result = subprocess.run(
        [sys.executable, "run.py", "--export-results", str(run_dir)],
        cwd=repo_root, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run_dir / "report" / "results.csv").exists()
    assert (run_dir / "report" / "results.parquet").exists()
    assert "results.csv" in result.stdout
