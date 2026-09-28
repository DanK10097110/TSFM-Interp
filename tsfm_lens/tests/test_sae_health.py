"""`derived.sae_health` -- the SAE section's dictionary-health table.

Added 2026-09-03. The table replaces eleven run-on `·`-separated per-target
paragraphs (a user review: "there is a lot of excess information ... I don't
want the viewer to be confused"), so what these tests protect is not the
layout but the two things the paragraphs got wrong: the verdicts were
authored rather than derived, and the reader could not compare targets
without diffing prose.

Every fixture is synthetic with a planted answer, and the load-bearing tests
are the negatives -- an alive dictionary that aligns badly, a gate whose
threshold is missing from its own verdict, and an insertion order that
disagrees with the sort.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tsfm_lens.report.derived import sae_health


def _write(tmp_path: Path, meta: dict) -> Path:
    import json
    run = tmp_path / "run"
    (run / "sae").mkdir(parents=True, exist_ok=True)
    (run / "sae" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return run


def _entry(dead: float, rho: float, p95: float, *, fidelity: float = 0.8,
           threshold: float | None = 0.30, passed: bool | None = None,
           win: float | None = 0.2, tok: float | None = 0.2) -> dict:
    e: dict = {"reconstruction_fidelity": fidelity, "dead_feature_rate": dead,
               "ground_truth_alignment": {
                   "mean_abs_rho_matched": rho,
                   "permutation_null": {"mean_abs_rho_null_p95": p95}}}
    if threshold is not None:
        e["dead_rate_gate"] = {
            "threshold": threshold, "value": dead,
            "passed": (dead <= threshold) if passed is None else passed}
    if win is not None:
        e["forecast_preservation"] = {"mase_delta": win}
    if tok is not None:
        e["forecast_preservation_token"] = {"mase_delta": tok}
    return e


def test_no_artifact_degrades_to_an_empty_frame(tmp_path):
    assert sae_health(tmp_path / "nonexistent").empty


def test_a_mostly_alive_dictionary_that_aligns_badly_is_not_reported_as_healthy(tmp_path):
    """The load-bearing negative, and a measured pattern rather than a
    hypothetical one: on this repo's own corpus the ~1%-dead dictionaries
    align to ground truth WORSE than the ~95%-dead ones. So the dead-rate
    verdict must not be able to speak for the alignment verdict."""
    run = _write(tmp_path, {"m/l": _entry(dead=0.01, rho=0.10, p95=0.12)})
    row = sae_health(run).iloc[0]
    assert row["dead-rate gate"].startswith("passes")
    assert "does NOT clear" in row["alignment vs null"]
    # And the shortfall is signed and printed, not just labelled.
    assert "-0.020" in row["alignment vs null"]


def test_the_gate_threshold_travels_with_its_own_verdict(tmp_path):
    """A bare "fails" is the invisible-threshold defect `derived.py` exists to
    prevent -- a reader has to be able to disagree with the arithmetic."""
    run = _write(tmp_path, {"m/l": _entry(dead=0.95, rho=0.3, p95=0.1,
                                          threshold=0.30)})
    assert "30%" in sae_health(run).iloc[0]["dead-rate gate"]


def test_an_absent_gate_is_not_a_passing_gate(tmp_path):
    """`passed` missing means nothing was configured to check, which is a
    third state -- collapsing it into "passes" asserts a check that never
    ran (`CLAUDE.md` sec 11.37)."""
    run = _write(tmp_path, {"m/l": _entry(dead=0.99, rho=0.3, p95=0.1,
                                          threshold=None)})
    verdict = sae_health(run).iloc[0]["dead-rate gate"]
    assert verdict == "no gate configured"
    assert "pass" not in verdict


def test_a_missing_permutation_null_is_not_comparable_rather_than_clearing(tmp_path):
    run = _write(tmp_path, {"m/l": {"reconstruction_fidelity": 0.8,
                                    "dead_feature_rate": 0.5,
                                    "ground_truth_alignment": {
                                        "mean_abs_rho_matched": 0.9}}})
    assert sae_health(run).iloc[0]["alignment vs null"] == "not comparable"


def test_rows_are_ordered_worst_dictionary_first_not_by_insertion(tmp_path):
    """Planted so that insertion order is the REVERSE of dead-rate order: a
    reduction that preserved dict order would pass every other assertion
    here."""
    run = _write(tmp_path, {"a/0": _entry(dead=0.05, rho=0.3, p95=0.1),
                            "b/0": _entry(dead=0.90, rho=0.3, p95=0.1),
                            "c/0": _entry(dead=0.50, rho=0.3, p95=0.1)})
    assert list(sae_health(run)["target"]) == ["b/0", "c/0", "a/0"]


def test_a_target_missing_forecast_preservation_keeps_its_row(tmp_path):
    """The dead rate is worth seeing even when preservation was not measured,
    so the cell is blank and the row survives -- dropping it would hide the
    least trustworthy dictionaries, which are exactly the ones whose extra
    stages tend not to have run."""
    run = _write(tmp_path, {"m/l": _entry(dead=0.97, rho=0.3, p95=0.1,
                                          win=None, tok=None)})
    df = sae_health(run)
    assert len(df) == 1
    assert df.iloc[0]["ΔMASE (window)"] is None
    assert df.iloc[0]["granularity gap"] is None
    assert df.iloc[0]["dead rate"] == pytest.approx(0.97)


def test_the_granularity_gap_is_zero_only_when_the_two_agree(tmp_path):
    """The gap is the architecture confound made visible: it is exactly 0 for
    a model whose token width equals the alignment window and large for one
    whose does not (ROADMAP.md sec 16 E15). A gap column that could not
    distinguish them would be decoration."""
    run = _write(tmp_path, {"same/0": _entry(dead=0.9, rho=0.3, p95=0.1,
                                             win=0.25, tok=0.25),
                            "diff/0": _entry(dead=0.9, rho=0.3, p95=0.1,
                                             win=2.34, tok=0.40)})
    gaps = dict(zip(sae_health(run)["target"], sae_health(run)["granularity gap"]))
    assert gaps["same/0"] == pytest.approx(0.0)
    assert gaps["diff/0"] == pytest.approx(1.94)


def test_sae_health_names_no_model_or_architecture():
    """The adaptivity contract: this must transfer to models nobody has run."""
    import inspect

    from tsfm_lens.report import derived as derived_mod
    src = inspect.getsource(derived_mod.sae_health)
    body = src.split('"""')[2] if src.count('"""') >= 2 else src
    for banned in ("TimesFM", "Chronos", "Sundial", "model_a", "model_b",
                   "models[0]", "models[1]", "stacked_xf", "encoder.block"):
        assert banned not in body, f"{banned!r} would tie this reduction to one run"


def _seed_floor(values: list[float]) -> dict:
    """A `seed_floor` block in the shape `sae/train.py` actually writes it.

    The nesting is load-bearing and is the whole point of this fixture: the
    spread lives at `seed_floor["spread"][metric]`, NOT at
    `seed_floor[metric]`. The first version of the sign column read the
    latter, so a run that HAD measured a floor rendered as one that had not.
    """
    import numpy as np
    a = np.asarray(values, dtype=float)
    row = {"n": len(values), "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
           "min": float(a.min()), "max": float(a.max())}
    return {"n_seeds": len(values), "per_seed": [], "spread": {"mase_delta_window": row}}


def test_a_measured_seed_spread_is_not_reported_as_unmeasured(tmp_path):
    """The negative that the key-path bug would fail.

    Two targets differ only in whether the mean clears its own sd. Neither
    may read as "no seed spread measured" -- that phrasing is reserved for a
    run that trained one seed, and using it for a measured-but-unresolvable
    delta is CLAUDE.md sec 11.37's confusion of absent with bad, in the
    direction that reads as MORE certain than the evidence.
    """
    clear = _entry(dead=0.9, rho=0.3, p95=0.1, win=1.0)
    clear["seed_floor"] = _seed_floor([0.90, 0.95, 1.00, 1.05, 1.10])
    swamped = _entry(dead=0.8, rho=0.3, p95=0.1, win=0.01)
    swamped["seed_floor"] = _seed_floor([-0.20, 0.35, 0.02, -0.30, 0.18])
    lone = _entry(dead=0.7, rho=0.3, p95=0.1, win=0.5)

    df = sae_health(_write(tmp_path, {"a/0": clear, "b/0": swamped, "c/0": lone}))
    sign = dict(zip(df["target"], df["ΔMASE sign"]))
    assert sign["a/0"].startswith("resolvable")
    assert "over 5 seeds" in sign["a/0"]
    assert sign["b/0"].startswith("NOT resolvable")
    assert "no seed spread measured" not in sign["b/0"]
    assert sign["c/0"] == "no seed spread measured (1 training)"


def test_the_sign_column_is_dropped_when_nothing_measured_a_spread(tmp_path):
    """A column carrying one value at every row is noise in a comparison
    table, and the section's own seed-floor block already states the
    run-level situation once. The condition is measured, not configured --
    so the column must reappear the moment any target has a spread, which
    the test above relies on."""
    run = _write(tmp_path, {"a/0": _entry(dead=0.9, rho=0.3, p95=0.1),
                            "b/0": _entry(dead=0.5, rho=0.3, p95=0.1)})
    assert "ΔMASE sign" not in sae_health(run).columns
