"""The SAE section's two new figures, and the reduction behind the heatmap.

Added 2026-09-03 on a user instruction ("good graphs in final report"). The
section rendered 26 tables against 4 plotly figures, so three questions that
are comparisons ACROSS eleven targets were being asked of columns of
decimals. What these tests protect is not the styling but the three places a
figure can quietly lie: a reference line drawn where it does not apply, a row
order that destroys the trend the figure exists to show, and a heatmap built
on the provenance field the whole residualization pass exists to remove.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsfm_lens.report.derived import sae_health, sae_structural_profile
from tsfm_lens.report.report import (_depth_ordered_targets, _sae_health_figure,
                                     _sae_structural_figure, _vocab_pretty)


def _write(tmp_path: Path, meta: dict) -> Path:
    run = tmp_path / "run"
    (run / "sae").mkdir(parents=True, exist_ok=True)
    (run / "sae" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return run


def _array_after(html: str, key: str) -> str:
    """The first `"<key>":[...]` array in plotly's emitted JSON, as raw text.

    Kept as one helper because the emitted form is not the obvious one: "/"
    arrives as \u002f, so ordering assertions have to be written against
    tokens that survive escaping.
    """
    frag = html[html.index(f'"{key}":['):]
    return frag[:frag.index("]")]


def _sep_entry(fields: list[str], *, dead: float = 0.5, threshold: float | None = 0.30,
               provenance: str = "generator_mixture") -> dict:
    """A target whose `separated` features track `fields`, one feature each.

    `provenance` is planted in every feature's provenance slot so a heatmap
    that read the wrong slot would be visibly wrong rather than plausibly
    wrong -- it is the field name ROADMAP.md sec 26 A3 found on 78.6% of
    matched features before residualization.
    """
    e: dict = {
        "reconstruction_fidelity": 0.8, "dead_feature_rate": dead,
        "ground_truth_alignment": {
            "mean_abs_rho_matched": 0.3,
            "permutation_null": {"mean_abs_rho_null_p95": 0.1},
            "separated": {"features": [
                {"feature": i,
                 "structural": {"field": f, "rho": 0.2 + 0.01 * i, "n": 400},
                 "provenance": {"field": provenance, "rho": 0.9, "n": 900}}
                for i, f in enumerate(fields)]}}}
    if threshold is not None:
        e["dead_rate_gate"] = {"threshold": threshold, "value": dead,
                               "passed": dead <= threshold}
    return e


# --------------------------------------------------------------- the reduction

def test_the_heatmap_reads_the_residualized_field_never_the_provenance_one(tmp_path):
    """The load-bearing negative.

    Every planted feature carries a provenance field alongside its
    structural one. A heatmap built on `provenance` -- or on a raw
    best-of-all-fields match -- would render a picture of how the benchmark
    was built, which is the exact defect sec 26 A2/A3 fixed upstream. So the
    provenance name must not appear anywhere in the profile.
    """
    run = _write(tmp_path, {"m/0": _sep_entry(["n_seasonalities", "ar_coeff_sum"])})
    prof = sae_structural_profile(run)
    assert set(prof["field"]) == {"n_seasonalities", "ar_coeff_sum"}
    assert "generator_mixture" not in set(prof["field"])


def test_shares_are_per_target_so_unequal_match_counts_stay_comparable(tmp_path):
    """A raw-count heatmap reads dictionary size as signal. Two targets that
    matched 2 and 6 features around the SAME single property must both read
    as 100% of their own matches, not as 2 vs 6."""
    run = _write(tmp_path, {"small/0": _sep_entry(["n_seasonalities"] * 2),
                            "big/0": _sep_entry(["n_seasonalities"] * 6)})
    prof = sae_structural_profile(run)
    by = dict(zip(prof["target"], prof["share_of_matched"]))
    assert by["small/0"] == pytest.approx(1.0)
    assert by["big/0"] == pytest.approx(1.0)
    counts = dict(zip(prof["target"], prof["n_features"]))
    assert counts == {"small/0": 2, "big/0": 6}
    for t in ("small/0", "big/0"):
        sub = prof[prof["target"] == t]
        assert sub["share_of_matched"].sum() == pytest.approx(1.0)


def test_a_target_with_no_separated_block_is_absent_not_zero(tmp_path):
    """A target whose artifact predates the residualization pass has NOT
    been measured to track nothing -- it has not been measured. A zero row
    would assert the former (CLAUDE.md sec 11.37: absent and bad are
    different outcomes)."""
    plain = {"reconstruction_fidelity": 0.8, "dead_feature_rate": 0.5,
             "ground_truth_alignment": {"mean_abs_rho_matched": 0.3}}
    run = _write(tmp_path, {"old/0": plain, "new/0": _sep_entry(["ar_coeff_sum"])})
    prof = sae_structural_profile(run)
    assert set(prof["target"]) == {"new/0"}


def test_no_separated_block_anywhere_gives_an_empty_frame_not_a_crash(tmp_path):
    run = _write(tmp_path, {"old/0": {"dead_feature_rate": 0.5}})
    assert sae_structural_profile(run).empty
    assert sae_structural_profile(tmp_path / "nope").empty


# ------------------------------------------------------------------ the figures

def test_one_gate_line_is_drawn_only_when_every_target_shares_a_threshold(tmp_path):
    """The load-bearing negative for the health figure.

    A single vertical reference line across eleven rows is only honest if
    every row was judged against it. Two targets gated at different
    thresholds must produce NO line rather than one that misdescribes
    whichever row it crosses -- the same rule sec 24's scorecard follows
    when it renders "not comparable" instead of a pass or a fail.
    """
    same = _write(tmp_path / "a", {"x/0": _sep_entry(["ar_coeff_sum"], threshold=0.30),
                                   "y/0": _sep_entry(["ar_coeff_sum"], threshold=0.30)})
    mixed = _write(tmp_path / "b", {"x/0": _sep_entry(["ar_coeff_sum"], threshold=0.30),
                                    "y/0": _sep_entry(["ar_coeff_sum"], threshold=0.55)})
    assert "gate 30%" in _sae_health_figure(sae_health(same))
    html = _sae_health_figure(sae_health(mixed))
    assert "gate 30%" not in html and "gate 55%" not in html


def test_an_ungated_target_draws_no_line_from_its_gated_neighbour(tmp_path):
    """`no gate configured` is a third state. A run mixing gated and ungated
    targets must not borrow the gated one's threshold as a run-wide line."""
    run = _write(tmp_path, {"x/0": _sep_entry(["ar_coeff_sum"], threshold=0.30),
                            "y/0": _sep_entry(["ar_coeff_sum"], threshold=None)})
    thresholds = sae_health(run).attrs["dead_rate_thresholds"]
    assert thresholds["y/0"] is None
    assert "gate 30%" in _sae_health_figure(sae_health(run))


def test_the_health_figure_colors_by_the_gates_own_verdict(tmp_path):
    """Not by a threshold re-derived in `report.py`. A target the artifact
    records as FAILING must be drawn as failing even if its rate would pass
    a differently-set bar, because the table beside it prints that verdict
    and the two must not disagree."""
    run = _write(tmp_path, {"pass/0": _sep_entry(["ar_coeff_sum"], dead=0.1),
                            "fail/0": _sep_entry(["ar_coeff_sum"], dead=0.9)})
    html = _sae_health_figure(sae_health(run))
    assert "#B04A5A" in html          # the failing color is present
    assert "#4E8D6E" in html          # so is the passing one
    both = _write(tmp_path / "c", {"a/0": _sep_entry(["ar_coeff_sum"], dead=0.1),
                                   "b/0": _sep_entry(["ar_coeff_sum"], dead=0.1)})
    assert "#B04A5A" not in _sae_health_figure(sae_health(both))


def test_the_heatmap_does_not_reorder_rows_by_any_data_column(tmp_path):
    """The second load-bearing negative.

    The heatmap's stated purpose is reading a DEPTH TREND down one model's
    own layers, so no data column may set the row order -- including the
    count of the leftmost field, which is the tempting default, since that
    reorders layers by strength and destroys exactly the structure the
    figure exists to show. These targets carry no trailing `.<int>`, so
    depth ordering cannot apply and artifact order is what must survive.
    Planted so alphabetical, count-sorted and artifact order all differ.
    """
    run = _write(tmp_path, {
        "m/L9": _sep_entry(["n_seasonalities"] * 5 + ["ar_coeff_sum"]),
        "m/L1": _sep_entry(["n_seasonalities"] + ["ar_coeff_sum"] * 5),
        "m/L5": _sep_entry(["n_seasonalities"] * 3 + ["ar_coeff_sum"] * 3)})
    html = _sae_structural_figure(run)
    # Plotly's category axis builds upward, so the artifact's first target is
    # emitted LAST in the y array; the figure reverses once for that reason.
    # Match on the layer token alone: plotly escapes "/" as \u002f in the
    # emitted JSON, so a substring carrying the separator never matches.
    y = _array_after(html, "y")
    assert y.index("L5") < y.index("L1") < y.index("L9"), y
    assert "generator_mixture" not in html


def test_the_heatmap_orders_fields_by_total_matches_across_the_run(tmp_path):
    """A reader scanning left to right should meet the properties this
    corpus actually exercises first. Planted so the busiest field is the one
    alphabetically last."""
    run = _write(tmp_path, {"m/0": _sep_entry(["ar_coeff_sum"] + ["trend_scale"] * 4)})
    html = _sae_structural_figure(run)
    x = _array_after(html, "x")
    # The axis carries the READER-FACING label, so the expected strings are
    # derived through the same transform the figure uses rather than guessed
    # -- a hardcoded raw field name passes only until a label is reworded,
    # and a hardcoded pretty one fails silently in the same way.
    assert x.index(_vocab_pretty("trend_scale")) < x.index(_vocab_pretty("ar_coeff_sum")), x


def test_both_figures_degrade_to_empty_string_on_an_empty_run(tmp_path):
    """A section builder returning "" is skipped; one raising takes the whole
    section down with it (`report.py`'s builder contract)."""
    import pandas as pd
    assert _sae_health_figure(pd.DataFrame()) == ""
    assert _sae_structural_figure(tmp_path / "nope") == ""


def test_rows_run_shallowest_to_deepest_within_each_model(tmp_path):
    """Artifact order is NOT depth order, which is why this is separate.

    Under `sae.targets: auto` the artifact's key order is `layer_screen`'s
    score ranking -- on `runs/full_report_run_large` that is `stacked_xf.`
    2, 6, 18, 10, 16, so a reader scanning the rows as depth would see 18
    followed by 10. Models keep first-appearance order; only depth moves.
    """
    got = _depth_ordered_targets(["A/blk.2", "A/blk.6", "A/blk.18",
                                  "A/blk.10", "A/blk.16", "B/e.7", "B/e.1"])
    assert got == ["A/blk.2", "A/blk.6", "A/blk.10", "A/blk.16", "A/blk.18",
                   "B/e.1", "B/e.7"], got


def test_a_layer_with_no_trailing_index_keeps_its_artifact_position(tmp_path):
    """The negative that keeps this from becoming a guess. Every adapter in
    this repo ends a block name with an index, but a future one need not,
    and inventing a depth for it would misrepresent the axis rather than
    decline to show it -- so an unparseable name holds its own slot instead
    of being sorted to an end."""
    got = _depth_ordered_targets(["A/tail", "A/blk.9", "A/head", "A/blk.1"])
    # The two indexed layers sort among themselves and lead (depth is known);
    # the two unindexed ones follow in the order the artifact wrote them.
    assert got == ["A/blk.1", "A/blk.9", "A/tail", "A/head"], got
