"""SAE dictionary admission: the held-out split, the three-bar gate, and
layer substitution (ROADMAP.md sec 28.19, user-requested 2026-09-11).

The user's complaint was that some dictionaries have low reconstruction
fidelity and/or a large ΔMASE, which undermines every feature claim read off
them, and that a layer failing those bars should be RETRIED at another layer
of the same model before being written off. Three mechanisms answer it and
each is tested here:

- `split_series` -- a held-out set of SERIES (invariant 2, never windows),
  drawn family-stratified.
- `admission_verdict` -- three bars, each recording the value and threshold
  that decided it, with a third `None` state for an unmeasured bar.
- `_next_substitute_layer` -- the screen's next-best CAPTURED layer.

Synthetic with planted answers throughout; `_next_substitute_layer` reads
exactly one method off a store, so a stub stands in (the same shape as
`tests/test_sae_target_resolution.py`, which pins the sibling resolution
path) and no checkpoint, zarr store or GPU is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.train import (  # noqa: E402
    _next_substitute_layer, _rows_for_series, admission_verdict, split_series)


class _StubStore:
    """Only `layers(model)` is ever called on the real store by the function
    under test -- see `_next_substitute_layer`'s body."""

    def __init__(self, mapping):
        self._m = mapping

    def layers(self, model):
        return list(self._m[model])


# ---------------------------------------------------------------------------
# split_series
# ---------------------------------------------------------------------------

def test_the_split_is_disjoint_exhaustive_and_series_level():
    s = split_series(100, 0.2, seed=0)
    assert s["n_heldout"] == 20 and s["n_train"] == 80
    assert set(s["train"]).isdisjoint(set(s["heldout"]))
    assert sorted(np.concatenate([s["train"], s["heldout"]])) == list(range(100))
    assert s["reason"] is None


def test_holdout_is_family_stratified_not_a_prefix():
    """The corpus is written GROUPED BY TASK, so a head slice or an
    unstratified draw silently over- or under-represents whichever family
    sorts first (sec 11.38 / sec 15 A4).

    Asserted across MANY seeds, deliberately. A single-seed version of this
    test was written first and was INERT -- deleting `strata=families` from
    the implementation left all of it passing, because an unstratified draw
    of 20 from 90/10 happens to return exactly 2 rare at seed 0. Proportional
    representation at EVERY seed is the property only stratification has;
    one seed's correct answer is a coin that landed the right way up.
    """
    families = np.array(["common"] * 90 + ["rare"] * 10)
    counts = set()
    for seed in range(25):
        s = split_series(100, 0.2, seed=seed, families=families)
        held = families[s["heldout"]]
        counts.add(int((held == "rare").sum()))
        assert s["n_heldout"] == 20
        # A prefix-based split would take series 0..19, i.e. all `common`.
        assert sorted(s["heldout"])[:3] != [0, 1, 2]
    assert counts == {2}, f"rare-family count varied across seeds: {sorted(counts)}"


def test_zero_frac_reproduces_the_old_all_train_behaviour_bit_for_bit():
    """Every SAE number recorded before this split existed was fitted on all
    series. `holdout_frac: 0` must therefore be exactly that, not a small
    split -- otherwise enabling the feature silently re-decides those numbers
    (sec 2.1)."""
    s = split_series(64, 0.0, seed=0)
    assert s["n_train"] == 64 and s["n_heldout"] == 0
    assert list(s["train"]) == list(range(64))
    assert s["heldout"].size == 0
    assert s["reason"] == "disabled by config"


@pytest.mark.parametrize("n", [1, 2, 5])
def test_a_corpus_too_small_to_split_says_so_rather_than_holding_out_one(n):
    """A one-series held-out set supports no mean worth reading. The state is
    NAMED, not silently rendered as a working split whose fidelity is one
    series' luck (sec 11.37)."""
    s = split_series(n, 0.2, seed=0)
    assert s["n_heldout"] == 0
    assert s["n_train"] == n
    assert "too small" in (s["reason"] or "")


def test_window_rows_follow_the_series_major_layout():
    """`load_all_windows` returns `[n_series * n_windows, D]` series-major.
    Getting this wrong mixes training windows into the held-out rows, which
    is invisible -- it just makes held-out fidelity match training fidelity,
    i.e. it silently disables the check."""
    rows = _rows_for_series(np.array([0, 2]), n_windows=3)
    assert list(rows) == [0, 1, 2, 6, 7, 8]
    assert _rows_for_series(np.array([], dtype=int), 3).size == 0


def test_train_and_holdout_rows_never_overlap():
    s = split_series(40, 0.25, seed=3)
    tr = set(_rows_for_series(s["train"], 5).tolist())
    ho = set(_rows_for_series(s["heldout"], 5).tolist())
    assert tr.isdisjoint(ho)
    assert len(tr) + len(ho) == 40 * 5


# ---------------------------------------------------------------------------
# admission_verdict
# ---------------------------------------------------------------------------

def _v(**kw):
    base = dict(fidelity_train=0.95, fidelity_heldout=0.93, delta_mase_token=0.05,
                min_fidelity=0.85, max_abs_delta_mase=0.25, dead_rate_passed=True)
    base.update(kw)
    return admission_verdict(**base)


def test_a_sound_dictionary_is_admitted_and_every_bar_records_its_threshold():
    v = _v()
    assert v["passed"] is True and v["failed_checks"] == []
    for c in v["checks"]:
        assert "rule" in c and c["passed"] is True
    fid = [c for c in v["checks"] if c["check"] == "reconstruction fidelity"][0]
    assert fid["threshold"] == 0.85 and fid["value"] == 0.93


def test_fidelity_is_judged_on_the_held_out_split_when_one_exists():
    """The plant: a dictionary that rebuilds its OWN training rows well
    (0.99) and generalizes badly (0.60). Judging it on the training value
    admits it; that is memorization, and it is exactly what the split was
    added to catch."""
    v = _v(fidelity_train=0.99, fidelity_heldout=0.60)
    assert v["passed"] is False
    fid = [c for c in v["checks"] if c["check"] == "reconstruction fidelity"][0]
    assert fid["split"] == "heldout" and fid["value"] == 0.60


def test_with_no_holdout_the_train_value_is_used_and_the_split_is_named():
    v = _v(fidelity_heldout=None, fidelity_train=0.91)
    fid = [c for c in v["checks"] if c["check"] == "reconstruction fidelity"][0]
    assert fid["split"] == "train" and fid["value"] == 0.91
    assert v["passed"] is True


def test_the_gate_reads_token_granularity_not_window():
    """The user's reported contradiction -- a target with the run's BEST
    fidelity and a large ΔMASE -- is the window/token confound: for a model
    whose token width differs from `alignment.window`, the window number
    carries a broadcast loss that is a property of the tokenizer, not the
    dictionary (measured on runs/full_report_run_4model: Chronos-Bolt
    block.4, fidelity 0.955, window +0.378, token -0.135). The gate takes
    the token number, so that target is admitted."""
    v = _v(delta_mase_token=-0.135)
    assert v["passed"] is True
    dm = [c for c in v["checks"] if c["check"] == "forecast preservation"][0]
    assert dm["value"] == -0.135 and "TOKEN" in dm["rule"]
    # The bar is on the MAGNITUDE: a large negative delta is as much a
    # behaviour change as a large positive one.
    assert _v(delta_mase_token=-0.9)["passed"] is False


def test_an_unmeasured_bar_makes_the_verdict_undecidable_not_a_pass():
    """sec 11.37: an absent measurement must not yield a confident answer in
    EITHER direction. `None` is a third state, and the reason names which bar
    is missing."""
    v = _v(delta_mase_token=None)
    assert v["passed"] is None
    assert "forecast preservation" in v["unmeasured_checks"]
    assert "not decidable" in v["reason"]
    assert np.isnan(np.nan)  # sanity: NaN is treated the same as absent below
    assert _v(delta_mase_token=float("nan"))["passed"] is None


def test_a_real_failure_beats_an_unmeasured_bar_in_the_verdict():
    """A dictionary with one failing bar and one unmeasured bar has FAILED --
    reporting it as undecidable would let a known-bad dictionary through on
    the strength of a second bar nobody could measure."""
    v = _v(fidelity_heldout=0.10, delta_mase_token=None)
    assert v["passed"] is False
    assert v["failed_checks"] == ["reconstruction fidelity"]


def test_the_dead_rate_gate_is_folded_in_rather_than_left_to_the_reader():
    assert _v(dead_rate_passed=False)["passed"] is False
    v = _v(dead_rate_passed=None)
    assert not any(c["check"] == "dead-feature rate" for c in v["checks"])


# ---------------------------------------------------------------------------
# _next_substitute_layer
# ---------------------------------------------------------------------------

# The screen's score DESCENDS as the store's layer order ASCENDS, so the
# best-scoring unattempted layer is never the last captured one. A first draft
# of this fixture used an interleaved ranking and was INERT: at every step its
# score-best and its "last captured layer" agreed, so it would have passed
# against a picker that reads position and ignores the screen entirely
# (sec 11.53's postscript -- a plant that changes nothing is indistinguishable
# from a mechanism that works). Verified to discriminate by executing both
# rules over this fixture.
_SEL = {"M": {"layers": ["b.0", "b.1", "b.2", "b.3"],
              "score_per_layer": [0.9, 0.7, 0.4, 0.1]}}


def test_substitution_follows_the_screens_own_ranking_not_layer_position():
    """Each assertion is a case where the screen's next choice and the naive
    "best captured layer is the last one" rule give DIFFERENT answers."""
    store = _StubStore({"M": ["b.0", "b.1", "b.2", "b.3"]})
    assert _next_substitute_layer(_SEL, store, "M", set()) == "b.0"       # not b.3
    assert _next_substitute_layer(_SEL, store, "M", {"b.0"}) == "b.1"     # not b.3
    assert _next_substitute_layer(_SEL, store, "M", {"b.0", "b.1"}) == "b.2"


def test_a_layer_the_screen_never_scored_sorts_after_every_scored_one():
    """`layer_screen` runs its own stride-1 store (sec 15 A1), so the two
    layer lists genuinely differ -- an unscored captured layer is a real
    state, and guessing a score for it would invent a ranking. `b.9` sorts
    last here DESPITE being last in the store, i.e. despite being exactly
    what a position rule would pick first."""
    store = _StubStore({"M": ["b.2", "b.9"]})           # b.9 unscored
    assert _next_substitute_layer(_SEL, store, "M", set()) == "b.2"
    assert _next_substitute_layer(_SEL, store, "M", {"b.2"}) == "b.9"


def test_exhausted_returns_none_rather_than_retrying_an_attempted_layer():
    """An infinite retry loop is the failure mode this guards; `None` is the
    honest answer and the caller logs and moves on, which is what the user
    asked for ('if those still don't work just move on')."""
    store = _StubStore({"M": ["b.0", "b.1"]})
    assert _next_substitute_layer(_SEL, store, "M", {"b.0", "b.1"}) is None


def test_a_model_absent_from_the_store_degrades_to_none():
    store = _StubStore({"M": ["b.0"]})
    assert _next_substitute_layer(_SEL, store, "OTHER", set()) is None
    assert _next_substitute_layer({}, store, "M", set()) == "b.0"


# ---------------------------------------------------------------------------
# The report half: held-out columns and the overlaid fidelity bar
# ---------------------------------------------------------------------------

def _meta(tmp_path: Path, entries: dict) -> Path:
    import json
    run = tmp_path / "run"
    (run / "sae").mkdir(parents=True, exist_ok=True)
    (run / "sae" / "meta.json").write_text(json.dumps(entries), encoding="utf-8")
    return run


_TRAINED = {
    "reconstruction_fidelity": 0.96, "dead_feature_rate": 0.12,
    "dead_rate_gate": {"passed": True, "threshold": 0.30},
    "ground_truth_alignment": {"mean_abs_rho_matched": 0.31,
                               "permutation_null": {"mean_abs_rho_null_p95": 0.22}},
    "forecast_preservation": {"mase_delta": 0.38},
    "forecast_preservation_token": {"mase_delta": 0.09},
}
_HELD = dict(_TRAINED, **{
    "reconstruction_fidelity_heldout": 0.71,
    "forecast_preservation_token_heldout": {"mase_delta": 0.31},
    "admission": {"passed": False, "reason": "failed: reconstruction fidelity",
                  "checks": [{"check": "reconstruction fidelity", "split": "heldout",
                              "value": 0.71, "threshold": 0.85,
                              "rule": "fidelity >= 0.85", "passed": False}]},
})


def test_health_table_gains_the_held_out_columns_when_they_were_measured(tmp_path):
    from tsfm_lens.report.derived import sae_health
    df = sae_health(_meta(tmp_path, {"M/b.0": _HELD}))
    assert df.loc[0, "reconstruction fidelity"] == 0.96
    assert df.loc[0, "reconstruction fidelity (held out)"] == 0.71
    assert df.loc[0, "ΔMASE (token, held out)"] == 0.31
    assert df.loc[0, "admission"].startswith("REFUSED")
    # The threshold reaches the figure as a NUMBER, never re-derived there.
    assert df.attrs["fidelity_thresholds"] == {"M/b.0": 0.85}


def test_a_run_predating_the_split_drops_those_columns_rather_than_blanking_them(tmp_path):
    """Backward compatibility AND sec 11.37: three columns of blanks would
    assert a held-out comparison that nobody performed."""
    from tsfm_lens.report.derived import sae_health
    df = sae_health(_meta(tmp_path, {"M/b.0": _TRAINED}))
    assert "reconstruction fidelity (held out)" not in df.columns
    assert "ΔMASE (token, held out)" not in df.columns
    assert "admission" not in df.columns
    assert df.loc[0, "reconstruction fidelity"] == 0.96      # unchanged


def test_one_target_missing_the_measurement_keeps_the_column(tmp_path):
    """Dropped only when EVERY target lacks it -- a single blank cell beside
    measured ones is itself the information."""
    from tsfm_lens.report.derived import sae_health
    df = sae_health(_meta(tmp_path, {"M/b.0": _HELD, "M/b.1": _TRAINED}))
    col = df["reconstruction fidelity (held out)"]
    assert col.notna().sum() == 1 and col.isna().sum() == 1


def _traces(html: str):
    import json
    dec = json.JSONDecoder()
    i = html.find("Plotly.newPlot")
    j = html.find("[", i)
    data, end = dec.raw_decode(html[j:])
    k = html.find("{", j + end)
    layout, _ = dec.raw_decode(html[k:])
    return data, layout


def test_the_fidelity_panel_overlays_held_out_on_the_same_row(tmp_path):
    """The user's request: 'overlaying train and test reconstruction fidelity
    on the same bar'. The bar stays TRAIN and the held-out value is a marker
    on the same y, so the generalization gap is a horizontal distance rather
    than a subtraction the reader performs."""
    from tsfm_lens.report.derived import sae_health
    from tsfm_lens.report.report import _sae_health_figure
    df = sae_health(_meta(tmp_path, {"M/b.0": _HELD}))
    data, layout = _traces(_sae_health_figure(df))
    panel2 = [t for t in data if t.get("xaxis") == "x2"]
    bar = [t for t in panel2 if t.get("type") == "bar"][0]
    marker = [t for t in panel2 if t.get("name") == "held-out series"]
    assert list(bar["x"]) == [0.96]
    assert marker and list(marker[0]["x"]) == [0.71]
    # Both live on the same category row -- that is what "same bar" means.
    assert list(bar["y"]) == list(marker[0]["y"]) == ["M/b.0"]
    # A gap this large is drawn in the warning colour, not the muted one.
    seg = [t for t in panel2 if t.get("type") != "bar" and t.get("name") is None]
    assert any(s["line"]["color"] == "#B04A5A" for s in seg)
    assert any("fitted vs held-out" in a.get("text", "")
               for a in layout["annotations"])


def test_the_panel_is_unchanged_for_a_run_with_no_held_out_series(tmp_path):
    from tsfm_lens.report.derived import sae_health
    from tsfm_lens.report.report import _sae_health_figure
    df = sae_health(_meta(tmp_path, {"M/b.0": _TRAINED}))
    data, layout = _traces(_sae_health_figure(df))
    panel2 = [t for t in data if t.get("xaxis") == "x2"]
    assert len(panel2) == 1 and panel2[0]["type"] == "bar"
    assert panel2[0].get("showlegend") is False
    assert not any("held-out" in a.get("text", "") for a in layout["annotations"])


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
