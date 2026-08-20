"""Multiplicity across model pairs and across correction families (§18 F8).

Two properties are load-bearing here and neither is obvious from reading the
code. First, **a two-model run must be bit-identical to the single-pair
correction that preceded all-pairs** -- the widening is supposed to happen only
when a third model actually adds comparisons, and a seeding mistake in the
generalization would have moved every recorded L0 number in the repo silently
(`CLAUDE.md` §2.1). Second, **the bootstrap p-floor puts a hard cap on how many
comparisons a correction family can hold**: p is floored at 1/n_boot, so the
smallest Holm-adjusted p reachable is n_tests/n_boot regardless of effect size,
and past alpha the family is unsatisfiable. That was found by measuring rather
than by trusting a demotion that looked textbook (see this file's own history
in `ROADMAP.md` §18 F8), and it is the thing most likely to be reintroduced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.l0_behavioral import _summarize
from tsfm_lens.analysis.stats import holm, paired_bootstrap
from tsfm_lens.config import ModelConfig, PipelineConfig
from tsfm_lens.report.report import (Finding, _multiplicity_block,
                                     _multiplicity_scopes, _other_pairs_block)
from tsfm_lens.utils import save_json

_FAMILIES = ["alpha", "beta", "gamma", "delta"]


def _metrics(seed: int = 7) -> pd.DataFrame:
    """Three models over four families; B is clearly worse on `beta` only."""
    rng = np.random.default_rng(seed)
    rows = []
    for fam in _FAMILIES:
        for i in range(30):
            sid = f"{fam}-{i}"
            for model, bias in (("A", 0.0), ("B", 0.35 if fam == "beta" else 0.01),
                                ("C", 0.5)):
                rows.append(dict(model=model, series_id=sid, family=fam, archetype=None,
                                 mase=float(1.0 + bias + rng.normal(0, 0.2)),
                                 smape=float(rng.uniform(5, 20)),
                                 pinball=float(rng.uniform(0.1, 0.5)),
                                 mae_over_mad=float(rng.uniform(0.5, 1.5)),
                                 mase_reliable=True))
    return pd.DataFrame(rows)


def _cfg(names, n_boot=400) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.models = [ModelConfig(name=n, adapter="mock_patch") for n in names]
    cfg.run.seed = 11
    cfg.stats.n_boot = n_boot
    cfg.stats.min_series = 5
    return cfg


def test_two_models_produce_exactly_one_pair_and_the_unwidened_correction():
    """With two models the joint (pair, family) set IS the family set, so every
    adjusted p equals the single-pair Holm that predates all-pairs."""
    m = _metrics()
    s = _summarize(m[m.model.isin(["A", "B"])].copy(), _cfg(["A", "B"]))

    assert s["multiplicity"]["n_pairs"] == 1
    assert s["multiplicity"]["n_tests"] == len(s["family_tests"])
    direct = holm({t["family"]: t["p"] for t in s["family_tests"]})
    assert {t["family"]: t["p_holm"] for t in s["family_tests"]} == pytest.approx(direct)


def test_third_model_widens_the_correction_and_can_only_demote():
    """Adding a model adds pairs, and the joint correction is over all of them.

    The two assertions are the ones that make "visibly demoted" possible at
    all: no family's adjusted p may DROP when comparisons are added, and the
    designated pair's strengths may only shrink. A per-pair correction (the
    obvious wrong implementation) would leave both unchanged.
    """
    m = _metrics()
    two = _summarize(m[m.model.isin(["A", "B"])].copy(), _cfg(["A", "B"]))
    three = _summarize(m.copy(), _cfg(["A", "B", "C"]))

    assert three["multiplicity"]["n_pairs"] == 3
    assert three["multiplicity"]["n_tests"] == 3 * two["multiplicity"]["n_tests"]

    p2 = {t["family"]: t["p_holm"] for t in two["family_tests"]}
    p3 = {t["family"]: t["p_holm"] for t in three["family_tests"]}
    assert set(p2) == set(p3)
    assert all(p3[f] >= p2[f] - 1e-12 for f in p2), (p2, p3)
    assert any(p3[f] > p2[f] + 1e-12 for f in p2), "correction did not widen at all"

    for model, fams in three["strengths"].items():
        assert set(fams) <= set(two["strengths"][model])


def test_raw_p_values_of_the_designated_pair_are_untouched_by_the_third_model():
    """Only the ADJUSTMENT widens. If the raw bootstrap p moved too, the seeds
    were re-derived and every recorded L0 number would have drifted."""
    m = _metrics()
    two = _summarize(m[m.model.isin(["A", "B"])].copy(), _cfg(["A", "B"]))
    three = _summarize(m.copy(), _cfg(["A", "B", "C"]))
    assert ([t["p"] for t in two["family_tests"]]
            == [t["p"] for t in three["family_tests"]])


def test_family_seed_convention_counts_skipped_families_too():
    """Pins the exact seed each family's bootstrap uses: `seed + 40 + fi`, with
    `fi` the family's index among ALL families in sort order -- including ones
    too small to test.

    This is the convention the pre-all-pairs code had, as a side effect of
    `enumerate(groupby(...))` advancing across the `continue`. Preserving it is
    the whole reason a two-model run reproduces bit-for-bit; a running counter
    over only the TESTED families would look tidier, be defensible in
    isolation, and silently move every recorded L0 p-value in the repo.
    Checked against `paired_bootstrap` called directly rather than against the
    old implementation, so the test states the convention instead of just
    comparing two versions of it.
    """
    m = _metrics()
    tiny = pd.DataFrame([dict(model=mm, series_id="tiny-0", family="aaa_tiny",
                              archetype=None, mase=1.0, smape=9.0, pinball=0.2,
                              mae_over_mad=1.0, mase_reliable=True)
                         for mm in ("A", "B")])
    both = pd.concat([tiny, m[m.model.isin(["A", "B"])]], ignore_index=True)
    cfg = _cfg(["A", "B"])
    out = _summarize(both.copy(), cfg)

    assert "aaa_tiny" not in {t["family"] for t in out["family_tests"]}, \
        "a 1-series family must be skipped, not tested"

    order = sorted(both["family"].unique())
    assert order[0] == "aaa_tiny", "the skipped family must sort FIRST to bite"
    got = {t["family"]: t["p"] for t in out["family_tests"]}
    for fi, fam in enumerate(order):
        grp = both[(both["family"] == fam)].pivot_table(
            index="series_id", columns="model", values="mase")
        if fam not in got:
            continue
        ref = paired_bootstrap((grp["B"] - grp["A"]).to_numpy(),
                               cfg.stats.n_boot, cfg.run.seed + 40 + fi, cfg.stats.ci)
        assert got[fam] == pytest.approx(ref["p"]), (
            f"{fam}: seed index {fi} does not reproduce the reported p -- the "
            f"skipped family was not counted")


def test_unsatisfiable_correction_is_named_not_silently_rendered(tmp_path):
    """n_tests/n_boot > alpha means no effect of any size can be significant.
    That must be stated, not left as a table of honest-looking non-results."""
    summary = {"multiplicity": {"scope": "l0.family", "method": "holm", "alpha": 0.05,
                                "n_models": 3, "n_pairs": 3, "n_tests": 18,
                                "n_boot": 150, "min_attainable_p_holm": 18 / 150,
                                "most_stringent_threshold": 0.05 / 18,
                                "designated_pair": ["A", "B"]}}
    findings: list = []
    html = _multiplicity_block(tmp_path, summary, findings)
    assert "Unsatisfiable correction" in html
    assert "360" in html, "must say how large n_boot needs to be"
    assert any("UNSATISFIABLE" in f.text for f in findings)


def test_a_satisfiable_correction_says_nothing_alarming(tmp_path):
    """The contrapositive, so the warning can't quietly become unconditional."""
    summary = {"multiplicity": {"scope": "l0.family", "method": "holm", "alpha": 0.05,
                                "n_models": 3, "n_pairs": 3, "n_tests": 18,
                                "n_boot": 2000, "min_attainable_p_holm": 18 / 2000,
                                "most_stringent_threshold": 0.05 / 18,
                                "designated_pair": ["A", "B"]}}
    findings: list = []
    html = _multiplicity_block(tmp_path, summary, findings)
    assert "Unsatisfiable" not in html
    assert "Multiplicity ledger" in html
    assert not any("UNSATISFIABLE" in f.text for f in findings)


def test_scopes_are_counted_separately_and_never_merged(tmp_path):
    """Three Holm families, counted, not pooled -- pooling would make a
    pre-registered confirmation pay for every exploratory look (§6.7)."""
    save_json(tmp_path / "confirm" / "confirmation.json",
              {"alpha": 0.05, "tests": [{"p_holm": 0.01, "n_boot": 500},
                                        {"p_holm": 0.4, "n_boot": 500}]})
    summary = {"multiplicity": {"scope": "l0.family", "method": "holm", "alpha": 0.05,
                                "n_models": 2, "n_pairs": 1, "n_tests": 6,
                                "n_boot": 500, "min_attainable_p_holm": 6 / 500,
                                "most_stringent_threshold": 0.05 / 6,
                                "designated_pair": ["A", "B"]},
               "per_archetype": {"alpha": 0.05,
                                 "tests": [{"p_holm": 0.02, "n_boot": 500}] * 3}}
    scopes = _multiplicity_scopes(tmp_path, summary)
    assert [sc["scope"] for sc in scopes] == ["l0.family", "l0.archetype",
                                              "confirm.hypotheses"]
    assert [sc["n_tests"] for sc in scopes] == [6, 3, 2]

    findings: list = []
    html = _multiplicity_block(tmp_path, summary, findings)
    assert "11 corrected comparisons" in html
    assert "3 independent correction families" in html


def test_other_pairs_block_is_empty_for_two_models(tmp_path):
    """A two-model report gains no new table at all -- there is no second pair,
    and this is the property that keeps every existing report unchanged."""
    findings: list = []
    summary = {"multiplicity": {"designated_pair": ["A", "B"], "n_pairs": 1},
               "pairwise": [{"a": "A", "b": "B", "family_tests": [], "strengths": {}}]}
    assert _other_pairs_block(summary, findings) == ""
    assert findings == []


def test_other_pairs_block_states_that_other_stages_never_saw_these_pairs(tmp_path):
    """The stronger half of the honesty claim: the extra pairs are unexamined
    everywhere else, which is not the same as weakly evidenced."""
    rows = [{"family": "beta", "ratio": 1.2, "mean": 0.3, "lo": 0.1, "hi": 0.5,
             "p": 0.001, "n_boot": 500, "p_holm": 0.01, "favored": "A"}]
    summary = {"multiplicity": {"designated_pair": ["A", "B"], "n_pairs": 3},
               "pairwise": [{"a": "A", "b": "B", "family_tests": [], "strengths": {}},
                            {"a": "A", "b": "C", "family_tests": rows,
                             "strengths": {"A": ["beta"], "C": []}}]}
    findings: list = []
    html = _other_pairs_block(summary, findings)
    assert "Paired family tests — A vs. C" in html
    assert "unexamined" in html
    assert any("stronger than C" in f.text for f in findings)
