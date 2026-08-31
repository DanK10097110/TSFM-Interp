"""The report's two-register legibility contract (user request, 2026-08-20).

Every figure must answer "what am I looking at" **without the reader clicking
anything**, and carry the detail behind one consistently-labelled dropdown.
A report whose subject lines are all collapsed is effectively unlabeled, and
the failure is invisible -- the HTML renders fine, it just cannot be read by
someone who did not build the pipeline. So it is pinned here rather than left
to inspection.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.report import _group_findings, _note, _scorecard  # noqa: E402

SMOKE = Path(__file__).resolve().parents[1] / "runs" / "smoke" / "report.html"


def test_a_note_shows_its_subject_and_hides_only_the_detail():
    html = _note("This chart shows X.", "Higher is better.", "Only on one corpus.")
    caption, _, detail = html.partition("<details")
    assert "This chart shows X." in caption          # visible without clicking
    assert "This chart shows X." not in detail
    assert "Higher is better." in detail             # detail stays collapsed
    assert "Only on one corpus." in detail
    assert "What does this mean?" in detail


def test_every_note_uses_the_same_dropdown_question():
    """One label, so a reader learns the affordance once and reuses it."""
    src = (Path(__file__).resolve().parents[1]
           / "tsfm_lens" / "report" / "report.py").read_text(encoding="utf-8")
    labels = set(re.findall(r'summary="(What[^"]*)"', src))
    assert labels, "no _note summary overrides found -- did the signature change?"
    assert all(l.startswith("What does this") for l in labels), labels


class _F:
    def __init__(self, stage, plain):
        self.stage, self.plain, self.text, self.caveat = stage, plain, plain, ""


def test_findings_group_by_stage_in_first_appearance_order_and_lose_nothing():
    fs = [_F("l0", "a"), _F("l1", "b"), _F("l0", "c"), _F("confirm", "d")]
    groups = _group_findings(fs)
    assert [g["stage"] for g in groups] == ["l0", "l1", "confirm"]
    # Grouping hides nothing. Each item is a head finding plus the look-alikes
    # collapsed under it, so "lose nothing" counts both -- the assertion this
    # file has always made, restated for the shape rather than relaxed.
    total = sum(1 + len(it["siblings"]) for g in groups for it in g["items"])
    assert total == len(fs)
    assert [it["finding"].plain for it in groups[0]["items"]] == ["a", "c"]
    assert groups[0]["label"].startswith("L0")


def test_lookalike_findings_collapse_under_the_first_and_none_is_dropped():
    """A stage that measures every model emits the same sentence per model.
    The first renders normally; the rest move into a collapsed block -- none
    is summarized away, which is what makes the collapse safe."""
    import tsfm_lens.report.report as R
    fs = [_F("l3", "In Alpha, 3 of 6 corruptions split."),
          _F("l3", "In Beta, 5 of 6 corruptions split."),
          _F("l3", "A different claim entirely.")]
    items = R._cluster_lookalikes(fs, ["Alpha", "Beta"])
    assert len(items) == 2
    assert items[0]["finding"].plain.startswith("In Alpha")
    assert [s.plain for s in items[0]["siblings"]] == ["In Beta, 5 of 6 corruptions split."]
    assert items[1]["siblings"] == []


def test_a_model_name_containing_digits_still_matches_its_siblings():
    """NEGATIVE, and the bug this was written from: masking numbers BEFORE
    model names turns `Chronos-2` into `Chronos-N`, so it never matches its
    own siblings and a repetition family looks like three unique claims."""
    import tsfm_lens.report.report as R
    names = ["Chronos-T5-Base", "Chronos-2", "TimesFM"]
    tmpls = {R._finding_template(f"In {m}, 4 of 6 corruptions split.", names)
             for m in names}
    assert len(tmpls) == 1


def test_findings_that_only_look_alike_after_masking_numbers_are_not_merged_wrongly():
    """NEGATIVE: masking must not merge claims that differ in their words."""
    import tsfm_lens.report.report as R
    fs = [_F("l0", "Alpha is more accurate on 3 families."),
          _F("l0", "Alpha is less accurate on 3 families.")]
    items = R._cluster_lookalikes(fs, ["Alpha"])
    assert len(items) == 2


class _Cfg:
    class _M:
        def __init__(self, name): self.name = name

    def __init__(self):
        self.models = [self._M("Alpha"), self._M("Beta")]


def _write_l0(tmp_path):
    (tmp_path / "l0").mkdir()
    (tmp_path / "l0" / "summary.json").write_text(json.dumps({
        "overall": [{"model": "Alpha", "mase": 1.0}, {"model": "Beta", "mase": 1.5}],
        "alpha": 0.05,
        "family_tests": [{"family": "seasonal", "favored": "Alpha", "ratio": 0.7,
                          "p_holm": 0.01}],
        "strengths": {"Alpha": ["seasonal"], "Beta": []}}), encoding="utf-8")


def test_the_scorecard_drops_the_rows_it_has_no_artifact_for(tmp_path):
    """It must never claim coverage the run does not have (CLAUDE.md sec 2.5)."""
    _write_l0(tmp_path)
    html = _scorecard(tmp_path, _Cfg())
    assert "Lowest overall MASE (Alpha)" in html
    assert "CKA" not in html            # no l1 artifact -> no such row
    assert "Crystallization" not in html  # no lens artifact -> no such row
    assert "captured FLOP" not in html    # no budget artifact -> no such row


def test_the_scorecard_is_empty_rather_than_confident_when_nothing_ran(tmp_path):
    assert _scorecard(tmp_path, _Cfg()) == ""


def test_every_scorecard_row_prints_the_rule_that_decided_its_verdict(tmp_path):
    """The point of the scorecard: no invisible threshold anywhere in it.

    The block it replaced chose between two authored sentences on a bare
    `cka > 4 * null`, so a reader could disagree with the English but never
    with the `4`. Here the rule text must be rendered in the same row as the
    verdict it produced, for every row, or the block has regressed to prose
    with a number in it.
    """
    _write_l0(tmp_path)
    (tmp_path / "l1").mkdir()
    (tmp_path / "l1" / "meta.json").write_text(json.dumps({
        "best_pair": {"layer_a": "a.0", "layer_b": "b.0", "cka": 0.4,
                      "ci": {"lo": 0.35, "hi": 0.45},
                      "null_ci": {"value": 0.02, "lo": 0.01, "hi": 0.03}}}),
        encoding="utf-8")
    html = _scorecard(tmp_path, _Cfg())
    rows = re.findall(r'<tr class="sc-row">(.*?)</tr>', html, re.S)
    assert len(rows) >= 3, len(rows)
    for row in rows:
        assert '<code>' in row, row          # the rule is printed...
        assert 'sc-verdict' in row, row      # ...beside the verdict it produced
    assert "value &ge; 4&times; reference" in html or "value ≥ 4× reference" in html


def test_a_scorecard_verdict_cannot_be_set_by_a_call_site(tmp_path):
    """`Verdict.verdict` is derived in __post_init__, never assigned by hand.

    This is the structural reason the block cannot drift back into authored
    conclusions: there is no parameter to write one into.
    """
    from tsfm_lens.report.derived import RULES, Verdict
    v = Verdict(measure="m", value=10.0, reference=2.0, reference_label="r",
                rule=RULES["ratio_at_least"](4.0), verdict="whatever the author says")
    assert v.verdict == "clears"
    below = Verdict(measure="m", value=3.0, reference=2.0, reference_label="r",
                    rule=RULES["ratio_at_least"](4.0))
    assert below.verdict == "does not clear"
    absent = Verdict(measure="m", value=3.0, reference=None, reference_label="r",
                     rule=RULES["ratio_at_least"](4.0))
    assert absent.verdict == "not comparable"


@pytest.mark.skipif(not SMOKE.exists(), reason="runs/smoke/report.html not built")
def test_the_rendered_report_captions_every_figure():
    """No *individual* figure is unlabelled -- checked positionally, not by count.

    This assertion used to compare totals (`n_caps >= n_figs`), and that is
    too weak: the first live `report.verbose: true` run passed it at 53
    captions against 70 figures on aggregate while 17 gallery panels -- the
    L3 per-series case studies and the per-family exemplars -- were in fact
    bare, because a handful of sections carry captions on tables and intros
    too and the surplus masked the deficit. Walk the document in order and
    require a caption immediately after each figure instead.
    """
    html = SMOKE.read_text(encoding="utf-8")
    seq = [
        "fig" if "plotly" in m.group(0) else "cap"
        for m in re.finditer(
            r'<div id="[0-9a-f-]{36}" class="plotly-graph-div"|<p class="figcap">',
            html,
        )
    ]
    n_figs = seq.count("fig")
    assert n_figs > 10, n_figs
    bare = [
        i for i, kind in enumerate(seq)
        if kind == "fig" and (i + 1 >= len(seq) or seq[i + 1] != "cap")
    ]
    assert not bare, f"{len(bare)} of {n_figs} figures have no caption after them"
    # The dropdown is per *explanation*, not per figure -- a gallery of near
    # identical panels shares one, deliberately (see `_figcap`) -- so this is
    # a presence check, not a per-figure count.
    assert html.count("What does this mean?") > 10


@pytest.mark.skipif(not SMOKE.exists(), reason="runs/smoke/report.html not built")
def test_the_rendered_report_opens_with_its_conclusion():
    html = SMOKE.read_text(encoding="utf-8")
    assert html.count('class="bottomline"') == 1
    body = html[html.index("<body"):]
    assert body.index('class="bottomline"') < body.index("How to read this report")
    assert body.index('class="bottomline"') < body.index('class="findings"')


def test_a_registered_finding_is_never_collapsed_behind_an_exploratory_one():
    """NEGATIVE. Headline mode hides every non-registered `<li>`, and a
    sibling nested inside one goes with it — so a pre-registered claim
    collapsed behind an exploratory look-alike would vanish from the one view
    that exists to show only pre-registered claims."""
    import tsfm_lens.report.report as R
    fs = [_F("confirm", "In Alpha, the claim held."),
          _F("confirm", "In Beta, the claim held.")]
    fs[1].registered = True
    items = R._cluster_lookalikes(fs, ["Alpha", "Beta"])
    assert len(items) == 2
    assert items[1]["finding"].registered is True
    assert items[0]["siblings"] == []


def test_lookalikes_are_gathered_even_when_not_adjacent():
    """A stage that loops family-then-model interleaves its per-model claims,
    so an adjacency-only pass leaves the reader meeting the same sentence
    three times, just further apart. Template order is first-appearance, so
    the stage's own ordering still decides what is read first."""
    import tsfm_lens.report.report as R
    fs = [_F("l3", "In Alpha, 3 of 6 split."),
          _F("l3", "Something else entirely."),
          _F("l3", "In Beta, 5 of 6 split.")]
    items = R._cluster_lookalikes(fs, ["Alpha", "Beta"])
    assert len(items) == 2
    assert items[0]["finding"].plain.startswith("In Alpha")
    assert len(items[0]["siblings"]) == 1
    assert items[1]["finding"].plain == "Something else entirely."
    # nothing lost
    assert sum(1 + len(i["siblings"]) for i in items) == len(fs)


def test_the_scorecard_collapses_its_rows_but_not_its_description(tmp_path):
    """Only the description is on the default path (user request, 2026-08-30).

    A scorecard rendered open at the top of a long report reads as the
    report's findings rather than as an index into them. Collapsed, it has to
    still say -- visibly, above the fold -- what it is and that the evidence
    is below; a `<details>` whose summary is the only visible text would
    trade one legibility defect for another.
    """
    _write_l0(tmp_path)
    html = _scorecard(tmp_path, _Cfg())
    before = html.split("<details", 1)[0]
    assert "summary of conclusions" in before, \
        "the description must render outside the collapsed block"
    assert "sections below" in before
    assert "<details" in html, "the rows must be collapsed"
    assert "<table" not in before, "the table must be inside the collapsed block"
    # And a `<details>` with no `open` attribute is closed by default.
    summary_tag = html.split("<details", 1)[1].split(">", 1)[0]
    assert "open" not in summary_tag, summary_tag


def test_the_collapsed_scorecard_still_contains_every_row(tmp_path):
    """Collapsing is a presentation change and must lose no content."""
    _write_l0(tmp_path)
    html = _scorecard(tmp_path, _Cfg())
    assert "Lowest overall MASE (Alpha)" in html
    assert "Verdict" in html and "Rule" in html
    assert "one run, not a" in html, "the footnote moved but must survive"
