"""The render-boundary stripper for internal planning-document citations.

The load-bearing tests here are the NEGATIVES. Deleting a citation is easy;
what makes this module safe to run over a fully rendered HTML document is
that it does not delete the sentence around one, does not eat a tag, does not
touch the `sec-*` anchors the report's own deep links depend on, and keeps the
half of a citation that told the reader what to *do*.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.sanitize import strip_internal_refs, strip_refs_in_place


@pytest.mark.parametrize("text", [
    "the noise floor (ROADMAP.md sec 18 F6), so it is not explained",
    "the noise floor (`ROADMAP.md` §18 F6), so it is not explained",
    "the noise floor (CLAUDE.md sec 11.34), so it is not explained",
    "the noise floor (sec 18 F6), so it is not explained",
])
def test_a_whole_citation_parenthetical_goes_and_the_sentence_survives(text):
    out = strip_internal_refs(text)
    assert out == "the noise floor, so it is not explained"


def test_the_actionable_half_of_a_citation_is_kept():
    """`(<doc> sec 18 F1 -- enable the l3 stage)` exists to tell a reader what
    to do. Dropping it whole would delete the instruction with the citation."""
    out = strip_internal_refs("all cross-depth figures (ROADMAP.md §18 F1 -- enable the l3 stage)")
    assert out == "all cross-depth figures (enable the l3 stage)"


def test_a_citation_among_other_clauses_takes_only_its_own_clause():
    """The reason the module works clause by clause instead of on the whole
    parenthetical: two of these three clauses describe the measurement."""
    out = strip_internal_refs(
        "(own-width-vs-error, equal-count deciles, `ROADMAP.md` sec 20 H4 baseline).")
    assert out == "(own-width-vs-error, equal-count deciles)."


def test_a_clause_before_a_semicolon_survives_a_citation_after_it():
    out = strip_internal_refs(
        "(diagonal-hit fraction; CLAUDE.md sec 6.3, sec 7 invariant 7) — near 1.0 is good.")
    assert out == "(diagonal-hit fraction) — near 1.0 is good."


def test_a_nested_paren_inside_a_reference_does_not_strand_a_tail():
    """`sec 16 E3(c)` carries its own parentheses, so the enclosing paren has
    to be found by balancing rather than by a non-greedy regex."""
    out = strip_internal_refs("a map (ROADMAP.md §16 E3(c) -- only measured adapters)")
    assert out == "a map (only measured adapters)"


def test_an_introduced_citation_takes_the_clause_it_introduced():
    out = strip_internal_refs("not because the analysis failed. See CLAUDE.md §12's envelope edge.")
    assert "See" not in out
    assert out.strip() == "not because the analysis failed."


def test_a_possessive_reference_becomes_the_article_not_a_deletion():
    """NEGATIVE: the sentence keeps going after `<ref>'s`, so scanning forward
    to the period -- correct for an introduced citation -- would delete the
    claim itself. Pinned because that is exactly what the first version did."""
    out = strip_internal_refs(
        "every number recorded before `ROADMAP.md` sec 6.2.1's Stage 0 gate was measured "
        "on a mostly-dead dictionary.")
    assert out == ("every number recorded before the Stage 0 gate was measured "
                   "on a mostly-dead dictionary.")


def test_html_section_anchors_are_not_touched():
    """NEGATIVE: `sec-` prefixed ids are the report's deep links, not prose.
    The `\\d` in the section pattern is what keeps them out of scope."""
    html = ('<section id="sec-l1"><h2 class="sec-title">Geometry</h2>'
            '<p>see ROADMAP.md sec 18 F1</p></section>')
    out = strip_internal_refs(html)
    assert 'id="sec-l1"' in out and 'class="sec-title"' in out
    assert "ROADMAP.md" not in out


def test_no_tag_is_ever_eaten():
    """NEGATIVE: a non-parenthetical deletion stops at `<`, so a citation at
    the end of a paragraph cannot swallow the markup that follows it."""
    html = "<p>coverage — see ROADMAP.md sec 18 F4</p><table><tr><td>x</td></tr></table>"
    out = strip_internal_refs(html)
    assert out.count("<") == html.count("<") - 0
    assert "<table><tr><td>x</td></tr></table>" in out


def test_text_with_no_reference_is_returned_unchanged():
    text = "Peak cross-model CKA 0.381 against a shuffled-series null of 0.022."
    assert strip_internal_refs(text) == text


def test_a_bare_section_number_alone_is_a_reference_too():
    assert strip_internal_refs("the depth axis (§18 F1) is measured") == \
        "the depth axis is measured"


def test_seconds_are_not_mistaken_for_a_section():
    """NEGATIVE: `sec` only counts when a digit follows it directly. The
    report writes durations as `29 s`, but a future one writing `29 sec`
    must not lose them."""
    assert strip_internal_refs("the run took 29 sec on CPU") == "the run took 29 sec on CPU"


def test_nested_structures_are_walked():
    obj = {"findings": [{"text": "a claim (ROADMAP.md sec 18 F6)", "n": 3},
                        {"text": "no reference here"}]}
    out = strip_refs_in_place(obj)
    assert out["findings"][0]["text"] == "a claim"
    assert out["findings"][0]["n"] == 3
    assert out["findings"][1]["text"] == "no reference here"


def test_the_rendered_report_has_no_internal_references_left():
    """The end-to-end check, against whatever report the suite last built."""
    run = Path(__file__).resolve().parents[1] / "runs" / "smoke" / "report.html"
    if not run.exists():
        pytest.skip("no smoke report built in this checkout")
    html = run.read_text(encoding="utf-8")
    assert not re.search(r"(CLAUDE|ROADMAP(_ARCHIVE)?)\.md", html)
    assert not re.search(r"\bsec\b\.?\s*\d", html)
    assert not re.search(r"§\s*\d", html)
    assert 'id="sec-' in html
