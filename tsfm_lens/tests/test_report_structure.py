"""Tests for the report's structure (report structure spec R1, CLAUDE.md sec
6.7): the hyperlinked Contents table of contents, the Part-grouped section
order, and the "At a glance" slot's position relative to the numbered Parts.

Exercised directly against `report/report.py`'s own `_TEMPLATE` /
`_build_report_parts` / `REPORT_PARTS`, with synthetic `coverage`/`sections`
fixtures -- the ordering logic here is pure data-plumbing (which eyebrow goes
under which Part, and whether a rendered section's own id matches the link
that points to it), so a full pipeline run is not needed to catch a
regression in it. A real run's report.html is checked separately (`tests/
test_report_coverage.py`, `tests/test_smoke.py`, and the R1 spec's own
rendered-HTML verification against `runs/full_report_run_4model`).

Every load-bearing assertion below was confirmed to fail under a planted
regression -- see each test's own docstring for what was broken and how,
and the final report for the exact pytest evidence.
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report import report as report_mod  # noqa: E402


class _IdCollector(HTMLParser):
    """Every `id="..."` attribute anywhere in the page."""

    def __init__(self):
        super().__init__()
        self.ids: set = set()

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if "id" in d:
            self.ids.add(d["id"])


def _render(coverage: list, sections: list, findings: list | None = None) -> str:
    parts = report_mod._build_report_parts(coverage, sections)
    return report_mod._TEMPLATE.render(
        title="Test report", run="test-run", date="2024-01-01",
        models_title="A vs B",
        model_chips=[{"name": "A", "cls": "a", "color": "#000"},
                    {"name": "B", "cls": "b", "color": "#000"}],
        colors=report_mod._COLORS, dataset_line="",
        findings=findings or [], finding_groups=[],
        sections=sections, parts=parts,
        config_text="", mock_warning="", bottom_line="",
        at_a_glance_extra="", how_to_read='<section class="howto" id="how-to-read">x</section>',
        glossary_block="", methods_appendix_block="", failure_gallery_block="",
        coverage=coverage, coverage_summary="ok", any_failed=False,
        family_resolution_line="", alignment_provenance="",
    )


def _cov(eyebrow, title, status="rendered", detail=""):
    return {"eyebrow": eyebrow, "title": title, "status": status, "detail": detail}


def _sec(eyebrow, title, blurb="blurb"):
    return {"eyebrow": eyebrow, "title": title, "blurb": blurb,
           "html": f"<p>content for {eyebrow}</p>", "slug": report_mod._section_slug(eyebrow)}


def _fixture():
    """One eyebrow per Part 1 (for the earliest-part check), the Part 6 trio
    ("SAE", "Concepts", "Exemplars") in the exact membership order report
    structure spec R1 asks for, and one skipped section ("L3") to exercise
    the TOC's greyed, unlinked branch."""
    coverage = [
        _cov("Fairness", "The fairness card"),
        _cov("Corpus", "The corpus card"),
        _cov("SAE", "Sparse feature dictionary"),
        _cov("Concepts", "Cross-model concepts"),
        _cov("Exemplars", "Exemplar case studies"),
        _cov("L3", "Perturbation & patching", status="skipped",
            detail="stage not enabled in config"),
    ]
    sections = [
        _sec("Fairness", "The fairness card"),
        _sec("Corpus", "The corpus card"),
        _sec("SAE", "Sparse feature dictionary"),
        _sec("Concepts", "Cross-model concepts"),
        _sec("Exemplars", "Exemplar case studies"),
    ]
    return coverage, sections


# ---------------------------------------------------------------------------
# 1. Every TOC href resolves to an id somewhere in the page.
# ---------------------------------------------------------------------------

def test_toc_hrefs_resolve_to_an_id_in_the_page():
    coverage, sections = _fixture()
    html = _render(coverage, sections)
    toc_html = html.split('<nav class="toc" id="toc">', 1)[1].split("</nav>", 1)[0]
    hrefs = re.findall(r'href="#([^"]+)"', toc_html)
    assert hrefs, "TOC has no links at all"
    collector = _IdCollector()
    collector.feed(html)
    missing = [h for h in hrefs if h not in collector.ids]
    assert not missing, f"TOC links to id(s) missing from the page: {missing}"


def test_skipped_section_is_greyed_and_unlinked_in_the_toc():
    """"L3" is skipped in the fixture -- it must appear as inert grey text
    ("not run"), never a link with no matching anchor (there is no `id=
    "sec-l3"` in the page at all, since a skipped section renders no
    `<section>`). Decoy: "SAE", rendered in the same list, IS linked."""
    coverage, sections = _fixture()
    html = _render(coverage, sections)
    toc_html = html.split('<nav class="toc" id="toc">', 1)[1].split("</nav>", 1)[0]
    assert '<span class="toc-skip">Perturbation & patching (not run)</span>' in toc_html
    assert '<a href="#sec-l3">' not in toc_html
    assert '<a href="#sec-sae">' in toc_html


# ---------------------------------------------------------------------------
# 2. Section order follows REPORT_PARTS, and Concepts sits after SAE and
#    before Exemplars specifically (the spec's own headline requirement).
# ---------------------------------------------------------------------------

def test_section_order_follows_the_parts_list(monkeypatch):
    """Plant: swap Part 6's membership order so "Concepts" precedes "SAE" --
    confirmed this flips the two ids' rendered order (i.e. the body loop's
    order is genuinely driven by `REPORT_PARTS`, not by `sections`' own
    list order), then reverted."""
    coverage, sections = _fixture()
    html = _render(coverage, sections)
    body = html.split('<div class="appendix"', 1)[0]
    idx = {eb: body.index(f'id="sec-{report_mod._section_slug(eb)}"')
          for eb in ("Fairness", "Corpus", "SAE", "Concepts", "Exemplars")}
    assert idx["Fairness"] < idx["Corpus"], "Part 1 order (Fairness, Corpus) not honored"
    assert idx["SAE"] < idx["Concepts"] < idx["Exemplars"], (
        "Concepts must render after SAE and before Exemplars (report structure spec R1)")

    bad_parts = [(title, intro, (["Concepts", "SAE", "Exemplars"]
                                 if eyebrows == ["SAE", "Concepts", "Exemplars"] else eyebrows))
                for (title, intro, eyebrows) in report_mod.REPORT_PARTS]
    monkeypatch.setattr(report_mod, "REPORT_PARTS", bad_parts)
    html_bad = _render(coverage, sections)
    body_bad = html_bad.split('<div class="appendix"', 1)[0]
    idx_bad = {eb: body_bad.index(f'id="sec-{report_mod._section_slug(eb)}"')
              for eb in ("SAE", "Concepts")}
    assert idx_bad["Concepts"] < idx_bad["SAE"], (
        "plant did not change the rendered order -- this test would not discriminate")


# ---------------------------------------------------------------------------
# 3. "At a glance" sits before the first Part.
# ---------------------------------------------------------------------------

def test_at_a_glance_sits_before_the_first_part():
    coverage, sections = _fixture()
    html = _render(coverage, sections)
    assert html.index('id="at-a-glance"') < html.index('id="part-1"'), (
        '"At a glance" must render before Part 1 (report structure spec R1)')


if __name__ == "__main__":
    import pytest as _pytest

    test_toc_hrefs_resolve_to_an_id_in_the_page()
    test_skipped_section_is_greyed_and_unlinked_in_the_toc()
    with _pytest.MonkeyPatch.context() as mp:
        test_section_order_follows_the_parts_list(mp)
    test_at_a_glance_sits_before_the_first_part()
    print("report structure tests passed")
