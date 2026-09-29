"""Progressive disclosure in the report (ROADMAP.md sec 21 J4 / H10).

A three-position control -- Headline / Standard / Methods -- lets a reader
choose how much of the report to see without the file losing its single
self-contained-HTML constraint (no new dependency, no server). Standard is
the default, so every prior reader's experience is unchanged unless they
click a button; these tests pin that default and the two other modes'
scope directly against the rendered template/output rather than trusting
the design description.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPORT_SRC = (Path(__file__).resolve().parents[1]
              / "tsfm_lens" / "report" / "report.py").read_text(encoding="utf-8")
SMOKE = Path(__file__).resolve().parents[1] / "runs" / "smoke" / "report.html"


def test_the_toggle_exists_and_defaults_to_standard():
    assert 'data-level="headline"' in REPORT_SRC
    assert 'data-level="standard"' in REPORT_SRC
    assert 'data-level="methods"' in REPORT_SRC
    # only the "standard" button starts with the active class -- no <script>
    # runs on load to change it, so this literally is the default state.
    standard_btn = re.search(r'<button[^>]*data-level="standard"[^>]*>', REPORT_SRC).group(0)
    assert 'class="active"' in standard_btn
    for level in ("headline", "methods"):
        btn = re.search(rf'<button[^>]*data-level="{level}"[^>]*>', REPORT_SRC).group(0)
        assert 'class="active"' not in btn


def test_headline_mode_css_scopes_to_fairness_l0_confirm_and_registered_findings():
    assert 'body[data-detail="headline"] section:not(.sec-headline){display:none}' in REPORT_SRC
    assert "body[data-detail=\"headline\"] .findings li:not(.registered){display:none}" in REPORT_SRC
    # exactly Fairness/L0/Confirm are tagged sec-headline -- not, e.g., L1/L2
    # which would silently smuggle un-headline-worthy sections back in.
    # (Matched on the trailing `.eyebrow in (...)` rather than a specific
    # loop-variable name, since the template's own variable is `it`, not `s`,
    # and pinning the variable name here would make this test fail on a
    # harmless rename instead of on the invariant it actually checks.)
    m = re.search(r"\.eyebrow in \(([^)]*)\)", REPORT_SRC)
    assert m is not None
    tagged = {t.strip().strip("'\"") for t in m.group(1).split(",")}
    assert tagged == {"Fairness", "L0", "Confirm"}


def test_methods_mode_expands_every_details_element_via_script_not_new_markup():
    assert "function tsfmSetDetail" in REPORT_SRC
    assert "d.open = true" in REPORT_SRC
    # the expansion targets the *existing* generic <details> selector, so it
    # covers notes/coverage/config without any new per-site markup.
    assert "querySelectorAll('details')" in REPORT_SRC


@pytest.mark.skipif(not SMOKE.exists(), reason="runs/smoke/report.html not built")
def test_rendered_report_tags_exactly_fairness_l0_confirm_as_headline_sections():
    html = SMOKE.read_text(encoding="utf-8")
    tagged = set(re.findall(r'<section id="sec-([a-z0-9-]+)" class="sec-headline">', html))
    assert tagged == {"fairness", "l0", "confirm"}
    # every other rendered section must NOT carry the headline class
    all_secs = set(re.findall(r'<section id="sec-([a-z0-9-]+)"', html))
    assert all_secs - tagged  # smoke renders more than just the 3 headline sections
    for slug in all_secs - tagged:
        assert f'<section id="sec-{slug}" class="sec-headline">' not in html


@pytest.mark.skipif(not SMOKE.exists(), reason="runs/smoke/report.html not built")
def test_rendered_report_finding_items_are_all_tagged_registered_or_exploratory():
    html = SMOKE.read_text(encoding="utf-8")
    n_findings = len(re.findall(r"<li class=\"(?:registered|exploratory)\">", html))
    assert n_findings > 0
    # every <li> inside the findings block carries exactly one of the two tags
    # (bounded below by "How to read this report" -- `_how_to_read()` renders
    # unconditionally, immediately after the findings appendix's closing
    # </details>. The findings appendix is the LAST thing rendered inside
    # REPORT_PARTS's loop of <section id="sec-..."> elements -- it is an
    # appendix after every numbered Part, not interleaved between sections --
    # so there is no later "<section id=\"sec-" to bound on, and the findings
    # <div> has nested <div>s/<details> of its own, so a naive first-</div> or
    # first-</details> split would truncate too early)
    findings_block = html.split('<details class="findings"', 1)[1].split('id="how-to-read"', 1)[0]
    untagged = re.findall(r"<li(?![^>]*class=\"(?:registered|exploratory)\")", findings_block)
    assert untagged == []


if __name__ == "__main__":
    test_the_toggle_exists_and_defaults_to_standard()
    test_headline_mode_css_scopes_to_fairness_l0_confirm_and_registered_findings()
    test_methods_mode_expands_every_details_element_via_script_not_new_markup()
    if SMOKE.exists():
        test_rendered_report_tags_exactly_fairness_l0_confirm_as_headline_sections()
        test_rendered_report_finding_items_are_all_tagged_registered_or_exploratory()
    print("All tests passed!")
