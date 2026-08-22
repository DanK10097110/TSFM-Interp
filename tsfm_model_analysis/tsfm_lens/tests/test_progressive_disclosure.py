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
    m = re.search(r"s\.eyebrow in \(([^)]*)\)", REPORT_SRC)
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
    # (bounded by the first rendered <section>, since the findings <div> has
    # nested <div>s of its own and a naive first-</div> split would truncate
    # too early)
    findings_block = html.split('<div class="findings">', 1)[1].split('<section id="sec-', 1)[0]
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
