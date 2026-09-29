"""ROADMAP.md sec 21 J6: the failure-mode gallery -- five real, frozen,
hand-written cases of an analysis in this report looking wrong before it
was diagnosed. Not derived from any live artifact (these are historical
narratives, several predating the fixes that make today's numbers
trustworthy), so the tests here check structure and citation, not numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens import failure_gallery
from tsfm_lens.report.report import _failure_gallery_block


def test_gallery_has_at_least_five_entries_all_fields_populated():
    entries = failure_gallery.gallery_entries()
    assert len(entries) >= 5
    for g in entries:
        assert g.title and g.looked_like and g.actually_was and g.lesson and g.source


def test_every_entry_cites_a_claude_md_or_roadmap_section():
    for g in failure_gallery.gallery_entries():
        assert "CLAUDE.md" in g.source or "ROADMAP.md" in g.source or "worked_example" in g.source, g.title


def test_gallery_entries_is_a_defensive_copy():
    a = failure_gallery.gallery_entries()
    a.pop()
    assert len(failure_gallery.gallery_entries()) == len(failure_gallery.GALLERY)


def test_report_block_renders_every_title_and_is_collapsed_by_default():
    html = _failure_gallery_block()
    assert '<details class="note failure-gallery">' in html
    assert "Failure-mode gallery" in html
    for g in failure_gallery.gallery_entries():
        assert g.title in html
        assert g.source in html
