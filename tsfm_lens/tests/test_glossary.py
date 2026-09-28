"""Tests for `ROADMAP.md` sec 21 J3's glossary.

The value of a glossary is entirely in it being *complete, non-drifting and
non-circular*, so the tests here check exactly those three things rather than
asserting the wording of any one entry (which would make every editorial
improvement a test failure for no gain):

- **Non-drifting.** The README block and the report's HTML both render from
  `GLOSSARY`, so a stale README is a test failure, not something a reader
  discovers later -- the same guarantee `test_stage_docs.py` gives sec 21 J2.
- **Complete where it is checkable.** The terms this repo's own report
  headings and the "How to read this report" preamble depend on must have
  entries; a glossary that omits the words the preamble itself uses is worse
  than none.
- **Loud on a miss.** `get()` raises rather than rendering a blank, per
  `CLAUDE.md` sec 2.5.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens import glossary
import render_glossary


def test_every_entry_has_all_three_fields_filled():
    for key, t in glossary.GLOSSARY.items():
        assert t.term.strip(), f"{key}: empty term"
        assert len(t.definition.strip()) > 40, f"{key}: definition too short to be a definition"
        assert t.where.strip(), f"{key}: no 'where it appears' pointer"


def test_definitions_are_one_sentence_each():
    """J3 asks for one sentence per term; a paragraph belongs in a `_note()` block.

    Counts sentence-ending periods that are followed by whitespace and a
    capital, so abbreviations and decimals inside a definition don't
    false-positive.
    """
    for key, t in glossary.GLOSSARY.items():
        breaks = re.findall(r"\.\s+[A-Z(]", t.definition)
        assert not breaks, f"{key}: definition looks like {len(breaks) + 1} sentences"


def test_get_raises_loudly_on_an_unknown_term():
    with pytest.raises(KeyError) as exc:
        glossary.get("no-such-term")
    assert "glossary.py" in str(exc.value)


def test_terms_are_alphabetized_for_lookup_not_dict_order():
    names = [t.term for t in glossary.terms()]
    assert names == sorted(names, key=str.lower)
    assert len(names) == len(glossary.GLOSSARY)


def test_readme_glossary_block_is_not_stale():
    """The one check that actually prevents drift between the two surfaces."""
    assert render_glossary.main.__module__  # the CLI is importable, not just present
    readme = (Path(render_glossary.__file__).resolve().parent / "README.md").read_text(encoding="utf-8")
    assert render_glossary._BEGIN in readme and render_glossary._END in readme
    assert render_glossary._splice(readme, render_glossary.generated_block()) == readme, (
        "README.md's glossary section is stale -- run `python render_glossary.py`")


def test_the_preamble_terms_the_report_leans_on_are_all_defined():
    """The preamble names terms without defining them; those are exactly the ones a lookup table must carry."""
    required = ["window", "relative-depth", "evidence-class", "mase",
                "series-resampling", "stitching-gain", "noise-floor"]
    for key in required:
        assert glossary.get(key).definition


def test_report_renders_the_glossary_into_its_html():
    from tsfm_lens.report.report import _glossary_block
    html = _glossary_block()
    assert "Glossary" in html and "gloss-where" in html
    # every term reaches the rendered page -- a silently truncated lookup
    # table is the failure mode a "does it contain the word glossary" test
    # would miss entirely
    for t in glossary.terms():
        assert t.term in html
