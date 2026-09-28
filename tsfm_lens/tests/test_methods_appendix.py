"""ROADMAP.md sec 21 J5: the advanced methods appendix. Most entries are
pulled verbatim from each estimator's own module docstring so the appendix
cannot drift from the code (`build_methods_appendix` raises rather than
degrading to a blank/placeholder row for a module-backed entry with no
docstring); two entries (SAE fidelity, `relative_decoder_norm`) are
hand-written because their source functions carry only one-line docstrings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens import methods_appendix
from tsfm_lens.report.report import _methods_appendix_block


def test_every_entry_has_nonempty_text():
    entries = methods_appendix.build_methods_appendix()
    assert len(entries) >= 8
    for e in entries:
        assert e["label"]
        assert e["text"] and len(e["text"]) > 40, e["label"]


def test_generated_entries_are_pulled_from_the_named_module_verbatim():
    entries = methods_appendix.build_methods_appendix()
    generated = [e for e in entries if e["generated"]]
    assert len(generated) >= 6
    for e in generated:
        import importlib
        mod = importlib.import_module(e["module"])
        assert e["text"] == mod.__doc__.strip()


def test_hand_written_entries_are_marked_and_cite_a_real_repo_finding():
    """The two hand-written entries exist precisely because their source
    docstrings are too thin -- they must not silently claim to be generated,
    and their failure-mode paragraphs must cite a real, already-measured
    number from this repo's own record rather than a generic warning."""
    entries = {e["label"]: e for e in methods_appendix.build_methods_appendix()}
    sae_entry = entries["SAE fidelity and dead-feature rate"]
    assert sae_entry["generated"] is False
    assert sae_entry["module"] is None
    assert "94.5" in sae_entry["text"] or "97.3" in sae_entry["text"]

    rdn_entry = entries["relative_decoder_norm (crosscoder shared/specific split)"]
    assert rdn_entry["generated"] is False
    assert "seed-fragile" in rdn_entry["text"] or "seed" in rdn_entry["text"]


def test_a_module_with_no_docstring_raises_rather_than_degrading(monkeypatch):
    """A silently blank appendix row is worse than a loud failure at
    report-build time (CLAUDE.md sec 2.5)."""
    import types
    fake = types.ModuleType("fake_no_doc_module")
    fake.__doc__ = None

    real_import = methods_appendix.importlib.import_module

    def fake_import(name):
        if name == "fake_no_doc_module":
            return fake
        return real_import(name)

    monkeypatch.setattr(methods_appendix.importlib, "import_module", fake_import)
    monkeypatch.setattr(methods_appendix, "_ESTIMATORS",
                        [methods_appendix.MethodEntry("broken", "fake_no_doc_module")])
    with pytest.raises(RuntimeError, match="no module docstring"):
        methods_appendix.build_methods_appendix()


def test_report_block_renders_every_label_and_is_collapsed_by_default():
    html = _methods_appendix_block()
    assert "<details class=\"note methods-appendix\">" in html
    assert "Methods appendix" in html
    entries = methods_appendix.build_methods_appendix()
    for e in entries:
        assert e["label"] in html
    # generated entries cite their source module; hand-written ones don't
    assert "own module docstring" in html
