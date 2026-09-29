"""Report prose must not name architectures that are not in the run.

`ROADMAP.md` §24 made the report's *conclusions* derived rather than authored.
This is the same discipline one level down, applied to the explanatory notes:
several of them illustrated a mechanism with a fixed architecture name
("an encoder-only model like Chronos-T5 caps out partway up the axis"), which
reads as a claim about a model that may not be in the run at all -- and on a
panel of three unfamiliar checkpoints it reads as a claim about the wrong one.

The helpers are unit-tested directly; the property they exist to enforce is
checked against a real rendered report.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.report import _name_phrase, _short_axis_clause

RUNS = Path(__file__).resolve().parents[1] / "runs"


def test_name_phrase_forms():
    assert _name_phrase([]) == ""
    assert _name_phrase([], "nothing") == "nothing"
    assert _name_phrase(["a"]) == "a"
    assert _name_phrase(["a", "b"]) == "a and b"
    assert _name_phrase(["a", "b", "c"]) == "a, b and c"


def test_short_axis_clause_is_empty_when_no_artifact_says_otherwise(tmp_path):
    """The load-bearing negative: with nothing measured, the note must state the
    mechanism with NO example rather than falling back to a written-in one.

    An absent artifact is the common case for a partial run, and inventing an
    illustration there is exactly the failure this replaced."""
    assert _short_axis_clause(tmp_path) == ""


def test_short_axis_clause_names_the_run_s_own_short_model(tmp_path):
    import json
    (tmp_path / "l3").mkdir()
    (tmp_path / "l3" / "meta.json").write_text(json.dumps({
        "rel_depth": {"deep": [0.0, 0.5, 1.0], "shallow": [0.0, 0.2, 0.43]}}),
        encoding="utf-8")
    clause = _short_axis_clause(tmp_path)
    assert "shallow" in clause and "does in this run" in clause
    assert "deep" not in clause


def test_short_axis_clause_survives_a_malformed_depth_entry(tmp_path):
    """A model whose coords are unreadable must be omitted, not crash the note
    and take the whole section down with it."""
    import json
    (tmp_path / "l3").mkdir()
    (tmp_path / "l3" / "meta.json").write_text(json.dumps({
        "rel_depth": {"broken": None, "shallow": [0.0, 0.4]}}), encoding="utf-8")
    assert "shallow" in _short_axis_clause(tmp_path)


@pytest.mark.parametrize("run_name", ["smoke", "smoke_panel"])
def test_no_finding_names_an_architecture_absent_from_the_run(run_name):
    """The property the user asked for, checked where conclusions actually live.

    `report/findings.json` is the machine-readable list of every claim the
    report makes. A mock-only run contains no real checkpoint, so an
    architecture name appearing in a claim there could only have been written
    at authoring time -- which is the definition of a hardcoded conclusion.

    Deliberately scoped to findings and NOT to the whole HTML: the report also
    renders a provenance block (installed library versions -- a fact about the
    environment), a failure-mode gallery (historical case studies from this
    repo's own record, which name the runs they happened on precisely so they
    can be checked), and a methods appendix citing prior measured results.
    Those name architectures correctly and on purpose; a whole-document grep
    would flag them and would have to be loosened until it caught nothing.
    """
    import json
    path = RUNS / run_name / "report" / "findings.json"
    if not path.exists():
        pytest.skip(f"{path} not built (gitignored run directory)")
    findings = json.loads(path.read_text(encoding="utf-8"))["findings"]
    assert findings, "a report with no findings would pass this test vacuously"
    blob = " ".join(
        str(f.get(k) or "") for f in findings
        for k in ("text", "plain", "caveat")).lower()
    offenders = {n: blob.count(n) for n in
                 ("timesfm", "chronos", "sundial", "patchtst", "moirai")
                 if blob.count(n)}
    assert not offenders, (
        f"{run_name}'s findings name architectures not in the run: {offenders}")


def test_rendered_methods_appendix_states_no_architecture_in_advance():
    """The methods appendix renders analysis module docstrings VERBATIM, so a
    docstring that illustrates a mechanism with a named architecture becomes a
    claim in the report about a model the run may not contain -- which is how
    `analysis/lens.py`'s "TimesFM front-loads then compresses" example reached
    a mock-only report. Entries citing this repo's own already-measured
    results are exempt: those are references, not predictions."""
    from tsfm_lens import methods_appendix
    bad = []
    for entry in methods_appendix.build_methods_appendix():
        if not entry.get("generated"):
            continue  # hand-written entries may cite recorded results by name
        if re.search(r"(?i)timesfm|chronos|sundial|patchtst|moirai", entry["text"]):
            bad.append(entry["module"])
    assert not bad, f"module docstrings rendered into the report name architectures: {bad}"
