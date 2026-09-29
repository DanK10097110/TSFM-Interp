"""Error messages that name a CLI flag are claims about the CLI.

Three separate guards in this repo told the reader to "rerun with `--force
<stage>`" for a stage that was **not in the selection**. `--force` only
bypasses a *selected* stage's skip predicate, so following any of those
instructions literally re-runs nothing and returns the identical error. Each
was correct behavior sitting behind an unreachable remedy, which is worse than
a plain refusal: the message convinces the reader the supported path exists
and sends them to work around it instead.

This test catches the class rather than the three known instances. It is
deliberately a source scan: the alternative is executing every guard, which
needs a broken store, a spent private corpus and a drifted registry.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "tsfm_lens"

# A message may name --force alone when the guard can ONLY fire for a stage
# that is already selected (the stage is running, so it was selected). Those
# are listed here with the reason, so adding one is a deliberate act.
_SELECTED_BY_CONSTRUCTION = {
    # run_confirm only executes when `confirm` is in the selection.
    "analysis/confirm.py",
    # A log line emitted while iterating the selection itself.
    "pipeline.py",
}


def _messages_naming_force() -> list:
    """Every user-facing message in the package that tells a reader to use --force.

    Whole *statements*, not lines: a remediation is routinely wrapped across
    several source lines, and judging a continuation line on its own reports
    the fix itself as the defect (the first version of this test did exactly
    that). Statements are recovered by grouping consecutive lines up to the
    one that balances brackets, which is enough for the string-concatenation
    style these messages are written in.
    """
    out = []
    for path in sorted(PKG.rglob("*.py")):
        rel = str(path.relative_to(PKG))
        lines = path.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith("#"):
                i += 1
                continue
            depth = 0
            start, chunk = i, []
            while i < len(lines):
                chunk.append(lines[i])
                depth += sum(lines[i].count(c) for c in "([{")
                depth -= sum(lines[i].count(c) for c in ")]}")
                i += 1
                if depth <= 0:
                    break
            stmt = "\n".join(chunk)
            if "--force" not in stmt:
                continue
            # argparse *declares* the flag; it does not instruct anyone.
            if "add_argument" in stmt:
                continue
            out.append((rel, start + 1, stmt))
    return out


def test_every_force_remediation_also_names_stages_or_is_exempt():
    """The load-bearing assertion: a flag named in an error must actually work."""
    offenders = []
    for rel, lineno, stmt in _messages_naming_force():
        if rel in _SELECTED_BY_CONSTRUCTION:
            continue
        # `--force X` is only actionable alongside `--stages`, or when the
        # message spells out that force alone is insufficient.
        if "--stages" in stmt or "--force all" in stmt:
            continue
        offenders.append(f"{rel}:{lineno}: {stmt.strip()[:140]}")
    assert not offenders, (
        "these messages tell a reader to use --force for a stage that may not be "
        "selected, where --force alone does nothing:\n  " + "\n  ".join(offenders))


def test_the_exemption_list_stays_honest():
    """An exemption must correspond to a real file, so a rename cannot hide a bug."""
    for rel in _SELECTED_BY_CONSTRUCTION:
        assert (PKG / rel).exists(), f"exempted {rel} no longer exists"


def test_scan_actually_finds_the_known_messages():
    """Without this, an over-narrow scan would pass by finding nothing at all."""
    found = _messages_naming_force()
    files = {rel for rel, _, _ in found}
    assert len(found) >= 3, f"scan found only {len(found)} statements; it is too narrow"
    assert "pipeline.py" in files
