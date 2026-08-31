"""Remove internal planning-document citations from anything a reader sees.

This repo's source deliberately carries heavy provenance: notes, findings and
module docstrings cite the internal planning documents that motivated them, and
that provenance is worth keeping in the source, where the people maintaining it
can follow a number back to the decision that produced it.

None of it means anything to someone reading a report. They have never seen
those documents, cannot open them, and a claim that trails off into a citation
they cannot follow reads as an unfinished sentence -- or worse, as though the
report is quoting evidence it has not shown them.

So the citations stay in the source and are stripped at the render boundary.
That direction matters. Editing them out of the ~76 strings that carry them
today would fix today's report and silently regress the first time anyone adds
a note in the house style, and it would not touch the methods appendix at all,
which renders analysis modules' `__doc__` verbatim and therefore inherits
whatever those docstrings say. Stripping on the way out is the only version of
this that stays true.

What survives is the actionable half. A citation of the form

    (<doc> sec 18 F6 -- enable l0.noise_floor_repeats >= 2)

exists to tell a reader what to *do*, and only its first half is internal, so
the parenthetical becomes `(enable l0.noise_floor_repeats >= 2)`. A citation
with no actionable remainder is dropped whole, along with the connector that
introduced it, so no orphaned " -- " or doubled space is left behind.
"""

from __future__ import annotations

import re

_DOC = r"(?:`?(?:ROADMAP(?:_ARCHIVE)?|CLAUDE)\.md`?)"
# A section number carries its item code with it: `sec 15 A11` and `§16 E3(c)`
# are each one reference, and matching only the number would strand the code.
_ITEM = r"(?:\s+[A-Z]\d{1,2}[a-z]?(?:\([a-z]\))?)?"
# The `\d` is load-bearing: it keeps `sec` out of the HTML this runs over,
# where `sec-l`, `sec">` and `class="sec-sae"` are section anchors, not prose.
_SECTION = r"(?:(?:\u00a7|\bsec\b\.?)\s*\d+(?:\.\d+)*" + _ITEM + r")"
# A doc name and the sections it introduces are ONE reference. Matching them
# separately let a forward scan stop at the "." inside a decimal section
# number, deleting "`<doc>` sec 6" and leaving ".2.1's" behind.
_REF = re.compile(_DOC + r"(?:[,;/]?\s*" + _SECTION + r")*|" + _SECTION)
_SEPARATORS = (" -- ", " — ", " – ")
_CONNECTORS = (" -- ", " — ", " – ", ", ", " ")
_INTRODUCERS = ("see ", "See ", "per ", "cf. ", "from ")
# A remainder that is itself only an internal pointer ("see §22.6", "parked")
# is no more useful to a reader than the citation it trailed.
_NOT_ACTIONABLE = re.compile(r"^(?:see\s*)?(?:§|sec\b|parked\b|item\b)", re.I)


def _enclosing_paren(s: str, i: int) -> tuple[int, int] | None:
    """Innermost balanced (...) containing index i, or None.

    Balanced rather than regex because real citations nest: a reference like
    `sec 16 E3(c)` carries its own parentheses, and a non-greedy `\\([^)]*\\)`
    would stop at the inner one and leave a stray tail behind.
    """
    depth = 0
    start = None
    for j in range(i, -1, -1):
        c = s[j]
        if c == ")":
            depth += 1
        elif c == "(":
            if depth == 0:
                start = j
                break
            depth -= 1
    if start is None:
        return None
    depth = 0
    for j in range(start, len(s)):
        c = s[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return start, j + 1
    return None


def _clean_clause(seg: str) -> str:
    """One comma-separated clause, reduced to whatever a reader can still use.

    A clause of the form `<doc> sec 18 F6 -- enable l0.noise_floor_repeats >= 2`
    exists to tell the reader what to do; only its first half is internal, so
    the instruction survives on its own. A clause that is nothing but a pointer
    -- a citation, a bare section number, "parked" -- has nothing to keep.
    """
    seg = seg.strip()
    if _NOT_ACTIONABLE.match(seg):
        return ""
    if not _REF.search(seg):
        return seg
    for sep in _SEPARATORS:
        if sep in seg:
            tail = seg.split(sep, 1)[1].strip()
            if tail and not _REF.search(tail) and not _NOT_ACTIONABLE.match(tail):
                return tail
    return ""


def _clean_parenthetical(inner: str) -> str:
    """Rebuild a parenthetical's contents with its internal clauses removed.

    Operating clause by clause rather than on the whole parenthetical is what
    keeps `(own-width-vs-error, equal-count deciles, <doc> sec 20 H4)` from
    losing the two clauses that describe the actual measurement.
    """
    parts = re.split(r"([,;] )", inner)
    out: list[str] = []
    for i in range(0, len(parts), 2):
        cleaned = _clean_clause(parts[i])
        if not cleaned:
            continue
        if out:
            out.append(parts[i - 1])
        out.append(cleaned)
    return "".join(out)


def strip_internal_refs(text: str) -> str:
    """Return `text` with every internal-document citation removed.

    Safe to run on a fully rendered HTML document: it only ever deletes a span
    that contains a citation, never rewrites markup, and stops a non-
    parenthetical deletion at the first `<` so it cannot eat a tag.
    """
    if not text:
        return text

    out = text
    guard = 0
    while guard < 20000:
        guard += 1
        m = _REF.search(out)
        if m is None:
            break

        span = _enclosing_paren(out, m.start())
        if span is not None and span[1] - span[0] <= 600:
            a, b = span
            kept = _clean_parenthetical(out[a + 1:b - 1])
            if kept:
                out = out[:a] + "(" + kept + ")" + out[b:]
                continue
            while a > 0 and out[a - 1] == " ":
                a -= 1
            if out[a - 1:a] in (",", ";"):
                a -= 1
            out = out[:a] + out[b:]
            continue

        start, end = m.start(), m.end()
        introduced = False
        for intro in _INTRODUCERS:
            if out[max(0, start - len(intro)):start] == intro:
                start -= len(intro)
                introduced = True
                break

        if introduced:
            # "See <ref> for the envelope edge." -- the introducer announced a
            # citation, so the rest of the sentence exists only to describe it.
            while end < len(out) and out[end] not in ".<)\n":
                end += 1
            if end < len(out) and out[end] == ".":
                end += 1
            out = out[:start] + out[end:]
            continue

        if out[end:end + 2] in ("'s", "\u2019s"):
            # "before <ref>'s Stage 0 gate" -- the sentence keeps going and
            # needs the possessive replaced, not deleted with everything after.
            out = out[:start] + "the" + out[end + 2:]
            continue

        for conn in _CONNECTORS:
            if out[max(0, start - len(conn)):start] == conn:
                start -= len(conn)
                break
        out = out[:start] + out[end:]

    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r" +([.,;)])", r"\1", out)
    out = re.sub(r"\(\s+", "(", out)
    return out


def strip_refs_in_place(obj):
    """Apply `strip_internal_refs` to every string in a nested dict/list."""
    if isinstance(obj, str):
        return strip_internal_refs(obj)
    if isinstance(obj, dict):
        return {k: strip_refs_in_place(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_refs_in_place(v) for v in obj]
    return obj
