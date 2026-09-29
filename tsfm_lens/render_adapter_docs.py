"""Regenerate `tsfm_lens/ADAPTERS.md` from `tsfm_lens/adapter_docs.py`.

Unlike `render_stage_docs.py`/`render_glossary.py`, this writes a whole
standalone file rather than splicing a generated block into a hand-authored
one -- `ADAPTERS.md` has no other rendering surface and no hand-written
prose of its own to preserve around it (`ROADMAP.md` sec 34.6 Item E2).

Usage:
    python render_adapter_docs.py            # rewrite ADAPTERS.md in place
    python render_adapter_docs.py --check    # exit 1 if ADAPTERS.md is stale
    python render_adapter_docs.py --print    # print the generated doc only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens import adapter_docs


def generated_doc() -> str:
    rows, error_rows = adapter_docs.build_adapter_table()
    return adapter_docs.render_markdown(rows, error_rows)


def main() -> int:
    args = sys.argv[1:]
    out_path = Path(__file__).resolve().parent / "tsfm_lens" / "ADAPTERS.md"
    doc = generated_doc()

    if "--print" in args:
        print(doc)
        return 0

    current = out_path.read_text(encoding="utf-8") if out_path.exists() else None

    if "--check" in args:
        if current != doc:
            print(f"ADAPTERS.md is stale relative to the current registry -- run "
                  f"`python {Path(__file__).name}` to regenerate it.", file=sys.stderr)
            return 1
        print("ADAPTERS.md matches the current registry.")
        return 0

    if current != doc:
        out_path.write_text(doc, encoding="utf-8")
        print(f"ADAPTERS.md regenerated ({out_path}).")
    else:
        print("ADAPTERS.md already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
