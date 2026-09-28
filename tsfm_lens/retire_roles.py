"""ROADMAP.md sec 30, Stage 4/5 (2026-09-11) -- the ONE irreversible step of
the roles-to-concepts supersession: rename `<run>/sae/roles.json` to
`<run>/sae/roles_injection.json`, carrying `superseded_by`/
`superseded_reason` alongside every original per-target record (never
deleting the evidence, only archiving it under a new name).

This is deliberately a standalone, manually-invoked script -- NOT a pipeline
stage, and NOT called automatically by `run_concepts`/`run_sae_roles.py`.
The rename is gated on the report actually reading `concepts.json` first
(`tsfm_lens/sae/concepts.py::run_concepts`'s own docstring), which is a
per-run fact this script does not verify beyond checking `concepts.json`
exists in the named run directory -- see `sae/concepts.py::
supersede_roles_artifact` for the full precondition list and the exact
verify-then-remove sequencing that keeps `roles.json` untouched on any
failure.

    python retire_roles.py --run runs/full_report_run_4model
    python retire_roles.py --run runs/full_report_run_4model --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.sae.concepts import SUPERSEDED_REASON, supersede_roles_artifact
from tsfm_lens.utils import load_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path,
                    help="run directory whose sae/roles.json is renamed "
                         "to sae/roles_injection.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="check preconditions and print what would happen "
                         "without writing or removing anything")
    args = ap.parse_args()

    run_dir = args.run
    roles_path = run_dir / "sae" / "roles.json"
    concepts_path = run_dir / "sae" / "concepts.json"
    archived_path = run_dir / "sae" / "roles_injection.json"

    if args.dry_run:
        print(f"run:              {run_dir}")
        print(f"roles.json:       {'exists' if roles_path.exists() else 'MISSING'}")
        print(f"concepts.json:    {'exists' if concepts_path.exists() else 'MISSING'}")
        print(f"roles_injection.json already present: {archived_path.exists()}")
        if roles_path.exists():
            n = len(load_json(roles_path))
            print(f"roles.json targets: {n}")
        print("\nsuperseded_reason that would be written:\n")
        print(SUPERSEDED_REASON)
        print("\n(--dry-run: nothing written or removed)")
        return

    out = supersede_roles_artifact(run_dir)
    print(f"wrote {out}")
    print(f"removed {roles_path}")


if __name__ == "__main__":
    main()
