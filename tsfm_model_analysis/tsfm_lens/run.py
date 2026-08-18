"""Thin CLI shim -- the real implementation is `tsfm_lens.run` (ROADMAP.md
E7a), so it can be installed as the `tsfm-lens` console script
(`[project.scripts]` in `pyproject.toml`). Kept here, at the documented
invocation location, so `python run.py --config configs/smoke.yaml` (run
from this directory, per CLAUDE.md sec 8) keeps working unchanged.

Examples:
    python run.py --config configs/smoke.yaml
    python run.py --config configs/default.yaml --stages l1,l2 --force l2
"""

from __future__ import annotations

from tsfm_lens.run import main

if __name__ == "__main__":
    main()
