"""Report-integration test for `report/sae_roles.py`'s pure table builder
`roles_summary_table` (ROADMAP.md sec 25.7, Component B(b)).

`report.py::_sae_roles_block` -- the renderer this file used to test end to
end -- was retired 2026-09-11 (ROADMAP.md sec 30, Stage 4): the SAE report
section now reads `sae/concepts.json` via `report/sae_concepts.py::
sae_concepts_block` instead of `sae/roles.json` via `_sae_roles_block`, per
sec 30.1's measured result that concepts (ablation-space clustering) beat
roles (injection-space clustering) at every target checked. `sae_roles.py`
itself is NOT retired -- its pure table builders stay unit-tested directly
(`test_sae_roles_also_moves.py`, `test_sae_role_evidence.py`), and this file
keeps the one check here that isn't already covered there: that the
report-facing `roles_summary_table` renders every row of a hypothetical
upstream artifact carrying a name collision, rather than silently
deduplicating two roles into one on-screen row. See
`tests/test_sae_concept_report.py` for the equivalent end-to-end coverage of
the new `sae_concepts_block` renderer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.response import CHANNELS  # noqa: E402


def _one_target_roles(model="Alpha", layer="layer.6"):
    candidates = [
        {"feature": 0, "clearing_channels": ["trend"]},
        {"feature": 1, "clearing_channels": ["trend"]},
        {"feature": 2, "clearing_channels": ["seasonal"]},
    ]
    roles = [
        {"role": 0, "name": "trend-slope ↑", "n_atoms": 2, "features": [0, 1],
         "dominant_channel": "trend", "dominant_effect_null_units": 2.75,
         "sign": 1, "structural_field": None, "structural_rho": None,
         "structural_n": None, "clears_null": True},
        {"role": 1, "name": "seasonal-magnitude ↑", "n_atoms": 1, "features": [2],
         "dominant_channel": "seasonal", "dominant_effect_null_units": 4.0,
         "sign": 1, "structural_field": "seasonal_amplitude_max",
         "structural_rho": 0.4, "structural_n": 200, "clears_null": True},
    ]
    rec = {"model": model, "layer": layer, "withheld": False, "skipped": False,
          "k": 2, "silhouette": 0.6, "non_modular": False, "non_modular_reason": "",
          "n_candidates": 3, "channel_columns": list(CHANNELS), "roles": roles,
          "candidates": candidates}
    return rec


def test_rendered_table_never_repeats_a_role_name_within_one_target():
    """Even if an upstream artifact somehow carried two roles with equal
    names (a defensive check, not an expected `run_sae_roles.py` output),
    the roles TABLE itself must not silently render a duplicate heading
    that reads as one role split across two rows with no way to tell them
    apart. Name uniqueness enforcement itself is `role_table`'s job
    (covered directly in `tests/test_sae_roles.py`); this checks the
    report-facing renderer is a straight pass-through that still shows both
    rows, distinguishable by their OTHER columns (atoms/features)."""
    from tsfm_lens.report.sae_roles import roles_summary_table
    rec = _one_target_roles("Alpha")
    df = roles_summary_table(rec)
    assert len(df) == len(rec["roles"])
