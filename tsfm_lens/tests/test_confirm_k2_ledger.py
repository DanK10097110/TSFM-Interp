"""The K2 concept-claim ledger in `confirm` must tolerate the `reliability_u1`
family that `claim_family_budget` reports when U1 claims are registered.

The first panel7_v2 confirm computed every private battery and then raised
`KeyError: 'reliability_u1'` while assembling this ledger (ROADMAP.md sec
38.2), because the K2 pass indexed a fixed four-family map with every budget
row. The planted budget below holds all four K2 families, an empty family
(skipped) and the U1 row (left to its own replication step).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis import confirm as confirm_mod  # noqa: E402
from tsfm_lens.analysis import hypotheses as hyp_mod  # noqa: E402


def _budget(cfg, registry):
    return [{"family": "concept_causal", "m": 3},
            {"family": "concept_atlas", "m": 0},
            {"family": "shared_input_agreement", "m": 2},
            {"family": "concept_structure", "m": 1},
            {"family": "reliability_u1", "m": 2}]


def test_ledger_skips_untested_families_and_keeps_k2_rows(monkeypatch):
    monkeypatch.setattr(hyp_mod, "claim_family_budget", _budget)
    out = {"ledger": [{"family": "concept_transfer", "m": 20}],
           "causal": {"n_tested": 3, "n_not_testable": 0},
           "agreement": {"n_tested": 1, "n_not_testable": 1}}
    rows = confirm_mod._k2_ledger(None, {}, out)
    fams = [r["family"] for r in rows]
    assert fams == ["concept_transfer", "concept_causal", "shared_input_agreement",
                    "concept_structure"]
    by = {r["family"]: r for r in rows}
    assert by["concept_causal"]["n_tested"] == 3
    assert by["shared_input_agreement"]["n_not_testable"] == 1
    assert by["concept_structure"]["n_tested"] is None
