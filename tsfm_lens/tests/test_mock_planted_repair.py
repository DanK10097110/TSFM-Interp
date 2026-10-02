"""The opt-in `repair` vocabulary of `mock_planted` (ROADMAP.md sec 39, R0).

Its own properties, with no SAE: the answer key (signs of the effect on MASE), the decoy
(fires on hard series, `beta = 0`), the oracle table (corpus rows hit, foreign rows counted
and inert), and the side-effect sign read from the input-side activation. K1 and L5 are
untouched by construction and covered by their own test files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from repair_fixture import HORIZON, QUANTILES, build_world  # noqa: E402
from tsfm_lens.analysis.stats import mase  # noqa: E402
from tsfm_lens.models.mock_planted import concept_table_repair  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return build_world(seed=0, dose=8.0)


def _concept(w, cid):
    return next(c for c in w["manifest"]["concepts"] if c["id"] == cid)


def test_answer_key_signs_and_decoy(world):
    table = {r["id"]: r for r in concept_table_repair()}
    assert table["repair_harmful"]["mase_effect_of_presence"] == 1
    assert table["repair_helpful"]["mase_effect_of_presence"] == -1
    assert table["repair_decoy"]["mase_effect_of_presence"] == 0
    decoy = _concept(world, "repair_decoy")
    assert decoy["beta"] == 0.0 and decoy["hardness_spearman"] > 0.2
    for cid in ("repair_harmful", "repair_helpful", "repair_sideeffect"):
        assert _concept(world, cid)["beta"] > 0
    assert _concept(world, "repair_sideeffect")["split_value"] > 0


def test_oracle_hits_the_corpus_and_counts_foreign_rows(world):
    w = world
    a, data = w["adapter"], w["data"]
    a.predict(data.contexts()[:16], HORIZON, QUANTILES)
    assert a.manifest()["oracle_misses"] == 0
    rng = np.random.default_rng(0)
    foreign = rng.normal(size=(4, data.contexts().shape[1]))
    a.predict(foreign, HORIZON, QUANTILES)
    assert a.manifest()["oracle_misses"] == 4


def test_the_presence_of_the_planted_concepts_moves_mase_the_planted_way(world):
    w = world
    data = w["data"]
    contexts, targets = data.contexts(), data.targets()
    point = np.concatenate([w["adapter"].predict(contexts[i:i + 64], HORIZON, QUANTILES)["point"]
                            for i in range(0, len(contexts), 64)])
    naive = np.repeat(contexts[:, -1:], HORIZON, axis=1)
    delta = mase(point, targets, contexts) - mase(naive, targets, contexts)
    for cid, sign in (("repair_harmful", 1), ("repair_helpful", -1)):
        a = np.asarray(_concept(w, cid)["series_activation"])
        top = np.argsort(-a)[:12]
        assert np.sign(delta[top].mean()) == sign, cid
