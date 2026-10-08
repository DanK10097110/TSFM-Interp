"""The pure pieces of `run_family_presence_known_answer.py`: the answer-key family cut, the
biased-null decoy and the pair status. Planted answers with the confusable cases on purpose:
a decoy concept whose atom must not enter any centroid, a signature no atom recovered, and a
decoy null that is only a decoy if it reproduces the observed effect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_family_presence_known_answer as drv  # noqa: E402
from tsfm_lens.analysis import confirm as confirm_mod  # noqa: E402
from tsfm_lens.analysis import family_claims as fc  # noqa: E402
from tsfm_lens.sae.concepts import CHANNELS  # noqa: E402


def _dir(i, d=8):
    v = np.zeros(d)
    v[i] = 1.0
    return v


def _manifest():
    return {"planted_layer": "blocks.2", "models": {"M": {"concepts": [
        {"id": "t", "cls": "shared", "kind": "trend", "sign": -1, "direction": _dir(0).tolist()},
        {"id": "l", "cls": "opposite", "kind": "level", "sign": 1, "direction": _dir(1).tolist()},
        {"id": "s", "cls": "unique", "kind": "seasonal", "sign": 1, "direction": _dir(2).tolist()},
        {"id": "inert", "cls": "decoy_input_only", "kind": None, "sign": 0,
         "direction": _dir(3).tolist()}]}}}


def _vec(channel, value):
    v = np.zeros(len(CHANNELS))
    v[CHANNELS.index(channel)] = value
    return v


def test_family_centroids_come_from_recovered_planted_atoms_only():
    """Atoms 0 and 1 recover the trend- and level+ concepts; atom 2 (seasonal) is NOT
    recovered (its row points elsewhere); atom 3 is the input-only decoy's atom and carries a
    big level effect. Planted regression: building the centroid from every causal feature
    pulls the level+ centroid toward the decoy atom's vector."""
    W = np.zeros((6, 8))
    W[0], W[1], W[3] = _dir(0), _dir(1), _dir(3)
    W[2] = _dir(7)
    dev = {("M", 0): _vec("trend", -2.0) + _vec("mase", 0.1), ("M", 1): _vec("level", -3.0),
           ("M", 2): _vec("seasonal", -1.0), ("M", 3): _vec("level", -3.0) + _vec("dispersion", 4.0)}
    doc, no_family = drv.family_table(_manifest(), {"M": W}, dev, {"M": [0, 1, 2, 3]})
    titles = {f["title"]: f for f in doc["families"]}
    assert set(titles) == {"trend-", "level+"} and "seasonal+" in no_family
    assert "level-" in no_family and "trend+" in no_family
    lev = np.array([titles["level+"]["mean_profile_ablation_units"][c] for c in CHANNELS])
    assert np.allclose(lev, drv.unit(_vec("level", -3.0)))
    assert [r["feature"] for r in doc["rows"]] == [0, 1, 2, 3]
    assert doc["params"]["assign_min"] == 0.5
    assert drv.planted_signatures(_manifest()["models"]["M"]) == {"trend-", "level+", "seasonal+"}


def test_decoy_null_reproduces_the_observed_effect_and_removes_the_claim():
    """A strongly present family (6 features along the centroid, quiet isotropic nulls) is
    claimed; the SAME batteries with the decoy null are not. Planted regression: leaving the
    null draws untouched keeps the claim."""
    rng = np.random.default_rng(0)
    by_feature = {}
    for f in range(6):
        obs = _vec("trend", 3.0) + rng.normal(0, 0.05, 9)
        null = rng.normal(0, 0.3, (40, 9))
        chans = {ch: {"available": True, "null_p95": 1.0, "signed_effect": float(obs[i]),
                      "null_draw_signed_means": [float(x) for x in null[:, i]]}
                 for i, ch in enumerate(CHANNELS)}
        by_feature[f] = {"feature": f, "scorable": True, "n_top_series": 8,
                         "n_channels_clearing": 1, "channels": chans}
    groups = {("profile_matched", 8, 40): {"M/L.0": {"withheld": False, "reason": "",
                                                       "by_feature": by_feature}}}
    claim = {"id": "family_presence::0::M", "stage": "family_presence", "family_id": 0,
             "family_title": "trend+", "model": "M", "targets": {"M/L.0": list(range(6))},
             "n_features": 6, "dev_centroid": _vec("trend", 1.0).tolist(), "min_cosine": 0.5,
             "dev_count": 6, "dev_basis": "cosine", "k_top_series": 8, "n_null_directions": 40,
             "null_mode": "profile_matched"}
    from types import SimpleNamespace
    cfg = SimpleNamespace(confirm=SimpleNamespace(alpha=0.05, family_presence_n_null=2000),
                          run=SimpleNamespace(seed=0))
    real = confirm_mod._confirm_family_presence(cfg, {"hypotheses": [claim]}, {}, groups)
    assert real["tests"][0]["verdict"] == "confirmed"
    bad = confirm_mod._confirm_family_presence(
        cfg, {"hypotheses": [claim]}, {}, drv.decoy_groups(groups, 1), tag="decoy")
    assert bad["tests"][0]["verdict"] == "not confirmed" and bad["tests"][0]["p"] > 0.2
    assert groups[("profile_matched", 8, 40)]["M/L.0"]["by_feature"][0]["channels"]["trend"][
        "null_draw_signed_means"] != drv.decoy_groups(groups, 1)[("profile_matched", 8, 40)][
        "M/L.0"]["by_feature"][0]["channels"]["trend"]["null_draw_signed_means"], "input is not mutated"
    v = fc.observed_and_null_vectors(drv.decoy_groups(groups, 1)[("profile_matched", 8, 40)][
        "M/L.0"]["by_feature"][0])
    assert float((v["null"] @ v["obs"]).min()) > 0.95


def test_status_of_names_every_non_detection():
    assert drv.status_of(None, True, False) == "no family"
    assert drv.status_of(None, False, True) == "not registered"
    assert drv.status_of({"status": "not testable"}, True, True) == "not testable"
    assert drv.status_of({"status": "tested", "verdict": "confirmed"}, True, True) == "claimed"
    assert drv.status_of({"status": "tested", "verdict": "not confirmed"}, True, True) == "not claimed"
