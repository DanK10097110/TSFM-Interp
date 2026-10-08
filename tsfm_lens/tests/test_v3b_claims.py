"""ROADMAP.md sec 41 (V3-B): the three opt-in claim types -- per-model effect-family
presence, centroid-level atlas agreement, and the external_real replication leg --
plus their null-mode awareness (sec 41.1).

Layers, as in `test_concept_causal_confirm.py`. UNIT tests feed the statistics and
the `_confirm_*` blocks hand-built batteries with a planted known answer AND the
confusable decoy beside it (an effect the model's own random directions also
produce, a concept whose members scatter but whose centroid holds, a family one
member of a model happens to touch). The END-TO-END tests run `register` and
`confirm` on the real mock dev run. Each load-bearing assertion names the planted
regression that must break it.
"""

from __future__ import annotations

import copy
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_concept_causal_confirm import (_atlas_claim, _causal_features,  # noqa: E402,F401
                                               _fresh_run, _stage_dev_artifacts, _vec_rec,
                                               dev_run)
from tsfm_lens.analysis import confirm as confirm_mod  # noqa: E402
from tsfm_lens.analysis import family_claims as fc  # noqa: E402
from tsfm_lens.analysis import hypotheses as hyp  # noqa: E402
from tsfm_lens.config import DataConfig  # noqa: E402
from tsfm_lens.manifest import resolve_config_keys  # noqa: E402
from tsfm_lens.pipeline import run_pipeline  # noqa: E402
from tsfm_lens.sae.ablation_run import ablation_path  # noqa: E402
from tsfm_lens.sae.response import CHANNELS  # noqa: E402
from tsfm_lens.utils import load_json, save_json  # noqa: E402


def _e(i, scale=1.0):
    v = np.zeros(len(CHANNELS))
    v[i] = scale
    return v


def _signed_rec(feature, obs, null, clearing=1, p95=1.0):
    """A private-battery record whose observed vector is `obs` and whose
    signed null draws are the rows of `null` (p95 = 1, so the units are the
    vector's own). `null[d, c]` is draw `d`'s signed mean delta on channel `c`."""
    null = np.asarray(null, dtype=np.float64)
    chans = {ch: {"available": True, "null_p95": p95, "signed_effect": float(obs[i]),
                  "null_draw_signed_means": [float(x) for x in null[:, i]]}
             for i, ch in enumerate(CHANNELS)}
    return {"feature": feature, "scorable": True, "n_top_series": 8,
            "n_channels_clearing": clearing, "channels": chans}


def _battery(by_feature, withheld=False, reason=""):
    return {"withheld": withheld, "reason": reason, "by_feature": by_feature}


# ---------------------------------------------------------------------------
# 1. The statistics.
# ---------------------------------------------------------------------------

def test_observed_and_null_vectors_keep_signs_scale_and_mask():
    """The null vector of draw d is `signed_draw_d / null_p95` per channel, with 0
    in exactly the channels the observed vector zeroed. Planted regression:
    taking `np.abs` of the draws (the unsigned `null_draw_means` convention)
    breaks the sign assertion."""
    rng = np.random.default_rng(0)
    draws = rng.normal(0, 1.0, (12, 9))
    obs = rng.normal(0, 1.0, 9)
    rec = _signed_rec(7, obs, draws, p95=2.0)
    rec["channels"][CHANNELS[3]]["null_p95"] = None
    out = fc.observed_and_null_vectors(rec)
    raw = draws / 2.0
    raw[:, 3] = 0.0
    assert np.allclose(out["null"], fc.unit(raw))
    assert (out["null"][:, 3] == 0).all()
    assert (np.sign(out["null"][:, [0, 1, 2]]) == np.sign(draws[:, [0, 1, 2]])).all()
    assert out["obs"][3] == 0 and abs(np.linalg.norm(out["obs"]) - 1) < 1e-12


def test_observed_and_null_vectors_refuse_unscorable_records():
    rng = np.random.default_rng(1)
    base = _signed_rec(1, rng.normal(size=9), rng.normal(size=(5, 9)))
    assert fc.observed_and_null_vectors(None) is None
    assert fc.observed_and_null_vectors(dict(base, scorable=False)) is None
    no_draws = copy.deepcopy(base)
    for ch in CHANNELS:
        no_draws["channels"][ch].pop("null_draw_signed_means")
    assert fc.observed_and_null_vectors(no_draws) is None
    ragged = copy.deepcopy(base)
    ragged["channels"][CHANNELS[2]]["null_draw_signed_means"] = [0.1, 0.2]
    assert fc.observed_and_null_vectors(ragged) is None


def test_presence_statistic_planted_effect_and_biased_null_decoy():
    """Planted: 6 features all pointing at the family (cosine 1), and random
    directions almost never do (null cosines ~ N(0, .3)): a tiny p. Decoy: the
    SAME 6 observed cosines, but this model's random directions land in the family
    97% of the time (a level-dominated null): the count is exactly what chance
    gives, so the p must be large. Planted regression: a null that ignores the
    draws (a constant small probability) makes the decoy p tiny too."""
    thr = 0.5
    obs = np.ones(6)
    rng = np.random.default_rng(0)
    quiet = [rng.normal(0, 0.3, 40) for _ in range(6)]
    loud = [np.where(rng.random(40) < 0.97, 0.9, 0.0) for _ in range(6)]
    real = fc.presence_statistic(obs, quiet, thr, 2000, np.random.default_rng(1))
    decoy = fc.presence_statistic(obs, loud, thr, 2000, np.random.default_rng(1))
    assert real["observed_count"] == decoy["observed_count"] == 6
    assert real["p"] < 0.01
    assert decoy["p"] > 0.5
    assert decoy["expected_null_count"] > 5.0 > real["expected_null_count"]


def test_presence_statistic_null_probability_is_smoothed_and_floored():
    """A feature none of whose draws reach the family gets `(1 + 0)/(D + 1)`, never
    0 ("impossible" from 20 draws), and a feature with no draws at all gets 1.
    Planted regression: the raw fraction k/D gives 0.0."""
    out = fc.presence_statistic(np.array([1.0, 1.0]), [np.full(20, -1.0), np.array([])], 0.5,
                                200, np.random.default_rng(0))
    assert out["null_probability"][0] == pytest.approx(1 / 21)
    assert out["null_probability"][1] == 1.0
    assert 0 < out["p"] <= 1
    floor = fc.presence_statistic(np.array([1.0]), [np.full(100000, -1.0)], 0.5, 100,
                                  np.random.default_rng(0))
    assert floor["p"] == pytest.approx(1 / 101), "plus-one p is floored at 1/(n_null+1), never 0"


def test_candidate_for_null_contract():
    """The artifact contract (sec 41.1): the artifact's own null is the legacy
    keys; another null lives under `by_null[<mode>]`; a missing one RAISES. Planted
    regression: falling back to the legacy fields for a missing null returns the
    wrong null's numbers silently."""
    legacy = {"feature": 3, "scorable": True, "channels": {"level": {"signed_effect": 1.0}},
              "n_channels_clearing": 2}
    both = dict(legacy, by_null={"profile_matched_cov": {
        "scorable": True, "channels": {"level": {"signed_effect": -9.0}},
        "n_channels_clearing": 0}})
    assert fc.candidate_for_null(both, None, "profile_matched") is both
    assert fc.candidate_for_null(both, "profile_matched", "profile_matched") is both
    cov = fc.candidate_for_null(both, "profile_matched_cov", "profile_matched")
    assert cov["channels"]["level"]["signed_effect"] == -9.0
    assert cov["n_channels_clearing"] == 0 and cov["feature"] == 3 and "by_null" not in cov
    with pytest.raises(fc.NullModeUnavailable, match="profile_matched_cov"):
        fc.candidate_for_null(legacy, "profile_matched_cov", "profile_matched")


def test_external_disjointness_refuses_overlap_and_mixed_methods():
    ext = ({"a", "b", "c"}, "m")
    ok = fc.check_external_disjoint(ext, {"dev": ({"x"}, "m"), "private": ({"y"}, "m")})
    assert ok["overlap"] == {"dev": 0, "private": 0}
    with pytest.raises(RuntimeError, match=r"1 series shared with private"):
        fc.check_external_disjoint(ext, {"dev": ({"x"}, "m"), "private": ({"c", "z"}, "m")})
    with pytest.raises(ValueError, match="hash both sides the same way"):
        fc.check_external_disjoint(ext, {"dev": ({"x"}, "other")})


# ---------------------------------------------------------------------------
# 2. Registration.
# ---------------------------------------------------------------------------

def _families_run(tmp_path, null_by_feature=None, with_ablation=True):
    """A synthetic dev run dir: two families (0 = level axis, 1 = seasonal axis),
    model `A` with 5 level features, model `B` with 5 seasonal features and ONE
    level feature (the decoy: it touches family 0 but is far below any min_dev).
    Layer names are dotted on purpose."""
    run_dir = tmp_path / "run"
    (run_dir / "sae").mkdir(parents=True)
    lvl, sea = CHANNELS.index("level"), CHANNELS.index("seasonal")
    rows, cands = [], {}
    spec = {("A", "blocks.0.mlp"): [(f, lvl) for f in range(5)],
            ("B", "blocks.1.mlp"): [(f, sea) for f in range(10, 15)] + [(20, lvl)]}
    for (model, layer), feats in spec.items():
        cl = []
        for f, axis in feats:
            vec = _e(axis, 3.0)
            vec[(axis + 3) % 9] += 0.1 * (f % 3)
            cl.append(_vec_rec(f, vec))
            rows.append({"model": model, "layer": layer, "feature": f,
                         "family": 0 if axis == lvl else 1})
        cands[(model, layer)] = cl
    families = [
        {"family": 0, "title": "Level raisers", "n_members": 6,
         "models": {"A": 5, "B": 1},
         "mean_profile_ablation_units": {ch: float(v) for ch, v in zip(CHANNELS, _e(lvl, 3.0))}},
        {"family": 1, "title": "Seasonality dampeners", "n_members": 5, "models": {"B": 5},
         "mean_profile_ablation_units": {ch: float(v) for ch, v in zip(CHANNELS, _e(sea, 3.0))}}]
    save_json(run_dir / "sae" / "concept_families.json", {
        "measured": True, "params": {"assign_min": 0.5}, "families": families, "rows": rows})
    if with_ablation:
        for (model, layer), cl in cands.items():
            p = ablation_path(run_dir, model, layer)
            p.parent.mkdir(parents=True, exist_ok=True)
            cl = copy.deepcopy(cl)
            for c in cl:
                if null_by_feature and c["feature"] in null_by_feature:
                    c["by_null"] = {"profile_matched_cov": null_by_feature[c["feature"]]}
            save_json(p, {"model": model, "layer": layer, "candidates": cl,
                          **({"ablation_null": "profile_matched"})})
    return run_dir


def _v3_cfg(**over):
    conf = dict(family_presence_min_dev=3, family_presence_n_null_directions=0,
                family_presence_n_null=2000, primary_null="", alpha=0.05,
                atlas_n_null=2000, causal_max_null=1000, agreement_n_null=1000,
                structure_n_boot=2000, register_reliability_claims=False,
                register_family_presence_claims=True, register_atlas_centroid_claims=False,
                atlas_centroid_min_members=5, register_requires_target_significance=False)
    conf.update(over)
    return SimpleNamespace(confirm=SimpleNamespace(**conf),
                           concepts=SimpleNamespace(top_k_series=8, n_null_directions=16,
                                                    n_registered_causal=8),
                           sae=SimpleNamespace(ablation_null="profile_matched"))


def test_family_presence_registers_only_pairs_with_dev_support(tmp_path):
    """(family 0, A) and (family 1, B) have 5 supporting features and register;
    (family 0, B) has ONE (the decoy) and must not, nor (family 1, A). Planted
    regression: dropping the `min_dev` filter registers the decoy."""
    run_dir = _families_run(tmp_path)
    entries, summary = hyp._family_presence_entries(run_dir, _v3_cfg())
    ids = sorted(e["id"] for e in entries)
    assert ids == ["family_presence::0::A", "family_presence::1::B"], ids
    e = {x["id"]: x for x in entries}["family_presence::0::A"]
    assert e["dev_count"] == 5 and e["dev_basis"] == "cosine" and e["n_features"] == 5
    assert e["targets"] == {"A/blocks.0.mlp": [0, 1, 2, 3, 4]}
    assert e["null_mode"] == "profile_matched" and e["min_cosine"] == 0.5
    assert e["n_null_directions"] == 16 and e["k_top_series"] == 8
    assert abs(np.linalg.norm(e["dev_centroid"]) - 1) < 1e-12
    assert any(k.endswith("concept_families.json") for k in e["artifacts"])
    assert any("blocks.0.mlp" in k.replace("_", ".") or "blocks" in k for k in e["artifacts"])
    assert summary["n_below_min_dev"] >= 1 and summary["n_registered"] == 2


def test_family_presence_partition_fallback_is_stated(tmp_path):
    """Without the dev ablation artifacts the dev support is the partition's own
    per-model member count, and the claim says so. Planted regression: reading
    `dev_basis` as always `cosine` leaves the fallback unlabeled."""
    run_dir = _families_run(tmp_path, with_ablation=False)
    entries, _ = hyp._family_presence_entries(run_dir, _v3_cfg())
    assert {e["dev_basis"] for e in entries} == {"partition"}
    assert {e["id"]: e["dev_count"] for e in entries} == {
        "family_presence::0::A": 5, "family_presence::1::B": 5}


def test_family_presence_partial_dev_vectors_keep_the_cosine_basis(tmp_path):
    """One of A's five causal features has no dev vector (e.g. unscorable in the artifact):
    the dev support is still the cosine count over the four that have one, not the
    partition's count of five, so dev and confirm share one statistic. Planted regression:
    the old all-or-nothing rule falls back to `partition` and reports 5."""
    run_dir = _families_run(tmp_path)
    p = ablation_path(run_dir, "A", "blocks.0.mlp")
    art = load_json(p)
    art["candidates"] = [c for c in art["candidates"] if c["feature"] != 4]
    save_json(p, art)
    entries, _ = hyp._family_presence_entries(run_dir, _v3_cfg())
    e = {x["id"]: x for x in entries}["family_presence::0::A"]
    assert e["dev_basis"] == "cosine" and e["dev_count"] == 4 and e["dev_n_with_vector"] == 4
    assert e["n_features"] == 5 and e["dev_partition_count"] == 5


def test_atlas_centroid_min_members_default_is_three():
    """Fixed to 3 before the V3 data (the centroid-only statistic is the registered test;
    the >= 2 models requirement stays). Planted regression: the earlier default of 5."""
    from tsfm_lens.config import ConfirmConfig
    assert ConfirmConfig().atlas_centroid_min_members == 3
    entries, summary = hyp._atlas_centroid_entries(
        [{"concept": 0, "members": [{"model": "A"}, {"model": "A"}, {"model": "B"}],
          "models": ["A", "B"], "dev_centroid": [1.0] + [0.0] * 8, "min_cosine": 0.9,
          "min_members": 3, "statement": "s", "id": "concept_atlas::0"}],
        ConfirmConfig().atlas_centroid_min_members, "profile_matched")
    assert len(entries) == 1


def test_primary_null_is_recorded_and_a_mismatching_family_space_refuses(tmp_path):
    run_dir = _families_run(tmp_path)
    entries, _ = hyp._family_presence_entries(run_dir, _v3_cfg(primary_null="profile_matched"))
    assert {e["null_mode"] for e in entries} == {"profile_matched"}
    with pytest.raises(ValueError, match="profile_matched_cov"):
        hyp._family_presence_entries(run_dir, _v3_cfg(primary_null="profile_matched_cov"))
    with pytest.raises(ValueError, match="not one of"):
        hyp.v3_null_mode(_v3_cfg(primary_null="nonsense"))


def test_dev_vectors_read_the_requested_null_or_refuse(tmp_path):
    """`_dev_vectors(null_mode=...)` on an artifact holding both nulls reads the
    requested one; on one holding only its own null it refuses. Planted
    regression: ignoring `null_mode` returns the legacy vectors for both."""
    cov_rec = _vec_rec(0, _e(1, 5.0))
    run_dir = _families_run(tmp_path, null_by_feature={0: {
        "scorable": True, "channels": cov_rec["channels"], "n_channels_clearing": 1}})
    legacy = hyp._dev_vectors(run_dir)
    assert np.argmax(legacy[("A", "blocks.0.mlp", 0)]) == CHANNELS.index("level")
    with pytest.raises(fc.NullModeUnavailable):
        hyp._dev_vectors(run_dir, "profile_matched_cov")
    only0 = hyp._dev_vectors(run_dir, "profile_matched")
    assert only0 == legacy
    art = load_json(ablation_path(run_dir, "A", "blocks.0.mlp"))
    for c in art["candidates"]:
        c.setdefault("by_null", {"profile_matched_cov": {
            "scorable": True, "channels": cov_rec["channels"], "n_channels_clearing": 1}})
    save_json(ablation_path(run_dir, "A", "blocks.0.mlp"), art)
    art = load_json(ablation_path(run_dir, "B", "blocks.1.mlp"))
    for c in art["candidates"]:
        c["by_null"] = {"profile_matched_cov": {
            "scorable": True, "channels": cov_rec["channels"], "n_channels_clearing": 1}}
    save_json(ablation_path(run_dir, "B", "blocks.1.mlp"), art)
    both = hyp._dev_vectors(run_dir, "profile_matched_cov")
    assert np.argmax(both[("A", "blocks.0.mlp", 0)]) == 1


def _atlas_entry(cid, n, models):
    members = [{"model": models[i % len(models)], "layer": "L.0", "feature": i}
               for i in range(n)]
    return {"id": f"concept_atlas::{cid}", "stage": "concept_atlas", "family": "concept_atlas",
            "statistic": "x", "concept": cid, "members": members,
            "models": sorted(set(m["model"] for m in members)),
            "dev_centroid": [1.0] + [0.0] * 8, "min_cosine": 0.9, "min_members": 3,
            "artifact": "a", "artifact_sha256": "b", "artifacts": {"a": "b"},
            "statement": "legacy", "replicable": True}


def test_atlas_centroid_registration_filters_by_members_and_models():
    """>= 5 members spanning >= 2 models register (frozen members, centroid and
    threshold carried over, the legacy id recorded); a 4-member multi-model concept
    and a 6-member single-model concept do not. Planted regression: dropping
    either filter registers the decoys."""
    entries, summary = hyp._atlas_centroid_entries(
        [_atlas_entry(0, 5, ["M", "N"]), _atlas_entry(1, 4, ["M", "N"]),
         _atlas_entry(2, 6, ["M"])], 5, "profile_matched_cov")
    assert [e["id"] for e in entries] == ["concept_atlas_centroid::0"]
    e = entries[0]
    assert e["stage"] == "concept_atlas_centroid" and e["legacy_claim_id"] == "concept_atlas::0"
    assert e["null_mode"] == "profile_matched_cov" and e["min_members"] == 5
    assert e["dev_centroid"] == [1.0] + [0.0] * 8 and e["min_cosine"] == 0.9
    assert len(e["members"]) == 5 and "pair" not in e["statistic"]
    assert summary["n_below_threshold"] == 2


def test_v3_registration_is_off_by_default_and_refuses_a_dangling_primary_null(tmp_path):
    run_dir = _families_run(tmp_path)
    off = _v3_cfg(register_family_presence_claims=False)
    assert hyp._v3_claim_entries(run_dir, off) == ([], {})
    with pytest.raises(ValueError, match="silently do nothing"):
        hyp._v3_claim_entries(run_dir, _v3_cfg(register_family_presence_claims=False,
                                               primary_null="profile_matched_cov"))


def test_new_config_fields_do_not_move_older_fingerprints(dev_run):
    """Every V3-B field is `omit_at_default`: at defaults the register and
    confirm fingerprint inputs are exactly as before; a non-default value enters
    them. Planted regression: removing the marker puts the field in the resolved
    dict at its default."""
    cfg = _fresh_run(dev_run)
    reg_keys = ("concepts.n_registered", "concepts.transfer_claim_mode",
                "confirm.register_family_presence_claims", "confirm.family_presence_min_dev",
                "confirm.register_atlas_centroid_claims", "confirm.atlas_centroid_min_members",
                "confirm.primary_null", "confirm.family_presence_n_null_directions")
    cfg.confirm.register_concept_claims = False
    own = resolve_config_keys(cfg, reg_keys)
    assert set(own) == {"concepts.n_registered", "concepts.transfer_claim_mode"}
    conf = resolve_config_keys(cfg, ("confirm",))["confirm"]
    new = ("register_family_presence_claims", "family_presence_min_dev", "family_presence_n_null",
           "family_presence_n_null_directions", "primary_null", "register_atlas_centroid_claims",
           "atlas_centroid_min_members", "external_path", "external_source",
           "external_max_series")
    assert not [n for n in new if n in conf], [n for n in new if n in conf]
    cfg.confirm.register_family_presence_claims = True
    cfg.confirm.external_path = "ext"
    cfg.confirm.primary_null = "profile_matched_cov"
    assert resolve_config_keys(cfg, reg_keys)["confirm.register_family_presence_claims"] is True
    conf = resolve_config_keys(cfg, ("confirm",))["confirm"]
    assert conf["external_path"] == "ext" and conf["primary_null"] == "profile_matched_cov"


def test_family_budget_row_appears_only_when_registered_and_refuses_unsatisfiable():
    """The `family_presence` Holm family is its own ledger row (only when
    registered), and `check_concept_claims_before_opening` refuses an
    unsatisfiable one. Planted regression: leaving the family out of
    `_V3_NULL_FIELD` drops the row and the refusal."""
    cfg = _v3_cfg(family_presence_n_null=10)
    claims = [{"id": f"family_presence::{i}::A", "stage": "family_presence"} for i in range(3)]
    none = hyp.claim_family_budget(cfg, {"hypotheses": []})
    assert "family_presence" not in {r["family"] for r in none}
    rows = hyp.claim_family_budget(cfg, {"hypotheses": claims})
    row = next(r for r in rows if r["family"] == "family_presence")
    assert row["m"] == 3 and row["min_attainable_p_holm"] == pytest.approx(3 / 11)
    assert row["satisfiable"] is False
    with pytest.raises(RuntimeError, match="family 'family_presence'"):
        confirm_mod.check_concept_claims_before_opening(cfg, {"hypotheses": claims})
    ok = _v3_cfg(family_presence_n_null=2000)
    confirm_mod.check_concept_claims_before_opening(ok, {"hypotheses": claims})


# ---------------------------------------------------------------------------
# 3. Confirmation statistics on planted private batteries.
# ---------------------------------------------------------------------------

def _fp_claim(model, family_id, n_feat, null_mode="profile_matched", centroid=None,
              dev_count=5):
    c = _e(0) if centroid is None else np.asarray(centroid)
    c = c / np.linalg.norm(c)
    tgt = f"{model}/L.0"
    return {"id": f"family_presence::{family_id}::{model}", "stage": "family_presence",
            "family_id": family_id, "family_title": f"fam{family_id}", "model": model,
            "targets": {tgt: list(range(n_feat))}, "n_features": n_feat,
            "dev_centroid": [float(v) for v in c], "min_cosine": 0.5, "dev_count": dev_count,
            "dev_basis": "cosine", "k_top_series": 8, "n_null_directions": 40,
            "null_mode": null_mode}


def _fp_cfg(n_null=2000):
    return SimpleNamespace(confirm=SimpleNamespace(alpha=0.05, family_presence_n_null=n_null),
                           run=SimpleNamespace(seed=0))


def _noise_null(rng, n=40, sd=0.3):
    return rng.normal(0, sd, (n, 9))


def _group(null_mode, by_target, k=8, nd=40):
    return {(null_mode, k, nd): by_target}


def test_family_presence_planted_present_absent_and_biased_null_decoy():
    """Model A: 6 features all along the family axis, random directions are
    isotropic: confirmed. Model B: 6 features in unrelated directions: not
    confirmed. Model C (the decoy): 6 features along the axis, but C's own random
    directions ALSO always raise that axis (a level-dominated null): the count is
    what chance gives, so NOT confirmed. Planted regression: scoring against a
    null that ignores C's draws confirms C."""
    rng = np.random.default_rng(0)
    A = {f: _signed_rec(f, _e(0, 3) + rng.normal(0, 0.05, 9), _noise_null(rng))
         for f in range(6)}
    B = {f: _signed_rec(f, rng.normal(0, 1, 9) * np.where(np.arange(9) == 0, 0.0, 1.0),
                        _noise_null(rng)) for f in range(6)}
    C = {}
    for f in range(6):
        loud = _noise_null(rng, sd=0.05)
        loud[:, 0] = np.abs(rng.normal(2.0, 0.2, 40))
        C[f] = _signed_rec(f, _e(0, 3) + rng.normal(0, 0.05, 9), loud)
    claims = [_fp_claim("A", 0, 6), _fp_claim("B", 0, 6), _fp_claim("C", 0, 6)]
    group = _group("profile_matched", {"A/L.0": _battery(A), "B/L.0": _battery(B),
                                       "C/L.0": _battery(C)})
    out = confirm_mod._confirm_family_presence(_fp_cfg(), {"hypotheses": claims}, {}, group)
    by = {t["model"]: t for t in out["tests"]}
    assert by["A"]["verdict"] == "confirmed" and by["A"]["observed_count"] == 6
    assert by["B"]["verdict"] == "not confirmed" and by["B"]["observed_count"] <= 2
    assert by["C"]["verdict"] == "not confirmed" and by["C"]["observed_count"] == 6
    assert by["C"]["expected_null_count"] > 5.0 > by["A"]["expected_null_count"]
    assert out["n_tested"] == 3 and out["n_confirmed"] == 1 and out["n_not_testable"] == 0


def test_family_presence_a_family_the_model_does_not_plant_is_not_claimed():
    """The same model (A) tested against two families: the one it carries
    (axis 0) is confirmed, the one it does not (axis 4) is not, with identical
    random-direction nulls. Planted regression: a statistic that counts any causal
    feature as present (ignores the centroid) confirms both."""
    rng = np.random.default_rng(2)
    A = {f: _signed_rec(f, _e(0, 3) + rng.normal(0, 0.05, 9), _noise_null(rng))
         for f in range(6)}
    claims = [_fp_claim("A", 0, 6), _fp_claim("A", 1, 6, centroid=_e(4))]
    group = _group("profile_matched", {"A/L.0": _battery(A)})
    out = confirm_mod._confirm_family_presence(_fp_cfg(), {"hypotheses": claims}, {}, group)
    v = {t["family_id"]: t["verdict"] for t in out["tests"]}
    assert v == {0: "confirmed", 1: "not confirmed"}, v


def test_family_presence_not_testable_states_are_not_failures():
    """A failed capture, a withheld reach probe, an unfired feature and missing
    signed draws are `not testable` with a reason, never `not confirmed`, and stay
    out of the Holm family."""
    rng = np.random.default_rng(3)
    ok = {f: _signed_rec(f, _e(0, 3), _noise_null(rng)) for f in range(6)}
    nodraws = {f: {**_signed_rec(f, _e(0, 3), _noise_null(rng))} for f in range(6)}
    for rec in nodraws.values():
        for ch in CHANNELS:
            rec["channels"][ch].pop("null_draw_signed_means")
    unfired = {f: {"feature": f, "scorable": False} for f in range(6)}
    claims = [_fp_claim(m, 0, 6) for m in "ABCDE"]
    group = _group("profile_matched", {
        "A/L.0": _battery(ok), "B/L.0": _battery({}, withheld=True, reason="no reach"),
        "D/L.0": _battery(nodraws), "E/L.0": _battery(unfired)})
    out = confirm_mod._confirm_family_presence(
        _fp_cfg(), {"hypotheses": claims}, {"C/L.0": "RuntimeError: boom"}, group)
    by = {t["model"]: t for t in out["tests"]}
    assert by["A"]["status"] == "tested"
    for m in "BCDE":
        assert by[m]["status"] == "not testable" and by[m]["verdict"] == "not testable", m
        assert by[m]["confirmed"] is False
    assert "no reach" in by["B"]["reason"] and "boom" in by["C"]["reason"]
    assert out["n_tested"] == 1 and out["n_not_testable"] == 4


def test_family_presence_is_scored_only_against_its_registered_null():
    """A claim registered under `profile_matched_cov` with batteries only under
    another null is `not testable` and says so; it is never scored against the
    other null's draws. Planted regression: taking the first group regardless of
    `null_mode` scores (and here confirms) it."""
    rng = np.random.default_rng(4)
    A = {f: _signed_rec(f, _e(0, 3), _noise_null(rng)) for f in range(6)}
    claim = _fp_claim("A", 0, 6, null_mode="profile_matched_cov")
    wrong = _group("profile_matched", {"A/L.0": _battery(A)})
    out = confirm_mod._confirm_family_presence(_fp_cfg(), {"hypotheses": [claim]}, {}, wrong)
    t = out["tests"][0]
    assert t["status"] == "not testable" and "profile_matched_cov" in t["reason"]
    right = _group("profile_matched_cov", {"A/L.0": _battery(A)})
    out = confirm_mod._confirm_family_presence(_fp_cfg(), {"hypotheses": [claim]}, {}, right)
    assert out["tests"][0]["verdict"] == "confirmed"


def _centroid_claim(cid, members, dev_centroid, null_mode="profile_matched", min_members=5):
    c = _atlas_claim(cid, members, dev_centroid, min_members=min_members)
    return dict(c, id=f"concept_atlas_centroid::{cid}", stage="concept_atlas_centroid",
                null_mode=null_mode)


def _scattered_members(rng, n, x=0.9):
    """`n` unit vectors with cosine `x` to e0 and orthogonal noise otherwise: the
    pairwise cosine is ~x^2 (< 0.9) while the centroid holds ~0.98."""
    out = []
    for _ in range(n):
        u = rng.normal(size=9)
        u[0] = 0
        u /= np.linalg.norm(u)
        out.append(3.0 * (x * _e(0) + np.sqrt(1 - x ** 2) * u))
    return out


def test_atlas_centroid_confirms_a_concept_the_pair_rule_refuses():
    """The sec 40.1 case: five members share one effect direction on average but
    scatter pairwise (pair cosines ~0.81 < 0.9), so the legacy rule (>= half the
    pairs >= min_cosine) cannot confirm, while the centroid cosine is ~0.98 and
    beats the member-set null. A second concept whose members point in unrelated
    directions is not confirmed by either rule. Planted regressions: re-adding the
    pair leg to the centroid claim makes the first concept fail; scoring the
    centroid against the wrong direction confirms the decoy."""
    rng = np.random.default_rng(5)
    by_m, by_n = {}, {}
    for f, v in enumerate(_scattered_members(rng, 5)):
        (by_m if f < 3 else by_n)[f] = _vec_rec(f, v)
    for f in range(10, 15):
        (by_m if f < 13 else by_n)[f] = _vec_rec(f, rng.normal(size=9) * 3)
    for f in range(100, 160):
        by_m[f] = _vec_rec(f, rng.normal(size=9))
        by_n[f + 100] = _vec_rec(f + 100, rng.normal(size=9))
    batteries = {"M/L.0": _battery(by_m), "N/L.1": _battery(by_n)}

    def mem(fs):
        return [{"model": "M" if f in by_m else "N", "layer": "L.0" if f in by_m else "L.1",
                 "feature": f} for f in fs]

    dev = _e(0).tolist()
    cfg = SimpleNamespace(confirm=SimpleNamespace(alpha=0.05, atlas_n_null=2000),
                          run=SimpleNamespace(seed=0))
    cen = [_centroid_claim(0, mem(range(5)), dev), _centroid_claim(1, mem(range(10, 15)), dev)]
    out = confirm_mod._confirm_atlas_centroid(cfg, {"hypotheses": cen}, {},
                                              {"profile_matched": batteries})
    v = {t["concept"]: t["verdict"] for t in out["tests"]}
    assert v == {0: "confirmed", 1: "not confirmed"}, v
    t0 = out["tests"][0]
    assert t0["private_centroid_cosine"] > 0.95 and "private_pair_fraction" not in t0
    assert t0["null_centroid_cosine_p95"] < 0.8
    legacy = [_atlas_claim(0, mem(range(5)), dev, min_members=3)]
    old = confirm_mod._confirm_atlas(cfg, {"hypotheses": legacy}, {}, batteries)
    assert old["tests"][0]["verdict"] == "not confirmed"
    assert old["tests"][0]["private_pair_fraction"] < 0.5
    assert old["tests"][0]["p_centroid"] == pytest.approx(t0["p"]), (
        "the centroid claim's null is the legacy centroid null, same seed")


def test_atlas_centroid_needs_enough_testable_members_and_its_own_null():
    rng = np.random.default_rng(6)
    by_m = {f: _vec_rec(f, v) for f, v in enumerate(_scattered_members(rng, 5))}
    by_m.update({f: _vec_rec(f, rng.normal(size=9)) for f in range(100, 160)})
    del by_m[0], by_m[1]
    members = [{"model": "M", "layer": "L.0", "feature": f} for f in range(5)]
    cfg = SimpleNamespace(confirm=SimpleNamespace(alpha=0.05, atlas_n_null=300),
                          run=SimpleNamespace(seed=0))
    claim = _centroid_claim(0, members, _e(0).tolist())
    out = confirm_mod._confirm_atlas_centroid(
        cfg, {"hypotheses": [claim]}, {}, {"profile_matched": {"M/L.0": _battery(by_m)}})
    t = out["tests"][0]
    assert t["status"] == "not testable" and "needs 5" in t["reason"]
    assert out["n_tested"] == 0 and out["n_not_testable"] == 1
    out = confirm_mod._confirm_atlas_centroid(
        cfg, {"hypotheses": [dict(claim, null_mode="profile_matched_cov")]}, {},
        {"profile_matched": {"M/L.0": _battery(by_m)}})
    assert "profile_matched_cov" in out["tests"][0]["reason"]


def test_atlas_centroid_needs_the_threshold_as_well_as_the_null():
    """Members whose centroid cosine to the dev centroid is ~0.7 beat a random
    member-set null easily (p tiny) but sit below `min_cosine` 0.9: a weak match
    that is more than chance is not a replication of the frozen direction.
    Planted regression: dropping the threshold confirms it."""
    rng = np.random.default_rng(7)
    by_m = {f: _vec_rec(f, v) for f, v in enumerate(_scattered_members(rng, 6, x=0.4))}
    by_m.update({f: _vec_rec(f, rng.normal(size=9)) for f in range(100, 160)})
    members = [{"model": "M", "layer": "L.0", "feature": f} for f in range(6)]
    members[3:] = [{"model": "N", "layer": "L.1", "feature": f} for f in range(3, 6)]
    by_n = {f: by_m.pop(f) for f in range(3, 6)}
    by_n.update({f + 100: _vec_rec(f + 100, rng.normal(size=9)) for f in range(100, 160)})
    cfg = SimpleNamespace(confirm=SimpleNamespace(alpha=0.05, atlas_n_null=2000),
                          run=SimpleNamespace(seed=0))
    out = confirm_mod._confirm_atlas_centroid(
        cfg, {"hypotheses": [_centroid_claim(0, members, _e(0).tolist(), min_members=5)]}, {},
        {"profile_matched": {"M/L.0": _battery(by_m), "N/L.1": _battery(by_n)}})
    t = out["tests"][0]
    assert t["status"] == "tested" and t["p"] < 0.05
    assert t["private_centroid_cosine"] < 0.9 and t["rule_holds"] is False
    assert t["verdict"] == "not confirmed" and out["n_confirmed"] == 0


# ---------------------------------------------------------------------------
# 4. The external_real leg.
# ---------------------------------------------------------------------------

class _Bundle:
    def __init__(self, seed, n=12, overlap_with=None, k=0):
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=(n, 40)).astype(np.float32)
        if overlap_with is not None and k:
            self.x[:k] = overlap_with.x[:k]
        self.n = n
        self.corpus_digest = f"digest{seed}"

    def contexts(self):
        return self.x[:, :30]

    def targets(self):
        return self.x[:, 30:]


def _ext_cfg(tmp_path, source="sealed", ext="ext_dir"):
    return SimpleNamespace(
        confirm=SimpleNamespace(external_path=ext, path="priv_dir", source=source,
                                external_source=source, require_seal=False,
                                external_max_series=100),
        data=DataConfig(source=source, path="dev_dir", context_len=30, horizon=10),
        run=SimpleNamespace(seed=0))


def _patch_loaders(monkeypatch, dev, private, ext):
    by_path = {"dev_dir": dev, "priv_dir": private, "ext_dir": ext}
    monkeypatch.setattr(confirm_mod, "load_benchmark",
                        lambda dc, seed=0: by_path[dc.path])
    monkeypatch.setattr(confirm_mod, "read_manifest", lambda p: None)


def test_external_leg_refuses_overlap_with_dev_or_private(monkeypatch, tmp_path):
    """A planted shared series (3 rows copied from private; elsewhere 2 from dev)
    stops the leg before any analysis. Planted regression: skipping the hash check
    lets both through."""
    dev, private = _Bundle(1), _Bundle(2)
    cfg = _ext_cfg(tmp_path)
    _patch_loaders(monkeypatch, dev, private, _Bundle(3))
    ok = confirm_mod._prepare_external(cfg, private, None, None)
    assert ok["overlap"]["overlap"] == {"dev": 0, "private": 0}
    assert ok["overlap"]["method"] == "series value sha256"
    _patch_loaders(monkeypatch, dev, private, _Bundle(3, overlap_with=private, k=3))
    with pytest.raises(RuntimeError, match=r"3 series shared with private"):
        confirm_mod._prepare_external(cfg, private, None, None)
    _patch_loaders(monkeypatch, dev, private, _Bundle(3, overlap_with=dev, k=2))
    with pytest.raises(RuntimeError, match=r"2 series shared with dev"):
        confirm_mod._prepare_external(cfg, private, None, None)


def test_external_leg_refuses_same_path_and_wrong_role(monkeypatch, tmp_path):
    cfg = _ext_cfg(tmp_path)
    _patch_loaders(monkeypatch, _Bundle(1), _Bundle(2), _Bundle(3))
    cfg.confirm.external_path = "priv_dir"
    with pytest.raises(RuntimeError, match="same corpus as confirm.path"):
        confirm_mod._prepare_external(cfg, _Bundle(2), None, None)
    cfg.confirm.external_path = "ext_dir"
    monkeypatch.setattr(confirm_mod, "read_manifest", lambda p: {"role": "synthetic"})
    with pytest.raises(RuntimeError, match="role 'synthetic'"):
        confirm_mod._prepare_external(cfg, _Bundle(2), None, None)


def test_external_leg_uses_manifest_hashes_without_loading_other_corpora(monkeypatch, tmp_path):
    """When all three manifests list sample hashes the overlap is decided from
    them (no uncapped reload of dev or private), and a shared hash refuses."""
    cfg = _ext_cfg(tmp_path)
    ext = _Bundle(3)
    _patch_loaders(monkeypatch, _Bundle(1), _Bundle(2), ext)

    def boom(*a, **k):
        raise AssertionError("must not reload corpora when manifests list hashes")

    monkeypatch.setattr(confirm_mod, "_uncapped_hashes", boom)
    man = {"dev_dir": {"sample_hashes": ["d1", "d2"]}, "priv_dir": {"sample_hashes": ["p1"]},
           "ext_dir": {"sample_hashes": ["e1", "e2"], "role": "external_real"}}
    monkeypatch.setattr(confirm_mod, "read_manifest", lambda p: man[p])
    out = confirm_mod._prepare_external(cfg, _Bundle(2), man["dev_dir"], man["priv_dir"])
    assert out["overlap"]["method"] == "sealed manifest sample_hashes"
    assert out["role_declared"] == "external_real"
    man["ext_dir"] = {"sample_hashes": ["e1", "p1"]}
    with pytest.raises(RuntimeError, match=r"1 series shared with private"):
        confirm_mod._prepare_external(cfg, _Bundle(2), man["dev_dir"], man["priv_dir"])


def test_confirmation_complete_waits_for_the_external_leg(tmp_path):
    """A confirmation without `external_replication` is not complete once an
    external corpus is configured (so the stage runs again and adds only that
    leg); without one configured it is complete as before."""
    run = tmp_path / "r"
    (run / "confirm").mkdir(parents=True)
    cfg = SimpleNamespace(run_dir=lambda: run, confirm=SimpleNamespace(external_path=""))
    assert confirm_mod.confirmation_complete(cfg) is False
    save_json(run / "confirm" / "confirmation.json", {"a": 1})
    assert confirm_mod.confirmation_complete(cfg) is True
    cfg.confirm.external_path = "ext"
    assert confirm_mod.confirmation_complete(cfg) is False
    save_json(run / "confirm" / "confirmation.json", {"a": 1, "external_replication": {}})
    assert confirm_mod.confirmation_complete(cfg) is True


# ---------------------------------------------------------------------------
# 5. Report.
# ---------------------------------------------------------------------------

def test_report_renders_the_new_blocks_and_the_external_leg_apart(tmp_path):
    from tsfm_lens.report.report import _sec_confirm
    run_dir = tmp_path / "run"
    (run_dir / "confirm").mkdir(parents=True)
    base = {"status": "tested", "n_registered": 1, "n_tested": 1, "n_confirmed": 1,
            "n_not_testable": 0}
    fam_test = {"id": "family_presence::0::A", "family_id": 0, "family_title": "Level raisers",
                "model": "A", "null_mode": "profile_matched_cov", "dev_count": 5,
                "observed_count": 4, "n_features_testable": 5, "n_features_frozen": 6,
                "expected_null_count": 0.31, "null_count_p95": 1.0, "p": 0.0005,
                "p_holm": 0.0005, "verdict": "confirmed", "status": "tested"}
    save_json(run_dir / "confirm" / "confirmation.json", {
        "n_private_series": 40, "alpha": 0.05, "tests": [], "n_registered": 2,
        "n_replicable": 2, "registry_sha256": "abc",
        "concept_replication": {
            "status": "tested", "transfer": {"status": "skipped", "tests": []},
            "family_presence": dict(base, n_null=2000, tests=[fam_test]),
            "atlas_centroid": dict(base, tests=[{
                "id": "c", "concept": 0, "n_members": 5, "n_testable_members": 5,
                "models": ["M", "N"], "null_mode": "profile_matched_cov",
                "private_centroid_cosine": 0.97, "null_centroid_cosine_p95": 0.6, "p": 0.001,
                "p_holm": 0.001, "verdict": "confirmed", "status": "tested"}]),
            "ledger": [{"family": "family_presence", "m": 1, "n_null": 2000,
                        "min_attainable_p_holm": 0.0005, "satisfiable": True}]},
        "external_replication": {
            "status": "tested", "role": "external_real", "counted_in_confirm_verdict": False,
            "n_series": 50, "repeated_look": True, "manifest_role_declared": "external_real",
            "overlap_check": {"method": "series value sha256", "overlap": {"dev": 0, "private": 0}},
            "transfer": {"status": "tested", "n_confirmed": 0, "n_tested": 0, "tests": []},
            "family_presence": dict(base, tests=[dict(fam_test, p_holm=0.2,
                                                      verdict="not confirmed")])}})
    findings: list = []
    html = _sec_confirm(run_dir, findings, 0)
    for h in ("Effect-family presence claims", "Atlas centroid claims",
              "External-real replication", "reported apart from the confirm verdict",
              "Repeated look: this leg was forced a second time",
              "profile_matched_cov"):
        assert h in html, h
    assert html.count("What does this mean?") >= 3
    texts = [f.text for f in findings]
    ext = [f for f in findings if f.text.startswith("EXTERNAL-REAL")]
    assert len(ext) == 1 and ext[0].registered is False
    assert any("effect-family presence claims: 1/1" in t for t in texts)
    assert not any("EXTERNAL" in t for t in texts if t.startswith("CONFIRM"))
    ledger_html = html.split("Causal-concept multiplicity ledger")[1]
    assert "family_presence" in ledger_html


def test_l6_reads_the_centroid_claim_per_concept():
    from tsfm_lens.report.derived import _l6_causal_tests
    rep = {"atlas_centroid": {"tests": [{"concept": 3, "verdict": "confirmed"},
                                        {"concept": 4, "verdict": "not confirmed"}]},
           "family_presence": {"tests": [{"family_id": 3, "verdict": "confirmed"}]}}
    assert _l6_causal_tests(3, rep) == {"atlas centroid": [{"concept": 3, "verdict": "confirmed"}]}
    assert _l6_causal_tests(9, rep) == {}


# ---------------------------------------------------------------------------
# 6. The signed null draws in the real battery.
# ---------------------------------------------------------------------------

def test_signed_null_draws_are_opt_in_and_aligned_with_the_unsigned_ones(dev_run, monkeypatch):
    """`keep_signed_null_draws` adds `null_draw_signed_means` (same length and order
    as `null_draw_means`, so |signed| <= the matching unsigned mean) and, left off,
    changes no key of a legacy artifact. Planted regression: appending the signed
    rows under a different filter shifts the draw order, breaking the bound."""
    from tests.test_concept_causal_confirm import _hub_and_private
    from tsfm_lens.utils import resolve_device

    cfg = _fresh_run(dev_run)
    run_dir = cfg.run_dir()
    target = cfg.sae.targets[0]
    hub, data = _hub_and_private(cfg)
    acts, _ = confirm_mod._capture_private_targets(
        cfg, hub, data, [f"{target['model']}/{target['layer']}"])
    feats = confirm_mod._frozen_features(
        run_dir, target["model"], target["layer"],
        acts[f"{target['model']}/{target['layer']}"], resolve_device(cfg.run.device))
    _private_battery = confirm_mod._private_battery
    cand = _causal_features(run_dir)[(target["model"], target["layer"])][:2]
    plain = _private_battery(cfg, hub, data, run_dir, target["model"], target["layer"], feats,
                             cand, 8, 6, True)
    signed = _private_battery(cfg, hub, data, run_dir, target["model"], target["layer"], feats,
                              cand, 8, 6, True, signed_null=True)
    n_checked = 0
    for f in cand:
        a, b = plain["by_feature"].get(f), signed["by_feature"].get(f)
        if not a or not a.get("scorable"):
            continue
        for ch, ca in a["channels"].items():
            assert "null_draw_signed_means" not in ca
            cb = b["channels"][ch]
            if "null_draw_means" not in ca:
                continue
            assert cb["null_draw_means"] == ca["null_draw_means"]
            s, u = np.asarray(cb["null_draw_signed_means"]), np.asarray(cb["null_draw_means"])
            assert s.shape == u.shape and (np.abs(s) <= u + 1e-9).all()
            n_checked += 1
    assert n_checked > 0


# ---------------------------------------------------------------------------
# 7. End to end on the mock dev run.
# ---------------------------------------------------------------------------

def _stage_families(run_dir: Path) -> None:
    """One family per model on the real dev vectors (the mean vector of that
    model's causal features), written in `concept_families.json`'s own shape."""
    vectors = hyp._dev_vectors(run_dir)
    feats = _causal_features(run_dir)
    families, rows = [], []
    for fid, ((model, layer), fs) in enumerate(sorted(feats.items())):
        mean = np.mean([vectors[(model, layer, f)] for f in fs], axis=0)
        families.append({"family": fid, "title": f"{model} effects", "n_members": len(fs),
                         "models": {model: len(fs)},
                         "mean_profile_ablation_units": {
                             ch: float(v) for ch, v in zip(CHANNELS, mean)}})
        rows += [{"model": model, "layer": layer, "feature": f, "family": fid} for f in fs]
    save_json(run_dir / "sae" / "concept_families.json", {
        "measured": True, "params": {"assign_min": 0.5}, "families": families, "rows": rows})


def _v3_run_cfg(dev_run, external=True):
    cfg = _fresh_run(dev_run)
    run_dir = cfg.run_dir()
    _stage_dev_artifacts(run_dir)
    _stage_families(run_dir)
    c = cfg.confirm
    c.register_family_presence_claims = True
    c.family_presence_min_dev = 1
    c.family_presence_n_null = 400
    c.family_presence_n_null_directions = 8
    c.register_atlas_centroid_claims = True
    c.atlas_centroid_min_members = 4
    if external:
        c.external_path, c.external_source, c.external_max_series = "ext_smoke", "smoke", 40
    return cfg


def test_end_to_end_register_confirm_external_and_report(dev_run):
    cfg = _v3_run_cfg(dev_run)
    run_dir = cfg.run_dir()
    cfg.report.enabled = True
    run_pipeline(cfg, stages=["register", "confirm"])
    reg = load_json(run_dir / "hypotheses.json")
    stages = {h["stage"] for h in reg["hypotheses"]}
    assert {"family_presence", "concept_atlas_centroid", "concept_atlas"} <= stages
    assert set(reg["v3_claim_candidates"]) == {"family_presence", "concept_atlas_centroid"}
    fam_claims = [h for h in reg["hypotheses"] if h["stage"] == "family_presence"]
    assert len({h["id"] for h in fam_claims}) == len(fam_claims) >= 2
    assert all(h["null_mode"] == "mean_magnitude" and h["n_null_directions"] == 8
               for h in fam_claims)

    conf = load_json(run_dir / "confirm" / "confirmation.json")
    cr = conf["concept_replication"]
    assert {"family_presence", "atlas_centroid", "atlas"} <= set(cr)
    fp = cr["family_presence"]
    tested = [t for t in fp["tests"] if t["status"] == "tested"]
    assert tested and fp["n_registered"] == len(fam_claims)
    for t in tested:
        assert {"observed_count", "expected_null_count", "p", "p_holm", "null_mode",
                "n_features_testable", "dev_count"} <= set(t)
        assert 0 <= t["observed_count"] <= t["n_features_testable"]
        assert t["verdict"] in ("confirmed", "not confirmed")
    ac = cr["atlas_centroid"]
    assert ac["n_registered"] == 1 and "private_pair_fraction" not in str(ac["tests"][0])
    ledger = {r["family"]: r for r in cr["ledger"]}
    assert {"family_presence", "concept_atlas_centroid"} <= set(ledger)
    assert ledger["family_presence"]["m"] == len(fam_claims)

    ext = conf["external_replication"]
    assert ext["role"] == "external_real" and ext["counted_in_confirm_verdict"] is False
    assert ext["repeated_look"] is False and ext["status"] == "tested"
    assert ext["overlap_check"]["overlap"] == {"dev": 0, "private": 0}
    assert ext["family_presence"]["n_registered"] == len(fam_claims)
    assert ext["n_series"] <= 40
    assert "external_replication" not in cr
    assert conf["n_registered"] == len(reg["hypotheses"]), "external is not counted"
    assert not any("external" in str(r.get("family", "")) for r in cr["ledger"])

    from tsfm_lens.report.report import _sec_confirm
    html = _sec_confirm(run_dir, [], 0)
    for h in ("Effect-family presence claims", "Atlas centroid claims",
              "External-real replication"):
        assert h in html, h

    with pytest.raises(RuntimeError, match="consumed once"):
        confirm_mod.run_confirm(cfg, hub=None)
    run_pipeline(cfg, stages=["confirm"], force={"confirm"})
    again = load_json(run_dir / "confirm" / "confirmation.json")
    assert again["repeated_look"] is True
    assert again["external_replication"]["repeated_look"] is True


def test_external_leg_added_later_does_not_reopen_the_private_split(dev_run, monkeypatch):
    """A confirmation made without an external corpus gains the leg when one is
    configured, and ONLY the leg: every legacy key is byte-identical, the private
    behavioural pass is not re-run, and the leg is not a repeated look. Planted
    regression: dropping `external_only` re-runs the whole confirmation (the
    monkeypatched private pass raises)."""
    cfg = _v3_run_cfg(dev_run, external=False)
    run_dir = cfg.run_dir()
    run_pipeline(cfg, stages=["register", "confirm"])
    before = load_json(run_dir / "confirm" / "confirmation.json")
    assert "external_replication" not in before
    assert confirm_mod.confirmation_complete(cfg) is True

    cfg.confirm.external_path = "ext_smoke"
    cfg.confirm.external_source, cfg.confirm.external_max_series = "smoke", 40
    assert confirm_mod.confirmation_complete(cfg) is False

    def _no_second_look(*a, **k):
        raise AssertionError("the private split was analysed a second time")

    monkeypatch.setattr(confirm_mod, "_private_behavioral", _no_second_look)
    monkeypatch.setattr(confirm_mod, "_replicate_registered_cka", _no_second_look)
    run_pipeline(cfg, stages=["confirm"], allow_stale=True)
    after = load_json(run_dir / "confirm" / "confirmation.json")
    ext = after.pop("external_replication")
    assert after == before
    assert ext["repeated_look"] is False and ext["family_presence"]["tests"]
    assert confirm_mod.confirmation_complete(cfg) is True


def test_end_to_end_defaults_add_nothing(dev_run):
    """With every V3-B flag at its default the registry and the confirmation gain
    no key and no ledger row."""
    cfg = _fresh_run(dev_run)
    run_pipeline(cfg, stages=["register", "confirm"])
    reg = load_json(cfg.run_dir() / "hypotheses.json")
    assert "v3_claim_candidates" not in reg
    assert not {"family_presence", "concept_atlas_centroid"} & {h["stage"]
                                                                for h in reg["hypotheses"]}
    conf = load_json(cfg.run_dir() / "confirm" / "confirmation.json")
    assert "external_replication" not in conf
    cr = conf["concept_replication"]
    assert not {"family_presence", "atlas_centroid"} & set(cr)
    assert not {"family_presence", "concept_atlas_centroid"} & {
        r["family"] for r in cr.get("ledger", [])}


def test_unmeasured_stability_is_skipped_with_a_reason_not_a_crash(tmp_path):
    """`concept_stability.json` holds a "not measured" string where the replicate SAEs did
    not run. That concept is not seed-stable, is left out, and the registration summary says
    how many were skipped; a measured concept in the same file still counts as stable.
    Planted regression: the old `(c.get("stability") or {}).get(...)` raises
    AttributeError on the string."""
    sae = tmp_path / "sae"
    sae.mkdir()
    save_json(sae / "concept_atlas.json", {"rows": [], "params": {}})
    save_json(sae / "atlas_transfer.json", {"tests": []})
    save_json(sae / "concept_stability.json", {"concepts": [
        {"concept": 0, "stability": "not measured: no replicate SAEs"},
        {"concept": 1, "stability": {"stable": True}},
        {"concept": 2, "stability": {"stable": False}},
        {"concept": 3}]})
    stable, unmeasured = hyp._stable_concept_ids(load_json(sae / "concept_stability.json"))
    assert stable == {1} and unmeasured == [0]
    ranking = hyp._concept_transfer_candidates(tmp_path, SimpleNamespace(n_registered=5))
    assert ranking["stability_not_measured"] == [0]
    assert "no measured seed stability" in ranking["reason"]
    assert hyp._stable_atlas_members(tmp_path)[0] == set()
    entries, summary = hyp._atlas_candidates(tmp_path, _v3_cfg())
    assert entries == [] and summary["n_stability_not_measured"] == 1
    assert summary["n_stable_concepts"] == 1
