"""ROADMAP.md sec 38.3.4: the per-target significance gate on K2 registration
(`confirm.register_requires_target_significance`).

Fixture, with a planted known answer and decoys. On the K2 mock dev run
(`patchy` / `steppy`) the ablation artifacts get a hand-written
`n_clearing_cells` and `empirical_chance` block per target, four targets in
all so that BH has something to correct:

- `patchy/<A>`: 40 clears in 100 cells at rate 0.02, p ~ 5e-41. SIGNIFICANT.
- `steppy/<B>`: 16 clears in 100 cells at rate 0.10, raw p 0.0399 but BH q
  0.0798. The DECOY: it clears the raw 0.05 and (at the nominal 0.05 rate)
  looks overwhelmingly significant, yet it is at chance after BH against its
  own rate. It carries causal claims, the atlas concept 0 member set and both
  agreement tests, so every claim type has something to lose here.
- two `patchy` twin layers at chance (p 0.549 and 0.679), the BH family's
  filler.

Each load-bearing assertion names the planted regression that breaks it. No
private split is opened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom, false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_concept_causal_confirm import (  # noqa: E402,F401
    _fresh_run, _stage_dev_artifacts, dev_run)
from tsfm_lens.analysis.hypotheses import build_registry, run_register  # noqa: E402
from tsfm_lens.analysis.target_significance import (  # noqa: E402
    TargetChanceMissing, target_significance)
from tsfm_lens.manifest import fingerprint_stage, resolve_config_keys  # noqa: E402
from tsfm_lens.sae.ablation_run import ablation_path  # noqa: E402
from tsfm_lens.utils import load_json, save_json  # noqa: E402

TWINS = ("blocks.7", "blocks.8")
CHANCE = {"A": (40, 100, 0.02), "B": (16, 100, 0.10), "C": (10, 100, 0.10), "D": (9, 100, 0.10)}


def _set_chance(run_dir: Path, model: str, layer: str, key: str) -> None:
    c, n, r = CHANCE[key]
    path = ablation_path(run_dir, model, layer)
    art = load_json(path)
    art["n_clearing_cells"] = c
    art["empirical_chance"] = {"rule": "planted", "n_cells": n, "expected_cells": n * r,
                               "rate": r}
    save_json(path, art)


def _planted(dev_run):
    """`(cfg, run_dir, names)` with the four-target planted run and a second,
    fully significant atlas concept and agreement test on target A, so the
    kept side of every claim type is non-empty."""
    cfg = _fresh_run(dev_run)
    run_dir = cfg.run_dir()
    staged = _stage_dev_artifacts(run_dir)
    (tp, lp), (ts, ls) = sorted(staged["feats"])
    feats = staged["feats"][(tp, lp)]
    assert len(feats) >= 4, "fixture needs four causal features on target A"
    for twin in TWINS:
        art = load_json(ablation_path(run_dir, tp, lp))
        art["layer"] = twin
        save_json(ablation_path(run_dir, tp, twin), art)
    _set_chance(run_dir, tp, lp, "A")
    _set_chance(run_dir, ts, ls, "B")
    _set_chance(run_dir, tp, TWINS[0], "C")
    _set_chance(run_dir, tp, TWINS[1], "D")

    atlas_p = run_dir / "sae" / "concept_atlas.json"
    atlas = load_json(atlas_p)
    for r in atlas["rows"]:
        if r["model"] == tp and r["layer"] == lp and r["feature"] in feats[2:4]:
            r["concept"] = 1
    atlas["concepts"].append({"concept": 1, "n_members": 2, "n_models": 1, "models": {tp: 2}})
    save_json(atlas_p, atlas)
    stab_p = run_dir / "sae" / "concept_stability.json"
    stab = load_json(stab_p)
    stab["concepts"].append({"concept": 1, "stability": {"stable": True}})
    save_json(stab_p, stab)

    sia_p = run_dir / "sae" / "shared_input_agreement.json"
    sia = load_json(sia_p)

    def _stat(obs):
        return {"observed": obs, "floor_p95_src": 0.4, "floor_p95_dst": 0.4,
                "floor_p05_src": -0.2, "floor_p05_dst": -0.2, "clears": obs > 0.4,
                "below_floor": obs < -0.2}

    sia["tests"].append({
        "src_target": f"{tp}/{lp}", "src_model": tp, "dst_target": f"{tp}/{lp}",
        "dst_model": tp, "dst_set_kind": "feature", "concept": 1, "src_features": [feats[2]],
        "dst_feature": feats[3], "dst_features": [feats[3]], "verdict": "same causal effect",
        "statistic_i": _stat(0.9), "statistic_ii": _stat(0.95)})
    save_json(sia_p, sia)
    cfg.concepts.n_registered_causal = 64
    return cfg, run_dir, {"A": f"{tp}/{lp}", "B": f"{ts}/{ls}",
                          "C": f"{tp}/{TWINS[0]}", "D": f"{tp}/{TWINS[1]}"}


def _by_type(reg: dict, stage: str) -> list:
    return [h for h in reg["hypotheses"] if h["stage"] == stage]


def _targets_of(h: dict) -> set:
    if h["stage"] == "concept_causal":
        return {h["target"]}
    if h["stage"] == "concept_atlas":
        return {f"{m['model']}/{m['layer']}" for m in h["members"]}
    return {h["src_target"], h["dst_target"]}


def _excluded(reg: dict) -> dict:
    s = reg["concept_claim_candidates"]
    return {t: s[t].get("excluded", []) for t in
            ("concept_causal", "concept_atlas", "shared_input_agreement")}


def test_the_planted_decoy_is_what_the_fixture_says_it_is(dev_run):
    """The decoy's numbers, from the one gate function. B clears raw p 0.05
    and the nominal-0.05 binomial, and fails only BH against its own rate.
    Planted regressions (nominal rate; raw p) both turn B significant."""
    cfg, run_dir, names = _planted(dev_run)
    tbl = target_significance([
        (m, l, load_json(ablation_path(run_dir, m, l)))
        for m, l in (n.split("/", 1) for n in names.values())])["targets"]
    assert tbl[names["A"]]["significant"] is True
    b = tbl[names["B"]]
    assert b["p"] < 0.05 < b["q_value"] and b["significant"] is False
    assert binom.sf(b["clearing_cells"] - 1, b["n_cells"], 0.05) < 1e-4
    assert not tbl[names["C"]]["significant"] and not tbl[names["D"]]["significant"]


def test_target_significance_is_binomial_vs_empirical_rate_then_bh():
    """The statistic itself against scipy, on artifacts with a known answer.
    Planted regressions: `binom.sf(c, ...)` off by one; the nominal 0.05 in
    place of the recorded rate; no BH."""
    arts = [("m", f"l{i}", {"n_clearing_cells": c, "empirical_chance": {
        "n_cells": n, "rate": r, "expected_cells": n * r}})
        for i, (c, n, r) in enumerate(CHANCE.values())]
    out = target_significance(arts)
    p = np.array([binom.sf(c - 1, n, r) for c, n, r in CHANCE.values()])
    q = false_discovery_control(p, method="bh")
    for (_m, layer, _a), pv, qv in zip(arts, p, q):
        row = out["targets"][f"m/{layer}"]
        assert row["p"] == pytest.approx(pv, rel=1e-12)
        assert row["q_value"] == pytest.approx(qv, rel=1e-12)
        assert row["significant"] is bool(qv <= 0.05)
    assert out["n_significant"] == 1 and out["n_targets"] == 4
    zero = target_significance([("m", "z", {"n_clearing_cells": 0, "empirical_chance": {
        "n_cells": 50, "rate": 0.03, "expected_cells": 1.5}})])
    assert zero["targets"]["m/z"]["p"] == 1.0


def test_gate_excludes_chance_targets_claims_and_records_each_reason(dev_run):
    """Every claim type loses what sits on a non-significant target and the
    loss is a record (id, claim type, target, p, q, reason), not a silent
    drop. Planted regressions: gate ignored (B's claims are registered);
    nominal 0.05 or raw p (B becomes significant); excluded claims dropped
    without a record (the `excluded` lists are empty)."""
    cfg, run_dir, names = _planted(dev_run)
    ungated = build_registry(cfg)
    sig_targets = {names["A"]}
    for stage in ("concept_causal", "concept_atlas", "shared_input_agreement"):
        assert any(_targets_of(h) - sig_targets for h in _by_type(ungated, stage)), (
            f"fixture must give {stage} a claim touching a non-significant target")
    cfg.confirm.register_requires_target_significance = True
    reg = build_registry(cfg)
    excluded = _excluded(reg)
    for stage in ("concept_causal", "concept_atlas", "shared_input_agreement"):
        kept = _by_type(reg, stage)
        assert kept, f"{stage}: the significant target's claims must survive"
        assert all(_targets_of(h) <= sig_targets for h in kept), stage
        assert excluded[stage], f"{stage}: exclusions must be recorded"
        for rec in excluded[stage]:
            assert rec["claim_type"] == stage and rec["targets"] and rec["reason"]
            for t in rec["targets"]:
                assert t["target"] not in sig_targets
                assert t["p"] is not None and t["q_value"] is not None and t["q"] == 0.05
                assert t["q_value"] > 0.05
    ids_before = {h["id"] for h in _by_type(ungated, "concept_causal")}
    ids_excl = {r["id"] for r in excluded["concept_causal"]}
    ids_after = {h["id"] for h in _by_type(reg, "concept_causal")}
    assert ids_excl and ids_excl <= ids_before and not (ids_excl & ids_after)
    assert {r["id"] for r in excluded["concept_atlas"]} == {"concept_atlas::0"}
    assert {h["id"] for h in _by_type(reg, "concept_atlas")} == {"concept_atlas::1"}
    assert {r["id"].split("::")[-1] for r in excluded["shared_input_agreement"]} == {
        "same causal effect", "acts differently"}
    cs = reg["concept_claim_candidates"]
    assert cs["concept_causal"]["n_before_gate"] == len(_by_type(ungated, "concept_causal"))
    assert cs["concept_causal"]["n_after_gate"] == len(_by_type(reg, "concept_causal"))
    assert cs["concept_atlas"]["n_before_gate"] == 2 and cs["concept_atlas"]["n_after_gate"] == 1
    assert cs["target_significance"]["n_significant"] == 1
    assert cs["target_significance"]["targets"][names["B"]]["significant"] is False


def test_gate_keeps_the_significant_targets_claims_unchanged(dev_run):
    """Gating removes only the non-significant targets' claims: each kept
    claim is entry-for-entry identical to its ungated twin (same id, same
    frozen fields). Planted regression: the gate re-ranks or rewrites kept
    claims (their dicts differ)."""
    cfg, run_dir, names = _planted(dev_run)
    ungated = {h["id"]: h for h in build_registry(cfg)["hypotheses"]}
    cfg.confirm.register_requires_target_significance = True
    reg = build_registry(cfg)
    kept = [h for h in reg["hypotheses"] if h["stage"] in
            ("concept_causal", "concept_atlas", "shared_input_agreement")]
    assert any(h["target"] == names["A"] for h in kept if h["stage"] == "concept_causal")
    for h in kept:
        assert h == ungated[h["id"]], h["id"]


def test_gate_does_not_apply_to_structure_claims_and_says_why(dev_run):
    """Structure claims are panel-level aggregates and stay registered with
    the gate on; the notes carry the reasoning. Planted regression: gating
    structure on target significance drops the claim and the note."""
    cfg, run_dir, names = _planted(dev_run)
    atlas_p = run_dir / "sae" / "concept_atlas.json"
    atlas = load_json(atlas_p)
    for c in atlas["concepts"]:
        c["n_models"] = 1
    save_json(atlas_p, atlas)
    before = {h["id"] for h in _by_type(build_registry(cfg), "concept_structure")}
    cfg.confirm.register_requires_target_significance = True
    reg = build_registry(cfg)
    after = {h["id"] for h in _by_type(reg, "concept_structure")}
    assert before == after and before
    note = reg["concept_claim_candidates"]["concept_structure"]["target_gate"]
    assert "not applied" in note and "condition on the outcome" in note


def test_missing_empirical_chance_refuses_loudly_and_never_uses_nominal(dev_run):
    """A measured target without `empirical_chance` makes registration raise,
    naming it, and `hypotheses.json` is not written. Planted regression: a
    fallback to the nominal 0.05 registers instead of raising."""
    cfg, run_dir, names = _planted(dev_run)
    cfg.confirm.register_requires_target_significance = True
    path = ablation_path(run_dir, *names["B"].split("/", 1))
    art = load_json(path)
    del art["empirical_chance"]
    save_json(path, art)
    with pytest.raises(TargetChanceMissing, match=names["B"]):
        build_registry(cfg)
    reg_path = run_dir / "hypotheses.json"
    reg_path.unlink(missing_ok=True)
    with pytest.raises(TargetChanceMissing, match="never used"):
        run_register(cfg)
    assert not reg_path.exists()
    art["empirical_chance"] = {"n_cells": 100, "rate": None, "expected_cells": 0.0}
    save_json(path, art)
    with pytest.raises(TargetChanceMissing):
        build_registry(cfg)


def test_gate_off_registry_is_byte_identical_and_carries_no_gate_keys(dev_run):
    """Opt-in: with the gate at its default the registry has no gate keys, no
    `excluded`, and is byte-identical across the explicit-False and default
    configs even though the artifacts now carry `empirical_chance`. Planted
    regression: an unconditional `target_significance` entry in the summary."""
    cfg, run_dir, names = _planted(dev_run)
    assert cfg.confirm.register_requires_target_significance is False
    default_bytes = json.dumps(build_registry(cfg), sort_keys=True)
    cfg.confirm.register_requires_target_significance = False
    assert json.dumps(build_registry(cfg), sort_keys=True) == default_bytes
    assert "excluded" not in default_bytes and "target_significance" not in default_bytes
    assert "n_before_gate" not in default_bytes and "target_gate" not in default_bytes
    assert names["B"] in default_bytes


def test_gate_without_concept_claims_is_a_loud_error_not_a_silent_noop(dev_run):
    """Gate on but `register_concept_claims` off would gate nothing; that
    must raise. Planted regression: returning `[], {}` first."""
    cfg, run_dir, names = _planted(dev_run)
    cfg.confirm.register_concept_claims = False
    cfg.confirm.register_requires_target_significance = True
    with pytest.raises(ValueError, match="silently do nothing"):
        build_registry(cfg)


def test_gate_field_is_omitted_at_default_and_fingerprints_register_when_on(dev_run):
    """`omit_at_default`: the field leaves the older fingerprints untouched
    and, when on, changes both `register`'s field-level key and `confirm`'s
    whole-section key. Planted regression: removing the marker."""
    cfg = _fresh_run(dev_run)
    key = "confirm.register_requires_target_significance"
    assert key not in resolve_config_keys(cfg, (key,))
    assert "register_requires_target_significance" not in resolve_config_keys(
        cfg, ("confirm",))["confirm"]
    fp = fingerprint_stage(resolve_config_keys(cfg, (key,)), {})
    fp_confirm = fingerprint_stage(resolve_config_keys(cfg, ("confirm",)), {})
    cfg.confirm.register_requires_target_significance = True
    assert resolve_config_keys(cfg, (key,))[key] is True
    assert fingerprint_stage(resolve_config_keys(cfg, (key,)), {}) != fp
    assert fingerprint_stage(resolve_config_keys(cfg, ("confirm",)), {}) != fp_confirm
    from tsfm_lens.pipeline import _stages
    assert key in {st.name: st for st in _stages()}["register"].config_keys
