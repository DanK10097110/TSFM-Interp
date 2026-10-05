"""ROADMAP.md sec 38.3.4: the L5 defined-firing rule at K2 registration and
confirmation (`confirm.register_requires_defined_firing`).

The dev agreement step is not rerun: the verdict is a pure function of a stored
test record, so `sae/shared_input_agreement.py::recompute_verdict_defined`
recomputes it from the record with `firing_defined` and `_verdict`, the one
implementation of the rule.

Planted known answer, with decoys, on four dev records:

- FALSE: the real false call's numbers. Statistic (i) -0.999 is below both
  floors' p05 (-0.486, -0.458), (ii) is 0.952 (inside its floor), and `level`
  clears on NEITHER side. Stored verdict `acts differently`; under the rule (i)
  is undefined, so the verdict is `no specific agreement`.
- ONE-SIDED: the decoy for a divergent implementation that defines (i) when
  EITHER side clears `level`. `level` clears on the source side only, (i) is
  below p05, (ii) is not. Under the rule it is `no specific agreement`; the
  divergent rule would keep it.
- GENUINE: `level` clears on both sides and (i) is -0.5, below both p05. It
  must be kept as `differs`.
- SAME: unaffected.

Each load-bearing assertion names the planted regression that breaks it. No
private split is opened.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_concept_causal_confirm import (  # noqa: E402,F401
    _fresh_run, _stage_dev_artifacts, dev_run)
from tsfm_lens.analysis import confirm as confirm_mod  # noqa: E402
from tsfm_lens.analysis.hypotheses import build_registry  # noqa: E402
from tsfm_lens.manifest import fingerprint_stage, resolve_config_keys  # noqa: E402
from tsfm_lens.sae import shared_input_agreement as sia  # noqa: E402
from tsfm_lens.utils import save_json  # noqa: E402

SAME, FALSE, GENUINE, ONE_SIDED = 0, 1, 2, 3
DIFFERS = "acts differently"
NO_AGREEMENT = "no specific agreement"


def _stat(obs, p05=(-0.2, -0.2), p95=(0.4, 0.4)):
    return {"observed": obs, "floor_p95_src": p95[0], "floor_p95_dst": p95[1],
            "floor_p05_src": p05[0], "floor_p05_dst": p05[1],
            "clears": obs > max(p95), "below_floor": obs < min(p05)}


def _record(concept, verdict, stat_i, stat_ii, clearing_src, clearing_dst, mask):
    return {"concept": concept, "verdict": verdict, "statistic_i": stat_i,
            "statistic_ii": stat_ii, "shape_mask": mask,
            "side_src": {"clearing_channels": clearing_src},
            "side_dst": {"clearing_channels": clearing_dst}}


def _planted_records() -> dict:
    return {
        SAME: _record(SAME, "same causal effect", _stat(0.9), _stat(0.95),
                      ["level", "trend"], ["level", "trend"], ["trend"]),
        FALSE: _record(FALSE, DIFFERS,
                       _stat(-0.999, p05=(-0.486, -0.458), p95=(0.5, 0.5)),
                       _stat(0.952, p95=(0.97, 0.97)),
                       ["dispersion", "trend"], ["dispersion", "trend"],
                       ["dispersion", "trend"]),
        GENUINE: _record(GENUINE, DIFFERS, _stat(-0.5), _stat(0.0),
                         ["level", "trend"], ["level", "dispersion"], ["trend", "dispersion"]),
        ONE_SIDED: _record(ONE_SIDED, DIFFERS, _stat(-0.379), _stat(-0.1),
                           ["level"], ["trend", "dispersion"], ["trend", "dispersion"]),
    }


def test_recompute_keeps_the_genuine_call_and_drops_the_undefined_ones():
    """Planted regressions: ignoring the rule keeps FALSE and ONE_SIDED as
    `acts differently`; a divergent implementation that defines (i) when
    EITHER side clears `level` keeps ONE_SIDED (FALSE alone cannot catch it,
    its `level` clears on neither side)."""
    rec = _planted_records()
    assert {k: sia.recompute_verdict_defined(r) for k, r in rec.items()} == {
        SAME: "same causal effect", FALSE: NO_AGREEMENT, GENUINE: DIFFERS,
        ONE_SIDED: NO_AGREEMENT}
    assert sia.record_firing_defined(rec[ONE_SIDED])["i"] is False
    assert sia.record_firing_defined(rec[GENUINE]) == {"i": True, "ii": True}


def test_recompute_reuses_firing_defined_and_verdict(monkeypatch):
    """One implementation: replacing `firing_defined` in the module changes the
    recomputed verdict. Planted regression: a second, inlined implementation
    inside `recompute_verdict_defined` would not follow the replacement."""
    rec = _planted_records()
    monkeypatch.setattr(sia, "firing_defined", lambda a, b, m: {"i": True, "ii": True})
    assert sia.recompute_verdict_defined(rec[FALSE]) == DIFFERS


def test_recompute_does_not_guess_a_missing_field_and_keeps_not_scorable():
    rec = _planted_records()[GENUINE]
    del rec["side_dst"]
    with pytest.raises(KeyError):
        sia.recompute_verdict_defined(rec)
    unscored = {"concept": 9, "verdict": "not scorable", "reason": "x"}
    assert sia.recompute_verdict_defined(unscored) == "not scorable"
    with pytest.raises(KeyError):
        sia.recompute_verdict_defined({"concept": 9, "verdict": DIFFERS})


def _planted_run(dev_run):
    """`(cfg, run_dir)`: the mock K2 dev run whose agreement artifact holds the
    four planted records over real targets and features."""
    cfg = _fresh_run(dev_run)
    run_dir = cfg.run_dir()
    staged = _stage_dev_artifacts(run_dir)
    (tp, lp), (ts, ls) = sorted(staged["feats"])
    members = staged["members"]
    base = {"src_target": f"{tp}/{lp}", "src_model": tp, "dst_target": f"{ts}/{ls}",
            "dst_model": ts, "dst_set_kind": "feature",
            "src_features": [members[0]["feature"]], "dst_feature": members[2]["feature"],
            "dst_features": [members[2]["feature"]]}
    save_json(run_dir / "sae" / "shared_input_agreement.json", {
        "tests": [dict(base, **r) for r in _planted_records().values()]})
    return cfg, run_dir


def _agreement(reg: dict) -> dict:
    return {str(h["concept"]): h
            for h in reg["hypotheses"] if h["stage"] == "shared_input_agreement"}


def test_registration_excludes_undefined_differs_and_keeps_the_rest(dev_run):
    """With the field on: FALSE and ONE_SIDED are not registered as `differs`
    and sit in `excluded` with a reason; GENUINE and SAME are registered and
    freeze the rule. Planted regression: the rule ignored at registration
    registers all four."""
    cfg, _ = _planted_run(dev_run)
    cfg.confirm.register_requires_defined_firing = True
    reg = build_registry(cfg)
    kept = _agreement(reg)
    assert set(kept) == {str(SAME), str(GENUINE)}
    assert all(h["requires_defined_firing"] is True for h in kept.values())
    assert kept[str(GENUINE)]["dev_verdict"] == DIFFERS
    assert kept[str(GENUINE)]["dev_firing_defined"] == {"i": True, "ii": True}
    s = reg["concept_claim_candidates"]["shared_input_agreement"]
    assert s["requires_defined_firing"] is True and s["n_excluded_undefined_firing"] == 2
    ex = {e["id"].split("::")[3]: e for e in s["excluded"]}
    assert set(ex) == {f"c{FALSE}", f"c{ONE_SIDED}"}
    assert all(e["recomputed_verdict"] == NO_AGREEMENT and "undefined" in e["reason"]
               and e["claim_type"] == "shared_input_agreement" for e in ex.values())
    assert ex[f"c{ONE_SIDED}"]["firing_defined"]["i"] is False


def test_registration_is_unchanged_with_the_field_off(dev_run):
    """Default: every stored `differs` is registered and no claim or summary
    gains a key. Planted regression: applying the rule regardless of the
    field drops FALSE and ONE_SIDED here."""
    cfg, _ = _planted_run(dev_run)
    assert cfg.confirm.register_requires_defined_firing is False
    reg = build_registry(cfg)
    kept = _agreement(reg)
    assert set(kept) == {str(SAME), str(FALSE), str(GENUINE), str(ONE_SIDED)}
    assert not any("requires_defined_firing" in h or "dev_firing_defined" in h
                   for h in kept.values())
    s = reg["concept_claim_candidates"]["shared_input_agreement"]
    assert "excluded" not in s and "requires_defined_firing" not in s


def test_field_requires_concept_claims_and_a_complete_record(dev_run):
    cfg, run_dir = _planted_run(dev_run)
    cfg.confirm.register_requires_defined_firing = True
    cfg.confirm.register_concept_claims = False
    with pytest.raises(ValueError, match="silently do nothing"):
        build_registry(cfg)
    cfg.confirm.register_concept_claims = True
    path = run_dir / "sae" / "shared_input_agreement.json"
    from tsfm_lens.utils import load_json
    doc = load_json(path)
    del doc["tests"][FALSE]["shape_mask"]
    save_json(path, doc)
    with pytest.raises(ValueError, match="the rule is not guessed at"):
        build_registry(cfg)


def test_field_is_omitted_at_default_and_fingerprints_register_when_on(dev_run):
    """`omit_at_default` and a field-level key on `register`. Planted
    regression: removing the marker puts the field into older fingerprints."""
    cfg = _fresh_run(dev_run)
    key = "confirm.register_requires_defined_firing"
    assert key not in resolve_config_keys(cfg, (key,))
    assert "register_requires_defined_firing" not in resolve_config_keys(
        cfg, ("confirm",))["confirm"]
    fp = fingerprint_stage(resolve_config_keys(cfg, (key,)), {})
    fp_confirm = fingerprint_stage(resolve_config_keys(cfg, ("confirm",)), {})
    cfg.confirm.register_requires_defined_firing = True
    assert resolve_config_keys(cfg, (key,))[key] is True
    assert fingerprint_stage(resolve_config_keys(cfg, (key,)), {}) != fp
    assert fingerprint_stage(resolve_config_keys(cfg, ("confirm",)), {}) != fp_confirm
    from tsfm_lens.pipeline import _stages
    assert key in {st.name: st for st in _stages()}["register"].config_keys


def _agreement_claim(i, verdict, frozen):
    h = {"id": f"shared_input_agreement::A/x::B/y::c{i}::f{i}::{verdict}",
         "stage": "shared_input_agreement", "concept": i, "src_target": "A/x",
         "dst_target": "B/y", "dst_feature": i, "dev_verdict": verdict,
         "src_model": "A", "dst_model": "B", "src_features": [1], "dst_features": [i],
         "dst_set_kind": "feature", "k_top_series": 4}
    if frozen:
        h["requires_defined_firing"] = True
    return h


def _private_row(i, obs_i, obs_ii, floor, clearing_src, clearing_dst, mask):
    f = sorted(floor)
    lo, hi = f[int(0.05 * len(f))], f[int(0.95 * len(f))]

    def stat(obs):
        return {"observed": obs, "floor_p95_src": hi, "floor_p95_dst": hi,
                "floor_p05_src": lo, "floor_p05_dst": lo, "clears": bool(obs > hi),
                "below_floor": bool(obs < lo), "n_floor_src": len(f), "n_floor_dst": len(f),
                "floor_values_src": list(floor), "floor_values_dst": list(floor)}
    return {"concept": i, "src_target": "A/x", "dst_target": "B/y", "dst_feature": i,
            "verdict": DIFFERS, "statistic_i": stat(obs_i), "statistic_ii": stat(obs_ii),
            "shape_mask": mask, "n_shared_series": 12,
            "side_src": {"clearing_channels": clearing_src},
            "side_dst": {"clearing_channels": clearing_dst}}


def test_confirm_applies_the_frozen_rule_to_the_private_verdict(monkeypatch):
    """Five `differs` claims on private rows:

    - 0 frozen, (i) far below p05 but `level` clears on neither side, (ii)
      defined and mid-floor: the false call. Not confirmed, p from (ii) alone.
    - 1 NOT frozen, the same private row: confirmed as before (the rule is read
      from the frozen spec, so older registries are unchanged).
    - 2 frozen, `level` clears on both sides, (i) below p05: confirmed.
    - 3 frozen, (i) undefined and far below, (ii) defined and below p05: the p
      comes from (ii) alone.
    - 4 frozen, both statistics far below p05 but neither defined: `not
      testable`, outside the Holm family.

    Planted regression: ignoring the frozen rule at confirm confirms claim 0,
    takes claim 3's p from (i) and puts claim 4 into the family."""
    rng = np.random.default_rng(0)
    floor = list(rng.normal(0.0, 0.1, 400))
    shape, other = ["trend"], ["dispersion"]
    rows = [
        _private_row(0, -0.9, 0.0, floor, other, other, other),
        _private_row(1, -0.9, 0.0, floor, other, other, other),
        _private_row(2, -0.9, 0.0, floor, ["level"], ["level"], shape),
        _private_row(3, -0.9, -0.3, floor, other + shape, other + shape, shape),
        _private_row(4, -0.9, -0.9, floor, ["level"], other, shape),
    ]
    claims = [_agreement_claim(0, DIFFERS, True), _agreement_claim(1, DIFFERS, False),
              _agreement_claim(2, DIFFERS, True), _agreement_claim(3, DIFFERS, True),
              _agreement_claim(4, DIFFERS, True)]
    monkeypatch.setattr(sia, "run_shared_input_agreement", lambda *a, **k: {"tests": rows})
    monkeypatch.setattr(sia, "reset_caches", lambda: None)
    cfg = SimpleNamespace(
        confirm=SimpleNamespace(alpha=0.05, agreement_n_null=400, path="", source="smoke"),
        run=SimpleNamespace(seed=0, device="cpu"))
    feats = {"A/x": {"pooled": None, "alive": None}, "B/y": {"pooled": None, "alive": None}}
    out = confirm_mod._confirm_agreement(cfg, None, None, None, {"hypotheses": claims},
                                         feats, {})
    t = {x["concept"]: x for x in out["tests"]}
    assert t[0]["firing_defined"] == {"i": False, "ii": True}
    assert t[0]["verdict"] == "not confirmed" and t[0]["rule_holds"] is False
    assert t[0]["p"] > 0.1
    assert t[1]["verdict"] == "confirmed" and t[1]["p"] < 0.01
    assert "firing_defined" not in t[1]
    assert t[2]["verdict"] == "confirmed" and t[2]["firing_defined"] == {"i": True, "ii": False}
    assert t[3]["firing_defined"] == {"i": False, "ii": True} and t[3]["rule_holds"] is True
    p_ii = confirm_mod._tail_p(-0.3, floor, "left")
    assert t[3]["p"] == pytest.approx(p_ii) and t[3]["p"] > t[1]["p"]
    assert t[4]["verdict"] == "not testable" and t[4]["confirmed"] is False
    assert "undefined" in t[4]["reason"]
    assert out["n_tested"] == 4 and out["n_not_testable"] == 1
