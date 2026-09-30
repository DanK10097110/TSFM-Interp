"""ROADMAP.md sec 38.2 (K2): registering and confirming the causal-concept
claims (`concept_causal`, `concept_atlas`, `shared_input_agreement`,
`concept_structure`).

Two layers, on purpose. The END-TO-END tests run the real `register` and
`confirm` stages on a mock run (`patchy` / `steppy`, real SAE checkpoints, a
real ablation battery) with a synthetic "private" split generated from the
smoke generators, so the plumbing (registry -> frozen checkpoints -> private
battery -> Holm -> `confirmation.json` -> report) is exercised as shipped. The
UNIT tests feed `_confirm_*` hand-built private batteries with a planted known
answer AND the confusable decoy beside it (an effect inside the null, an
input-only match, a vector-incoherent concept), because a statistic that
confirms everything passes a test that has no decoy. Each load-bearing
assertion names the planted regression that must break it.

No test opens a sealed private split: the "private" corpus is `source: smoke`
with a shifted seed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_concept_stage import _cfg as _stage_cfg  # noqa: E402
from tsfm_lens.analysis import confirm as confirm_mod  # noqa: E402
from tsfm_lens.analysis.hypotheses import (CONCEPT_CLAIM_STAGES, build_registry,  # noqa: E402
                                           check_registry_freshness,
                                           claim_family_budget, run_register)
from tsfm_lens.analysis.power import mde_ablation_effect  # noqa: E402
from tsfm_lens.extraction.store import ActivationStore  # noqa: E402
from tsfm_lens.manifest import fingerprint_stage, resolve_config_keys  # noqa: E402
from tsfm_lens.pipeline import run_pipeline  # noqa: E402
from tsfm_lens.sae.ablation_run import ablation_path, checkpoint_path  # noqa: E402
from tsfm_lens.utils import load_json, save_json  # noqa: E402


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Fixture: a real mock dev run, copied per test, with dev artifacts staged
# for the claim types the mock atlas is too small to produce by itself.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dev_run():
    out = tempfile.mkdtemp()
    cfg = _stage_cfg(out, name="k2dev")
    run_pipeline(cfg, stages=["extract"])
    store = ActivationStore(cfg.run_dir() / "activations.zarr", mode="r")
    cfg.sae.targets = [{"model": "patchy", "layer": store.layers("patchy")[-1]},
                       {"model": "steppy", "layer": store.layers("steppy")[-1]}]
    run_pipeline(cfg, stages=["sae", "concepts"])
    return cfg.run_dir(), cfg


def _k2_cfg(out_dir: str, name: str, targets: list):
    cfg = _stage_cfg(out_dir, name=name)
    cfg.sae.targets = targets
    cfg.confirm.enabled = True
    cfg.confirm.source = "smoke"
    cfg.confirm.max_series = 60
    cfg.confirm.require_seal = False
    cfg.confirm.register_concept_claims = True
    cfg.confirm.causal_n_null = 30
    cfg.confirm.causal_max_null = 300
    cfg.confirm.atlas_n_null = 300
    cfg.confirm.agreement_n_null = 200
    cfg.confirm.structure_n_boot = 300
    cfg.concepts.n_registered_causal = 8
    return cfg


def _fresh_run(dev_run, tmp_path=None):
    rd, cfg = dev_run
    work = Path(tempfile.mkdtemp())
    shutil.copytree(rd, work / rd.name)
    return _k2_cfg(str(work), rd.name, cfg.sae.targets)


def _causal_features(run_dir: Path) -> dict:
    out = {}
    for f in sorted(run_dir.glob("sae/*/*_ablation.json")):
        art = load_json(f)
        out[(art["model"], art["layer"])] = [
            int(c["feature"]) for c in art["candidates"]
            if c.get("scorable") and c.get("n_channels_clearing")]
    return out


def _stage_dev_artifacts(run_dir: Path, null_majority: bool = False) -> dict:
    """Write the dev artifacts the mock atlas is too small to produce: one
    seed-stable cross-model concept over real causal features, one shared-
    input `same causal effect` test and one `acts differently` test over real
    targets/features, and (optionally) a dev ablation set that is mostly
    causally null. Returns the members used."""
    feats = _causal_features(run_dir)
    (tp, lp), (ts, ls) = sorted(feats)
    members = ([{"model": tp, "layer": lp, "feature": f} for f in feats[(tp, lp)][:2]]
               + [{"model": ts, "layer": ls, "feature": f} for f in feats[(ts, ls)][:2]])
    member_keys = {(m["model"], m["layer"], m["feature"]) for m in members}
    rows = [dict(m, concept=0, pc1=0.0, pc2=0.0) for m in members]
    for (mo, la), fs in feats.items():
        rows += [{"model": mo, "layer": la, "feature": f, "concept": None, "pc1": 0.0,
                  "pc2": 0.0} for f in fs if (mo, la, f) not in member_keys]
    save_json(run_dir / "sae" / "concept_atlas.json", {
        "params": {"min_cosine": 0.9, "min_members": 3}, "rows": rows,
        "concepts": [{"concept": 0, "n_members": 4, "n_models": 2,
                      "models": {tp: 2, ts: 2}}]})
    save_json(run_dir / "sae" / "concept_stability.json", {
        "measured": True, "concepts": [{"concept": 0, "stability": {"stable": True}}]})
    save_json(run_dir / "sae" / "atlas_transfer.json", {"k_top_series": 4, "tests": []})

    def _stat(obs, p05=-0.2, p95=0.4):
        return {"observed": obs, "floor_p95_src": p95, "floor_p95_dst": p95,
                "floor_p05_src": p05, "floor_p05_dst": p05, "clears": obs > p95,
                "below_floor": obs < p05}

    base = {"src_target": f"{tp}/{lp}", "src_model": tp,
            "dst_target": f"{ts}/{ls}", "dst_model": ts, "dst_set_kind": "feature",
            "src_features": [members[0]["feature"]]}
    save_json(run_dir / "sae" / "shared_input_agreement.json", {"tests": [
        dict(base, concept=0, dst_feature=members[2]["feature"],
             dst_features=[members[2]["feature"]], verdict="same causal effect",
             statistic_i=_stat(0.9), statistic_ii=_stat(0.95)),
        dict(base, concept=1, dst_feature=members[3]["feature"],
             dst_features=[members[3]["feature"]], verdict="acts differently",
             statistic_i=_stat(-0.6), statistic_ii=_stat(-0.5))]})
    if null_majority:
        for f in run_dir.glob("sae/*/*_ablation.json"):
            art = load_json(f)
            cands = [c for c in art["candidates"] if c.get("scorable")]
            for c in cands[: int(len(cands) * 0.8)]:
                c["n_channels_clearing"] = 0
            save_json(f, art)
    return {"members": members, "feats": feats}


# ---------------------------------------------------------------------------
# 1. Registration.
# ---------------------------------------------------------------------------

def test_default_registration_is_unchanged_and_has_no_concept_claims(dev_run):
    """Opt-in: with `confirm.register_concept_claims` at its default the
    registry has none of the new entries and none of the new top-level keys
    (byte-identical to the pre-K2 registry -- invariant 13)."""
    cfg = _fresh_run(dev_run)
    cfg.confirm.register_concept_claims = False
    reg = build_registry(cfg)
    assert not [h for h in reg["hypotheses"] if h["stage"] in CONCEPT_CLAIM_STAGES]
    assert set(reg) == {"hypotheses", "n_replicable", "concept_transfer_candidates",
                        "concept_knob_candidates"}
    assert all("artifacts" not in h for h in reg["hypotheses"])


def test_concept_causal_claim_ids_unique_per_model_layer_feature(dev_run):
    """The same feature number and channel exists at two layers of one model
    and at both models; every id must still be distinct because it carries
    model, layer, feature AND channel. Planted regression: an id built
    without the layer collapses the two layers of `patchy` into duplicates,
    which `build_registry`'s refusal turns into a `ValueError`."""
    cfg = _fresh_run(dev_run)
    cfg.concepts.n_registered_causal = 64
    run_dir = cfg.run_dir()
    src = next(p for p in sorted(run_dir.glob("sae/patchy/*_ablation.json")))
    art = load_json(src)
    twin_layer = "blocks.4"
    art["layer"] = twin_layer
    save_json(ablation_path(run_dir, "patchy", twin_layer), art)
    reg = build_registry(cfg)
    ids = [h["id"] for h in reg["hypotheses"] if h["stage"] == "concept_causal"]
    assert len(ids) == len(set(ids))
    layers = {h["layer"] for h in reg["hypotheses"] if h["stage"] == "concept_causal"
              and h["model"] == "patchy"}
    assert {twin_layer} < layers and len(layers) == 2
    twin = [h for h in reg["hypotheses"] if h["stage"] == "concept_causal"
            and h["model"] == "patchy"]
    keys = [(h["feature"], h["channel"]) for h in twin]
    assert len(keys) > len(set(keys)), "fixture must contain the colliding decoy"
    assert all(h["layer"] in h["id"] and f"f{h['feature']}" in h["id"]
               and h["channel"] in h["id"] and h["model"] in h["id"]
               for h in reg["hypotheses"] if h["stage"] == "concept_causal")


def test_registration_minimum_per_model_and_stable_preference(dev_run):
    cfg = _fresh_run(dev_run)
    run_dir = cfg.run_dir()
    staged = _stage_dev_artifacts(run_dir)
    cfg.concepts.n_registered_causal = 4
    reg = build_registry(cfg)
    causal = [h for h in reg["hypotheses"] if h["stage"] == "concept_causal"]
    per_model = {}
    for h in causal:
        per_model[h["model"]] = per_model.get(h["model"], 0) + 1
    assert min(per_model.values()) >= 4, "at least 4 per model even when the cut is 4 in total"
    stable = {(m["model"], m["layer"], m["feature"]) for m in staged["members"]}
    for model in per_model:
        chosen = [h for h in causal if h["model"] == model]
        n_stable_available = sum(1 for k in stable if k[0] == model)
        assert sum(h["seed_stable_atlas_member"] for h in chosen) >= min(
            n_stable_available, len(chosen)), "seed-stable atlas members are preferred"


def test_registry_pins_every_artifact_including_checkpoints(dev_run):
    """Each K2 claim records the sha256 of every dev artifact it reads, `.pt`
    included, and freshness refuses a retrained checkpoint. Planted
    regression: leaving the checkpoint out of `artifacts` makes the refusal
    below not fire."""
    cfg = _fresh_run(dev_run)
    run_dir = cfg.run_dir()
    _stage_dev_artifacts(run_dir)
    reg = build_registry(cfg)
    k2 = [h for h in reg["hypotheses"] if h["stage"] in CONCEPT_CLAIM_STAGES]
    assert {h["stage"] for h in k2} >= {"concept_causal", "concept_atlas",
                                        "shared_input_agreement"}
    for h in k2:
        pts = [k for k in h["artifacts"] if k.endswith(".pt")]
        assert pts, h["id"]
        for rel, sha in h["artifacts"].items():
            assert sha == _sha(run_dir / rel)
    causal = next(h for h in k2 if h["stage"] == "concept_causal")
    ckpt = checkpoint_path(run_dir, causal["model"], causal["layer"])
    assert ckpt.relative_to(run_dir).as_posix() in causal["artifacts"]
    check_registry_freshness(cfg, reg)
    ckpt.write_bytes(ckpt.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match=r"\.pt"):
        check_registry_freshness(cfg, reg)


def test_new_fields_do_not_move_older_fingerprints(dev_run):
    """`omit_at_default`: defaults leave the register/confirm fingerprint
    inputs exactly as they were before K2 (§11.51), while a non-default
    value enters them. Planted regression: dropping the `omit_at_default`
    marker from a field puts it in the resolved dict at its default."""
    cfg = _fresh_run(dev_run)
    cfg.confirm.register_concept_claims = False
    cfg.confirm.causal_max_null = 1000
    cfg.confirm.causal_n_null = 200
    cfg.confirm.atlas_n_null = 2000
    cfg.confirm.agreement_n_null = 1000
    cfg.confirm.structure_n_boot = 2000
    cfg.concepts.n_registered_causal = 32
    reg_keys = ("concepts.n_registered", "concepts.transfer_claim_mode",
                "confirm.register_concept_claims", "concepts.n_registered_causal",
                "concepts.n_registered_agreement_differs")
    own = resolve_config_keys(cfg, reg_keys)
    assert set(own) == {"concepts.n_registered", "concepts.transfer_claim_mode"}
    conf = resolve_config_keys(cfg, ("confirm",))["confirm"]
    for name in ("register_concept_claims", "causal_n_null", "causal_max_null",
                 "atlas_n_null", "agreement_n_null", "structure_n_boot"):
        assert name not in conf, name
    fp_default = fingerprint_stage(own, {})
    cfg.confirm.register_concept_claims = True
    own_on = resolve_config_keys(cfg, reg_keys)
    assert own_on["confirm.register_concept_claims"] is True
    assert fingerprint_stage(own_on, {}) != fp_default
    cfg.confirm.causal_max_null = 5000
    assert resolve_config_keys(cfg, ("confirm",))["confirm"]["causal_max_null"] == 5000


def test_agreement_claim_ids_carry_concept_and_destination_feature(dev_run):
    """Two `same causal effect` tests between the SAME target pair (different
    concept, different destination feature) must not collapse to one id (the
    spec's `{src}::{dst}::{verdict}` form would). Planted regression: an id
    without the concept and feature raises the duplicate refusal."""
    cfg = _fresh_run(dev_run)
    run_dir = cfg.run_dir()
    staged = _stage_dev_artifacts(run_dir)
    path = run_dir / "sae" / "shared_input_agreement.json"
    doc = load_json(path)
    twin = copy.deepcopy(doc["tests"][0])
    twin["concept"] = 7
    twin["dst_feature"] = staged["members"][3]["feature"]
    twin["dst_features"] = [twin["dst_feature"]]
    doc["tests"].append(twin)
    save_json(path, doc)
    reg = build_registry(cfg)
    ids = [h["id"] for h in reg["hypotheses"] if h["stage"] == "shared_input_agreement"]
    same = [i for i in ids if i.endswith("::same causal effect")]
    assert len(same) == 2 and len(set(ids)) == len(ids)
    assert any("::c7::f" in i for i in same) and any("::c0::f" in i for i in same)


# ---------------------------------------------------------------------------
# 2. The private battery's own plumbing.
# ---------------------------------------------------------------------------

def test_confirm_battery_uses_private_ground_truth_periods(dev_run, monkeypatch):
    """`run_ablation_target(ground_truth_path=...)` reads the PRIVATE
    corpus's ground truth, never `cfg.data.path`. Planted regression: reading
    `cfg.data.path` makes the recorded path the dev one."""
    cfg = _fresh_run(dev_run)
    cfg.data.path = "DEV_CORPUS_PATH"
    cfg.confirm.source, cfg.confirm.path = "sealed", "PRIVATE_CORPUS_PATH"
    seen = []

    def _fake_gt(path):
        seen.append(path)
        raise FileNotFoundError("recorded only")

    from tsfm_lens.sae import ablation_run
    monkeypatch.setattr(ablation_run, "load_ground_truth_table", _fake_gt)
    assert confirm_mod._private_ground_truth_path(cfg) == "PRIVATE_CORPUS_PATH"
    from tsfm_lens.data import load_benchmark
    import dataclasses
    private = load_benchmark(dataclasses.replace(cfg.data, source="smoke", path="",
                                                 max_series=40), cfg.run.seed + 1000)
    run_dir = cfg.run_dir()
    (model, layer), fs = next(iter(sorted(_causal_features(run_dir).items())))
    from tsfm_lens.models import ModelHub
    from tsfm_lens.utils import resolve_device, resolve_dtype
    device = resolve_device(cfg.run.device)
    hub = ModelHub(cfg.models, cfg.data, device, resolve_dtype(cfg.run.dtype, device))
    acts, _ = confirm_mod._capture_private_targets(cfg, hub, private, [f"{model}/{layer}"])
    feats = confirm_mod._frozen_features(run_dir, model, layer, acts[f"{model}/{layer}"],
                                         device)
    confirm_mod._private_battery(cfg, hub, private, run_dir, model, layer, feats, fs[:2],
                                 4, 4, False)
    assert seen == ["PRIVATE_CORPUS_PATH"], seen
    assert "DEV_CORPUS_PATH" not in seen
    cfg.confirm.source = "smoke"
    assert confirm_mod._private_ground_truth_path(cfg) == "", (
        "a non-sealed private source must not fall back to the DEV table")


# ---------------------------------------------------------------------------
# 3. Frozen dev SAE, withheld targets, refusal before opening.
# ---------------------------------------------------------------------------

def _hub_and_private(cfg, n=40):
    import dataclasses
    from tsfm_lens.data import load_benchmark
    from tsfm_lens.models import ModelHub
    from tsfm_lens.utils import resolve_device, resolve_dtype
    device = resolve_device(cfg.run.device)
    hub = ModelHub(cfg.models, cfg.data, device, resolve_dtype(cfg.run.dtype, device))
    private = load_benchmark(dataclasses.replace(cfg.data, source="smoke", path="",
                                                 max_series=n), cfg.run.seed + 1000)
    return hub, private


def _register_and_replicate(cfg, staged=True):
    """Register from dev artifacts, then run `_replicate_causal_concept_claims`
    on a synthetic private split; returns `(registry, replication)`."""
    run_dir = cfg.run_dir()
    if staged:
        _stage_dev_artifacts(run_dir)
    run_register(cfg)
    registry = load_json(run_dir / "hypotheses.json")
    hub, private = _hub_and_private(cfg)
    acts, errors = confirm_mod._capture_private_targets(
        cfg, hub, private, confirm_mod._needed_targets(registry))
    rep = confirm_mod._replicate_causal_concept_claims(
        cfg, hub, private, registry, {"status": "skipped", "ledger": []}, acts, errors)
    return registry, rep


def test_confirm_causal_uses_frozen_dev_sae_not_retrained(dev_run, monkeypatch):
    """The private battery encodes and ablates with the dev checkpoint
    (`sae/<model>/<layer>.pt`), byte-for-byte unchanged, and never trains.
    Planted regression: a `train_sae` / `search_dict_size` call anywhere on
    this path trips the guard below."""
    cfg = _fresh_run(dev_run)
    run_dir = cfg.run_dir()
    _stage_dev_artifacts(run_dir)
    before = {p: _sha(p) for p in run_dir.glob("sae/*/*.pt")}
    from tsfm_lens.sae import ablation_run
    from tsfm_lens.sae import train as train_mod

    def _refuse(*a, **k):
        raise AssertionError("confirm must not retrain an SAE")

    for name in ("train_sae", "search_dict_size", "train_sae_replicate", "run_sae"):
        monkeypatch.setattr(train_mod, name, _refuse)
    loaded = []
    real_load = train_mod.load_sae_checkpoint

    def _spy(path):
        loaded.append(Path(path))
        return real_load(path)

    monkeypatch.setattr(train_mod, "load_sae_checkpoint", _spy)
    monkeypatch.setattr(ablation_run, "load_sae_checkpoint", _spy)
    registry, rep = _register_and_replicate(cfg, staged=False)
    causal = rep["causal"]
    assert causal["n_tested"] > 0
    used = {p.resolve() for p in loaded}
    for h in [h for h in registry["hypotheses"] if h["stage"] == "concept_causal"]:
        assert checkpoint_path(run_dir, h["model"], h["layer"]).resolve() in used
    assert {p: _sha(p) for p in run_dir.glob("sae/*/*.pt")} == before


def test_withheld_private_target_is_not_testable_not_failed(dev_run, monkeypatch):
    """A target whose private reach probe fails is `not testable`: not
    `confirmed`, not `not confirmed`, and outside the Holm family. Planted
    regression: mapping a withheld target to `confirmed: False` with verdict
    `not confirmed` breaks the verdict and the family-size assertions."""
    cfg = _fresh_run(dev_run)
    from tsfm_lens.sae import response as response_mod
    real = response_mod.reach_probe

    def _probe(cfg_, adapter, layer, *a, **k):
        if adapter.name == "patchy":
            return {"reachable": False, "reason": "planted: patchy is unreachable on private"}
        return real(cfg_, adapter, layer, *a, **k)

    monkeypatch.setattr(response_mod, "reach_probe", _probe)
    registry, rep = _register_and_replicate(cfg)
    causal = rep["causal"]
    withheld = [t for t in causal["tests"] if t["model"] == "patchy"]
    live = [t for t in causal["tests"] if t["model"] == "steppy"]
    assert withheld and live
    for t in withheld:
        assert t["status"] == "not testable" and t["verdict"] == "not testable"
        assert t["confirmed"] is False and "p_holm" not in t
        assert "planted: patchy" in t["reason"]
    assert causal["n_not_testable"] == len(withheld)
    assert causal["n_tested"] == len([t for t in live if t["status"] == "tested"])
    assert all(t["verdict"] in ("confirmed", "not confirmed") for t in live
               if t["status"] == "tested")
    ledger = {r["family"]: r for r in rep["ledger"]}
    assert ledger["concept_causal"]["n_not_testable"] == len(withheld)


def _write_registry(cfg, n_causal: int):
    run_dir = cfg.run_dir()
    _stage_dev_artifacts(run_dir)
    cfg.concepts.n_registered_causal = n_causal
    run_register(cfg)


def test_confirm_refuses_unsatisfiable_holm_family_before_opening(dev_run, monkeypatch):
    """`m / (n_null + 1) > alpha` for a registered family refuses BEFORE the
    private split is opened. Planted regression: removing the check lets
    `_load_private` run, which the guard below turns into a failure."""
    cfg = _fresh_run(dev_run)
    _write_registry(cfg, 8)
    cfg.confirm.causal_max_null = 20          # 8/21 > 0.05
    opened = []
    monkeypatch.setattr(confirm_mod, "_load_private",
                        lambda c: opened.append(1) or (_ for _ in ()).throw(
                            AssertionError("private split was opened")))
    from tsfm_lens.models import ModelHub
    with pytest.raises(RuntimeError, match="concept_causal.*cannot reach alpha"):
        confirm_mod.run_confirm(cfg, hub=None)
    assert not opened
    assert not (cfg.run_dir() / "confirm").exists() or not list(
        (cfg.run_dir() / "confirm").glob("confirmation.json"))
    rows = {r["family"]: r for r in claim_family_budget(cfg, load_json(
        cfg.run_dir() / "hypotheses.json"))}
    assert rows["concept_causal"]["satisfiable"] is False
    cfg.confirm.causal_max_null = 1000
    assert claim_family_budget(cfg, load_json(cfg.run_dir() / "hypotheses.json"))[0][
        "satisfiable"] is True


def test_confirm_refuses_ablation_null_mismatch_before_opening(dev_run, monkeypatch):
    """Claims frozen under one ablation null are never tested under another."""
    cfg = _fresh_run(dev_run)
    _write_registry(cfg, 8)
    registry = load_json(cfg.run_dir() / "hypotheses.json")
    assert {h["ablation_null"] for h in registry["hypotheses"]
            if h["stage"] == "concept_causal"} == {"mean_magnitude"}
    cfg.sae.ablation_null = "profile_matched"
    monkeypatch.setattr(confirm_mod, "_load_private",
                        lambda c: (_ for _ in ()).throw(AssertionError("opened")))
    with pytest.raises(RuntimeError, match="ablation null"):
        confirm_mod.run_confirm(cfg, hub=None)


def test_confirm_refuses_agreement_claims_frozen_under_another_null(dev_run, monkeypatch):
    """Registered agreement claims record the null mode and are covered by the same
    refusal: with every causal claim frozen under `profile_matched` and only the
    agreement claims frozen under the legacy null, a `profile_matched` config still
    refuses. Planted regression: dropping the agreement claims from the frozen set."""
    cfg = _fresh_run(dev_run)
    _write_registry(cfg, 8)
    path = cfg.run_dir() / "hypotheses.json"
    registry = load_json(path)
    agreement = [h for h in registry["hypotheses"] if h["stage"] == "shared_input_agreement"]
    assert agreement and {h["ablation_null"] for h in agreement} == {"mean_magnitude"}
    for h in registry["hypotheses"]:
        if h["stage"] == "concept_causal":
            h["ablation_null"] = "profile_matched"
    save_json(path, registry)
    cfg.sae.ablation_null = "profile_matched"
    monkeypatch.setattr(confirm_mod, "_load_private",
                        lambda c: (_ for _ in ()).throw(AssertionError("opened")))
    with pytest.raises(RuntimeError, match="ablation null"):
        confirm_mod.run_confirm(cfg, hub=None)


def test_doctor_row_matches_the_confirm_refusal(dev_run):
    from tsfm_lens.doctor import check_concept_claim_budget, run_preflight
    cfg = _fresh_run(dev_run)
    _write_registry(cfg, 8)
    assert check_concept_claim_budget(cfg).status == "pass"
    assert any(c.name == "concept claim budget" for c in run_preflight(cfg))
    cfg.confirm.causal_max_null = 20
    bad = check_concept_claim_budget(cfg)
    assert bad.status == "fail" and "concept_causal" in bad.detail
    cfg.confirm.register_concept_claims = False
    assert not any(c.name == "concept claim budget" for c in run_preflight(cfg)), (
        "the default preflight output must not gain a row")


# ---------------------------------------------------------------------------
# 4. Agreement: the lower-tail rule, on private draws.
# ---------------------------------------------------------------------------

def _agreement_claim(i, verdict):
    return {"id": f"shared_input_agreement::A/x::B/y::c{i}::f{i}::{verdict}",
            "stage": "shared_input_agreement", "concept": i, "src_target": "A/x",
            "dst_target": "B/y", "dst_feature": i, "dev_verdict": verdict,
            "src_model": "A", "dst_model": "B", "src_features": [1], "dst_features": [i],
            "dst_set_kind": "feature", "k_top_series": 4}


def _private_test_row(i, obs_i, obs_ii, floor, verdict):
    def stat(obs):
        f = sorted(floor)
        return {"observed": obs, "floor_p95_src": f[int(0.95 * len(f))],
                "floor_p95_dst": f[int(0.95 * len(f))], "floor_p05_src": f[int(0.05 * len(f))],
                "floor_p05_dst": f[int(0.05 * len(f))],
                "clears": bool(obs > f[int(0.95 * len(f))]),
                "below_floor": bool(obs < f[int(0.05 * len(f))]),
                "n_floor_src": len(f), "n_floor_dst": len(f),
                "floor_values_src": list(floor), "floor_values_dst": list(floor)}
    return {"concept": i, "src_target": "A/x", "dst_target": "B/y", "dst_feature": i,
            "verdict": verdict, "statistic_i": stat(obs_i), "statistic_ii": stat(obs_ii),
            "n_shared_series": 12}


def test_agreement_acts_differently_needs_lower_tail_on_private(monkeypatch):
    """An `acts differently` claim is confirmed only by falling BELOW both
    floors' p05. The decoy (claim 1) sits well inside the floor: it does not
    clear the p95 either, and a p95-style rule would call that disagreement.
    Planted regression: `rule_holds = not clears` (a p95 rule) confirms the
    decoy."""
    rng = np.random.default_rng(0)
    floor = list(rng.normal(0.0, 0.1, 400))
    rows = [
        _private_test_row(0, -0.9, -0.8, floor, "acts differently"),      # real: below p05
        _private_test_row(1, 0.02, 0.03, floor, "no specific agreement"),  # decoy: mid-floor
        _private_test_row(2, 0.9, 0.9, floor, "same causal effect"),       # real agreement
        _private_test_row(3, 0.02, 0.03, floor, "no specific agreement"),  # decoy for "same"
    ]
    claims = [_agreement_claim(0, "acts differently"), _agreement_claim(1, "acts differently"),
              _agreement_claim(2, "same causal effect"), _agreement_claim(3, "same causal effect")]
    from tsfm_lens.sae import shared_input_agreement as sia
    monkeypatch.setattr(sia, "run_shared_input_agreement",
                        lambda *a, **k: {"tests": rows})
    monkeypatch.setattr(sia, "reset_caches", lambda: None)
    cfg = SimpleNamespace(
        confirm=SimpleNamespace(alpha=0.05, agreement_n_null=400, path="", source="smoke"),
        run=SimpleNamespace(seed=0, device="cpu"))
    registry = {"hypotheses": claims}
    feats = {"A/x": {"pooled": None, "alive": None}, "B/y": {"pooled": None, "alive": None}}
    out = confirm_mod._confirm_agreement(cfg, None, None, None, registry, feats, {})
    by_id = {t["id"]: t for t in out["tests"]}
    v = {t["concept"]: t["verdict"] for t in out["tests"]}
    assert v == {0: "confirmed", 1: "not confirmed", 2: "confirmed", 3: "not confirmed"}, v
    assert by_id[claims[1]["id"]]["rule_holds"] is False
    assert by_id[claims[1]["id"]]["private_statistics"]["statistic_i"]["clears"] == False  # noqa: E712
    assert by_id[claims[0]["id"]]["p"] < 0.01 and by_id[claims[1]["id"]]["p"] > 0.1


def test_agreement_unscorable_private_side_is_not_testable(monkeypatch):
    from tsfm_lens.sae import shared_input_agreement as sia
    row = {"concept": 0, "src_target": "A/x", "dst_target": "B/y", "dst_feature": 0,
           "verdict": "not scorable", "reason": "B/y has no channel clearing its null"}
    monkeypatch.setattr(sia, "run_shared_input_agreement", lambda *a, **k: {"tests": [row]})
    monkeypatch.setattr(sia, "reset_caches", lambda: None)
    cfg = SimpleNamespace(
        confirm=SimpleNamespace(alpha=0.05, agreement_n_null=100, path="", source="smoke"),
        run=SimpleNamespace(seed=0, device="cpu"))
    registry = {"hypotheses": [_agreement_claim(0, "acts differently")]}
    feats = {"A/x": {"pooled": None, "alive": None}, "B/y": {"pooled": None, "alive": None}}
    out = confirm_mod._confirm_agreement(cfg, None, None, None, registry, feats, {})
    t = out["tests"][0]
    assert t["verdict"] == "not testable" and t["confirmed"] is False
    assert out["n_tested"] == 0 and out["n_not_testable"] == 1


# ---------------------------------------------------------------------------
# 5. Causal / atlas / structure statistics on planted private batteries.
# ---------------------------------------------------------------------------

def _rec(feature, channels, clearing=1, scorable=True):
    return {"feature": feature, "scorable": scorable, "n_top_series": 8,
            "n_channels_clearing": clearing, "channels": channels}


def _chan(effect, signed, null_draws, p95=None, rows=None):
    d = np.asarray(null_draws)
    return {"available": True, "effect": effect, "signed_effect": signed,
            "null_p95": float(np.quantile(d, 0.95)) if p95 is None else p95,
            "clears_null": bool(effect > np.quantile(d, 0.95)),
            "null_draw_means": [float(x) for x in d],
            "row_abs_effects": rows if rows is not None else [abs(signed)] * 8}


def _causal_cfg(n0=200, n_max=200):
    return SimpleNamespace(
        confirm=SimpleNamespace(alpha=0.05, causal_n_null=n0, causal_max_null=n_max),
        run=SimpleNamespace(seed=0))


def _causal_claim(feature, channel="level", sign=1):
    return {"id": f"concept_causal::M::L.0::f{feature}::{channel}", "stage": "concept_causal",
            "model": "M", "layer": "L.0", "target": "M/L.0", "feature": feature,
            "channel": channel, "sign": sign, "dev_effect": 1.0, "dev_null_p95": 0.3,
            "dev_effect_over_null_p95": 3.3, "ablation_null": "mean_magnitude",
            "k_top_series": 8, "atlas_concept": None}


def test_causal_planted_effect_confirms_and_decoys_do_not():
    """Planted: feature 1 moves the channel far beyond its null and with the
    dev sign (confirmed); feature 2 has a real effect but the WRONG sign vs
    dev (sign decoy); feature 3's effect sits inside the null (magnitude
    decoy). Only feature 1 is confirmed, and every test carries an MDE."""
    rng = np.random.default_rng(0)
    null = rng.normal(0.30, 0.02, 200)
    by_feature = {
        1: _rec(1, {"level": _chan(1.0, 1.0, null)}),
        2: _rec(2, {"level": _chan(1.0, -1.0, null)}),
        3: _rec(3, {"level": _chan(0.30, 0.30, null)}),
    }
    registry = {"hypotheses": [_causal_claim(1), _causal_claim(2), _causal_claim(3)]}
    batteries = {"M/L.0": {"withheld": False, "reason": "", "by_feature": by_feature}}
    out = confirm_mod._confirm_concept_causal(_causal_cfg(), None, None, None, registry, {},
                                              {}, batteries)
    v = {t["feature"]: t["verdict"] for t in out["tests"]}
    assert v == {1: "confirmed", 2: "not confirmed", 3: "not confirmed"}, v
    by = {t["feature"]: t for t in out["tests"]}
    assert by[2]["sign_matches_dev"] is False and by[2]["p"] < 0.01
    assert by[3]["p"] > 0.05
    for t in out["tests"]:
        assert "mde" in t and "non_replication_reading" in t
    assert by[1]["p_holm"] < 0.05


def test_causal_adaptive_redraw_resolves_floor_hits():
    """A claim whose exact p sits on 1/(n+1) is redrawn at the larger null;
    the p can then go below the first floor, and the method is recorded."""
    rng = np.random.default_rng(1)
    small = rng.normal(0.3, 0.02, 30)
    big = rng.normal(0.3, 0.02, 300)
    claim = _causal_claim(1)
    first = {1: _rec(1, {"level": _chan(1.0, 1.0, small)})}
    second = {1: _rec(1, {"level": _chan(1.0, 1.0, big)})}
    batteries = {"M/L.0": {"withheld": False, "reason": "", "by_feature": first},
                 ("adaptive", "M/L.0", 1): {"withheld": False, "reason": "",
                                            "by_feature": second}}
    out = confirm_mod._confirm_concept_causal(_causal_cfg(30, 300), None, None, None,
                                              {"hypotheses": [claim]}, {}, {}, batteries)
    t = out["tests"][0]
    assert t["p_method"] == "adaptive" and t["n_null"] == 300
    assert t["p"] == pytest.approx(1 / 301) and t["p"] < 1 / 31
    assert t["verdict"] == "confirmed"


def test_causal_unfired_feature_and_missing_channel_are_not_testable():
    null = np.random.default_rng(2).normal(0.3, 0.02, 50)
    by_feature = {1: _rec(1, {}, clearing=0, scorable=False),
                  2: _rec(2, {"level": {"available": False, "reason": "no finite rows"}})}
    registry = {"hypotheses": [_causal_claim(1), _causal_claim(2)]}
    batteries = {"M/L.0": {"withheld": False, "reason": "", "by_feature": by_feature}}
    out = confirm_mod._confirm_concept_causal(_causal_cfg(), None, None, None, registry, {},
                                              {}, batteries)
    assert [t["verdict"] for t in out["tests"]] == ["not testable", "not testable"]
    assert out["n_tested"] == 0 and out["n_confirmed"] == 0
    del null


def _atlas_claim(cid, members, dev_centroid, min_cosine=0.9, min_members=3):
    return {"id": f"concept_atlas::{cid}", "stage": "concept_atlas", "concept": cid,
            "members": members, "models": sorted({m["model"] for m in members}),
            "dev_centroid": dev_centroid, "min_cosine": min_cosine, "min_members": min_members}


def _vec_rec(feature, vec, clearing=1):
    """A battery record whose `ablation_vector` is exactly `vec` (p95 = 1)."""
    from tsfm_lens.sae.response import CHANNELS
    chans = {ch: {"null_p95": 1.0, "signed_effect": float(v), "available": True}
             for ch, v in zip(CHANNELS, vec)}
    return _rec(feature, chans, clearing=clearing)


def test_atlas_coherent_concept_confirms_and_incoherent_decoy_does_not():
    """Planted: concept 0's three members keep one direction on private data
    (confirmed); concept 1's members were coherent on dev but point in
    unrelated directions on private data (decoy). The private causal pool
    (the null) is a broad mix of directions, so coherence is informative."""
    rng = np.random.default_rng(3)
    dirn = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0]) * 4.0
    by_feature = {}
    pool = {}
    for f in range(100, 160):
        by_feature[f] = _vec_rec(f, rng.normal(0, 1.0, 9))
    for f in (1, 2, 3):
        by_feature[f] = _vec_rec(f, dirn + rng.normal(0, 0.05, 9))
    for f, v in ((4, np.eye(9)[1]), (5, np.eye(9)[4]), (6, -np.eye(9)[7])):
        by_feature[f] = _vec_rec(f, v * 3.0)
    dev_centroid = (dirn / np.linalg.norm(dirn)).tolist()
    mem = lambda fs: [{"model": "M", "layer": "L.0", "feature": f} for f in fs]  # noqa: E731
    claims = [_atlas_claim(0, mem([1, 2, 3]), dev_centroid),
              _atlas_claim(1, mem([4, 5, 6]), dev_centroid)]
    cfg = SimpleNamespace(confirm=SimpleNamespace(alpha=0.05, atlas_n_null=400),
                          run=SimpleNamespace(seed=0))
    batteries = {"M/L.0": {"withheld": False, "reason": "", "by_feature": by_feature}}
    out = confirm_mod._confirm_atlas(cfg, {"hypotheses": claims}, {}, batteries)
    v = {t["concept"]: t["verdict"] for t in out["tests"]}
    assert v == {0: "confirmed", 1: "not confirmed"}, v
    t0 = next(t for t in out["tests"] if t["concept"] == 0)
    t1 = next(t for t in out["tests"] if t["concept"] == 1)
    assert t0["private_centroid_cosine"] > 0.99 and t0["private_pair_fraction"] == 1.0
    assert t1["private_pair_fraction"] < 0.5


def test_atlas_withheld_member_target_makes_the_concept_not_testable():
    by_feature = {f: _vec_rec(f, np.eye(9)[0] + 0.01 * f) for f in range(1, 10)}
    mem = [{"model": "M", "layer": "L.0", "feature": 1},
           {"model": "M", "layer": "L.0", "feature": 2},
           {"model": "N", "layer": "L.1", "feature": 3}]
    cfg = SimpleNamespace(confirm=SimpleNamespace(alpha=0.05, atlas_n_null=50),
                          run=SimpleNamespace(seed=0))
    batteries = {"M/L.0": {"withheld": False, "reason": "", "by_feature": by_feature},
                 "N/L.1": {"withheld": True, "reason": "planted", "by_feature": {}}}
    out = confirm_mod._confirm_atlas(cfg, {"hypotheses": [_atlas_claim(
        0, mem, np.eye(9)[0].tolist())]}, {}, batteries)
    assert out["tests"][0]["verdict"] == "not testable"
    assert out["n_confirmed"] == 0 and out["n_not_testable"] == 1


def _struct_claims(rate_dev=0.6):
    a = {"id": "concept_structure::no_concept_in_all_models", "stage": "concept_structure",
         "statistic": "atlas_models_per_concept", "p_valued": False,
         "dev_models": ["M", "N"], "dev_max_models_per_concept": 1,
         "min_cosine": 0.9, "min_members": 2,
         "pool": [{"model": "M", "layer": "L.0", "feature": f} for f in (1, 2, 3)]
                 + [{"model": "N", "layer": "L.1", "feature": f} for f in (4, 5, 6)]}
    b = {"id": "concept_structure::majority_prominent_features_causally_null",
         "stage": "concept_structure", "statistic": "causally_null_rate", "p_valued": True,
         "dev_rate": rate_dev, "candidates": {"M/L.0": list(range(20, 60)),
                                              "N/L.1": list(range(20, 60))}}
    return a, b


def test_structure_claims_planted_and_decoy():
    """(a) The private atlas keeps one concept per model (no all-model
    concept): confirmed; the decoy variant makes a mixed concept spanning
    both models: not confirmed. (b) 90% causally null on private: confirmed;
    the decoy at 52% with the same n has a lower bound below 0.5: not."""
    cfg = SimpleNamespace(confirm=SimpleNamespace(alpha=0.05, structure_n_boot=400),
                          run=SimpleNamespace(seed=0))

    def _batt(null_share, mixed):
        by_m, by_n = {}, {}
        for f in (1, 2, 3):
            by_m[f] = _vec_rec(f, np.eye(9)[0] * 3 + (0.01 * f if not mixed else 0))
        for f in (4, 5, 6):
            by_n[f] = _vec_rec(f, (np.eye(9)[3] if not mixed else np.eye(9)[0]) * 3)
        n_null = int(round(40 * null_share))
        for tgt, d in (("M", by_m), ("N", by_n)):
            for i, f in enumerate(range(20, 60)):
                d[f] = _vec_rec(f, np.eye(9)[i % 9], clearing=0 if i < n_null else 1)
        return {"M/L.0": {"withheld": False, "reason": "", "by_feature": by_m},
                "N/L.1": {"withheld": False, "reason": "", "by_feature": by_n}}

    a, b = _struct_claims()
    real = confirm_mod._confirm_structure(cfg, {"hypotheses": [a, b]}, {}, _batt(0.9, False))
    v = {t["id"].split("::")[1]: t["verdict"] for t in real["tests"]}
    assert v == {"no_concept_in_all_models": "confirmed",
                 "majority_prominent_features_causally_null": "confirmed"}, v
    decoy = confirm_mod._confirm_structure(cfg, {"hypotheses": [a, b]}, {}, _batt(0.52, True))
    v = {t["id"].split("::")[1]: t["verdict"] for t in decoy["tests"]}
    assert v == {"no_concept_in_all_models": "not confirmed",
                 "majority_prominent_features_causally_null": "not confirmed"}, v
    tb = next(t for t in decoy["tests"] if t["p_valued"])
    assert tb["lower_95_one_sided"] < 0.5 and tb["resampling_unit"].startswith("feature")


def test_structure_a_is_not_testable_when_a_model_has_no_private_rows():
    cfg = SimpleNamespace(confirm=SimpleNamespace(alpha=0.05, structure_n_boot=50),
                          run=SimpleNamespace(seed=0))
    a, _b = _struct_claims()
    batteries = {"M/L.0": {"withheld": False, "reason": "", "by_feature": {
        f: _vec_rec(f, np.eye(9)[0] * 3) for f in (1, 2, 3)}},
        "N/L.1": {"withheld": True, "reason": "planted", "by_feature": {}}}
    out = confirm_mod._confirm_structure(cfg, {"hypotheses": [a]}, {}, batteries)
    t = out["tests"][0]
    assert t["verdict"] == "not testable" and "vacuously" in t["reason"]


# ---------------------------------------------------------------------------
# 6. Power.
# ---------------------------------------------------------------------------

def test_mde_ablation_effect_shrinks_with_rows_and_names_its_failure_states():
    rng = np.random.default_rng(0)
    null = rng.normal(0.30, 0.01, 300)
    few = mde_ablation_effect(0.3 + 0.4 * np.linspace(-1, 1, 4), null, m=8, n_sim=300)
    many = mde_ablation_effect(0.3 + 0.4 * np.linspace(-1, 1, 64), null, m=8, n_sim=300)
    assert few["mde"] is not None and many["mde"] is not None
    # Same per-row spread, so only k differs: the mean of 64 rows is 4x tighter
    # than the mean of 4, and the MDE must reflect that (measured 0.465 vs
    # 0.362). Planted regression: resampling a fixed 4 rows regardless of k
    # erases the gap.
    assert many["mde"] < few["mde"] - 0.05, (few["mde"], many["mde"])
    assert many["mde"] > float(np.quantile(null, 0.99)), "MDE sits above the null's own tail"
    assert mde_ablation_effect(np.ones(8), null[:20], m=8)["reason"] == "unsatisfiable_correction"
    assert mde_ablation_effect(np.ones(2), null)["reason"] == "n_below_minimum"
    assert mde_ablation_effect(np.ones(8), np.full(300, 0.3))["reason"] == "degenerate_null"
    out = mde_ablation_effect(rng.normal(0.3, 0.05, 8).clip(0), null, m=8, null_p95=0.35,
                              n_sim=200)
    assert out["mde_over_null_p95"] == pytest.approx(out["mde"] / 0.35)
    assert 0.0 <= out["achieved_power_at_observed"] <= 1.0


# ---------------------------------------------------------------------------
# 7. L6 and the report, on synthetic confirmations.
# ---------------------------------------------------------------------------

def test_l6_reachable_for_causal_claims_and_not_testable_is_not_a_failure():
    from tsfm_lens.report.derived import concept_verdicts
    profiles = {"concepts": [
        {"concept": 0, "name": "c0", "n_models": 2, "sharing_class": "shared (same effect, same inputs)"},
        {"concept": 1, "name": "c1", "n_models": 2, "sharing_class": "partially shared"},
        {"concept": 2, "name": "c2", "n_models": 1, "sharing_class": "single-model"}]}
    rep = {"status": "tested",
           "causal": {"tests": [
               {"atlas_concept": 0, "verdict": "confirmed"},
               {"atlas_concept": 1, "verdict": "not testable"}]},
           "atlas": {"tests": [{"concept": 1, "verdict": "not testable"},
                               {"concept": 2, "verdict": "not confirmed"}]},
           "agreement": {"tests": []}}
    by = {r["concept"]: r["rungs"][5] for r in concept_verdicts(profiles, None, None, None, rep)}
    assert by[0]["status"] == "reached" and "causal: 1/1" in by[0]["detail"]
    assert by[1]["status"] == "not measured" and "not testable" in by[1]["detail"]
    assert by[2]["status"] == "not reached"


def test_confirm_report_renders_the_four_subtables(tmp_path):
    from tsfm_lens.report.report import _sec_confirm
    run_dir = tmp_path / "run"
    (run_dir / "confirm").mkdir(parents=True)
    base = {"status": "tested", "n_registered": 1, "n_tested": 1, "n_confirmed": 1,
            "n_not_testable": 0}
    save_json(run_dir / "confirm" / "confirmation.json", {
        "n_private_series": 40, "alpha": 0.05, "tests": [], "n_registered": 4,
        "n_replicable": 4, "registry_sha256": "abc",
        "concept_replication": {
            "status": "tested",
            "transfer": {"status": "skipped", "tests": []},
            "causal": dict(base, tests=[{
                "id": "c", "model": "M", "layer": "L.0", "feature": 3, "channel": "level",
                "sign": 1, "dev_effect_over_null_p95": 3.0, "private_effect": 0.9, "p": 0.003,
                "p_method": "adaptive", "p_holm": 0.01, "verdict": "confirmed",
                "status": "tested", "mde": {"mde": 0.2, "mde_over_null_p95": 0.9},
                "non_replication_reading": "confirmed with power"}]),
            "atlas": dict(base, tests=[{
                "id": "a", "concept": 0, "n_members": 3, "n_testable_members": 3,
                "models": ["M"], "private_pair_fraction": 1.0,
                "private_centroid_cosine": 0.99, "p": 0.01, "p_holm": 0.01,
                "verdict": "confirmed", "status": "tested"}]),
            "agreement": dict(base, tests=[{
                "id": "g", "concept": 0, "src_target": "M/L.0", "dst_target": "N/L.1",
                "dst_feature": 1, "dev_verdict": "acts differently",
                "private_verdict": "acts differently", "p": 0.01, "p_holm": 0.01,
                "verdict": "confirmed", "status": "tested"}]),
            "structure": dict(base, tests=[{
                "id": "concept_structure::no_concept_in_all_models", "p_valued": False,
                "private_n_concepts": 2, "private_max_models_per_concept": 1,
                "p": None, "verdict": "confirmed", "status": "tested"}]),
            "ledger": [{"family": "concept_causal", "m": 1, "n_null": 300,
                        "min_attainable_p_holm": 0.0033, "satisfiable": True}]}})
    findings: list = []
    html = _sec_confirm(run_dir, findings, 0)
    for h in ("Causal feature claims", "Atlas concept claims",
              "Shared-input agreement claims", "Structure claims",
              "Causal-concept multiplicity ledger"):
        assert h in html, h
    assert html.count("What does this mean?") >= 4
    assert sum(1 for f in findings if "CONFIRM" in f.text) >= 4


# ---------------------------------------------------------------------------
# 8. End to end: register -> synthetic private split -> confirm -> report.
# ---------------------------------------------------------------------------

def test_end_to_end_register_confirm_writes_new_keys_and_renders(dev_run):
    cfg = _fresh_run(dev_run)
    run_dir = cfg.run_dir()
    _stage_dev_artifacts(run_dir, null_majority=True)
    cfg.report.enabled = True
    run_pipeline(cfg, stages=["register", "confirm"])
    conf = load_json(run_dir / "confirm" / "confirmation.json")
    cr = conf["concept_replication"]
    for key in ("causal", "atlas", "agreement", "structure"):
        assert key in cr, key
        assert cr[key]["status"] == "tested"
    assert cr["status"] == "tested"
    assert {"transfer", "knob", "ledger"} <= set(cr)
    reg = load_json(run_dir / "hypotheses.json")
    assert conf["n_registered"] == len(reg["hypotheses"])
    assert conf["registry_sha256"] == _sha(run_dir / "hypotheses.json")
    causal = cr["causal"]
    assert causal["n_registered"] == len([h for h in reg["hypotheses"]
                                          if h["stage"] == "concept_causal"])
    tested = [t for t in causal["tests"] if t["status"] == "tested"]
    assert tested, "the mock targets are reachable on private data"
    for t in tested:
        assert {"p", "p_holm", "p_method", "mde", "sign_matches_dev"} <= set(t)
        assert t["verdict"] in ("confirmed", "not confirmed")
    assert {r["family"] for r in cr["ledger"]} >= {"concept_causal", "concept_atlas",
                                                   "shared_input_agreement",
                                                   "concept_structure", "concept_knob"}
    assert all(r.get("satisfiable") for r in cr["ledger"] if r["family"] in CONCEPT_CLAIM_STAGES)
    ag = cr["agreement"]
    assert ag["n_registered"] == 2
    assert {t["dev_verdict"] for t in ag["tests"]} == {"same causal effect", "acts differently"}
    assert all(t["verdict"] in ("confirmed", "not confirmed", "not testable")
               for blk in (cr["atlas"], cr["structure"], ag) for t in blk["tests"])
    st = {t["id"].split("::")[1]: t for t in cr["structure"]["tests"]}
    assert "majority_prominent_features_causally_null" in st
    from tsfm_lens.report.report import _sec_confirm
    html = _sec_confirm(run_dir, [], 0)
    for h in ("Causal feature claims", "Atlas concept claims", "Shared-input agreement claims",
              "Structure claims"):
        assert h in html, h


def test_end_to_end_default_confirm_has_no_new_keys(dev_run):
    """With `register_concept_claims` off, the same run's confirmation gains no
    key and the registry has no K2 entry (byte-identical default)."""
    cfg = _fresh_run(dev_run)
    cfg.confirm.register_concept_claims = False
    run_pipeline(cfg, stages=["register", "confirm"])
    conf = load_json(cfg.run_dir() / "confirm" / "confirmation.json")
    assert not ({"causal", "atlas", "agreement", "structure"} & set(conf["concept_replication"]))
    reg = load_json(cfg.run_dir() / "hypotheses.json")
    assert set(reg) == {"hypotheses", "n_replicable", "concept_transfer_candidates",
                        "concept_knob_candidates"}
