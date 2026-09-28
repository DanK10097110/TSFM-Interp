"""ROADMAP.md sec 37.10 P7: registering `concept_transfer` claims from dev
concept-atlas artifacts and replicating them once on a fresh private epoch.

Mirrors `tests/test_hypotheses.py`'s fixture style (a small `build_config`
run directory with hand-written JSON artifacts standing in for a real
`sae`/`concepts` stage run) and `tests/test_confirm_provenance.py`'s
`_sec_confirm` end-to-end rendering pattern. The heavy per-target capture/
encode step (`_capture_private_window_acts`/`_encode_series_pooled`) is
monkeypatched to a deterministic lookup rather than run through a real
model + SAE checkpoint -- this module tests the STATISTICAL and
bookkeeping machinery (ranking, the private-strata transfer test, Holm,
the two-family ledger, the freshness guard, the report block), the same
scope `tests/test_shared_input_agreement.py` carves out for its own
per-target `TargetContext`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config
from tsfm_lens.analysis import confirm as confirm_mod
from tsfm_lens.analysis.hypotheses import (build_registry, check_registry_freshness,
                                           run_register)
from tsfm_lens.config import config_from_dict
from tsfm_lens.extraction.store import ActivationStore, save_meta
from tsfm_lens.sae.transfer import matched_draws, series_strata, transfer_one_fixed_feature
from tsfm_lens.utils import load_json, save_json


def _atlas_fixture(run_dir: Path, tests: list, stable_concepts: set) -> None:
    """Write a minimal `sae/concept_atlas.json` + `sae/atlas_transfer.json`
    + `sae/concept_stability.json` triple. `tests` is a list of dicts with
    keys `concept`, `src_target`, `src_model`, `dst_target`, `dst_model`,
    `feature`, `auc`, `null_p95`, `rev_auc`, `rev_null_p95`,
    `reciprocal_fdr`; every distinct (concept, src_target) gets one atlas
    row per feature `0` (feature identity does not matter to the code under
    test -- only that `sae/concept_atlas.json["rows"]` has an entry so
    `src_features` resolves to something non-empty).
    """
    sae_dir = run_dir / "sae"
    sae_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for t in tests:
        key = (t["concept"], t["src_target"])
        if key in seen:
            continue
        seen.add(key)
        model, layer = t["src_target"].split("/", 1)
        rows.append({"model": model, "layer": layer, "feature": 0, "concept": t["concept"]})
    concepts = sorted({t["concept"] for t in tests})
    save_json(sae_dir / "concept_atlas.json", {"rows": rows,
                                               "concepts": [{"concept": c} for c in concepts]})
    save_json(sae_dir / "atlas_transfer.json", {"k_top_series": 8, "tests": tests})
    # Real schema (`sae/stability.py::run_concept_stability`): the flag is
    # nested under `stability.stable`, not top-level -- a real smoke-config
    # end-to-end run (ROADMAP.md sec 37.10 P7) caught `_concept_transfer_
    # candidates` reading the wrong (flat) key against the real artifact, so
    # this fixture is written in the shape production code actually emits,
    # not the shape the bug expected.
    save_json(sae_dir / "concept_stability.json", {
        "concepts": [{"concept": c, "stability": {"stable": c in stable_concepts}}
                    for c in concepts]})


def _base_test_row(concept=0, src_target="modelA/layerA", src_model="modelA",
                   dst_target="modelB/layerB", dst_model="modelB", feature=5,
                   auc=0.9, null_p95=0.6, rev_auc=0.85, rev_null_p95=0.55,
                   reciprocal_fdr=True) -> dict:
    return {"concept": concept, "src_target": src_target, "src_model": src_model,
           "dst_target": dst_target, "dst_model": dst_model, "feature": feature,
           "auc": auc, "null_p95": null_p95, "rev_auc": rev_auc,
           "rev_null_p95": rev_null_p95, "reciprocal_fdr": reciprocal_fdr}


def _cfg(out=None, n_registered=20) -> object:
    cfg = config_from_dict(build_config(out or tempfile.mkdtemp()))
    cfg.concepts.n_registered = n_registered
    return cfg


# ---------------------------------------------------------------------------
# Registration / ranking.
# ---------------------------------------------------------------------------

def test_registration_cap_respected():
    """25 stable, reciprocal-FDR candidates; `n_registered=5` keeps exactly
    the top 5 by (weaker-leg) dev AUC margin, and records the full ranked
    candidate list plus the cut."""
    cfg = _cfg(n_registered=5)
    tests = []
    for i in range(25):
        # margin increases with i -> candidate i=24 is the strongest.
        tests.append(_base_test_row(concept=i, src_target=f"modelA/layer{i}",
                                    dst_target=f"modelB/layer{i}",
                                    auc=0.6 + i * 0.01, null_p95=0.5,
                                    rev_auc=0.6 + i * 0.01, rev_null_p95=0.5))
    _atlas_fixture(cfg.run_dir(), tests, stable_concepts=set(range(25)))
    reg = build_registry(cfg)
    ct = [h for h in reg["hypotheses"] if h["stage"] == "concept_transfer"]
    assert len(ct) == 5
    assert reg["concept_transfer_candidates"]["cut"] == 5
    assert len(reg["concept_transfer_candidates"]["candidates"]) == 25
    # Strongest 5 margins are concepts 20..24.
    assert {h["concept"] for h in ct} == set(range(20, 25))


def test_registration_excludes_non_reciprocal_and_unstable():
    """Two DISTINCT, non-confounded exclusion reasons: concept 1 is
    reciprocal-FDR but its atlas concept is unstable; concept 2 is stable
    but its test did not survive reciprocal FDR. Only concept 0 (both
    conditions met) is registered."""
    cfg = _cfg()
    tests = [
        _base_test_row(concept=0, src_target="modelA/layerA", dst_target="modelB/layerA",
                       reciprocal_fdr=True),
        _base_test_row(concept=1, src_target="modelA/layerB", dst_target="modelB/layerB",
                       reciprocal_fdr=True),  # reciprocal, but unstable -- excluded
        _base_test_row(concept=2, src_target="modelA/layerC", dst_target="modelB/layerC",
                       reciprocal_fdr=False),  # stable, but not reciprocal -- excluded
    ]
    _atlas_fixture(cfg.run_dir(), tests, stable_concepts={0, 2})  # concept 1 unstable
    reg = build_registry(cfg)
    ct = [h for h in reg["hypotheses"] if h["stage"] == "concept_transfer"]
    assert [h["concept"] for h in ct] == [0]


def test_registration_degrades_when_concepts_artifacts_absent():
    """A confirm-only run (no `concepts` stage, e.g. `configs/smoke.yaml`)
    must still register its l0/l1/l2/l3 claims -- concept_transfer degrades
    to an empty list with a stated reason, never a hard failure."""
    cfg = _cfg()
    reg = build_registry(cfg)
    assert [h for h in reg["hypotheses"] if h["stage"] == "concept_transfer"] == []
    assert "not found" in reg["concept_transfer_candidates"]["reason"]


def test_knob_family_registered_empty_with_reason():
    cfg = _cfg()
    reg = build_registry(cfg)
    knob = reg["concept_knob_candidates"]
    assert knob["candidates"] == [] and knob["cut"] == 0
    assert "37.9" in knob["reason"] and "empty" in knob["reason"]
    assert [h for h in reg["hypotheses"] if h["stage"] == "concept_knob"] == []


# ---------------------------------------------------------------------------
# Registration precedes private access (A15's freshness guard).
# ---------------------------------------------------------------------------

def test_registration_precedes_private_access():
    """Mutating `sae/atlas_transfer.json` (the concept_transfer claims'
    registered artifact) after `register` must make `run_confirm` refuse
    BEFORE it ever loads the private corpus."""
    cfg = _cfg()
    (cfg.run_dir() / "l0").mkdir(parents=True)
    save_json(cfg.run_dir() / "l0" / "summary.json", {"strengths": {}, "mase_ratio": {}})
    _atlas_fixture(cfg.run_dir(), [_base_test_row()], stable_concepts={0})
    run_register(cfg)

    # A dev stage ran again after registration (e.g. `concepts` re-run).
    at_path = cfg.run_dir() / "sae" / "atlas_transfer.json"
    doc = load_json(at_path)
    doc["tests"][0]["auc"] = 0.99
    save_json(at_path, doc)

    called = {"private_loaded": False}

    def _poison(*a, **k):
        called["private_loaded"] = True
        raise AssertionError("private corpus must not be loaded when the registry is stale")

    orig = confirm_mod._load_private
    confirm_mod._load_private = _poison
    try:
        try:
            confirm_mod.run_confirm(cfg, hub=None)
            raise AssertionError("expected RuntimeError for a stale registry")
        except RuntimeError as e:
            assert "sae/atlas_transfer.json" in str(e)
            assert "hash changed" in str(e)
    finally:
        confirm_mod._load_private = orig
    assert called["private_loaded"] is False


# ---------------------------------------------------------------------------
# Private replication -- monkeypatched capture/encode, real statistics.
# ---------------------------------------------------------------------------

def _patch_capture(pooled_lookup: dict):
    """Replace the per-target capture+encode pair with a deterministic
    lookup keyed by "model/layer" -- the tests below exercise the
    STATISTICAL machinery (`matched_draws`/`transfer_one`/Holm/the ledger),
    not a real forward pass through a mock adapter or a trained SAE
    checkpoint (`tests/test_shared_input_agreement.py` already covers that
    boundary for the sibling P5b module this one reuses `transfer.py`
    alongside).
    """
    def _fake_capture(cfg, hub, private, model, layer):
        return None  # unused by the fake encode below

    def _fake_encode(run_dir, model, layer, acts, device):
        return pooled_lookup[f"{model}/{layer}"]

    orig_capture = confirm_mod._capture_private_window_acts
    orig_encode = confirm_mod._encode_series_pooled
    confirm_mod._capture_private_window_acts = _fake_capture
    confirm_mod._encode_series_pooled = _fake_encode
    return orig_capture, orig_encode


def _unpatch_capture(orig):
    confirm_mod._capture_private_window_acts, confirm_mod._encode_series_pooled = orig


def _private_meta(n: int, archetype: str = "onlyA") -> SimpleNamespace:
    meta = pd.DataFrame({"archetype": [archetype] * n, "family": ["fam"] * n})
    return SimpleNamespace(meta=meta)


def _registry_with_claims(claims: list) -> dict:
    """`claims`: list of dicts with `id`, `concept`, `src_target`,
    `dst_target`, `dst_model`, `src_features`, `dst_feature`,
    `k_top_series`, `dev_auc`, `dev_auc_margin` -- exactly the fields
    `_replicate_registered_concepts` reads off a registered entry."""
    hyps = []
    for c in claims:
        hyps.append({"stage": "concept_transfer", "family": "concept_transfer", **c})
    return {"hypotheses": hyps, "n_replicable": len(hyps)}


def test_planted_real_transfer_confirms_and_spurious_does_not():
    """A real transfer: dst's feature ranks the SAME 40 series the same way
    src's concept does (near-perfect AUC both legs) -> confirmed. A
    spurious one: dst's feature is independent noise -> not confirmed."""
    rng = np.random.default_rng(0)
    n = 40
    src_pooled = np.zeros((n, 1))
    src_pooled[:, 0] = np.arange(n, dtype=float)  # monotone -> top-8 = 32..39

    dst_real = np.zeros((n, 6))
    dst_real[:, 5] = np.arange(n, dtype=float) + rng.normal(0, 0.01, n)  # same order

    dst_spurious = np.zeros((n, 6))
    dst_spurious[:, 5] = rng.permutation(n).astype(float)  # unrelated order

    cfg = _cfg()
    cfg.confirm.concept_transfer_n_null = 300
    private = _private_meta(n)
    claims = [
        {"id": "concept_transfer::modelA/layerA::0::modelB", "concept": 0,
         "src_target": "modelA/layerA", "dst_target": "modelB/layerB", "dst_model": "modelB",
         "src_features": [0], "dst_feature": 5, "k_top_series": 8,
         "dev_auc": 0.95, "dev_auc_margin": 0.3},
        {"id": "concept_transfer::modelA/layerA::0::modelC", "concept": 0,
         "src_target": "modelA/layerA", "dst_target": "modelC/layerC", "dst_model": "modelC",
         "src_features": [0], "dst_feature": 5, "k_top_series": 8,
         "dev_auc": 0.95, "dev_auc_margin": 0.3},
    ]
    registry = _registry_with_claims(claims)
    orig = _patch_capture({"modelA/layerA": src_pooled, "modelB/layerB": dst_real,
                          "modelC/layerC": dst_spurious})
    try:
        out = confirm_mod._replicate_registered_concepts(cfg, hub=None, private=private,
                                                          registry=registry)
    finally:
        _unpatch_capture(orig)

    tests = {t["id"]: t for t in out["transfer"]["tests"]}
    real = tests["concept_transfer::modelA/layerA::0::modelB"]
    spurious = tests["concept_transfer::modelA/layerA::0::modelC"]
    assert real["status"] == "tested" and spurious["status"] == "tested"
    assert real["verdict"] == "confirmed"
    assert spurious["verdict"] == "not confirmed"
    assert real["private_auc"] > spurious["private_auc"]
    assert out["transfer"]["n_confirmed"] == 1
    assert out["transfer"]["n_tested"] == 2


def test_holm_family_counted_separately_in_ledger():
    """`concept_transfer` and `concept_knob` are separate ledger rows --
    `concept_knob` is always empty (P7 does not register or replicate it),
    but its `m`/`satisfiable` must not be conflated with the transfer
    family's."""
    rng = np.random.default_rng(1)
    n = 30
    src_pooled = np.zeros((n, 1)); src_pooled[:, 0] = np.arange(n, dtype=float)
    dst_pooled = np.zeros((n, 3)); dst_pooled[:, 1] = np.arange(n, dtype=float)
    cfg = _cfg()
    cfg.confirm.concept_transfer_n_null = 300
    private = _private_meta(n)
    claims = [{"id": "concept_transfer::A/l::0::B", "concept": 0, "src_target": "A/l",
              "dst_target": "B/l", "dst_model": "B", "src_features": [0], "dst_feature": 1,
              "k_top_series": 6, "dev_auc": 0.9, "dev_auc_margin": 0.2}]
    registry = _registry_with_claims(claims)
    orig = _patch_capture({"A/l": src_pooled, "B/l": dst_pooled})
    try:
        out = confirm_mod._replicate_registered_concepts(cfg, hub=None, private=private,
                                                          registry=registry)
    finally:
        _unpatch_capture(orig)
    by_family = {row["family"]: row for row in out["ledger"]}
    assert set(by_family) == {"concept_transfer", "concept_knob"}
    assert by_family["concept_transfer"]["m"] == 1
    assert by_family["concept_knob"]["m"] == 0
    assert by_family["concept_knob"]["satisfiable"] is None
    assert by_family["concept_transfer"]["satisfiable"] is True
    assert by_family["concept_transfer"]["n_null"] == 300
    assert by_family["concept_knob"]["n_null"] is None


def test_empty_knob_family_skips_gracefully():
    cfg = _cfg()
    private = _private_meta(10)
    registry = _registry_with_claims([])  # no concept_transfer claims either
    out = confirm_mod._replicate_registered_concepts(cfg, hub=None, private=private,
                                                      registry=registry)
    assert out["status"] == "skipped"
    assert out["transfer"]["status"] == "skipped"
    assert out["knob"]["status"] == "empty"
    assert out["knob"]["n_registered"] == 0
    ledger = {row["family"]: row for row in out["ledger"]}
    assert ledger["concept_knob"]["m"] == 0


def test_private_strata_used():
    """Load-bearing: if the strata array fed to `matched_draws` does not
    match the corpus the pooled features were actually computed on (here:
    a `private.meta` with far fewer rows than the pooled arrays -- as if
    dev's meta had been used by mistake), the mismatch must be detected,
    never silently absorbed into a plausible-looking result."""
    n = 40
    src_pooled = np.zeros((n, 1)); src_pooled[:, 0] = np.arange(n, dtype=float)
    dst_pooled = np.zeros((n, 3)); dst_pooled[:, 1] = np.arange(n, dtype=float)
    cfg = _cfg()
    cfg.confirm.concept_transfer_n_null = 200
    claims = [{"id": "concept_transfer::A/l::0::B", "concept": 0, "src_target": "A/l",
              "dst_target": "B/l", "dst_model": "B", "src_features": [0], "dst_feature": 1,
              "k_top_series": 8, "dev_auc": 0.9, "dev_auc_margin": 0.2}]
    registry = _registry_with_claims(claims)
    orig = _patch_capture({"A/l": src_pooled, "B/l": dst_pooled})
    try:
        # Correct: private.meta has as many rows as the pooled arrays (40).
        out_ok = confirm_mod._replicate_registered_concepts(
            cfg, hub=None, private=_private_meta(n), registry=registry)
        assert out_ok["transfer"]["tests"][0]["status"] == "tested"

        # Wrong: feed a 5-row meta (as if dev's smaller corpus's strata had
        # been used instead of private's own) -- top_series/S picks indices
        # up to 39, so strata[S] must fail rather than quietly succeed.
        raised = False
        try:
            confirm_mod._replicate_registered_concepts(
                cfg, hub=None, private=_private_meta(5), registry=registry)
        except (IndexError, ValueError):
            raised = True
        assert raised, ("feeding a strata source that does not match the private corpus "
                        "the pooled features came from must raise, not silently succeed")
    finally:
        _unpatch_capture(orig)


# ---------------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------------

def test_report_renders_concept_replication_block(tmp_path):
    from tsfm_lens.report.report import _sec_confirm

    run_dir = tmp_path / "run"
    (run_dir / "confirm").mkdir(parents=True)
    save_json(run_dir / "confirm" / "confirmation.json", {
        "n_private_series": 40, "alpha": 0.05, "tests": [],
        "n_registered": 0, "n_replicable": 0, "registry_sha256": "abc",
        "concept_replication": {
            "status": "tested",
            "transfer": {"status": "tested", "n_registered": 2, "n_tested": 2,
                        "n_confirmed": 1, "n_null": 300,
                        "tests": [
                            {"concept": 0, "src_target": "modelA/layerA",
                             "dst_target": "modelB/layerB", "dst_model": "modelB",
                             "dev_auc": 0.95, "dev_auc_margin": 0.3,
                             "private_auc": 0.97, "p_holm": 0.01,
                             "status": "tested", "verdict": "confirmed"},
                            {"concept": 1, "src_target": "modelA/layerC",
                             "dst_target": "modelC/layerC", "dst_model": "modelC",
                             "dev_auc": 0.9, "dev_auc_margin": 0.2,
                             "private_auc": 0.5, "p_holm": 0.8,
                             "status": "tested", "verdict": "not confirmed"},
                        ]},
            "knob": {"status": "empty", "n_registered": 0, "tests": [],
                    "reason": "no dev response survived BH (sec 37.9)"},
            "ledger": [
                {"family": "concept_transfer", "m": 2, "n_null": 300,
                 "min_attainable_p_holm": 0.0066, "satisfiable": True},
                {"family": "concept_knob", "m": 0, "n_null": None,
                 "min_attainable_p_holm": None, "satisfiable": None},
            ],
        },
    })
    html = _sec_confirm(run_dir, [], 0)
    assert "Concept transfer replication" in html
    assert "1 of 2 tested confirmed" in html
    assert "confirmed" in html and "not confirmed" in html
    assert "Concept multiplicity ledger" in html
    assert "concept_knob" in html


def test_report_renders_not_measured_when_key_missing(tmp_path):
    """A `confirmation.json` from before this item existed (no
    `concept_replication` key at all) must render an honest "not measured"
    line, not a KeyError."""
    from tsfm_lens.report.report import _sec_confirm

    run_dir = tmp_path / "run"
    (run_dir / "confirm").mkdir(parents=True)
    save_json(run_dir / "confirm" / "confirmation.json", {
        "n_private_series": 40, "alpha": 0.05, "tests": [],
        "n_registered": 0, "n_replicable": 0, "registry_sha256": "abc",
    })
    html = _sec_confirm(run_dir, [], 0)
    assert "Concept replication: not measured" in html


def test_l6_rung_filled_from_concept_replication():
    from tsfm_lens.report.derived import concept_verdicts

    profiles = {"concepts": [
        {"concept": 0, "name": "c0", "n_models": 2,
         "sharing_class": "shared (same effect, same inputs)"},
        {"concept": 1, "name": "c1", "n_models": 1, "sharing_class": "single-model"},
    ]}
    concept_replication = {
        "status": "tested",
        "transfer": {"tests": [
            {"concept": 0, "verdict": "confirmed"},
        ]},
    }
    rows = concept_verdicts(profiles, None, None, None, concept_replication)
    by_concept = {r["concept"]: r for r in rows}
    l6_c0 = by_concept[0]["rungs"][5]
    l6_c1 = by_concept[1]["rungs"][5]
    assert l6_c0["status"] == "reached"
    assert "confirmed on private data" in l6_c0["detail"]
    assert l6_c1["status"] == "not measured"
    assert "no registered concept_transfer claim" in l6_c1["detail"]


def test_l6_rung_not_measured_without_artifact():
    from tsfm_lens.report.derived import concept_verdicts

    profiles = {"concepts": [{"concept": 0, "name": "c0", "n_models": 1,
                              "sharing_class": "single-model"}]}
    rows = concept_verdicts(profiles, None, None, None, None)
    assert rows[0]["rungs"][5]["status"] == "not measured"


if __name__ == "__main__":
    test_registration_cap_respected()
    test_registration_excludes_non_reciprocal_and_unstable()
    test_registration_degrades_when_concepts_artifacts_absent()
    test_knob_family_registered_empty_with_reason()
    test_registration_precedes_private_access()
    test_planted_real_transfer_confirms_and_spurious_does_not()
    test_holm_family_counted_separately_in_ledger()
    test_empty_knob_family_skips_gracefully()
    test_private_strata_used()
    print("concept confirm tests passed (report tests need tmp_path -- run via pytest)")


def test_same_concept_to_several_layers_of_one_model_gets_distinct_ids():
    """One source concept transferring to two LAYERS of the same destination
    model is two claims. The reference run registered 20 claims under 9
    distinct ids when the id named only the destination model, and `holm()`
    (keyed by id) silently ran over 9 p-values instead of 20."""
    cfg = _cfg()
    tests = [_base_test_row(concept=0, dst_target="modelB/layer.1", auc=0.95, rev_auc=0.9),
             _base_test_row(concept=0, dst_target="modelB/layer.2", auc=0.9, rev_auc=0.85)]
    _atlas_fixture(cfg.run_dir(), tests, stable_concepts={0})
    reg = build_registry(cfg)
    ct = [h for h in reg["hypotheses"] if h["stage"] == "concept_transfer"]
    assert len(ct) == 2
    assert len({h["id"] for h in ct}) == 2


def test_replication_refuses_duplicate_claim_ids():
    """A registry whose claims share an id (one built before the id named
    the destination target) must fail loudly, not Holm-correct a shrunken
    family."""
    n = 30
    src_pooled = np.zeros((n, 1)); src_pooled[:, 0] = np.arange(n, dtype=float)
    dst_pooled = np.zeros((n, 3)); dst_pooled[:, 1] = np.arange(n, dtype=float)
    cfg = _cfg()
    cfg.confirm.concept_transfer_n_null = 300
    claim = {"id": "concept_transfer::A/l::0::B", "concept": 0, "src_target": "A/l",
             "dst_target": "B/l", "dst_model": "B", "src_features": [0], "dst_feature": 1,
             "k_top_series": 6, "dev_auc": 0.9, "dev_auc_margin": 0.2}
    registry = _registry_with_claims([claim, {**claim, "dst_target": "B/m"}])
    orig = _patch_capture({"A/l": src_pooled, "B/l": dst_pooled, "B/m": dst_pooled})
    try:
        raised = False
        try:
            confirm_mod._replicate_registered_concepts(cfg, hub=None, private=_private_meta(n),
                                                       registry=registry)
        except ValueError as exc:
            raised = "duplicate ids" in str(exc)
    finally:
        _unpatch_capture(orig)
    assert raised


# ---------------------------------------------------------------------------
# ROADMAP.md sec 37.10 P7b -- concepts.transfer_claim_mode: search | frozen.
# ---------------------------------------------------------------------------

_SEARCH_ENTRY_KEYS = {
    "id", "stage", "family", "concept", "src_target", "src_model", "src_features",
    "dst_target", "dst_model", "dst_feature", "dev_auc", "dev_null_p95",
    "dev_rev_auc", "dev_rev_null_p95", "dev_auc_margin", "k_top_series",
    "artifact", "artifact_sha256", "statement", "replicable",
}


def test_search_mode_unchanged():
    """`concepts.transfer_claim_mode` defaults to `"search"`, and it must
    reproduce P7's own entry shape byte-for-byte, whether left at the
    default or set explicitly: none of P7b's new keys (`mode`, `dev_p`,
    `dev_rev_p`) leak into a search-mode registration entry or a search-mode
    replication test row.

    Plant: routing the default mode through `_concept_transfer_frozen_
    entries` (an inverted `if mode == "frozen":` in `hypotheses.py`) adds
    `mode`/`dev_p`/`dev_rev_p` to the registered entry and fails the
    key-set assertion below (also raises, since frozen registration needs a
    real store this fixture never builds -- either way, FAILS).
    """
    cfg = _cfg()
    assert cfg.concepts.transfer_claim_mode == "search"
    tests = [_base_test_row(concept=0, auc=0.9, rev_auc=0.85)]
    _atlas_fixture(cfg.run_dir(), tests, stable_concepts={0})
    reg_default = build_registry(cfg)
    cfg.concepts.transfer_claim_mode = "search"
    reg_explicit = build_registry(cfg)
    for reg in (reg_default, reg_explicit):
        ct = [h for h in reg["hypotheses"] if h["stage"] == "concept_transfer"]
        assert len(ct) == 1
        assert set(ct[0].keys()) == _SEARCH_ENTRY_KEYS
        assert "mode" not in ct[0]

    n = 40
    src_pooled = np.zeros((n, 1)); src_pooled[:, 0] = np.arange(n, dtype=float)
    dst_pooled = np.zeros((n, 6)); dst_pooled[:, 5] = np.arange(n, dtype=float)
    claims = [{"id": "concept_transfer::A/l::0::B", "concept": 0, "src_target": "A/l",
              "dst_target": "B/l", "dst_model": "B", "src_features": [0], "dst_feature": 5,
              "k_top_series": 8, "dev_auc": 0.9, "dev_auc_margin": 0.2}]
    registry = _registry_with_claims(claims)
    cfg2 = _cfg()
    cfg2.confirm.concept_transfer_n_null = 300
    orig = _patch_capture({"A/l": src_pooled, "B/l": dst_pooled})
    try:
        out = confirm_mod._replicate_registered_concepts(cfg2, hub=None,
                                                          private=_private_meta(n),
                                                          registry=registry)
    finally:
        _unpatch_capture(orig)
    t = out["transfer"]["tests"][0]
    assert "mode" not in t and "dev_p" not in t
    assert t["verdict"] == "confirmed"


def test_frozen_mode_uses_single_feature_null():
    """Decoy: the destination has TWO candidate features. Feature 2 matches
    the source's own order almost perfectly; feature 5 (dev's frozen choice
    here) is independent noise. Search mode's argmax must find feature 2
    and confirm; frozen mode is locked onto feature 5 and must NOT confirm.

    Plant: frozen mode falling back to argmax (i.e. calling `transfer_one`
    instead of `transfer_one_fixed_feature` for a `mode == "frozen"` claim)
    makes it pick feature 2 too, and `frozen_t["verdict"] == "not confirmed"`
    fails.
    """
    rng = np.random.default_rng(3)
    n = 40
    src_pooled = np.zeros((n, 1)); src_pooled[:, 0] = np.arange(n, dtype=float)
    dst = np.zeros((n, 6))
    dst[:, 2] = np.arange(n, dtype=float) + rng.normal(0, 0.01, n)   # decoy: near-perfect match
    dst[:, 5] = rng.permutation(n).astype(float)                    # frozen feature: noise

    cfg = _cfg()
    cfg.confirm.concept_transfer_n_null = 300
    private = _private_meta(n)

    search_claim = {"id": "concept_transfer::A/l::0::B", "concept": 0, "src_target": "A/l",
                    "dst_target": "B/l", "dst_model": "B", "src_features": [0], "dst_feature": 5,
                    "k_top_series": 8, "dev_auc": 0.95, "dev_auc_margin": 0.3}
    frozen_claim = {**search_claim, "id": "concept_transfer_frozen::A/l::0::B::f5",
                   "mode": "frozen"}

    orig = _patch_capture({"A/l": src_pooled, "B/l": dst})
    try:
        out_search = confirm_mod._replicate_registered_concepts(
            cfg, hub=None, private=private, registry=_registry_with_claims([search_claim]))
        out_frozen = confirm_mod._replicate_registered_concepts(
            cfg, hub=None, private=private, registry=_registry_with_claims([frozen_claim]))
    finally:
        _unpatch_capture(orig)

    search_t = out_search["transfer"]["tests"][0]
    frozen_t = out_frozen["transfer"]["tests"][0]
    assert search_t["verdict"] == "confirmed", "search must find the decoy feature and confirm"
    assert search_t["private_feature"] == 2
    assert frozen_t["mode"] == "frozen"
    assert frozen_t["verdict"] == "not confirmed", "frozen is locked to the noisy feature 5"
    assert frozen_t["private_feature"] == 5


def test_frozen_null_is_not_max_over_features():
    """`transfer_one_fixed_feature`'s forward null is feature 0's OWN AUC
    distribution over the matched draws -- never a max over the whole
    destination dictionary. A max over 25 independently-noisy features
    stochastically dominates any single one of them, so using it here would
    inflate `null_p95` well above the single-feature value.

    Plant: replacing the null-column selection with `auc_from_ranks(
    dst_ranks, fwd_draws).max(axis=1)` (the search-mode null) makes
    `null_p95` jump to the maxed value and the exact-match assertion fails.
    """
    from tsfm_lens.sae.transfer import auc_from_ranks

    rng = np.random.default_rng(7)
    n, n_feat = 60, 25
    strata = series_strata(pd.DataFrame({"archetype": ["s"] * n, "family": ["fam"] * n}))
    by_stratum = {v: np.flatnonzero(strata == v) for v in np.unique(strata)}
    src_scores = rng.normal(0, 1, n)
    S = np.argsort(-src_scores)[:10]
    dst_ranks = np.column_stack([rankdata(rng.normal(0, 1, n)) for _ in range(n_feat)])
    fwd_seed = 123
    fwd_draws = matched_draws(S, strata, by_stratum, 500, np.random.default_rng(fwd_seed))

    feature = 0
    null_single = auc_from_ranks(dst_ranks[:, [feature]], fwd_draws)[:, 0]
    null_all = auc_from_ranks(dst_ranks, fwd_draws)
    p95_single = float(np.percentile(null_single, 95))
    p95_maxed = float(np.percentile(null_all.max(axis=1), 95))
    assert p95_maxed > p95_single + 0.02, "sanity: max over 25 noisy features must dominate one"

    result = transfer_one_fixed_feature(src_scores, dst_ranks, S, fwd_draws, strata, by_stratum,
                                        feature=feature, k=10, n_draws=500, seed=99,
                                        p_method="exact", fwd_seed=fwd_seed)
    assert result["null_p95"] == pytest.approx(p95_single)
    assert result["null_p95"] != pytest.approx(p95_maxed)


def test_frozen_ids_unique_and_carry_feature():
    """One source concept transferring (FROZEN) to two layers of the same
    destination model must get two DISTINCT ids, each carrying its own
    frozen feature -- the frozen-mode counterpart of `test_same_concept_to_
    several_layers_of_one_model_gets_distinct_ids`'s search-mode regression
    (20 registered claims collapsed to 9 ids when the id named only the
    destination model, sec 37.10 P7's `b53802b`)."""
    cfg = _cfg()
    cfg.concepts.transfer_claim_mode = "frozen"
    run_dir = cfg.run_dir()
    n = 40
    tests = [_base_test_row(concept=0, src_target="modelA/layerA", dst_target="modelB/layerB1",
                            feature=3, auc=0.9, rev_auc=0.85),
             _base_test_row(concept=0, src_target="modelA/layerA", dst_target="modelB/layerB2",
                            feature=4, auc=0.85, rev_auc=0.8)]
    _atlas_fixture(run_dir, tests, stable_concepts={0})

    rng = np.random.default_rng(11)
    store = ActivationStore.create(run_dir / "activations.zarr", n_series=n,
                                   n_windows=4, window=32, context_len=128)
    src_pooled = rng.normal(0, 0.1, size=(n, 1))
    src_pooled[:, 0] = np.arange(n, dtype=float)
    store.init_sae_layer("modelA", "layerA", 1)
    store.write_sae_batch("modelA", "layerA", 0, np.repeat(src_pooled[:, None, :], 4, axis=1))
    for layer, feat in (("layerB1", 3), ("layerB2", 4)):
        dst_pooled = rng.normal(0, 0.1, size=(n, 6))
        dst_pooled[:, feat] = np.arange(n, dtype=float)
        store.init_sae_layer("modelB", layer, 6)
        store.write_sae_batch("modelB", layer, 0, np.repeat(dst_pooled[:, None, :], 4, axis=1))
    save_meta(run_dir, pd.DataFrame({"archetype": ["s"] * n, "family": ["fam"] * n}))

    reg = build_registry(cfg)
    ct = [h for h in reg["hypotheses"] if h["stage"] == "concept_transfer"]
    assert len(ct) == 2
    ids = {h["id"] for h in ct}
    assert len(ids) == 2
    for h in ct:
        assert h["mode"] == "frozen"
        assert f"f{h['dst_feature']}" in h["id"]
    assert {h["dst_feature"] for h in ct} == {3, 4}


def test_report_renders_mode_and_feature_columns(tmp_path):
    """The confirm section's concept-transfer table gets a `mode` column and
    a `feature` column: `dev -> private` for search rows, the frozen claim's
    own id for frozen rows."""
    from tsfm_lens.report.report import _sec_confirm

    run_dir = tmp_path / "run"
    (run_dir / "confirm").mkdir(parents=True)
    save_json(run_dir / "confirm" / "confirmation.json", {
        "n_private_series": 40, "alpha": 0.05, "tests": [],
        "n_registered": 0, "n_replicable": 0, "registry_sha256": "abc",
        "concept_replication": {
            "status": "tested",
            "transfer": {"status": "tested", "n_registered": 2, "n_tested": 2,
                        "n_confirmed": 1, "n_null": 2000,
                        "tests": [
                            {"concept": 0, "src_target": "modelA/layerA",
                             "dst_target": "modelB/layerB", "dst_model": "modelB",
                             "dev_auc": 0.95, "dev_auc_margin": 0.3,
                             "dev_dst_feature": 5, "private_feature": 2,
                             "private_auc": 0.97, "p_holm": 0.01,
                             "status": "tested", "verdict": "confirmed"},
                            {"concept": 1, "src_target": "modelA/layerC",
                             "dst_target": "modelC/layerC", "dst_model": "modelC",
                             "dev_auc": 0.9, "dev_auc_margin": 0.2,
                             "dev_dst_feature": 7, "private_feature": 7,
                             "private_auc": 0.5, "p_holm": 0.8, "mode": "frozen",
                             "id": "concept_transfer_frozen::modelA/layerC::1::modelC/layerC::f7",
                             "status": "tested", "verdict": "not confirmed"},
                        ]},
            "knob": {"status": "empty", "n_registered": 0, "tests": [],
                    "reason": "no dev response survived BH (sec 37.9)"},
            "ledger": [
                {"family": "concept_transfer", "m": 2, "n_null": 2000,
                 "min_attainable_p_holm": 0.001, "satisfiable": True},
                {"family": "concept_knob", "m": 0, "n_null": None,
                 "min_attainable_p_holm": None, "satisfiable": None},
            ],
        },
    })
    html = _sec_confirm(run_dir, [], 0)
    assert "<th>mode</th>" in html
    assert "<th>feature</th>" in html
    assert ">search<" in html and ">frozen<" in html
    assert "5 → 2" in html
    assert "concept_transfer_frozen::modelA/layerC::1::modelC/layerC::f7" in html
