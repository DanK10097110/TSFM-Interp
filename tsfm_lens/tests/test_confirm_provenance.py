"""`confirm` records the corpus it consumed (`ROADMAP.md` §34 item B5).

`confirmation.json` used to name no particular corpus at all -- no digest,
no epoch, no composition -- despite §4.5's epoch mechanism meaning there can
be several private corpora for one benchmark. `BenchmarkData.corpus_digest`
already existed upstream and was simply dropped. These tests pin: the two
checks B5.2 names (fail loudly on a non-private corpus, warn-only on an
epoch mismatch), the one it names NOT to write (a dev-vs-private digest
equality check, which would fire on every correct run), that the new output
fields do not move the stage's config fingerprint, and that every field
degrades to a stated reason rather than a bare `None` when there is no
manifest to read at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.confirm import (
    _check_private_provenance,
    _load_private,
    _private_provenance,
    _read_provenance_manifests,
    run_confirm,
)
from tsfm_lens.config import ModelConfig, PipelineConfig
from tsfm_lens.data import BenchmarkData
from tsfm_lens.manifest import fingerprint_stage, resolve_config_keys
from tsfm_lens.utils import save_json

_NAME_A, _NAME_B = "A", "B"


def _write_jsonl_corpus(path: Path, n: int = 5, length: int = 10) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with open(path / "corpus.jsonl", "w") as fh:
        for i in range(n):
            row = {"values": rng.normal(0, 1, length).tolist(), "sample_id": f"s{i}"}
            fh.write(json.dumps(row) + "\n")


def _write_manifest(path: Path, visibility: str, epoch: int = 0,
                    global_digest: str = "deadbeef") -> None:
    save_json(path / "manifest.json", {
        "visibility": visibility, "epoch": epoch, "n_samples": 5,
        "sample_hashes": [], "global_digest": global_digest, "determinism": {},
        "extra": {},
    })


def _cfg(tmp_path: Path) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.models = [ModelConfig(name=_NAME_A, adapter="mock_patch"),
                 ModelConfig(name=_NAME_B, adapter="mock_patch")]
    cfg.run.out_dir = str(tmp_path)
    cfg.run.name = "run"
    cfg.run.seed = 3
    cfg.data.context_len = 8
    cfg.data.horizon = 2
    cfg.data.source = "smoke"           # dev side: no manifest, kept simple
    cfg.confirm.alpha = 0.05
    cfg.confirm.source = "jsonl"        # private side: `_load_corpus_rows`
    cfg.confirm.require_seal = False    # never reads manifest.json itself --
    #                                     exactly why a pre-fix confirm "runs
    #                                     happily" against a public corpus.
    return cfg


# --- T-B5.1: confirm against a public corpus raises, naming the visibility --

def test_confirm_against_a_public_corpus_raises_naming_the_visibility(tmp_path):
    private_dir = tmp_path / "public_corpus"
    _write_jsonl_corpus(private_dir)
    _write_manifest(private_dir, visibility="public")
    dev_manifest, private_manifest = None, __import__(
        "tsfm_lens.analysis.corpus_card", fromlist=["read_manifest"]
    ).read_manifest(str(private_dir))
    assert private_manifest is not None, "the manifest we just wrote must actually be read"
    with pytest.raises(RuntimeError, match="public"):
        _check_private_provenance(private_manifest, dev_manifest)


def test_confirm_against_a_public_corpus_raises_end_to_end(tmp_path):
    """The literal load-bearing negative: point `confirm` (not just the
    helper) at the public split. The check fires before `_private_behavioral`
    is ever reached, so this needs no working model hub at all."""
    cfg = _cfg(tmp_path)
    cfg.confirm.path = str(tmp_path / "public_corpus")
    _write_jsonl_corpus(Path(cfg.confirm.path))
    _write_manifest(Path(cfg.confirm.path), visibility="public")
    save_json(cfg.run_dir() / "hypotheses.json", {"hypotheses": []})

    with pytest.raises(RuntimeError, match="visibility"):
        run_confirm(cfg, hub=None)


def test_the_load_bearing_negative_actually_fails_against_pre_fix_code(tmp_path):
    """Reproduce the PRE-FIX condition directly: `_load_private` alone (the
    only corpus-loading step that existed before this item) never reads
    manifest.json for a `jsonl`-source corpus at all -- `_load_corpus_rows`'s
    jsonl branch parses `corpus.jsonl` only. So pointing the pre-fix pipeline
    at a public corpus loaded it and ran happily, which is the exact
    "confirm current code, which runs happily" half of the load-bearing
    negative the loop's own binding rules require."""
    cfg = _cfg(tmp_path)
    cfg.confirm.path = str(tmp_path / "public_corpus")
    _write_jsonl_corpus(Path(cfg.confirm.path))
    _write_manifest(Path(cfg.confirm.path), visibility="public")

    # No exception: `_load_private` (pre-fix behavior, still the only thing
    # that ran before this item's `_check_private_provenance` call was added
    # to `run_confirm`) has no visibility awareness whatsoever.
    private = _load_private(cfg)
    assert private.n == 5


# --- T-B5.2: dev/private digests differing must NOT raise (sec 11.35) ------

def test_differing_dev_and_private_digests_does_not_raise(tmp_path):
    """The false-refusal direction B5.2 explicitly bans: two different,
    correctly-sealed splits legitimately have different global digests --
    they are DIFFERENT corpora by construction (dev vs. private). An
    equality check here would fire on every single correct run."""
    dev_manifest = {"visibility": "public", "epoch": 0, "global_digest": "dev-digest-aaa"}
    private_manifest = {"visibility": "private", "epoch": 0, "global_digest": "private-digest-bbb"}
    assert dev_manifest["global_digest"] != private_manifest["global_digest"]
    _check_private_provenance(private_manifest, dev_manifest)  # must not raise


def test_check_private_provenance_never_compares_digests():
    """Source-level pin: the function this item added must contain no digest
    equality comparison at all -- not merely one that happens not to fire on
    the fixture above."""
    import inspect
    from tsfm_lens.analysis import confirm as confirm_mod
    src = inspect.getsource(confirm_mod._check_private_provenance)
    assert "global_digest" not in src, (
        "digest comparison does not belong in the provenance CHECK -- dev and "
        "private digests must differ on every correct run (sec 11.35)")


def test_epoch_mismatch_warns_but_does_not_raise(tmp_path, caplog):
    import logging
    dev_manifest = {"visibility": "public", "epoch": 0}
    private_manifest = {"visibility": "private", "epoch": 1}
    with caplog.at_level(logging.WARNING):
        _check_private_provenance(private_manifest, dev_manifest)  # must not raise
    assert any("epoch" in r.message for r in caplog.records)


def test_matching_epoch_does_not_warn(caplog):
    import logging
    dev_manifest = {"visibility": "public", "epoch": 5}
    private_manifest = {"visibility": "private", "epoch": 5}
    with caplog.at_level(logging.WARNING):
        _check_private_provenance(private_manifest, dev_manifest)
    assert not any("epoch" in r.message for r in caplog.records)


def test_no_manifest_on_either_side_is_not_an_error():
    """Smoke/jsonl on both sides: nothing to check, nothing raised or warned."""
    _check_private_provenance(None, None)  # must not raise


# --- B5.1: every new field is populated, or null with a stated reason ------

def _benchmark_data(n_tier_a: int = 3, n_tier_b: int = 2) -> BenchmarkData:
    import pandas as pd
    n = n_tier_a + n_tier_b
    meta = pd.DataFrame({
        "series_id": [f"s{i}" for i in range(n)],
        "family": ["fam1"] * n_tier_a + ["fam2"] * n_tier_b,
        "tier": ["synthetic"] * n_tier_a + ["real_derived"] * n_tier_b,
        "archetype": [None] * n,
        "generator": ["parametric"] * n,
    })
    return BenchmarkData(np.zeros((n, 10), dtype=np.float32), meta, 8, 2,
                        {}, corpus_digest="private-digest-xyz")


def test_every_new_field_is_populated_when_a_manifest_exists():
    private = _benchmark_data()
    dev_manifest = {"visibility": "public", "epoch": 3, "global_digest": "dev-digest"}
    private_manifest = {"visibility": "private", "epoch": 3, "global_digest": "priv-digest",
                        "extra": {"audit": {"gate": {"gate_effective": True}}}}
    out = _private_provenance(private, "/some/path", dev_manifest, private_manifest)
    assert out["private_corpus_path"] == "/some/path"
    assert out["private_corpus_digest"] == "private-digest-xyz"
    assert out["private_manifest_epoch"] == 3
    assert out["private_visibility"] == "private"
    assert out["private_composition"]["by_tier"] == {"synthetic": 3, "real_derived": 2}
    assert out["private_composition"]["by_family"] == {"fam1": 3, "fam2": 2}
    assert out["dev_corpus_digest"] == "dev-digest"
    assert out["dev_manifest_epoch"] == 3
    assert out["private_audit"]["state"] == "measured"
    assert out["exchangeability"]["available"] is False
    assert "B3" in out["exchangeability"]["reason"]
    assert out["private_manifest_reason"] is None


def test_TB3_4_exchangeability_is_populated_end_to_end_when_cross_split_json_exists(tmp_path):
    """B3.4's actual wiring: when `benchmark_validation` has persisted a
    real cross_split.json next to the private corpus (matching this run's
    digest), `_private_provenance` must surface its real verdict -- not
    just the degraded 'not run yet' reason the test above pins for the
    absent case."""
    private = _benchmark_data()  # corpus_digest == "private-digest-xyz"
    save_json(tmp_path / "cross_split.json", {
        "private_corpus_digest": "private-digest-xyz",
        "model_free": True,
        "n_public": 965, "n_private": 967,
        "near_duplicates": {"near_duplicate_fraction": 5e-05},
        "equivalence": {
            "overall_verdict": "inconclusive",
            "tost": {"verdict": "inconclusive",
                     "detail": "n too small to establish equivalence at margin 0.2 SD for: CO_trev_1_num"},
            "energy_distance": {"p": 0.511744, "significant_difference": False},
        },
    })
    dev_manifest = {"visibility": "public", "epoch": 0, "global_digest": "dev-digest"}
    private_manifest = {"visibility": "private", "epoch": 0}
    out = _private_provenance(private, str(tmp_path), dev_manifest, private_manifest)
    exch = out["exchangeability"]
    assert exch["available"] is True
    assert exch["provenance"] == "verified"
    assert exch["report"]["equivalence"]["overall_verdict"] == "inconclusive"
    assert exch["report"]["equivalence"]["energy_distance"]["p"] == pytest.approx(0.511744)


def test_TB3_4_exchangeability_refuses_a_cross_split_json_for_a_different_private_corpus(tmp_path):
    """Load-bearing negative for the wiring itself: a stale/mismatched
    cross_split.json (wrong digest) must not be surfaced as this run's
    verdict."""
    private = _benchmark_data()  # corpus_digest == "private-digest-xyz"
    save_json(tmp_path / "cross_split.json", {
        "private_corpus_digest": "some-other-corpus-digest",
        "equivalence": {"overall_verdict": "equivalent"},
    })
    out = _private_provenance(private, str(tmp_path), None, {"visibility": "private"})
    assert out["exchangeability"]["available"] is False
    assert "private-digest-xyz" in out["exchangeability"]["reason"]
    assert "some-other-corpus-digest" in out["exchangeability"]["reason"]


def test_manifest_derived_fields_are_null_with_a_stated_reason_when_no_manifest_exists():
    """The smoke/jsonl failure mode: every manifest-derived field is `None`,
    but the reason for that `None` is recorded rather than a bare null."""
    private = _benchmark_data()
    out = _private_provenance(private, "/some/path", None, None)
    assert out["private_manifest_epoch"] is None
    assert out["private_visibility"] is None
    assert out["dev_corpus_digest"] is None
    assert out["dev_manifest_epoch"] is None
    assert out["private_corpus_digest"] == "private-digest-xyz"  # from BenchmarkData, not manifest
    assert out["private_manifest_reason"] is not None
    assert "no sealed manifest" in out["private_manifest_reason"]
    # `private_audit`'s own three-state vocabulary already carries a reason
    # (reused from `corpus_card.py`, sec 11.37) -- never a bare null either.
    assert out["private_audit"]["state"] == "not_recorded"
    assert out["private_audit"]["reason"] is not None
    # Composition is NOT manifest-derived -- it must stay populated even
    # with no manifest at all, since it comes straight from `private.meta`.
    assert out["private_composition"]["by_tier"] == {"synthetic": 3, "real_derived": 2}


def test_adaptivity_names_no_model_or_architecture():
    """The same contract every other pure reduction in this repo holds to."""
    import inspect
    from tsfm_lens.analysis import confirm as confirm_mod
    src = inspect.getsource(confirm_mod._private_provenance)
    body = src.split('"""')[2] if src.count('"""') >= 2 else src
    for banned in ("TimesFM", "Chronos", "Sundial", "model_a", "model_b",
                   "models[0]", "models[1]"):
        assert banned not in body


# --- T-B5.3: the new output fields must not move the stage fingerprint -----

def test_new_output_fields_do_not_move_the_confirm_stage_fingerprint(tmp_path):
    """Outputs, not config -- the resolved fingerprint input for `confirm`
    must be bit-identical whether or not this item's fields exist, verified
    the way §11.51's fourth defect was (comparing the resolved input itself,
    not merely asserting the stage still runs)."""
    cfg = _cfg(tmp_path)
    own_resolved = resolve_config_keys(cfg, ("confirm",))
    fp = fingerprint_stage(own_resolved, {})
    # No `ConfirmConfig` field was added by this item -- confirm's
    # `config_keys` is `("confirm",)`, i.e. the whole config SECTION, so an
    # unchanged section must fingerprint identically across two builds of
    # the same cfg regardless of what `run_confirm` later writes to
    # `confirmation.json` (an output artifact, never a fingerprint input).
    own_resolved_again = resolve_config_keys(_cfg(tmp_path), ("confirm",))
    fp_again = fingerprint_stage(own_resolved_again, {})
    assert fp == fp_again


def test_confirm_config_keys_unchanged_by_this_item():
    """Pins that the provenance item (B5) did not (and must not) add a new
    `ConfirmConfig` field -- ITS new fields are `confirmation.json` OUTPUTS.
    `concept_transfer_n_null` (ROADMAP.md sec 37.10 P7) is a genuine
    exception to that rule, not a violation of it: it is an actual stage
    INPUT (`_replicate_registered_concepts`'s null-draw count), so it
    belongs in the pinned set rather than being read back out of the
    artifact the way B5's fields are. ROADMAP.md sec 38.2 (K2) added six
    more genuine stage inputs (the causal-concept claim knobs), each marked
    `omit_at_default` so an older run's fingerprint is unchanged
    (`tests/test_concept_causal_confirm.py::test_new_fields_do_not_move_
    older_fingerprints`)."""
    from tsfm_lens.config import ConfirmConfig
    import dataclasses
    names = {f.name for f in dataclasses.fields(ConfirmConfig)}
    assert names == {"enabled", "source", "path", "max_series", "require_seal", "alpha",
                     "concept_transfer_n_null", "register_concept_claims", "causal_n_null",
                     "causal_max_null", "atlas_n_null", "agreement_n_null",
                     "structure_n_boot"}


# --- consumer audit (sec 11.40): confirmation.json's four named consumers --

def test_report_py_reads_confirm_by_key_not_exhaustive_keys():
    import inspect
    from tsfm_lens.report import report as report_mod
    src = inspect.getsource(report_mod._sec_confirm)
    assert "conf.keys()" not in src
    assert "set(conf)" not in src


def test_analysis_card_survives_extra_confirm_keys(tmp_path):
    from tsfm_lens.report.analysis_card import build_analysis_card
    run_dir = tmp_path / "run"
    (run_dir / "confirm").mkdir(parents=True)
    save_json(run_dir / "confirm" / "confirmation.json", {
        "n_registered": 4, "n_replicable": 2, "alpha": 0.05, "n_private_series": 40,
        "private_corpus_digest": "abc123", "private_visibility": "private",
        "exchangeability": {"available": False, "reason": "not implemented"},
    })
    card = build_analysis_card(run_dir)
    assert card["confirm"]["n_registered"] == 4
    assert card["confirm"]["private_corpus_digest"] == "abc123"


def test_meta_report_reads_confirm_by_key(tmp_path):
    from tsfm_lens.report.meta_report import summarize_run
    run_dir = tmp_path / "run"
    (run_dir / "confirm").mkdir(parents=True)
    save_json(run_dir / "confirm" / "confirmation.json", {
        "tests": [{"status": "tested", "confirmed": True}], "overall": {"p": 0.01},
        "private_corpus_digest": "abc123",
    })
    summary = summarize_run(run_dir)
    assert summary["confirm_tests"] == [{"status": "tested", "confirmed": True}]


# --- B5.3: the report's Confirm section renders a provenance line --------

def test_report_confirm_section_renders_provenance_when_manifest_present():
    from tsfm_lens.report.report import _confirm_provenance_block
    html = _confirm_provenance_block({
        "private_visibility": "private", "private_manifest_epoch": 3,
        "private_corpus_digest": "abcdef0123456789", "dev_manifest_epoch": 3,
        "private_audit": {"state": "measured", "reason": None},
        "private_manifest_reason": None,
    })
    assert "private" in html and "3" in html and "abcdef0123456789"[:16] in html
    assert "measured" in html
    assert "different epoch" not in html  # epochs match -- no mismatch note


def test_report_confirm_section_flags_an_epoch_mismatch():
    from tsfm_lens.report.report import _confirm_provenance_block
    html = _confirm_provenance_block({
        "private_visibility": "private", "private_manifest_epoch": 3,
        "private_corpus_digest": "abc", "dev_manifest_epoch": 1,
        "private_audit": {"state": "not_recorded", "reason": None},
        "private_manifest_reason": None,
    })
    assert "different epoch" in html


def test_report_confirm_section_states_the_no_manifest_reason():
    from tsfm_lens.report.report import _confirm_provenance_block
    html = _confirm_provenance_block({"private_manifest_reason": "no sealed manifest (smoke source)"})
    assert "not recorded" in html
    assert "no sealed manifest" in html
    assert "distributionally identical" in html


def test_report_confirm_section_degrades_for_a_pre_b5_artifact():
    """A `confirmation.json` with none of B5's keys at all (a run from before
    this item existed) must render the same honest "not recorded" line, not
    a KeyError and not a fabricated "measured" claim."""
    from tsfm_lens.report.report import _confirm_provenance_block
    html = _confirm_provenance_block({"n_private_series": 10, "alpha": 0.05})
    assert "not recorded" in html


def test_sec_confirm_end_to_end_includes_the_provenance_block(tmp_path):
    from tsfm_lens.report.report import _sec_confirm
    run_dir = tmp_path / "run"
    (run_dir / "confirm").mkdir(parents=True)
    save_json(run_dir / "confirm" / "confirmation.json", {
        "n_private_series": 40, "alpha": 0.05, "tests": [],
        "n_registered": 0, "n_replicable": 0, "registry_sha256": "abc",
        "private_visibility": "private", "private_manifest_epoch": 0,
        "private_corpus_digest": "xyz", "dev_manifest_epoch": 0,
        "private_audit": {"state": "measured", "reason": None},
        "private_manifest_reason": None,
    })
    html = _sec_confirm(run_dir, [], 0)
    assert "Private corpus" in html
    assert "measured" in html


def test_replication_summary_survives_extra_confirm_keys(tmp_path):
    from tsfm_lens.report.derived import replication_summary
    run_dir = tmp_path / "run"
    (run_dir / "confirm").mkdir(parents=True)
    save_json(run_dir / "confirm" / "confirmation.json", {
        "tests": [{"status": "tested", "confirmed": True}],
        "private_corpus_digest": "abc123", "private_visibility": "private",
        "private_audit": {"state": "measured"},
    })
    df = replication_summary(run_dir)
    assert len(df) == 1
