"""The `concepts` pipeline stage (ROADMAP.md sec 37.4 P1): ablation battery ->
concept clustering -> cross-model transfer -> deterministic descriptions, as
one stage over mock adapters on CPU.

The load-bearing tests are the script/stage identity (the refactor's whole
point is that the two entry points cannot drift) and the fingerprint scoping
(the clustering and transfer knobs stay in `sae:` but must fingerprint THIS
stage, and a `concepts:` edit must not touch the `sae` stage's fingerprint).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_smoke import build_config  # noqa: E402
from tsfm_lens.config import config_from_dict  # noqa: E402
from tsfm_lens.manifest import resolve_config_keys  # noqa: E402
from tsfm_lens.pipeline import _stage_by_name, run_pipeline  # noqa: E402
from tsfm_lens.sae.concept_stage import _transfer_skip_reason  # noqa: E402
from tsfm_lens.utils import load_json  # noqa: E402


def _cfg(out_dir: str, name: str = "concept_stage"):
    cfg = config_from_dict(build_config(out_dir))
    cfg.run.name = name
    cfg.sae.enabled = True
    cfg.sae.epochs = 3
    cfg.sae.dict_size_mult = 2
    cfg.sae.k = 4
    cfg.sae.forecast_preservation_max_series = 16
    cfg.sae.persist_features = True
    cfg.concepts.enabled = True
    # The legacy single-fit sentinel (sec 32 D5): the mocks' few causal
    # candidates cannot satisfy the default size floor, and the chain needs
    # concepts to exercise transfer at all.
    cfg.sae.concept_min_members = 1
    cfg.concepts.n_features_per_rule = 6
    cfg.concepts.n_null_directions = 4
    cfg.concepts.max_series = 16
    cfg.concepts.top_k_series = 4
    return cfg


@pytest.fixture(scope="module")
def run_dir():
    out = tempfile.mkdtemp()
    cfg = _cfg(out)
    run_pipeline(cfg, stages=["extract"])
    from tsfm_lens.extraction.store import ActivationStore
    store = ActivationStore(cfg.run_dir() / "activations.zarr", mode="r")
    cfg.sae.targets = [{"model": "patchy", "layer": store.layers("patchy")[-1]},
                       {"model": "steppy", "layer": store.layers("steppy")[-1]}]
    run_pipeline(cfg, stages=["sae", "concepts"])
    return cfg.run_dir(), cfg


def test_stage_chain_writes_all_artifacts(run_dir):
    rd, cfg = run_dir
    for rel in ("sae/concepts.json", "sae/transfer.json", "sae/concept_stage.json"):
        assert (rd / rel).exists(), rel
    assert len(list((rd / "sae").glob("*/*_ablation.json"))) == 2
    record = load_json(rd / "sae" / "concept_stage.json")
    assert record["n_concepts"] > 0
    assert record["transfer"]["status"] == "ran" and record["transfer"]["n_pairs"] > 0
    # The mocks have no ground-truth table, so no `separated` block and no
    # description packets; the stage must say so rather than leave a file.
    if record["describe"]["status"] != "ran":
        assert not (rd / "sae" / "descriptions.json").exists()
    assert set(record["ablation"]) == {t["model"] + "/" + t["layer"] for t in cfg.sae.targets}
    transfer = load_json(rd / "sae" / "transfer.json")
    assert record["transfer"]["n_pairs"] == len(transfer["pairs"])


def test_atlas_artifact_and_block_written(run_dir):
    """ROADMAP.md sec 37.15 q6 (option b): the fifth step, an ADDITIVE
    cross-model concept atlas alongside the four steps
    `test_stage_chain_writes_all_artifacts` already covers. On this fixture
    (2 mock architectures, `atlas_enabled` at its default) the atlas has
    enough causal, non-withheld candidates pooled across both models to
    actually run -- so this pins the `"ran"` branch's own internal
    consistency (the record's counts match the artifact's), not just that
    SOME status was recorded. The `"skipped"` branch (an empty pool, or
    `concepts.atlas_enabled: false`) is covered separately in
    `tests/test_concept_atlas.py` via `_atlas_skip_reason` and the
    empty-artifact driver test -- this fixture is not built to exercise it,
    since forcing it here would mean starving the fixture of the very
    causal candidates `test_stage_chain_writes_all_artifacts` above needs."""
    rd, cfg = run_dir
    record = load_json(rd / "sae" / "concept_stage.json")
    atlas = record["atlas"]
    assert atlas["status"] == "ran", atlas
    assert (rd / "sae" / "concept_atlas.json").exists()
    art = load_json(rd / "sae" / "concept_atlas.json")
    assert atlas["n_features"] == art["n_features"]
    assert atlas["n_assigned"] == art["n_assigned"]
    assert atlas["n_concepts"] == len(art["concepts"])
    assert atlas["n_multi_model"] == sum(1 for c in art["concepts"] if c["n_models"] >= 2)
    # Every pooled row must come from one of THIS run's two mock models --
    # the atlas must not silently pick up an unrelated model's leftover
    # ablation file from a previous run sharing the same `out_dir`.
    assert set(r["model"] for r in art["rows"]) <= {"patchy", "steppy"}


def test_atlas_disabled_removes_stale_artifact(run_dir):
    """Planted regression companion to the positive test above: disabling
    `concepts.atlas_enabled` on a rerun into the SAME run directory must
    drop the `concept_atlas.json` this fixture's default run just wrote,
    exactly `transfer.json`'s own `_drop_stale` discipline
    (`test_skipped_transfer_removes_a_stale_artifact` above)."""
    rd, cfg = run_dir
    import shutil
    import tempfile
    work = Path(tempfile.mkdtemp()) / rd.name
    shutil.copytree(rd, work)
    assert (work / "sae" / "concept_atlas.json").exists()
    cfg2 = _cfg(str(work.parent), name=work.name)
    cfg2.sae.targets = cfg.sae.targets
    cfg2.concepts.atlas_enabled = False
    run_pipeline(cfg2, stages=["concepts"], force={"concepts"}, allow_stale=True)
    record = load_json(work / "sae" / "concept_stage.json")
    assert record["atlas"]["status"] == "skipped"
    assert record["atlas"]["reason"] == "concepts.atlas_enabled is false"
    assert record["atlas"]["removed_stale"] is True
    assert not (work / "sae" / "concept_atlas.json").exists()


def test_stability_wired_into_stage_default_seeds(run_dir):
    """ROADMAP.md sec 37 P2: `concepts.n_sae_seeds` defaults to 3 (the
    primary plus 2 replicates), so this fixture's default run (no override
    in `_cfg`) must have trained and persisted both replicates for every
    trained target and run `concept_stability` end to end -- not merely
    left the knob wired but unexercised."""
    rd, cfg = run_dir
    from tsfm_lens.extraction.store import ActivationStore
    record = load_json(rd / "sae" / "concept_stage.json")
    stability = record["stability"]
    assert stability["status"] == "ran", stability
    assert stability["n_sae_seeds"] == 3
    assert "replicate_train_seconds" in stability
    assert (rd / "sae" / "concept_stability.json").exists()
    art = load_json(rd / "sae" / "concept_stability.json")
    assert art["measured"] is True
    assert art["n_sae_seeds"] == 3 and art["replicates"] == 2
    assert stability["n_concepts"] == art["n_concepts"]
    assert stability["n_stable"] == art["n_stable"]
    # Both replicates were actually trained and persisted as SIBLING
    # checkpoints/store groups, never overwriting the primary's own.
    store = ActivationStore(rd / "activations.zarr", mode="r")
    for target in cfg.sae.targets:
        model, layer = target["model"], target["layer"]
        assert store.has_sae_features(model, layer, replicate=0)
        assert store.has_sae_features(model, layer, replicate=1)
        assert store.has_sae_features(model, layer, replicate=2)
        from tsfm_lens.sae.train import sanitize
        assert (rd / "sae" / sanitize(model) / f"{sanitize(layer)}@r1.pt").exists()
        assert (rd / "sae" / sanitize(model) / f"{sanitize(layer)}@r2.pt").exists()
        # negative: a replicate must not silently overwrite the primary's own
        # checkpoint -- if it did, this file would be missing entirely
        # (`run_sae` always writes it with no `@r` suffix).
        assert (rd / "sae" / sanitize(model) / f"{sanitize(layer)}.pt").exists()


def test_stability_disabled_trains_no_replicates():
    """`concepts.n_sae_seeds: 1` must cost nothing extra: no replicate
    checkpoint, no replicate store group, and `concept_stability.json`
    records `"not measured"` rather than a partial or fabricated result."""
    from tsfm_lens.extraction.store import ActivationStore
    from tsfm_lens.sae.train import sanitize

    work = tempfile.mkdtemp()
    cfg = _cfg(work, name="stability_disabled")
    cfg.concepts.n_sae_seeds = 1
    run_pipeline(cfg, stages=["extract"])
    store = ActivationStore(cfg.run_dir() / "activations.zarr", mode="r")
    cfg.sae.targets = [{"model": "patchy", "layer": store.layers("patchy")[-1]},
                       {"model": "steppy", "layer": store.layers("steppy")[-1]}]
    run_pipeline(cfg, stages=["sae", "concepts"])
    rd = cfg.run_dir()
    record = load_json(rd / "sae" / "concept_stage.json")
    assert record["stability"]["status"] == "not_measured"
    art = load_json(rd / "sae" / "concept_stability.json")
    assert art["measured"] is False
    store2 = ActivationStore(rd / "activations.zarr", mode="r")
    for target in cfg.sae.targets:
        model, layer = target["model"], target["layer"]
        assert not (rd / "sae" / sanitize(model) / f"{sanitize(layer)}@r1.pt").exists()
        assert not store2.has_sae_features(model, layer, replicate=1)


def test_script_and_stage_produce_identical_ablation_json(run_dir):
    """Re-running `run_sae_ablation.py --all` with its CLI defaults set to the
    stage's knobs must reproduce the stage's artifacts byte for byte."""
    rd, cfg = run_dir
    stage_bytes = {p: p.read_bytes() for p in sorted((rd / "sae").glob("*/*_ablation.json"))}
    c = cfg.concepts
    subprocess.run(
        [sys.executable, str(ROOT / "run_sae_ablation.py"), "--run", str(rd), "--all",
         "--top-k-series", str(c.top_k_series), "--n-null-directions", str(c.n_null_directions),
         "--max-series", str(c.max_series), "--keep-forecasts", str(c.keep_forecasts),
         "--n-features-per-rule", str(c.n_features_per_rule), "--seed", str(cfg.run.seed)],
        check=True, cwd=str(ROOT), capture_output=True)
    for p, b in stage_bytes.items():
        assert p.read_bytes() == b, f"{p.name}: script and stage disagree"


def test_ablation_defaults_match_the_script(run_dir):
    """The config defaults ARE the script's CLI defaults, so a default stage
    run and a default script run are the same measurement."""
    from tsfm_lens.config import ConceptsConfig
    src = (ROOT / "run_sae_ablation.py").read_text(encoding="utf-8")
    d = ConceptsConfig()
    for flag, value in (("--top-k-series", d.top_k_series),
                        ("--n-null-directions", d.n_null_directions),
                        ("--max-series", d.max_series),
                        ("--keep-forecasts", d.keep_forecasts),
                        ("--n-features-per-rule", d.n_features_per_rule)):
        line = next(l for l in src.splitlines() if f'"{flag}"' in l)
        assert f"default={value}" in line, f"{flag}: {line.strip()}"


def test_preflight_refuses_without_persisted_features(run_dir):
    rd, cfg = run_dir
    bad = _cfg(str(rd.parent), name=rd.name)
    bad.sae.persist_features = False
    with pytest.raises(ValueError, match="persist_features"):
        run_pipeline(bad, stages=["concepts"], force={"concepts"})


def test_skipped_transfer_removes_a_stale_artifact(run_dir):
    """A clustering change that leaves no concepts must not leave the
    previous `transfer.json` beside the new `concepts.json`."""
    rd, cfg = run_dir
    import shutil
    work = Path(tempfile.mkdtemp()) / rd.name
    shutil.copytree(rd, work)
    cfg2 = _cfg(str(work.parent), name=work.name)
    cfg2.sae.targets = cfg.sae.targets
    cfg2.sae.concept_min_members = 50  # no k is admissible -> no concepts
    assert (work / "sae" / "transfer.json").exists()
    run_pipeline(cfg2, stages=["concepts"], force={"concepts"}, allow_stale=True)
    record = load_json(work / "sae" / "concept_stage.json")
    assert record["n_concepts"] == 0
    assert record["transfer"]["status"] == "skipped"
    assert record["transfer"]["removed_stale"] is True
    assert not (work / "sae" / "transfer.json").exists()


def test_transfer_skip_reasons():
    cfg = _cfg(tempfile.mkdtemp())
    c1 = [{"concept": 0}]
    two = {"targets": {"patchy/b.1": {"model": "patchy", "concepts": c1},
                       "steppy/b.1": {"model": "steppy"}}}
    persisted = {k: {"features_persisted": True} for k in two["targets"]}
    assert _transfer_skip_reason(cfg, two, persisted) is None

    one = {"targets": {"patchy/b.1": {"model": "patchy", "concepts": c1},
                       "patchy/b.2": {"model": "patchy"}}}
    assert "span 1 model" in _transfer_skip_reason(cfg, one, {})

    empty = {"targets": {k: {"model": v["model"]} for k, v in two["targets"].items()}}
    assert "no target has any concepts" in _transfer_skip_reason(cfg, empty, persisted)

    assert "persisted" in _transfer_skip_reason(cfg, two, {"patchy/b.1": {"features_persisted": True}})

    cfg.sae.transfer_enabled = False
    assert "transfer_enabled" in _transfer_skip_reason(cfg, two, persisted)

    solo = _cfg(tempfile.mkdtemp())
    solo.models = solo.models[:1]
    assert _transfer_skip_reason(solo, two, persisted).startswith("run shape solo")


def test_fingerprint_scoping():
    """`sae.concept_*`/`sae.transfer_*` move the concepts stage's fingerprint;
    a `concepts:` edit does not move the `sae` stage's."""
    cfg = _cfg(tempfile.mkdtemp())
    sae_keys, concept_keys = _stage_by_name("sae").config_keys, _stage_by_name("concepts").config_keys
    sae_before = resolve_config_keys(cfg, sae_keys)
    concepts_before = resolve_config_keys(cfg, concept_keys)

    cfg.concepts.top_k_series += 1
    assert resolve_config_keys(cfg, sae_keys) == sae_before
    assert resolve_config_keys(cfg, concept_keys) != concepts_before

    concepts_mid = resolve_config_keys(cfg, concept_keys)
    cfg.sae.concept_min_members += 1
    assert resolve_config_keys(cfg, concept_keys) != concepts_mid
    cfg.sae.transfer_seed += 1
    assert resolve_config_keys(cfg, concept_keys) != resolve_config_keys(
        _cfg(tempfile.mkdtemp()), concept_keys)


def test_existing_sae_fields_not_moved():
    """Removing these from `sae:` would change every existing run's `sae`
    fingerprint (CLAUDE.md sec 11.51)."""
    import dataclasses
    from tsfm_lens.config import SAEConfig
    names = {f.name for f in dataclasses.fields(SAEConfig)}
    for field in ("concept_causal_only", "concept_k", "concept_min_silhouette",
                  "concept_min_members", "transfer_enabled", "transfer_top_k",
                  "transfer_n_null", "transfer_seed", "describe_from_exemplars"):
        assert field in names, field
