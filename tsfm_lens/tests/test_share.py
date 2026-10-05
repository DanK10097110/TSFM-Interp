"""`share.py`: prune and bundle a run directory, and the pruned-run pipeline guard.

The fixture is a fake run directory with a planted, known answer and decoys:
a `foo.pt.json` data file that merely looks like a checkpoint, a `.zarr`
symlink pointing OUTSIDE the run (its target must survive a prune), a backup
render, and a JSON over the size cap. Sizes are planted so freed bytes are
asserted exactly rather than "greater than zero".
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens import share  # noqa: E402

SIZES = {
    "sae/m/a.b_0.pt": 1000,
    "sae/m/a.b_0@r1.pt": 2000,
    "sae/m/plain.pt": 400,
    "activations.zarr/act/m/a.b_0/0.0": 5000,
    "activations.zarr/act/m/a.b_0/0.1": 7000,
    "activations.zarr/.zattrs": 100,
    "layer_screen/screen_activations.zarr/x/0": 300,
}
EXPECTED_HEAVY = {"sae/m/a.b_0.pt", "sae/m/a.b_0@r1.pt", "sae/m/plain.pt",
                  "activations.zarr", "layer_screen/screen_activations.zarr", "link.zarr"}
OUTSIDE_BYTES = 9000


def _write(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * n)


@pytest.fixture()
def fake_run(tmp_path):
    run = tmp_path / "run_a"
    for rel, n in SIZES.items():
        _write(run / rel, n)
    outside = tmp_path / "outside_store"
    _write(outside / "chunk", OUTSIDE_BYTES)
    (run / "link.zarr").symlink_to(outside, target_is_directory=True)
    (run / "report.html").write_text("<html>report</html>", encoding="utf-8")
    (run / "report_preconfirm.html").write_text("<html>old</html>", encoding="utf-8")
    (run / "notes.pre_fix.json").write_text("{}", encoding="utf-8")
    (run / "l0").mkdir()
    (run / "l0" / "summary.json").write_text('{"a": 1}', encoding="utf-8")
    (run / "l1").mkdir()
    (run / "l1" / "cka.npz").write_bytes(b"n" * 50)
    (run / "sae" / "m" / "foo.pt.json").write_text('{"decoy": true}', encoding="utf-8")
    (run / "sae" / "m" / "a.b_0_ablation.json").write_text('{"k": 1}', encoding="utf-8")
    _write(run / "big.json", 3 * 1024 * 1024)
    _write(run / "weights.bin", 10)
    return run


def _rel(run, paths):
    return {Path(p).relative_to(run).as_posix() for p in paths}


def test_heavy_paths_exact_set_and_decoy(fake_run):
    heavy = _rel(fake_run, share.heavy_paths(fake_run))
    assert heavy == EXPECTED_HEAVY
    assert "sae/m/foo.pt.json" not in heavy


def test_dry_run_deletes_nothing_and_sizes_are_planted(fake_run):
    before = sorted(p.as_posix() for p in fake_run.rglob("*"))
    out = share.prune_run(fake_run, dry_run=True)
    assert sorted(p.as_posix() for p in fake_run.rglob("*")) == before
    assert not (fake_run / "pruned.json").exists()
    planted = sum(SIZES.values())
    link_size = (fake_run / "link.zarr").lstat().st_size
    assert out["freed_bytes"] == planted + link_size
    assert out["symlink_target_bytes_kept"] == OUTSIDE_BYTES
    kinds = {r["path"]: r["kind"] for r in out["removed"]}
    assert kinds["sae/m/a.b_0.pt"] == "sae_checkpoint"
    assert kinds["activations.zarr"] == "activation_store"
    assert kinds["layer_screen/screen_activations.zarr"] == "layer_screen_store"
    assert kinds["link.zarr"] == "zarr_store"


def test_prune_deletes_heavy_keeps_symlink_target_and_records(fake_run, tmp_path):
    out = share.prune_run(fake_run, dry_run=False)
    assert not (fake_run / "activations.zarr").exists()
    assert not (fake_run / "link.zarr").exists() and not (fake_run / "link.zarr").is_symlink()
    assert (tmp_path / "outside_store" / "chunk").stat().st_size == OUTSIDE_BYTES
    assert not list(fake_run.rglob("*.pt"))
    for keep in ("report.html", "l0/summary.json", "sae/m/foo.pt.json", "big.json"):
        assert (fake_run / keep).exists(), keep
    rec = json.loads((fake_run / "pruned.json").read_text(encoding="utf-8"))
    assert set(rec) == {"pruned_at", "removed", "freed_bytes", "note", "store_attrs_error"}
    assert "index axis" in rec["store_attrs_error"]
    assert rec["freed_bytes"] == out["freed_bytes"] == sum(r["bytes"] for r in rec["removed"])
    assert {r["path"] for r in rec["removed"]} == EXPECTED_HEAVY
    assert "only --stages report" in rec["note"]


def test_second_prune_merges(fake_run):
    first = share.prune_run(fake_run, dry_run=False)
    _write(fake_run / "sae" / "m2" / "late.pt", 123)
    second = share.prune_run(fake_run, dry_run=False)
    assert second["freed_bytes"] == 123
    rec = json.loads((fake_run / "pruned.json").read_text(encoding="utf-8"))
    assert rec["freed_bytes"] == first["freed_bytes"] + 123
    assert len(rec["removed"]) == len(EXPECTED_HEAVY) + 1


def test_bundle_inclusion_exclusion_and_manifest(fake_run):
    zpath = share.bundle_run(fake_run, max_file_mb=1.0)
    assert zpath == fake_run.parent / "run_a_share.zip"
    with zipfile.ZipFile(zpath) as zf:
        names = {n.split("/", 1)[1] for n in zf.namelist()}
        manifest = json.loads(zf.read("run_a/SHARE_MANIFEST.json"))
    included = {e["path"] for e in manifest["included"]}
    assert included == {"report.html", "l0/summary.json", "l1/cka.npz", "sae/m/foo.pt.json",
                        "sae/m/a.b_0_ablation.json"}
    assert names == included | {"SHARE_MANIFEST.json"}
    reasons = {e["path"]: e["reason"] for e in manifest["excluded"]}
    assert reasons["activations.zarr"] == "heavy cache"
    assert reasons["link.zarr"] == "heavy cache"
    assert reasons["sae/m/a.b_0.pt"] == "heavy cache"
    assert reasons["big.json"] == "over size cap"
    assert reasons["report_preconfirm.html"] == "backup render"
    assert reasons["notes.pre_fix.json"] == "backup render"
    assert reasons["weights.bin"] == "extension"
    assert manifest["excluded_by_reason"]["heavy cache"]["count"] == len(EXPECTED_HEAVY)
    assert manifest["included_bytes"] == sum(e["bytes"] for e in manifest["included"])


def test_bundle_keeps_report_over_cap(fake_run):
    _write(fake_run / "report.html", 2 * 1024 * 1024)
    zpath = share.bundle_run(fake_run, out=fake_run.parent / "o.zip", max_file_mb=0.5)
    with zipfile.ZipFile(zpath) as zf:
        assert "run_a/report.html" in zf.namelist()


def test_bundle_without_report_refuses(fake_run):
    (fake_run / "report.html").unlink()
    with pytest.raises(FileNotFoundError, match="report.html"):
        share.bundle_run(fake_run)


def _cfg_for(tmp_path):
    from tests.test_smoke import build_config
    from tsfm_lens.config import config_from_dict
    cfg = config_from_dict(build_config(str(tmp_path)))
    cfg.run.name = "guard"
    return cfg


def test_guard_refuses_other_stages_allows_report(tmp_path):
    from tsfm_lens.pipeline import _refuse_on_pruned_run
    cfg = _cfg_for(tmp_path)
    run = cfg.run_dir()
    _write(run / "sae" / "m" / "a.pt", 10)
    _refuse_on_pruned_run(cfg, ["l1"])
    share.prune_run(run, dry_run=False)
    with pytest.raises(ValueError) as e:
        _refuse_on_pruned_run(cfg, ["l1"])
    msg = str(e.value)
    assert "l1" in msg and "pruned on" in msg and "sae_checkpoint" in msg
    assert "--stages report" in msg and "new run directory" in msg
    with pytest.raises(ValueError):
        _refuse_on_pruned_run(cfg, ["report", "l1"])
    with pytest.raises(ValueError):
        _refuse_on_pruned_run(cfg, None)
    _refuse_on_pruned_run(cfg, ["report"])


def test_missing_checkpoint_gets_rendered_reason_only_when_pruned(tmp_path):
    from tsfm_lens.report.report import _feature_cards_for
    cfg = _cfg_for(tmp_path)
    cfg.run_dir().mkdir(parents=True)
    args = (cfg, None, "m", "a.b", {"ground_truth_alignment": {"rows": [0]}}, {}, None)
    with pytest.raises(Exception) as before:
        _feature_cards_for(*args)
    assert "cache pruned" not in str(before.value)
    share.prune_run(cfg.run_dir(), dry_run=False)
    with pytest.raises(RuntimeError, match="cache pruned: the SAE checkpoint for m/a.b"):
        _feature_cards_for(*args)


def test_cli_prune_dry_run_default_and_yes(fake_run, monkeypatch, capsys):
    from tsfm_lens import run as run_mod
    monkeypatch.setattr(sys, "argv", ["run.py", "--prune", str(fake_run)])
    run_mod.main()
    out = capsys.readouterr().out
    assert "dry run: nothing deleted" in out and "sae_checkpoint" in out
    assert (fake_run / "activations.zarr").exists()
    monkeypatch.setattr(sys, "argv", ["run.py", "--prune", str(fake_run), "--yes"])
    run_mod.main()
    assert "freed" in capsys.readouterr().out
    assert not (fake_run / "activations.zarr").exists()
    monkeypatch.setattr(sys, "argv", ["run.py", "--bundle", str(fake_run),
                                      "--bundle-max-file-mb", "1"])
    run_mod.main()
    out = capsys.readouterr().out
    assert "files included" in out and "over size cap" in out


@pytest.fixture(scope="module")
def pruned_smoke(tmp_path_factory):
    """Real mock run (SAE + concepts + confirm), report rendered, pruned, report rerun."""
    from tests.test_concept_stage import _cfg
    from tsfm_lens.extraction.store import ActivationStore
    from tsfm_lens.pipeline import run_pipeline
    out = str(tmp_path_factory.mktemp("share_e2e"))
    cfg = _cfg(out, "share_e2e")
    run_pipeline(cfg, stages=["extract"])
    store = ActivationStore(cfg.run_dir() / "activations.zarr", mode="r")
    cfg.sae.targets = [{"model": "patchy", "layer": store.layers("patchy")[-1]},
                       {"model": "steppy", "layer": store.layers("steppy")[-1]}]
    run_pipeline(cfg)
    run = cfg.run_dir()
    run_pipeline(cfg, stages=["report"])
    before = json.loads((run / "report" / "coverage.json").read_text(encoding="utf-8"))
    before_bytes = share.run_dir_bytes(run)
    html_before = (run / "report.html").read_text(encoding="utf-8")
    summary = share.prune_run(run, dry_run=False)
    run_pipeline(cfg, stages=["report"])
    after = json.loads((run / "report" / "coverage.json").read_text(encoding="utf-8"))
    html_after = (run / "report.html").read_text(encoding="utf-8")
    return cfg, run, before, after, summary, before_bytes, (html_before, html_after)


def test_pruned_run_report_has_identical_sections(pruned_smoke):
    cfg, run, before, after, summary, _, (html_b, html_a) = pruned_smoke
    key = lambda cov: [(s["eyebrow"], s["title"], s["status"]) for s in cov["sections"]]
    assert key(after) == key(before)
    assert after["summary"] == before["summary"]
    assert not any(s["status"] == "failed" for s in after["sections"])
    assert (run / "report.html").stat().st_size > 10_000
    import re

    def norm(h):
        h = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "UUID", h)
        h = re.sub(r"\d{4}-\d\d-\d\d[T ][\d:.]+", "TS", h)
        return re.sub(r"\b\d+(\.\d+)?s<br>", "Ns<br>", h)
    import difflib
    diff = [ln for ln in difflib.unified_diff(norm(html_b).replace(">", ">\n").split("\n"),
                                              norm(html_a).replace(">", ">\n").split("\n"),
                                              lineterm="", n=0)
            if not ln.startswith(("---", "+++", "@@"))]
    assert not diff, [d[:200] for d in diff[:6]]
    assert not (run / "activations.zarr").exists()


def test_pruned_run_removed_the_planted_caches(pruned_smoke):
    cfg, run, _, _, summary, before_bytes, _ = pruned_smoke
    kinds = {r["kind"] for r in summary["removed"]}
    assert kinds == {"activation_store", "sae_checkpoint"}
    assert sum(r["kind"] == "sae_checkpoint" for r in summary["removed"]) == 6
    assert summary["freed_bytes"] > 0
    assert (run / share.STORE_ATTRS_NAME).exists()


def test_pruned_run_rejects_l1_but_report_still_works(pruned_smoke):
    from tsfm_lens.pipeline import run_pipeline
    cfg, run, *_ = pruned_smoke
    with pytest.raises(ValueError, match="pruned on"):
        run_pipeline(cfg, stages=["l1"])
    assert not (run / "activations.zarr").exists()


def test_pruned_store_keeps_block_depth_axis_metadata(pruned_smoke):
    cfg, run, *_ = pruned_smoke
    store = share.open_store_or_stub(run)
    assert isinstance(store, share.PrunedStore)
    assert store.models() == ["patchy", "steppy"]
    assert store.stack_meta("patchy") and store.layers("patchy")
