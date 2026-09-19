"""Tests for `scaffold_adapter.py` / `run.py --new-adapter` (`ROADMAP.md` sec
34.6 Item E3).

Unit tests exercise the pure rendering/validation functions against tmp
directories, never the real `models/contrib/`/`configs/` -- except the one
integration test at the bottom, which is the load-bearing negative for the
whole item: it scaffolds a REAL throwaway adapter via the real CLI, runs the
five printed commands' first two (the ones E3's own acceptance criterion
says must succeed right now, against the pristine tier-0 scaffold) via
subprocess, and confirms a report actually renders -- then removes every
file and directory it created, in a `finally` block, so the repo is left
exactly as it found it (the same discipline `test_adapter_docs.py`'s
T-E2.2 negative already established for this package).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tsfm_lens.scaffold_adapter import (
    ScaffoldError, class_name_for, command_plan, render_adapter_source,
    render_config_source, scaffold, validate_name,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]  # tsfm_lens/ (holds run.py, configs/)


def test_class_name_for_maps_snake_case_to_camel_case_plus_adapter_suffix():
    assert class_name_for("my_model") == "MyModelAdapter"
    assert class_name_for("timer") == "TimerAdapter"


def test_class_name_for_does_not_double_the_adapter_suffix():
    # A contributor typing `my_model_adapter` as the NAME should not get
    # `MyModelAdapterAdapter` -- ugly, and a plausible first guess a
    # contributor would make.
    assert class_name_for("my_model_adapter") == "MyModelAdapter"


def test_validate_name_rejects_non_identifier_naming_why():
    with pytest.raises(ScaffoldError, match="not a valid Python identifier"):
        validate_name("my-model", known_adapters=set())


def test_validate_name_rejects_a_python_keyword():
    with pytest.raises(ScaffoldError, match="not a valid Python identifier"):
        validate_name("class", known_adapters=set())


def test_validate_name_rejects_a_digit_leading_name():
    with pytest.raises(ScaffoldError, match="not a valid Python identifier"):
        validate_name("2fast", known_adapters=set())


def test_validate_name_rejects_a_name_already_registered_naming_the_registry():
    with pytest.raises(ScaffoldError, match="already registered") as exc_info:
        validate_name("mock_patch", known_adapters={"mock_patch", "timesfm"})
    # The load-bearing negative for THIS check specifically: the error must
    # name what's already registered, not just say "no" (CLAUDE.md sec
    # 11.43 -- an error message is a claim about the code and must be
    # useful, not just present).
    assert "mock_patch" in str(exc_info.value)


def test_validate_name_accepts_a_free_valid_name():
    validate_name("brand_new_model", known_adapters={"mock_patch"})  # must not raise


def test_render_adapter_source_substitutes_name_class_and_checkpoint():
    source = render_adapter_source("timer2", "org/timer2-ckpt", "Timer2Adapter")
    assert 'ADAPTER_NAME = "timer2"' in source
    assert 'ADAPTER_CLASS = "Timer2Adapter"' in source
    assert "class Timer2Adapter(MockBlackBoxAdapter):" in source
    assert "org/timer2-ckpt" in source
    # The template's own placeholder name/class must be fully gone, not just
    # supplemented -- a leftover "TemplateAdapter" would silently ship a
    # dead class alongside the real one.
    assert "TemplateAdapter" not in source
    assert 'ADAPTER_NAME = "template"' not in source


def test_render_adapter_source_fails_loudly_if_the_template_text_drifts(monkeypatch):
    """Load-bearing negative: a substitution that silently no-ops must not ship.

    Points the module at a stand-in "template" missing one of the four
    expected literal strings and confirms `render_adapter_source` refuses
    rather than returning a file that still says `ADAPTER_CLASS =
    "TemplateAdapter"` under the hood.
    """
    import tsfm_lens.scaffold_adapter as scaffold_adapter_mod

    fake_template = scaffold_adapter_mod._PKG_DIR / "_test_e3_fake_template.py"
    fake_template.write_text(
        'ADAPTER_NAME = "not_the_real_marker"\n', encoding="utf-8")
    try:
        monkeypatch.setattr(scaffold_adapter_mod, "TEMPLATE_PATH", fake_template)
        with pytest.raises(ScaffoldError, match="out of sync with the template"):
            render_adapter_source("foo", "org/ckpt", "FooAdapter")
    finally:
        fake_template.unlink(missing_ok=True)


def test_render_config_source_pairs_the_new_adapter_against_mock_patch():
    text = render_config_source("timer2", "org/timer2-ckpt")
    assert "adapter: mock_patch" in text
    assert "adapter: timer2" in text
    assert 'checkpoint: "org/timer2-ckpt"' in text
    assert "name: smoke_timer2" in text


def test_render_config_source_has_no_absolute_paths():
    # CLAUDE.md invariant 11: no config field may hardcode an absolute path.
    text = render_config_source("timer2", "org/timer2-ckpt")
    assert "out_dir: runs" in text
    assert "/common/" not in text
    assert not any(line.strip().startswith("/") for line in text.splitlines())


def test_command_plan_has_five_commands_referencing_the_right_name_and_config():
    commands = command_plan("timer2")
    assert len(commands) == 5
    for i, cmd in enumerate(commands, start=1):
        assert cmd.startswith(f"{i}.")
        assert "configs/smoke_timer2.yaml" in cmd
        assert " timer2" in cmd or "timer2" in cmd


def test_command_plan_orders_checklist_before_alignment_and_spans():
    # E4.2's own dependency order: construct/checklist first, span/alignment
    # checks only meaningful once tier 1 is real.
    commands = command_plan("timer2")
    idx = {}
    for i, cmd in enumerate(commands):
        for name in ("--check-adapter", "--discover-layers", "--check-alignment"):
            if name in cmd and name not in idx:
                idx[name] = i  # first occurrence only -- --check-adapter recurs at the end
    assert idx["--check-adapter"] < idx["--discover-layers"] < idx["--check-alignment"]


def test_scaffold_writes_both_files_under_tmp_dirs(tmp_path):
    contrib_dir = tmp_path / "contrib"
    configs_dir = tmp_path / "configs"
    result = scaffold("timer2", "org/timer2-ckpt", known_adapters=set(),
                      contrib_dir=contrib_dir, configs_dir=configs_dir)
    assert (contrib_dir / "timer2_adapter.py").exists()
    assert (configs_dir / "smoke_timer2.yaml").exists()
    assert result.class_name == "Timer2Adapter"
    assert result.adapter_rel_path == "models/contrib/timer2_adapter.py"
    assert result.config_rel_path == "configs/smoke_timer2.yaml"
    assert len(result.commands) == 5


def test_scaffold_refuses_to_overwrite_an_existing_adapter_file_naming_it(tmp_path):
    contrib_dir = tmp_path / "contrib"
    configs_dir = tmp_path / "configs"
    contrib_dir.mkdir(parents=True)
    existing = contrib_dir / "timer2_adapter.py"
    existing.write_text("# already here\n", encoding="utf-8")
    with pytest.raises(ScaffoldError, match="refusing to overwrite") as exc_info:
        scaffold("timer2", "org/timer2-ckpt", known_adapters=set(),
                contrib_dir=contrib_dir, configs_dir=configs_dir)
    assert str(existing) in str(exc_info.value)
    # Untouched, not silently regenerated:
    assert existing.read_text(encoding="utf-8") == "# already here\n"


def test_scaffold_refuses_to_overwrite_an_existing_config_and_writes_neither_file(tmp_path):
    """Load-bearing negative for the "never a partial write" claim.

    Pre-creates only the CONFIG half of the pair (the adapter file does not
    exist yet) and confirms `scaffold` still refuses -- and, critically,
    that it does not go ahead and write the adapter file anyway before
    hitting the config collision.
    """
    contrib_dir = tmp_path / "contrib"
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True)
    (configs_dir / "smoke_timer2.yaml").write_text("# already here\n", encoding="utf-8")
    with pytest.raises(ScaffoldError, match="refusing to overwrite"):
        scaffold("timer2", "org/timer2-ckpt", known_adapters=set(),
                contrib_dir=contrib_dir, configs_dir=configs_dir)
    assert not (contrib_dir / "timer2_adapter.py").exists()


def test_scaffold_refuses_a_name_collision_before_writing_anything(tmp_path):
    contrib_dir = tmp_path / "contrib"
    configs_dir = tmp_path / "configs"
    with pytest.raises(ScaffoldError, match="already registered"):
        scaffold("mock_patch", "org/ckpt", known_adapters={"mock_patch"},
                contrib_dir=contrib_dir, configs_dir=configs_dir)
    assert not contrib_dir.exists() or not any(contrib_dir.iterdir())
    assert not configs_dir.exists() or not any(configs_dir.iterdir())


def test_scaffold_refuses_an_invalid_identifier_before_writing_anything(tmp_path):
    contrib_dir = tmp_path / "contrib"
    configs_dir = tmp_path / "configs"
    with pytest.raises(ScaffoldError, match="not a valid Python identifier"):
        scaffold("my-model", "org/ckpt", known_adapters=set(),
                contrib_dir=contrib_dir, configs_dir=configs_dir)
    assert not contrib_dir.exists() or not any(contrib_dir.iterdir())
    assert not configs_dir.exists() or not any(configs_dir.iterdir())


def test_scaffolded_adapter_file_is_syntactically_valid_python(tmp_path):
    # A defensive smoke check independent of the integration test below:
    # the generated file must at least compile, regardless of environment.
    contrib_dir = tmp_path / "contrib"
    configs_dir = tmp_path / "configs"
    result = scaffold("timer2", "org/timer2-ckpt", known_adapters=set(),
                      contrib_dir=contrib_dir, configs_dir=configs_dir)
    source = (contrib_dir / "timer2_adapter.py").read_text(encoding="utf-8")
    compile(source, str(contrib_dir / "timer2_adapter.py"), "exec")


# ---------------------------------------------------------------------------
# Integration test against the REAL repo layout and the REAL CLI -- E3's own
# acceptance criterion, run for real rather than asserted.
# ---------------------------------------------------------------------------

def test_new_adapter_cli_scaffolds_and_reaches_a_rendered_report():
    """E3's acceptance criterion: scaffold, then run the printed commands
    against a mock checkpoint substitute and reach a report.

    Runs command 1 (`--check-adapter`, must pass at tier 0) and command 2
    (the full pipeline run) exactly as `--new-adapter` prints them, via a
    real subprocess against the real `run.py`. Commands 3-5 are correctly
    NOT run here -- they are labelled "once tier 1/N is real" and this
    adapter never leaves tier 0, so running them would test something the
    scaffolder never claimed.
    """
    name = "_test_e3_throwaway"
    contrib_dir = _REPO_ROOT / "tsfm_lens" / "models" / "contrib"
    configs_dir = _REPO_ROOT / "configs"
    adapter_file = contrib_dir / f"{name}_adapter.py"
    config_file = configs_dir / f"smoke_{name}.yaml"
    run_dir = _REPO_ROOT / "runs" / f"smoke_{name}"
    run_py = str(_REPO_ROOT / "run.py")

    def run(args):
        return subprocess.run([sys.executable, run_py] + args, cwd=str(_REPO_ROOT),
                              capture_output=True, text=True, timeout=180)

    try:
        scaffold_run = run(["--new-adapter", name, "--checkpoint", "org/fake-ckpt"])
        assert scaffold_run.returncode == 0, scaffold_run.stdout + scaffold_run.stderr
        assert adapter_file.exists()
        assert config_file.exists()
        out = scaffold_run.stdout
        assert f"wrote models/contrib/{name}_adapter.py" in out
        assert f"wrote configs/smoke_{name}.yaml" in out
        assert "run these next, in order" in out

        # Re-running with the same name must refuse (E3's own failure mode),
        # not silently regenerate over the file just written. Which of the
        # two refusal messages fires depends on whether the fresh subprocess's
        # own `import tsfm_lens.models` re-discovers the contrib file the
        # first invocation just wrote (making it "already registered" via
        # validate_name's collision check) before the file-existence check in
        # `scaffold()` ever runs -- both are correct, and either is fine here.
        rescaffold = run(["--new-adapter", name, "--checkpoint", "org/fake-ckpt"])
        assert rescaffold.returncode != 0
        combined = rescaffold.stdout + rescaffold.stderr
        assert "refusing to overwrite" in combined or "already registered" in combined

        # Command 1: --check-adapter must pass right now, at tier 0.
        check = run(["--config", f"configs/smoke_{name}.yaml",
                    "--check-adapter", name])
        assert check.returncode == 0, check.stdout + check.stderr
        assert "overall: pass" in check.stdout

        # Command 2: the full pipeline must reach a rendered report naming
        # this adapter -- "a contributor's first hour should end at a
        # rendered HTML, not at a stack trace" (E3's own acceptance text).
        pipeline = run(["--config", f"configs/smoke_{name}.yaml", "--no-preflight"])
        assert pipeline.returncode == 0, pipeline.stdout + pipeline.stderr
        report_path = run_dir / "report.html"
        assert report_path.exists(), pipeline.stdout + pipeline.stderr
        report_html = report_path.read_text(encoding="utf-8")
        assert name in report_html
    finally:
        adapter_file.unlink(missing_ok=True)
        config_file.unlink(missing_ok=True)
        cache = contrib_dir / "__pycache__"
        if cache.exists():
            for pyc in cache.glob(f"{name}_adapter*.pyc"):
                pyc.unlink(missing_ok=True)
        if run_dir.exists():
            import shutil
            shutil.rmtree(run_dir)
