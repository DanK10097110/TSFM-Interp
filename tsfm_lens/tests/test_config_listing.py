"""`run.py --list-configs` (the config-sprawl entry point).

Forty-six flat YAML files with no index is a real cost for someone who did not
write them, and the obvious fix -- a hand-maintained `configs/README.md` --
is the shape of claim this repo has repeatedly had to correct as stale
(`CLAUDE.md` sec 11.34). So the listing is DERIVED from the files, and these
tests pin the derivations rather than any particular rendered text.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.run import _config_summary, _print_config_listing

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_shape_is_derived_from_model_count_not_from_the_filename():
    """`smoke_three_model.yaml` is a panel because it declares three models.

    The load-bearing negative is the pair: a config whose NAME says nothing
    about its shape must still be classified, and a config named for one
    shape must not be believed over its own contents.
    """
    assert _config_summary(CONFIGS / "smoke_solo.yaml")["shape"] == "solo"
    assert _config_summary(CONFIGS / "smoke.yaml")["shape"] == "pair"
    assert _config_summary(CONFIGS / "smoke_three_model.yaml")["shape"] == "panel"
    assert _config_summary(CONFIGS / "smoke_panel.yaml")["shape"] == "panel"


def test_purpose_sentence_does_not_truncate_at_a_dotted_filename():
    """The first version split on ANY period, so every header citing
    "ROADMAP.md" rendered as "Phase 4 (ROADMAP" -- a listing that made the
    configs referencing the roadmap the least legible ones."""
    purpose = _config_summary(CONFIGS / "smoke_panel.yaml")["purpose"]
    assert "ROADMAP.md" in purpose
    assert not purpose.endswith("(ROADMAP")


def test_a_config_outside_the_schema_is_listed_with_a_note_not_dropped(tmp_path):
    """A config with a `ladder:` block (like the dev-branch-only
    `scaling_ladder_chronos.yaml`) is deliberately not in the config schema,
    so `load_config` rejects it. A listing that silently omitted the configs
    needing the most explanation would be worse than none, which is why the
    summary parses with `yaml.safe_load` rather than the real loader.

    Built as a synthetic fixture (rather than loading the real
    `scaling_ladder_chronos.yaml`) so this test does not require that
    dev-branch-only config to exist on `main`; `_config_summary` takes any
    path, not just one inside `configs/`."""
    cfg = tmp_path / "scaling_ladder_fake.yaml"
    cfg.write_text(
        "# A scaling-ladder sweep seed (ROADMAP.md reference).\n"
        "ladder:\n"
        "  sizes: [1, 2, 3]\n",
        encoding="utf-8")
    row = _config_summary(cfg)
    assert row["name"] == "scaling_ladder_fake"
    assert "--emit-configs" in row["note"]


def test_listing_covers_every_yaml_in_the_directory():
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_config_listing(CONFIGS)
    out = buf.getvalue()
    names = sorted(p.stem for p in CONFIGS.glob("*.yaml"))
    example_names = sorted(p.stem for p in (CONFIGS / "examples").glob("*.yaml"))
    assert f"{len(names) + len(example_names)} configs" in out
    missing = [n for n in names if n not in out]
    assert not missing, f"configs absent from the listing: {missing}"


def test_examples_subdirectory_configs_are_listed_with_a_prefix():
    """`configs/examples/` holds the curated, doc-verified starting points
    (`configs/examples/README.md`). They must show up in `--list-configs`
    too -- a listing that only covers the ~50 dev configs and silently
    excludes the ones a newcomer is pointed at would defeat the point of
    curating them -- and with an `examples/` prefix so the printed name is
    the actual `--config` path to pass."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_config_listing(CONFIGS)
    out = buf.getvalue()
    assert "examples/short_1model" in out
    assert "examples/smoke_mock" in out


def test_ladder_subdirectory_is_still_excluded(tmp_path):
    """A `_ladder/` subdirectory of machine-expanded scaling-ladder configs
    (like the dev-branch-only `configs/_ladder/`) must never show up in the
    listing; adding `examples/` must not accidentally widen the scan to every
    subdirectory.

    Built as a synthetic `tmp_path` config directory (rather than depending on
    the real, dev-branch-only `configs/_ladder/`) so this test does not
    require that directory to exist on `main`. The plant that justifies this
    test: making `_print_config_listing` recurse (`config_dir.rglob` instead
    of `config_dir.glob`) makes it fail, confirmed by hand while writing it."""
    (tmp_path / "_ladder").mkdir()
    (tmp_path / "_ladder" / "ladder_expanded_size7.yaml").write_text(
        "run: {name: ladder_expanded_size7}\n", encoding="utf-8")
    (tmp_path / "normal_config.yaml").write_text("run: {name: normal_config}\n", encoding="utf-8")

    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_config_listing(tmp_path)
    out = buf.getvalue()
    assert "normal_config" in out
    assert "ladder_expanded_size7" not in out


def test_disabled_stages_are_reported_from_the_config_itself(tmp_path):
    """The "off:" annotation is what tells a reader that e.g. an extract-only
    config has a narrowed stage list -- read from its own `enabled: false`
    flags, so it cannot describe a stage list the file does not have.

    Built as a synthetic fixture (rather than loading the real, dev-branch-
    only `configs/crosscoder_stage0.yaml`) so this test does not require that
    config to exist on `main`."""
    cfg = tmp_path / "extract_only.yaml"
    cfg.write_text(
        "run: {name: extract_only}\n"
        "l0: {enabled: false}\n"
        "l1: {enabled: false}\n"
        "report: {enabled: false}\n",
        encoding="utf-8")
    row = _config_summary(cfg)
    assert row["stages"].startswith("off: ")
    assert "report" in row["stages"]
    assert _config_summary(CONFIGS / "smoke_panel.yaml")["stages"] == ""
