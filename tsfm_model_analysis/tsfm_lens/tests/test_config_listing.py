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


def test_a_config_outside_the_schema_is_listed_with_a_note_not_dropped():
    """`scaling_ladder_chronos.yaml`'s `ladder:` block is deliberately not in
    the config schema, so `load_config` rejects it. A listing that silently
    omitted the configs needing the most explanation would be worse than none,
    which is why the summary parses with `yaml.safe_load`."""
    row = _config_summary(CONFIGS / "scaling_ladder_chronos.yaml")
    assert row["name"] == "scaling_ladder_chronos"
    assert "--emit-configs" in row["note"]


def test_listing_covers_every_yaml_in_the_directory():
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_config_listing(CONFIGS)
    out = buf.getvalue()
    names = sorted(p.stem for p in CONFIGS.glob("*.yaml"))
    assert f"{len(names)} configs" in out
    missing = [n for n in names if n not in out]
    assert not missing, f"configs absent from the listing: {missing}"


def test_disabled_stages_are_reported_from_the_config_itself():
    """The "off:" annotation is what tells a reader that e.g. a crosscoder
    config is extract-only -- read from its own `enabled: false` flags, so it
    cannot describe a stage list the file does not have."""
    row = _config_summary(CONFIGS / "crosscoder_stage0.yaml")
    assert row["stages"].startswith("off: ")
    assert "report" in row["stages"]
    assert _config_summary(CONFIGS / "smoke_panel.yaml")["stages"] == ""
