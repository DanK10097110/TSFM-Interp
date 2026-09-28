"""Unknown config keys raise instead of being silently dropped
(ROADMAP.md sec 15 A21).

`config.py::_build` used to iterate the target dataclass's own fields and
copy across only the keys it recognized, discarding anything else without a
word -- so a misspelled knob (`capture_layer_stide`) ran the whole pipeline
at the default value while the YAML on disk, and `config_resolved.yaml`
copied next to the run, both read as though the setting were in force. These
tests pin the fix across a top-level section, a nested section (`l3.patching`),
and a `models[*]` entry -- the three shapes `config_from_dict` builds through.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config
from tsfm_lens.config import config_from_dict


def test_unknown_top_level_section_key_raises_with_suggestion():
    d = build_config(tempfile.mkdtemp())
    d["alignment"]["sanity_chek"] = True  # missing 'c'
    with pytest.raises(TypeError, match=r"sanity_chek.*sanity_check"):
        config_from_dict(d)


def test_unknown_key_with_no_close_match_still_raises():
    d = build_config(tempfile.mkdtemp())
    d["alignment"]["totally_unrelated_gibberish_xyz"] = 1
    with pytest.raises(TypeError, match="totally_unrelated_gibberish_xyz"):
        config_from_dict(d)


def test_unknown_key_in_l3_patching_raises():
    d = build_config(tempfile.mkdtemp())
    d["l3"]["patching"]["windo_stride"] = 2  # typo for window_stride
    with pytest.raises(TypeError, match=r"l3\.patching.*windo_stride"):
        config_from_dict(d)


def test_unknown_key_in_l3_own_section_raises():
    d = build_config(tempfile.mkdtemp())
    d["l3"]["max_seriess"] = 10  # typo for max_series
    with pytest.raises(TypeError, match="max_seriess"):
        config_from_dict(d)


def test_unknown_key_in_a_models_entry_raises_and_names_the_index():
    d = build_config(tempfile.mkdtemp())
    d["models"][1]["batch_siz"] = 8  # typo for batch_size
    with pytest.raises(TypeError, match=r"models\[1\].*batch_siz"):
        config_from_dict(d)


def test_valid_config_with_every_declared_key_still_builds():
    """The fix must not reject anything actually declared."""
    d = build_config(tempfile.mkdtemp())
    cfg = config_from_dict(d)
    assert cfg.l3.patching.window_stride == 1
    assert cfg.models[1].batch_size == 64


def test_none_section_still_falls_back_to_defaults():
    """`key: null` (or an absent section) must keep working -- only actual
    unrecognized keys are an error, not an empty/absent section."""
    d = build_config(tempfile.mkdtemp())
    d["l3"] = None
    cfg = config_from_dict(d)
    assert cfg.l3.max_series == 512  # class default, not build_config's override


def test_non_mapping_section_raises_a_clear_type_error():
    d = build_config(tempfile.mkdtemp())
    d["alignment"] = "not-a-mapping"
    with pytest.raises(TypeError, match=r"alignment.*mapping"):
        config_from_dict(d)
