"""The committed Stage 0 winner config (ROADMAP.md §6.2.1's fourth exit criterion).

Stage 0's exit criteria require the winning configuration to live in a config
file rather than in a session log's CLI line. A config file only satisfies
that if it is *executable* and *faithful* -- a YAML nobody parses is a note,
not a configuration. These tests check both properties without loading a
checkpoint or touching the GPU: that `run_crosscoder_stage0.py` turns the file
into exactly the sweep row the recorded numbers were measured with, and that
the numbers recorded inside the file are the ones the gate actually scored.

The `expected:` block is checked against the run artifact rather than trusted,
because `CLAUDE.md` §11.24's lesson is precisely that a config's *meaning* can
drift while its bytes stay identical. When the artifact is absent (a fresh
clone has no `runs/`), that check skips rather than failing -- it is a
consistency check on this machine's record, not a claim a stranger can verify.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs" / "crosscoder_stage0_winner.yaml"


def _harness():
    """Import `run_crosscoder_stage0.py` by path -- it is a CLI script at the
    package root, not an importable module of `tsfm_lens`. Registering it in
    `sys.modules` before executing matters: `dataclasses` resolves a class's
    annotations through `sys.modules[cls.__module__]`, so `Row` fails to build
    without it."""
    spec = importlib.util.spec_from_file_location(
        "run_crosscoder_stage0", ROOT / "run_crosscoder_stage0.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_crosscoder_stage0"] = module
    spec.loader.exec_module(module)
    return module


def _params() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_winner_config_parses_into_the_row_the_numbers_were_measured_with():
    row = _harness().row_from_params(_params(), label="winner")
    assert row.k == 48, "k=48 is the smallest k that cleared the gate; see §6.2.1"
    assert row.dict_size == 1280
    assert row.aux_k == 64
    assert row.aux_coef == pytest.approx(0.03125)
    assert row.epochs == 60
    assert row.k_is_swept, "the config owns its k; a stray --k must not override it"


def test_winner_config_names_the_layer_pair_and_store_it_was_measured_on():
    """A config that omits where it was measured is not reproducible. Paths
    stay relative per `CLAUDE.md` invariant 11."""
    params = _params()
    assert params["run"] == "runs/crosscoder_stage0_bigdata"
    assert not Path(params["run"]).is_absolute()
    assert params["layer_a"] == "stacked_xf.4"
    assert params["layer_b"] == "encoder.block.10"
    assert params["eff_dims"] == [28.15, 13.93]


def test_row_from_params_rejects_an_unknown_key_instead_of_ignoring_it():
    """A typo must fail loudly rather than silently run a different
    experiment (`CLAUDE.md` §2.5)."""
    bad = {"train": {"k": 48, "dictionary_size": 1280}}
    with pytest.raises(ValueError, match="dictionary_size"):
        _harness().row_from_params(bad)


def test_recorded_expectations_match_the_measured_artifact():
    """The `expected:` block must be the sweep's real output, not a summary
    someone retyped."""
    artifact = ROOT / "runs" / "crosscoder_stage0_bigdata" / "crosscoder_stage0_ksweep.json"
    if not artifact.exists():
        pytest.skip("k-sweep artifact not present in this checkout")
    import json

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    params = _params()
    label = f"k={params['train']['k']}:aux_k=64,coef=0.03125"
    row = next(r for r in payload["rows"] if r["label"] == label)
    exp, cross = params["expected"], row["crosscoder"]
    fid = cross["fidelity"]
    assert fid[payload["model_a"]] == pytest.approx(exp["crosscoder"]["fidelity_a"])
    assert fid[payload["model_b"]] == pytest.approx(exp["crosscoder"]["fidelity_b"])
    assert cross["dead_feature_rate"] == pytest.approx(exp["crosscoder"]["dead_feature_rate"])
    assert cross["n_alive"] == exp["crosscoder"]["n_alive"]
    assert cross["l0_actual"] == pytest.approx(exp["crosscoder"]["l0_actual"])
    assert cross["verdict"]["passes"], "the committed winner must be a passing row"
    base = row["baseline"]
    assert base[payload["model_a"]]["verdict"]["passes"] is exp["baseline_a_passes"]
    assert base[payload["model_b"]]["verdict"]["passes"] is exp["baseline_b_passes"]
