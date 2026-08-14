"""The committed config that clears Stage 0 (ROADMAP.md §6.2.1 finding (17)).

`test_stage0_winner_config.py` pins the same two properties for the k sweep's
winner -- executable, and faithful to the artifact. This file does the same for
the configuration that actually satisfies all three artifacts at once, and adds
the property that only became load-bearing with this run: a gate verdict of
"passes" is a claim about *five seeds*, not about the one seed the config
commits. A config recording only seed 0 would read as a 1/1 result, so the
five-seed bounds are recorded too and checked against every seed's artifact.

The `expected:` block is checked rather than trusted, per `CLAUDE.md` §11.24 --
a config's meaning can drift while its bytes stay identical. When the artifacts
are absent (a fresh clone has no `runs/`), those checks skip: they are a
consistency check on this machine's record, not a claim a stranger can verify.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs" / "crosscoder_stage0_gate.yaml"
ARTIFACTS = ROOT / "runs" / "crosscoder_stage0_bigdata"


def _harness():
    spec = importlib.util.spec_from_file_location(
        "run_crosscoder_stage0", ROOT / "run_crosscoder_stage0.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_crosscoder_stage0"] = module
    spec.loader.exec_module(module)
    return module


def _params() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _seed_rows(dict_size: int = 1024) -> list:
    """Every seed's `gate:dict=<n>` row, or an empty list if unrun here."""
    rows = []
    for f in sorted(ARTIFACTS.glob("gate_seed*.json")):
        payload = json.loads(f.read_text(encoding="utf-8"))
        row = next((r for r in payload["rows"]
                    if r["label"] == f"gate:dict={dict_size}"), None)
        if row is not None:
            rows.append((payload, row))
    return rows


def test_gate_config_parses_into_the_row_the_numbers_were_measured_with():
    row = _harness().row_from_params(_params(), label="gate")
    assert row.dict_size == 1024, "896 passed at only 3 of 5 seeds; see finding (17)"
    assert row.k == 48
    assert row.aux_k == 64
    assert row.aux_coef == pytest.approx(0.03125)
    assert row.epochs == 60
    assert row.k_is_swept, "the config owns its k; a stray --k must not override it"


def test_baseline_sizes_are_top_level_so_they_reach_the_exit_criteria_record():
    """`row_from_params` reads `train:`, but `main()` reads the *top level* for
    the sizing that lands in the artifact's `exit_criteria`. Under `train:`
    alone the row would train correctly while the stored record claimed
    `baseline_sizing: matched` -- an artifact that misreports which rule scored
    it (`CLAUDE.md` §11.24)."""
    params = _params()
    assert params["baseline_dict_sizes"] == [576, 512]
    assert "baseline_dict_sizes" not in params["train"]


def test_gate_config_names_the_layer_pair_and_store_it_was_measured_on():
    params = _params()
    assert params["run"] == "runs/crosscoder_stage0_bigdata"
    assert not Path(params["run"]).is_absolute()
    assert params["layer_a"] == "stacked_xf.4"
    assert params["layer_b"] == "encoder.block.10"
    assert params["eff_dims"] == [28.15, 13.93]


def test_the_committed_baseline_sizes_are_the_strict_ones():
    """The sizing decision only holds if the sizes are not chosen for ease.
    TimesFM's 576 is the smallest measured size above its own 563 alive floor,
    so it must keep 97.7% of its atoms alive -- a stricter bar than the 0.30
    dead-rate ceiling imposes at any larger size."""
    m = _harness()
    tf_dict, ch_dict = _params()["baseline_dict_sizes"]
    tf_floor = m.min_alive_for(28.15)
    assert tf_dict > tf_floor
    assert tf_floor / tf_dict > (1.0 - m.MAX_DEAD_RATE), \
        "the alive floor, not the dead-rate bar, is what binds TimesFM at 576"
    assert ch_dict >= m.min_alive_for(13.93)


def test_recorded_seed0_expectations_match_the_measured_artifact():
    rows = _seed_rows()
    if not rows:
        pytest.skip("gate sweep artifacts not present in this checkout")
    payload, row = next(((p, r) for p, r in rows if p["seed"] == _params()["seed"]),
                        (None, None))
    if row is None:
        pytest.skip("the committed seed was not among the artifacts")
    exp, cross = _params()["expected"], row["crosscoder"]
    assert cross["fidelity"][payload["model_a"]] == pytest.approx(exp["crosscoder"]["fidelity_a"])
    assert cross["fidelity"][payload["model_b"]] == pytest.approx(exp["crosscoder"]["fidelity_b"])
    assert cross["dead_feature_rate"] == pytest.approx(exp["crosscoder"]["dead_feature_rate"])
    assert cross["n_alive"] == exp["crosscoder"]["n_alive"]
    assert cross["l0_actual"] == pytest.approx(exp["crosscoder"]["l0_actual"])
    assert cross["verdict"]["passes"], "the committed config must be a passing row"
    base = row["baseline"]
    assert base[payload["model_a"]]["verdict"]["passes"] is exp["baseline_a_passes"]
    assert base[payload["model_b"]]["verdict"]["passes"] is exp["baseline_b_passes"]
    assert row["baseline_sizing"] == "own", "scored under the gate, not the control"


def test_the_five_seed_claim_is_the_one_the_artifacts_support():
    """The gate verdict rests on every seed passing, so the config's recorded
    bounds have to be the real worst case across seeds -- not seed 0's numbers
    relabelled."""
    rows = _seed_rows()
    if len(rows) < 5:
        pytest.skip(f"need 5 gate seeds, found {len(rows)}")
    exp = _params()["expected"]["five_seed"]
    crosses = [r["crosscoder"] for _, r in rows]
    assert sum(c["verdict"]["passes"] for c in crosses) == exp["n_passing"] == len(rows)
    assert max(c["dead_feature_rate"] for c in crosses) == \
        pytest.approx(exp["dead_feature_rate_max"])
    assert min(c["n_alive"] for c in crosses) == exp["n_alive_min"]
    assert min(min(c["fidelity"].values()) for c in crosses) == \
        pytest.approx(exp["fidelity_min"])
    # Every bound must still clear the bar it is measured against, or the
    # recorded worst case would be documenting a failure as a pass.
    m = _harness()
    assert exp["dead_feature_rate_max"] <= m.MAX_DEAD_RATE
    assert exp["fidelity_min"] >= m.MIN_FIDELITY
    assert exp["n_alive_min"] >= m.min_alive_for((28.15, 13.93))


def test_the_smaller_dictionary_is_recorded_as_failing_rather_than_omitted():
    """896 is the size the single-seed record made look best. It passes at 3 of
    5 seeds, and on fidelity alone -- the config's comment says so, and this
    pins that claim to the artifacts so the comment cannot go quietly stale."""
    rows = _seed_rows(896)
    if len(rows) < 5:
        pytest.skip(f"need 5 gate seeds, found {len(rows)}")
    crosses = [r["crosscoder"] for _, r in rows]
    assert sum(c["verdict"]["passes"] for c in crosses) == 3
    for c in crosses:
        if not c["verdict"]["passes"]:
            assert c["verdict"]["fidelity_ok"] is False
            assert c["verdict"]["dead_ok"] and c["verdict"]["alive_ok"]
