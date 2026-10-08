"""The dual-null battery (ROADMAP.md sec 41.1): both nulls in one pass.

Known answers: the `wired` fixture of `test_profile_matched_null.py` (a linear
head reading one direction, a real atom, a dense leaky decoy). A dual-null run's
primary values must equal a single-null run of the primary byte for byte, and
`by_null[<other>]` must equal a single-null run of the other, because each mode
draws from its own generator seeded as a single-null run seeds it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_profile_matched_null import DICT, _run, wired  # noqa: E402,F401
from tsfm_lens.analysis import family_claims as fc  # noqa: E402
from tsfm_lens.sae import response as R  # noqa: E402

PRIMARY, SECOND = "profile_matched_cov", "profile_matched"


def _j(x):
    return json.dumps(x, sort_keys=True, default=float)


def _strip(cand):
    return {k: v for k, v in cand.items() if k not in ("by_null",)}


KW = dict(empirical_chance=True, keep_null_draws=True, keep_signed_null_draws=True)


@pytest.fixture
def runs(wired):
    dual = _run(wired, null_mode=PRIMARY, extra_null_modes=(SECOND,), **KW)
    one_p = _run(wired, null_mode=PRIMARY, **KW)
    one_s = _run(wired, null_mode=SECOND, **KW)
    return dual, one_p, one_s


def test_primary_legacy_keys_equal_a_single_null_run(runs):
    dual, one_p, _ = runs
    assert dual["ablation_null"] == PRIMARY and dual["ablation_nulls"] == [PRIMARY, SECOND]
    assert _j([_strip(c) for c in dual["candidates"]]) == _j([_strip(c) for c in one_p["candidates"]])
    for k in ("n_clearing_cells", "n_shape_clearing_cells", "empirical_chance", "feature_chance"):
        assert _j(dual[k]) == _j(one_p[k])


def test_secondary_block_equals_a_single_null_run_of_that_mode(runs):
    """Planted regression: sharing one generator across modes shifts the second
    mode's draws, so its block no longer equals the single-null run."""
    dual, _, one_s = runs
    got = 0
    for cd, cs in zip(dual["candidates"], one_s["candidates"]):
        blk = cd["by_null"][SECOND]
        if not cs.get("scorable"):
            assert blk["scorable"] is False
            continue
        got += 1
        for k in ("channels", "n_channels_clearing", "shape_channels", "n_shape_channels_clearing",
                  "feature_chance"):
            assert _j(blk[k]) == _j(cs[k]), k
    assert got > 0
    assert dual["by_null_summary"][SECOND]["n_clearing_cells"] == one_s["n_clearing_cells"]
    assert dual["by_null_summary"][PRIMARY]["n_clearing_cells"] == dual["n_clearing_cells"]
    assert _j(dual["by_null_summary"][SECOND]["empirical_chance"]) == _j(one_s["empirical_chance"])


def test_the_two_nulls_actually_differ(runs):
    dual, _, _ = runs
    c = dual["candidates"][1]
    assert c["channels"]["level"]["null_p95"] != c["by_null"][SECOND]["channels"]["level"]["null_p95"]


def test_real_ablation_forward_runs_once(wired):
    calls = {"n": 0}
    orig = wired.predict

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    wired.predict = counting
    _run(wired, null_mode=PRIMARY)
    single = calls["n"]
    calls["n"] = 0
    _run(wired, null_mode=PRIMARY, extra_null_modes=(SECOND,))
    dual = calls["n"]
    scored = sum(1 for c in _run(wired, null_mode=PRIMARY)["candidates"] if c.get("scorable"))
    assert dual - single == scored * 16, "one extra mode costs n_null forwards per scored feature, no more"


def test_reader_contract_candidate_for_null(runs):
    dual, one_p, _ = runs
    c = dual["candidates"][1]
    out = fc.candidate_for_null(c, SECOND, PRIMARY)
    assert out["channels"] == c["by_null"][SECOND]["channels"] and "by_null" not in out
    assert fc.candidate_for_null(c, None, PRIMARY) is c
    with pytest.raises(fc.NullModeUnavailable):
        fc.candidate_for_null(one_p["candidates"][1], SECOND, PRIMARY)
    for cand in dual["candidates"]:
        assert fc.candidate_for_null(cand, SECOND, PRIMARY)


def test_single_null_artifact_is_unchanged_by_the_extra_modes_argument(wired):
    a = _run(wired, null_mode=SECOND)
    b = _run(wired, null_mode=SECOND, extra_null_modes=())
    assert _j(a) == _j(b) and "by_null_summary" not in a
    assert all("by_null" not in c for c in a["candidates"])


def test_a_mean_magnitude_secondary_survives_a_profile_primary(wired):
    """The chunk-level null must not be overwritten by the per-feature draws."""
    dual = _run(wired, null_mode=SECOND, extra_null_modes=("mean_magnitude",))
    legacy = _run(wired, null_mode="mean_magnitude")
    n = 0
    for cd, cl in zip(dual["candidates"], legacy["candidates"]):
        if cl.get("scorable"):
            n += 1
            assert _j(cd["by_null"]["mean_magnitude"]["channels"]) == _j(cl["channels"])
    assert n > 0


def test_unknown_extra_mode_raises(wired):
    with pytest.raises(ValueError, match="nonsense"):
        _run(wired, extra_null_modes=("nonsense",))


def _draws(n, k, rng):
    return [np.abs(rng.normal(size=k)) for _ in range(n)]


def test_joint_chance_keeps_channel_dependence():
    """Two channels with IDENTICAL draws clear together: the joint rate equals one
    channel's rate, the union bound is twice it, the independent estimate sits
    between. Planted regression: summing the channels in the joint rate breaks
    the equality."""
    rng = np.random.default_rng(1)
    d = _draws(40, 1, rng)
    r = R._lodo_pseudo_clear_rate(d)
    assert 0 < r < 0.5
    assert R._lodo_joint_pseudo_clear_rate({"a": d, "b": list(d)}) == r
    ind = _draws(40, 1, np.random.default_rng(2))
    joint = R._lodo_joint_pseudo_clear_rate({"a": d, "b": ind})
    assert r <= joint <= r + R._lodo_pseudo_clear_rate(ind) + 1e-12
    per = {"a": {"available": True, "empirical_chance": r}, "b": {"available": True, "empirical_chance": r},
           "c": {"available": False}}
    rec = R.feature_chance_record(per, {"a": d, "b": list(d)})
    assert rec["p_union_bound"] == pytest.approx(2 * r)
    assert rec["p_independent"] == pytest.approx(1 - (1 - r) ** 2)
    assert rec["p_draw_level"] == r
    assert rec["p_draw_level"] < rec["p_independent"] < rec["p_union_bound"]


def test_joint_chance_refuses_misaligned_draw_lists():
    rng = np.random.default_rng(1)
    assert R._lodo_joint_pseudo_clear_rate({"a": _draws(10, 1, rng), "b": _draws(9, 1, rng)}) is None
    assert R._lodo_joint_pseudo_clear_rate({}) is None


def test_feature_chance_blocks_in_artifact(runs):
    dual, _, _ = runs
    fcb = dual["feature_chance"]
    n_sc = sum(1 for c in dual["candidates"] if c.get("scorable"))
    assert fcb["n_features"] == n_sc
    assert fcb["observed_clearing_ge1"] == sum(1 for c in dual["candidates"]
                                               if c.get("n_channels_clearing", 0) > 0)
    assert fcb["expected_independent"] <= fcb["expected_union_bound"] + 1e-9
    sec = dual["by_null_summary"][SECOND]["feature_chance"]
    assert sec["n_features"] == n_sc


def test_config_ablation_nulls():
    from tsfm_lens.config import config_from_dict, load_config
    from tsfm_lens.manifest import resolve_config_keys
    from tsfm_lens.pipeline import _stages
    from tsfm_lens.sae.ablation_run import cfg_extra_null_modes, cfg_null_mode
    cfg = load_config(str(ROOT / "configs" / "known_answer.yaml"))
    stage = {s.name: s for s in _stages()}["concepts"]
    assert "sae.ablation_nulls" in stage.config_keys
    assert "sae.ablation_nulls" not in resolve_config_keys(cfg, stage.config_keys)
    c = config_from_dict({"sae": {"ablation_nulls": [PRIMARY, SECOND]}})
    assert c.sae.ablation_null == PRIMARY and c.sae.ablation_nulls == (PRIMARY, SECOND)
    assert resolve_config_keys(c, stage.config_keys)["sae.ablation_nulls"] == (PRIMARY, SECOND)
    with pytest.raises(ValueError, match="primary"):
        config_from_dict({"sae": {"ablation_nulls": [PRIMARY, SECOND], "ablation_null": SECOND}})
    with pytest.raises(ValueError, match="unknown null"):
        config_from_dict({"sae": {"ablation_nulls": ["bogus"], "ablation_null": "bogus"}})
    assert cfg_null_mode(c) == PRIMARY and cfg_extra_null_modes(c) == [SECOND]


def test_a_silent_atom_still_carries_a_by_null_block(wired):
    """Decoy: atom 5 is silent on every series, so it is unscorable. A reader asking
    for the secondary null must still find a block (`scorable: False`) instead of
    `NullModeUnavailable`. Planted regression: dropping the block for unscorable
    candidates makes the contract reader raise on them."""
    from tests.test_profile_matched_null import Z_ALL, _Cfg, _Data, _SAE
    acts = Z_ALL.max(axis=1).astype(np.float64)
    acts[:, 5] = 0.0
    out = R.feature_ablation_fingerprints(
        _Cfg, wired, "blocks.0", _SAE(), _Data(), "cpu",
        candidates=[{"feature": f, "rules": []} for f in range(DICT)], activations=acts,
        top_k_series=3, n_null_directions=4, null_mode=PRIMARY, extra_null_modes=(SECOND,))
    silent = out["candidates"][5]
    assert silent["scorable"] is False
    assert fc.candidate_for_null(silent, SECOND, PRIMARY)["scorable"] is False
    assert out["candidates"][6]["by_null"][SECOND]["scorable"] is True
