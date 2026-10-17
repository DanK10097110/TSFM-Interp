"""Re-measurement driver for the paper's four-model numbers (ROADMAP.md sec 47.5): planted known answers with decoys."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_dagger_remeasure as rdr  # noqa: E402
from tsfm_lens.sae.concepts import CHANNELS  # noqa: E402

LAYER = "encoder.block.3"


def _channels(signed: list, p95: float = 1.0) -> dict:
    return {ch: {"available": True, "effect": abs(float(v)), "signed_effect": float(v), "null_p95": p95,
                 "clears_null": bool(abs(v) > p95)} for ch, v in zip(CHANNELS, signed)}


def _cand(feature: int, signed: list, n_clear: int, scorable: bool = True, forecasts=None, by_null=None,
          feature_chance=None) -> dict:
    c = {"feature": feature, "scorable": scorable, "channels": _channels(signed), "n_channels_clearing": n_clear,
         "forecasts": forecasts or []}
    if by_null is not None:
        c["by_null"] = by_null
    if feature_chance is not None:
        c["feature_chance"] = feature_chance
    return c


def _write(run: Path, model: str, art: dict) -> None:
    d = run / "sae" / model
    d.mkdir(parents=True, exist_ok=True)
    (d / (rdr_sanitize(art["layer"]) + "_ablation.json")).write_text(json.dumps(art), encoding="utf-8")


def rdr_sanitize(layer: str) -> str:
    return layer.replace(".", "_")


def _panel(delta_ramp, recon: float, sd_scale: float = 1.0) -> dict:
    t = np.arange(40)
    ctx = sd_scale * np.where(t % 2 == 0, 1.0, -1.0)
    base = np.linspace(0, 1, 8)
    with_f = base
    return {"context": ctx.tolist(), "with_feature": with_f.tolist(),
            "without_feature": (with_f + np.asarray(delta_ramp)).tolist(),
            "unpatched": (with_f + recon).tolist()}


def test_mn15_uses_max_over_horizon_not_mean(tmp_path):
    ramp = np.linspace(0.0, 0.1, 8)
    flat = _panel(ramp, 0.2)
    zero_sd = _panel(ramp, 0.2)
    zero_sd["context"] = [3.0] * 40
    no_without = _panel(ramp, 0.2)
    del no_without["without_feature"]
    cand = _cand(1, [0] * 9, 0, forecasts=[flat, zero_sd, no_without])
    _write(tmp_path, "M", {"model": "M", "layer": LAYER, "candidates": [cand]})
    r = rdr.mn15(tmp_path)
    sd = float(np.std(np.asarray(flat["context"])))
    assert r["n_panels"] == 1
    assert r["median_effect_sd"] == pytest.approx(0.1 / sd)
    assert r["median_recon_sd"] == pytest.approx(0.2 / sd)
    assert r["recon_over_effect"] == pytest.approx(2.0)
    assert r["share_effect_below_tenth_of_recon"] == 0.0


def test_mn15_share_below_tenth(tmp_path):
    small = _panel(np.full(8, 0.01), 0.5)
    big = _panel(np.full(8, 0.4), 0.5)
    _write(tmp_path, "M", {"model": "M", "layer": LAYER, "candidates": [_cand(1, [0] * 9, 0, forecasts=[small, big])]})
    assert rdr.mn15(tmp_path)["share_effect_below_tenth_of_recon"] == 0.5


def test_participation_ratio_rank_one_and_isotropic_and_centering():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(200, 1))
    assert rdr.participation_ratio(u @ np.ones((1, 9))) == pytest.approx(1.0)
    iso = rng.normal(size=(20000, 9))
    assert rdr.participation_ratio(iso) == pytest.approx(9.0, rel=0.05)
    offset = 50.0 + 0.1 * rng.normal(size=(5000, 9))
    assert rdr.participation_ratio(offset) > 8.0
    assert rdr.participation_ratio(offset, center=False) < 1.1
    with pytest.raises(ValueError):
        rdr.participation_ratio(np.ones((1, 9)))


def test_effect_matrix_selects_the_requested_null_and_is_causal_only(tmp_path):
    prim_only = _cand(1, [4.0] + [0.0] * 8, 1, by_null={"iso": {"channels": _channels([0.5] + [0.0] * 8), "n_channels_clearing": 0}})
    iso_only = _cand(2, [0.2] + [0.0] * 8, 0, by_null={"iso": {"channels": _channels([3.0, 1.0] + [0.0] * 7), "n_channels_clearing": 1}})
    unscorable = _cand(3, [0.0] * 9, 0, scorable=False, by_null={"iso": {"scorable": False, "reason": "never fires"}})
    _write(tmp_path, "M", {"model": "M", "layer": LAYER, "candidates": [prim_only, iso_only, unscorable]})
    Xp, rp = rdr.effect_matrix(tmp_path, None)
    Xi, ri = rdr.effect_matrix(tmp_path, "iso")
    assert [r["feature"] for r in rp] == [1] and Xp[0, 0] == 4.0
    assert [r["feature"] for r in ri] == [2] and Xi[0, 0] == 3.0 and Xi[0, 1] == 1.0
    with pytest.raises(KeyError):
        rdr.null_view({"by_null": {}}, "iso")


def test_effect_matrix_matches_library_pooling(tmp_path):
    from tsfm_lens.sae.concept_atlas import pooled_features

    rng = np.random.default_rng(3)
    cands = [_cand(i, list(rng.normal(size=9) * 3), 1) for i in range(12)]
    _write(tmp_path, "M", {"model": "M", "layer": LAYER, "candidates": cands})
    X, rows = rdr.effect_matrix(tmp_path)
    Xl, rl = pooled_features(tmp_path)
    assert np.array_equal(X, Xl) and rows == rl


def test_chance_multiple_is_ratio_of_sums_and_refuses_missing_expectation(tmp_path):
    def art(layer, cells_iso, exp_iso, cells_cov, exp_cov):
        return {"model": "M", "layer": layer, "candidates": [],
                "by_null_summary": {"iso": {"n_clearing_cells": cells_iso, "empirical_chance": {"expected_cells": exp_iso}},
                                    "cov": {"n_clearing_cells": cells_cov, "empirical_chance": {"expected_cells": exp_cov}}}}
    _write(tmp_path, "M", art("a.1", 10, 1.0, 2, 2.0))
    _write(tmp_path, "N", art("b.1", 10, 9.0, 4, 2.0))
    r = rdr.chance_multiple(tmp_path)
    assert r["iso"]["ratio"] == pytest.approx(20 / 10.0)
    assert r["cov"]["ratio"] == pytest.approx(6 / 4.0)
    assert r["iso"]["n_targets"] == 2
    legacy = tmp_path.parent / (tmp_path.name + "_legacy")
    _write(legacy, "M", {"model": "M", "layer": "a.1", "candidates": [], "n_clearing_cells": 7, "chance_expected_cells": 3.5})
    lr = rdr.chance_multiple(legacy)["legacy"]
    assert lr["clearing_cells"] == 7 and lr["ratio"] is None and lr["expected_cells"] is None


def test_poisson_binomial_known_values():
    assert rdr._poisson_binomial_upper(1, [0.5, 0.5]) == pytest.approx(0.75)
    assert rdr._poisson_binomial_upper(2, [0.5, 0.5]) == pytest.approx(0.25)
    assert rdr._poisson_binomial_upper(0, [0.3, 0.9]) == pytest.approx(1.0)


def test_feature_chance_multiple_sums_draw_level_not_independent(tmp_path):
    fc = {"p_draw_level": 0.25, "p_independent": 0.5, "p_union_bound": 0.9}
    cands = [_cand(i, [2.0] + [0.0] * 8, 1 if i < 2 else 0, feature_chance=fc) for i in range(4)]
    cands.append(_cand(9, [0.0] * 9, 0, scorable=False, feature_chance=fc))
    _write(tmp_path, "M", {"model": "M", "layer": LAYER, "candidates": cands})
    r = rdr.feature_chance_multiple(tmp_path)
    assert r["n_scorable"] == 4 and r["clearing_ge1"] == 2
    assert r["expected_draw_level"] == pytest.approx(1.0) and r["ratio_draw_level"] == pytest.approx(2.0)
    assert r["expected_independent"] == pytest.approx(2.0)


def test_atlas_null_recovers_planted_clusters_and_beats_shuffle(tmp_path):
    rng = np.random.default_rng(0)
    cands = []
    fid = 0
    for axis in range(3):
        for _ in range(6):
            v = np.full(9, 0.0)
            v[axis] = 6.0
            v += 0.05 * rng.normal(size=9)
            cands.append(_cand(fid, list(v), 1))
            fid += 1
    _write(tmp_path, "M", {"model": "M", "layer": LAYER, "candidates": cands})
    r = rdr.atlas_null(tmp_path, min_cosine=0.9, n_null=50)
    assert r["real_concepts"] == 3 and r["n_assigned"] == 18
    assert r["null_mean"] < 2.5 < r["real_concepts"]
    noise = tmp_path.parent / (tmp_path.name + "_noise")
    _write(noise, "M", {"model": "M", "layer": LAYER, "candidates": [_cand(i, list(rng.normal(size=9) * 3), 1) for i in range(18)]})
    rn = rdr.atlas_null(noise, min_cosine=0.9, n_null=50)
    assert rn["real_concepts"] <= rn["null_p95"]


def _patching_fixture(run: Path) -> None:
    (run / "l3").mkdir(parents=True)
    corrs = ["deseasonalize", "noise", "level_shift"]
    arr = np.zeros((3, 2, 4), dtype=np.float32)
    arr[0, :, -1] = [0.9, 0.9]
    arr[1, :, -1] = [0.1, 0.4]
    arr[1, :, 0] = [0.05, 0.0]
    arr[2, :, -1] = [0.7, 0.7]
    np.savez(run / "l3" / "patching.npz", restoration_windows_TimesFM=arr)
    (run / "l3" / "patching.json").write_text(json.dumps(
        {"TimesFM": {"corruptions": corrs, "layers": ["stacked_xf.0", "stacked_xf.16"], "n_realized": 320}}), encoding="utf-8")


def test_patching_by_depth_looks_up_corruption_by_name(tmp_path):
    _patching_fixture(tmp_path)
    r = rdr.patching_by_depth(tmp_path)
    assert r["restoration_last_window"] == pytest.approx([0.1, 0.4])
    assert r["restoration_first_window"] == pytest.approx([0.05, 0.0])
    assert r["window"] == 3 and r["n_series"] == 320
    with pytest.raises(KeyError):
        rdr.patching_by_depth(tmp_path, corruption="warp")


def test_make_configs_are_timesfm_only_and_unclamped(tmp_path):
    yaml = pytest.importorskip("yaml")
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump({
        "run": {"name": "x", "out_dir": "runs"},
        "models": [{"name": "TimesFM", "batch_size": 32}, {"name": "Chronos-2", "batch_size": 24}],
        "l3": {"patching": {"max_series": 16}}}), encoding="utf-8")
    paths = rdr.make_configs(out_dir=tmp_path / "cfg", source=src)
    big = yaml.safe_load(Path(paths["de01_n320"]).read_text(encoding="utf-8"))
    small = yaml.safe_load(Path(paths["de01_check_n16"]).read_text(encoding="utf-8"))
    assert [m["name"] for m in big["models"]] == ["TimesFM"]
    assert big["models"][0]["batch_size"] >= big["l3"]["patching"]["max_series"] == 320
    assert small["models"][0]["batch_size"] == 32 and small["l3"]["patching"]["max_series"] == 16
    assert big["run"]["name"] == "de01_n320" and big["run"]["out_dir"].endswith("dagger_remeasure")


@pytest.mark.skipif(not (rdr.RUNS / rdr.FOUR_MODEL / "sae").exists(), reason="four-model artifacts not present")
def test_known_answers_on_the_four_model_artifacts():
    r = rdr.check()
    assert r["a"]["ok"] and r["b"]["ok"] and r["e"]["ok"]


def test_patching_composition_is_family_stratified(tmp_path):
    pd = pytest.importorskip("pandas")
    yaml = pytest.importorskip("yaml")
    fam = np.array(["a"] * 600 + ["b"] * 300 + ["c"] * 100)
    pd.DataFrame({"family": fam}).to_parquet(tmp_path / "meta.parquet")
    (tmp_path / "config_resolved.yaml").write_text(yaml.safe_dump({
        "run": {"seed": 0}, "models": [{"batch_size": 100}],
        "l3": {"max_series": 500, "patching": {"max_series": 200}}}), encoding="utf-8")
    r = rdr.patching_composition(tmp_path)
    assert r["n"] == 100 and sum(r["realized"].values()) == 100
    assert set(r["realized"]) == {"a", "b", "c"}
    assert abs(r["realized"]["c"] - 10) <= 2 and abs(r["realized"]["a"] - 60) <= 3
