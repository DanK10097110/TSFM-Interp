"""The dual-null battery through the real `concepts` stage on mock models
(ROADMAP.md sec 41.1), and the restricted-result / feature-chance reporting.

Known answers: concepts / atlas built from a dual-null run must equal those of a
single-null run under the primary (the downstream stages read the primary's legacy
keys), the secondary's counts appear beside the primary's, and the restriction drops
exactly the targets planted as un-admitted or mis-aligned.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_concept_stage import _cfg  # noqa: E402
from tsfm_lens.analysis import battery_robustness as BR  # noqa: E402
from tsfm_lens.pipeline import run_pipeline  # noqa: E402
from tsfm_lens.utils import load_json  # noqa: E402

PRIMARY, SECOND = "profile_matched_cov", "profile_matched"
SCRATCH = ROOT / "runs" / "_scratch_v3e1"


@pytest.fixture(scope="module")
def stage_runs():
    out = SCRATCH / "stage"
    shutil.rmtree(SCRATCH, ignore_errors=True)
    out.mkdir(parents=True)
    cfg = _cfg(str(out), name="dual_null_stage")
    cfg.sae.ablation_null = PRIMARY
    cfg.sae.ablation_empirical_chance = True
    run_pipeline(cfg, stages=["extract"])
    from tsfm_lens.extraction.store import ActivationStore
    store = ActivationStore(cfg.run_dir() / "activations.zarr", mode="r")
    cfg.sae.targets = [{"model": "patchy", "layer": store.layers("patchy")[-1]},
                       {"model": "steppy", "layer": store.layers("steppy")[-1]}]
    run_pipeline(cfg, stages=["sae"])
    rd = cfg.run_dir()
    t0 = time.monotonic()
    run_pipeline(cfg, stages=["concepts"])
    t_single = time.monotonic() - t0
    keep = {rel: load_json(rd / rel) for rel in
            ("sae/concepts.json", "sae/concept_atlas.json", "sae/transfer.json")
            if (rd / rel).exists()}
    single_art = {p.name: load_json(p) for p in (rd / "sae").glob("*/*_ablation.json")}
    cfg.sae.ablation_nulls = (PRIMARY, SECOND)
    cfg.sae.keep_signed_null_draws = True
    t0 = time.monotonic()
    run_pipeline(cfg, stages=["concepts"], force={"concepts"}, allow_stale=True)
    t_dual = time.monotonic() - t0
    yield {"rd": rd, "cfg": cfg, "single": keep, "single_art": single_art,
           "t_single": t_single, "t_dual": t_dual}
    shutil.rmtree(SCRATCH, ignore_errors=True)


def test_downstream_stages_consume_the_primary_null(stage_runs):
    rd = stage_runs["rd"]
    for rel, doc in stage_runs["single"].items():
        assert json.dumps(load_json(rd / rel), sort_keys=True) == json.dumps(doc, sort_keys=True), rel


def test_dual_artifact_primary_keys_equal_single_run_and_carry_by_null(stage_runs):
    rd = stage_runs["rd"]
    for p in (rd / "sae").glob("*/*_ablation.json"):
        dual, one = load_json(p), stage_runs["single_art"][p.name]
        assert dual["ablation_null"] == PRIMARY and dual["ablation_nulls"] == [PRIMARY, SECOND]
        for k in ("n_clearing_cells", "empirical_chance", "feature_chance"):
            assert json.dumps(dual[k], sort_keys=True) == json.dumps(one[k], sort_keys=True)
        assert all(SECOND in c["by_null"] for c in dual["candidates"])
    assert stage_runs["t_dual"] > 0


def test_keep_signed_null_draws_config_reaches_the_battery(stage_runs):
    """`sae.keep_signed_null_draws` is a real config field: the dual run's artifacts carry
    the signed draws for BOTH nulls, the single run's (flag off) carry none. Planted
    regression: not forwarding the field leaves the draws out."""
    rd = stage_runs["rd"]
    n = 0
    for p in (rd / "sae").glob("*/*_ablation.json"):
        for c in load_json(p)["candidates"]:
            if not c.get("scorable"):
                continue
            for blk in (c, c["by_null"][SECOND]):
                for v in blk["channels"].values():
                    if v.get("available"):
                        assert v["null_draw_signed_means"] and v["null_draw_means"]
                        n += 1
        assert all("null_draw_signed_means" not in v for c in stage_runs["single_art"][p.name]["candidates"]
                   if c.get("scorable") for v in c["channels"].values())
    assert n > 0


def test_robustness_artifact_shows_both_nulls_and_feature_chance(stage_runs):
    doc = load_json(stage_runs["rd"] / "sae" / "battery_robustness.json")
    assert doc["primary_null"] == PRIMARY and doc["nulls"] == [PRIMARY, SECOND]
    for mode in (PRIMARY, SECOND):
        h = doc["by_null"][mode]["headline"]
        assert h["n_targets"] == 2
        fc = h["feature_chance"]
        assert fc["measured"] and fc["n_features"] == h["n_scorable_features"]
        assert fc["observed_clearing_ge1"] == h["n_causal_features"]
        assert fc["expected_independent"] <= fc["expected_union_bound"] + 1e-9
        assert h["expected_cells_empirical"] is not None
    prim_n = sum(r["n_clearing_cells"] for r in doc["by_null"][PRIMARY]["per_target"])
    assert prim_n == doc["by_null"][PRIMARY]["headline"]["n_clearing_cells"]


def test_report_block_renders_both_nulls_and_the_restriction(stage_runs):
    html = __import__("tsfm_lens.report.sae_concepts", fromlist=["x"]).battery_robustness_block(
        stage_runs["rd"])
    assert "primary" in html and "secondary" in html
    assert PRIMARY in html and SECOND in html
    assert "restricted: admitted + aligned" in html and "union bound" in html
    assert html.count("<table") == 2


def _write_run(tmp, targets, admission, align, atlas=None):
    import numpy as np
    from tsfm_lens.sae.train import sanitize
    meta = {}
    arts = []
    for (model, layer), adm in zip(targets, admission):
        meta[f"{model}/{layer}"] = {"admission": {"passed": adm, "reason": f"adm={adm}"}}
    (tmp / "sae").mkdir(parents=True)
    (tmp / "alignment").mkdir()
    (tmp / "sae" / "meta.json").write_text(json.dumps(meta))
    (tmp / "alignment" / "alignment_check.json").write_text(json.dumps(align))
    for i, (model, layer) in enumerate(targets):
        d = tmp / "sae" / sanitize(model)
        d.mkdir(exist_ok=True)
        ch = {"available": True, "clears_null": True, "empirical_chance": 0.1}
        cand = {"feature": 0, "scorable": True, "channels": {"level": ch},
                "n_channels_clearing": 1, "feature_chance": {
                    "n_channels": 1, "p_union_bound": 0.1, "p_independent": 0.1, "p_draw_level": 0.1}}
        art = {"model": model, "layer": layer, "withheld": False, "ablation_null": PRIMARY,
               "n_clearing_cells": 1, "empirical_chance": {"rate": 0.1, "n_cells": 1,
                                                           "expected_cells": 0.1},
               "feature_chance": {"n_features": 1, "observed_clearing_ge1": 1,
                                  "expected_union_bound": 0.1, "expected_independent": 0.1,
                                  "expected_draw_level": 0.1, "n_features_draw_level": 1},
               "candidates": [cand]}
        (d / f"{sanitize(layer)}_ablation.json").write_text(json.dumps(art))
    if atlas:
        (tmp / "sae" / "concept_atlas.json").write_text(json.dumps(atlas))


def _align(hit_by_model):
    return {m: {"per_layer": hits, "resolvable_ceiling": {"ceiling": 1.0},
                "min_diagonal_frac_threshold": 0.5} for m, hits in hit_by_model.items()}


def test_restriction_drops_exactly_the_planted_unadmitted_and_misaligned_targets():
    """Four targets: A good, B un-admitted, C mis-aligned (hit 0.2 < 0.5), D admission
    undecided (None). Only A is kept; unrestricted counts keep all four. Planted
    regression: treating None as admitted/aligned keeps D (and an unprobed layer)."""
    tmp = SCRATCH / "planted"
    shutil.rmtree(tmp, ignore_errors=True)
    targets = [("M", "blocks.1"), ("M", "blocks.2"), ("M", "blocks.3"), ("M", "blocks.4"),
               ("M", "blocks.9")]
    adm = [True, False, True, None, True]
    align = _align({"M": {"blocks.1": 1.0, "blocks.2": 1.0, "blocks.3": 0.2, "blocks.4": 1.0}})
    rows = ([{"model": "M", "layer": "blocks.1", "feature": i, "concept": 0} for i in range(3)]
            + [{"model": "M", "layer": "blocks.3", "feature": i, "concept": 1} for i in range(3)]
            + [{"model": "M", "layer": "blocks.2", "feature": 9, "concept": 0}])
    atlas = {"params": {"min_members": 3}, "rows": rows,
             "concepts": [{"concept": 0}, {"concept": 1}]}
    _write_run(tmp, targets, adm, align, atlas)
    try:
        doc = BR.build_battery_robustness(tmp)
        res = doc["restriction"]
        assert [k for k, v in res["targets"].items() if v["kept"]] == ["M/blocks.1"]
        assert res["n_admitted"] == 3 and res["n_admission_undecided"] == 1
        assert res["n_aligned"] == 3 and res["n_alignment_unmeasured"] == 1
        unr = doc["by_null"][PRIMARY]["headline"]
        assert unr["n_targets"] == 5 and unr["n_clearing_cells"] == 5
        rest = doc["restricted"]["headline_by_null"][PRIMARY]
        assert rest["n_targets"] == 1 and rest["n_clearing_cells"] == 1
        assert rest["feature_chance"]["observed_clearing_ge1"] == 1
        a = doc["restricted"]["membership"]["atlas"]
        assert a["n_concepts"] == 2 and a["n_concepts_surviving"] == 1
        assert a["n_features"] == 7 and a["n_features_kept"] == 3
        c0 = [r for r in a["concepts"] if r["concept"] == 0][0]
        assert c0["n_members"] == 4 and c0["n_members_kept"] == 3
        assert [r for r in a["concepts"] if r["concept"] == 1][0]["survives"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_alignment_status_uses_the_ceiling_not_the_raw_hit():
    tmp = SCRATCH / "ceil"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "alignment").mkdir(parents=True)
    rec = {"per_layer": {"l": 0.3}, "resolvable_ceiling": {"ceiling": 0.5},
           "min_diagonal_frac_threshold": 0.5}
    (tmp / "alignment" / "alignment_check.json").write_text(json.dumps({"M": rec}))
    try:
        st = BR.alignment_status(tmp, "M", "l")
        assert st["aligned"] is True and st["hit_over_ceiling"] == pytest.approx(0.6)
        assert BR.alignment_status(tmp, "M", "missing")["aligned"] is None
        assert BR.alignment_status(tmp, "X", "l")["aligned"] is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
