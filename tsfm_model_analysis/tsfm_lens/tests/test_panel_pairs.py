"""Panel runs measure EVERY pair, not the first two (`ROADMAP.md` sec 24.3 sub-item 3).

`tests/test_run_shape.py` covers the *gate* -- which stages a run shape drops.
This covers the other half, which is where the risk actually is: three
artifact families (`l1/cka.npz`, `l2/stitching.json`,
`clustering/comparison.json`) that every recorded number in `ROADMAP.md` lives
in had to grow per-pair entries without moving a single existing key.

Two failure modes shape the tests below, and they point in opposite
directions:

1. **A panel silently reporting the reference pair.** That is the defect this
   item exists to remove and it is invisible in the output -- six pairs'
   worth of models produce a report that looks complete and describes two of
   them. Tested by asserting the pair COUNT and that a late pair's numbers
   actually differ from pair 0's, not merely that a key exists.
2. **A pair run moving.** The rule this repo already validated for `l3`'s
   `models` key is: add a canonical key, leave the legacy keys untouched,
   read new-then-legacy. Tested by asserting the legacy key is not just
   present but IDENTICAL to pair 0's suffixed entry -- a reader that falls
   back to it must get the same array, not a plausible one.

Everything here is synthetic or reads an already-built run directory; no
checkpoint, no GPU, and nothing that trains.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import ModelConfig, PipelineConfig, load_config
from tsfm_lens.report import derived

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
PANEL = ROOT / "runs" / "smoke_panel"

needs_panel_run = pytest.mark.skipif(
    not (PANEL / "l1" / "meta.json").exists(),
    reason="runs/smoke_panel not built (gitignored); "
           "`python run.py --config configs/smoke_panel.yaml` to create it")


def _cfg(n: int) -> PipelineConfig:
    cfg = load_config(CONFIGS / "smoke.yaml")
    base = cfg.models[0]
    cfg.models = [ModelConfig(name=f"m{i}", adapter=base.adapter,
                              layer_regex=base.layer_regex) for i in range(n)]
    return cfg


# --- the seam itself -------------------------------------------------------

def test_pairs_enumeration_and_reference_pair():
    assert _cfg(1).comparison_pairs() == []
    two = _cfg(2)
    assert [(a.name, b.name) for a, b in two.comparison_pairs()] == [("m0", "m1")]
    four = [(a.name, b.name) for a, b in _cfg(4).comparison_pairs()]
    assert len(four) == 6
    # Pair 0 must stay the designated pair: sec 18 F8's L0 seeding convention
    # gives pair 0 its historical bootstrap seeds so a two-model run
    # reproduces bit-for-bit, and reordering here would move every recorded
    # L0 p-value in ROADMAP.md.
    assert four[0] == ("m0", "m1")
    assert len(set(four)) == 6


def test_panel_config_ships_and_enables_every_cross_model_stage():
    # `smoke_panel.yaml`'s whole discipline is that nothing is disabled by
    # hand, so a regression to `comparison_pair()` shows up as a report that
    # compares two of four models rather than as a skipped section. If a
    # future edit switches one of these off, this config stops testing the
    # thing it exists to test and says so here instead of silently.
    cfg = load_config(CONFIGS / "smoke_panel.yaml")
    assert cfg.run_shape() == "panel"
    assert len(cfg.models) == 4
    assert len(cfg.comparison_pairs()) == 6
    for stage in ("l1", "l2", "clustering", "exemplars", "confirm"):
        assert getattr(cfg, stage).enabled, f"{stage} must stay on in smoke_panel.yaml"
    # sec 24.3 sub-item 4: 6 pairs x 6 families = 36 jointly Holm-corrected
    # tests, and a bootstrap p is floored at 1/n_boot, so the smallest
    # attainable adjusted p is 36/n_boot. Below this the correction is
    # unsatisfiable ARITHMETIC and every non-result is a floor artifact
    # rather than evidence (`CLAUDE.md` sec 6.6).
    assert 36 / cfg.stats.n_boot < cfg.stats.alpha


# --- artifacts, against a real built run -----------------------------------

@needs_panel_run
def test_l1_measures_every_pair_and_keeps_the_legacy_key_identical():
    meta = json.loads((PANEL / "l1" / "meta.json").read_text())
    arrays = np.load(PANEL / "l1" / "cka.npz")
    assert meta["run_shape"] == "panel"
    assert len(meta["pairs"]) == 6

    first = meta["pairs"][0]
    key0 = f'cka_window__{first["model_a"]}__{first["model_b"]}'
    # Not "the legacy key exists" -- the legacy key must BE pair 0's array.
    # A reader falling back to it has to get the same numbers, not merely
    # numbers of the same shape.
    assert np.array_equal(arrays["cka_window"], arrays[key0])
    assert np.array_equal(arrays["cka_family"],
                          arrays[f'cka_family__{first["model_a"]}__{first["model_b"]}'])
    assert meta["model_a"] == first["model_a"] and meta["model_b"] == first["model_b"]
    assert meta["best_pair"] == first["best_pair"]

    seen = set()
    for rec in meta["pairs"]:
        key = f'cka_window__{rec["model_a"]}__{rec["model_b"]}'
        assert key in arrays.files
        assert arrays[key].shape == (len(rec["layers_a"]), len(rec["layers_b"]))
        seen.add((rec["model_a"], rec["model_b"]))
    assert len(seen) == 6

    # The real defect this item removes is a panel that reports the reference
    # pair six times. Distinct matrices are the evidence that did not happen.
    mats = [arrays[f'cka_window__{r["model_a"]}__{r["model_b"]}'] for r in meta["pairs"]]
    assert any(m.shape != mats[0].shape or not np.array_equal(m, mats[0]) for m in mats[1:])


@needs_panel_run
def test_l2_measures_both_directions_of_every_pair():
    data = json.loads((PANEL / "l2" / "stitching.json").read_text())
    assert len(data["pairs"]) == 6
    assert len(data["directions"]) == 12
    for rec in data["pairs"]:
        a, b = rec["model_a"], rec["model_b"]
        assert rec["directions"] == [f"{a}->{b}", f"{b}->{a}"]
        for d in rec["directions"]:
            assert d in data["directions"]
        # `best_gain` on the pair record is the better of its two directions,
        # which are fit independently and are not expected to agree.
        assert rec["best_gain"] == max(data["directions"][d]["best_gain"]
                                       for d in rec["directions"])
    first = data["pairs"][0]
    assert (data["model_a"], data["model_b"]) == (first["model_a"], first["model_b"])
    # `layers` is keyed per model and must now cover every model in the panel,
    # since a later pair's heatmap axes are read from it.
    assert set(data["layers"]) == {m for r in data["pairs"]
                                   for m in (r["model_a"], r["model_b"])}


@needs_panel_run
def test_clustering_clusters_every_model_and_scores_every_pair():
    clusters = json.loads((PANEL / "clustering" / "clusters.json").read_text())
    comp = json.loads((PANEL / "clustering" / "comparison.json").read_text())
    emb = pd.read_parquet(PANEL / "clustering" / "embedding.parquet")
    assert len(clusters) == 4
    assert set(emb["model"].unique()) == set(clusters)
    assert len(comp["pairs"]) == 6
    first = comp["pairs"][0]
    for key in ("model_a", "model_b", "ami", "contingency", "rows_a", "cols_b"):
        assert comp[key] == first[key], f"legacy key {key!r} must describe pair 0"
    amis = [r["ami"]["value"] for r in comp["pairs"]]
    assert len(set(amis)) > 1, "six identical AMIs would mean one pair scored six times"


@needs_panel_run
def test_exemplars_rank_by_spread_across_every_model():
    meta = json.loads((PANEL / "exemplars" / "exemplars.json").read_text())
    recs = meta["exemplars"]
    assert set(meta["models"]) == {"patchy", "steppy", "wavy", "encdecy"}
    # A panel has no privileged pair to take a signed gap between, so the
    # ranking quantity is the spread across every model. `gap` must be absent
    # rather than quietly computed from an arbitrary two of the four.
    assert all("spread" in r and "gap" not in r for r in recs)
    for r in recs:
        vals = [r[m] for m in meta["models"]]
        assert r["spread"] == pytest.approx(max(vals) - min(vals), rel=1e-6)
    # `spread` for two models is |gap|, so the derived summary's rank labels
    # (largest disagreement / closest agreement / mid-range) must still be
    # produced -- they were previously gated on a `gap` column existing.
    summary = derived.exemplar_summary(PANEL)
    assert "selection" in summary.columns
    assert not summary["selection"].isna().any()


@needs_panel_run
def test_cka_partner_column_covers_models_outside_the_reference_pair():
    # `_cka_partners` used to return {} for any model not in pair 0, so on a
    # panel the two non-reference models had an empty "closest layer in
    # another model" column -- an absent measurement rendered as a blank,
    # which reads as "not measured for this model" when it was.
    l1 = json.loads((PANEL / "l1" / "meta.json").read_text())
    for model in ("patchy", "steppy", "wavy", "encdecy"):
        partners = derived._cka_partners(PANEL, l1, model)
        assert partners, f"{model} has no partner column"
        # With more than one candidate model a bare layer name is ambiguous,
        # so the partner must name the model too.
        assert all(":" in name for name, _ in partners.values())
    assert derived._cka_partners(PANEL, l1, "not_in_this_run") == {}


@needs_panel_run
def test_panel_report_renders_every_pair_and_states_the_fairness_scope():
    html = (PANEL / "report.html").read_text(encoding="utf-8")
    coverage = json.loads((PANEL / "report" / "coverage.json").read_text())
    assert not [s for s in coverage["sections"] if s["status"] == "failed"]
    for heading in ("Peak similarity, every pair", "Stitching gain, every pair",
                    "Partition agreement, every pair",
                    "Layer-pair similarity, every pair"):
        assert heading in html, heading
    # The fairness card's rows are inherently two-column, so a panel must say
    # which pair it is describing rather than let a reader read four models
    # off a two-model table (`CLAUDE.md` sec 2.5: state the absence). Asserted
    # on the card's OWN sentence, not on the phrase "designated reference
    # pair" -- that phrase also appears in L4's every-pair note, so the loose
    # version of this assertion passed before the card said anything at all.
    assert "This is a panel run of 4 models." in html
    assert "not on this card" in html
    for name in ("patchy", "steppy", "wavy", "encdecy"):
        assert name in html
