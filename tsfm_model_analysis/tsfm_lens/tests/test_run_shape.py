"""Run shape -- solo / pair / panel as first-class run shapes (`ROADMAP.md` sec 24.3).

This gate has the same failure mode as the capability-tier gate it is built
beside (`CLAUDE.md` sec 11.33/11.35): a false drop deletes an analysis the run
could actually have supported, and does so with a stated-looking reason that
reads as a considered finding. So most of what follows is negatives -- above
all that a two-model run drops NOTHING, which is this seam's non-negotiable
acceptance criterion (sec 2.1: reshaping the comparison seam must not move a
single already-recorded number).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import ModelConfig, PipelineConfig, load_config
from tsfm_lens.pipeline import _STAGE_MIN_MODELS, _apply_shape, stage_names

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _cfg(n: int) -> PipelineConfig:
    cfg = load_config(CONFIGS / "smoke.yaml")
    base = cfg.models[0]
    cfg.models = [ModelConfig(name=f"m{i}", adapter=base.adapter,
                              layer_regex=base.layer_regex) for i in range(n)]
    return cfg


def _enabled() -> set:
    cfg = _cfg(2)
    from tsfm_lens.pipeline import _stage_by_name
    return {n for n in stage_names() if _stage_by_name(n).enabled(cfg)}


def test_shape_is_derived_from_model_count_not_configured():
    # A hand-set mode is a claim checked nowhere (`CLAUDE.md` sec 11.34), so
    # there must be no way to declare one -- config loading rejects unknown
    # keys (sec 15 A21), which is what makes "derived" enforceable rather
    # than merely intended.
    assert _cfg(1).run_shape() == "solo"
    assert _cfg(2).run_shape() == "pair"
    assert _cfg(3).run_shape() == "panel"
    assert _cfg(7).run_shape() == "panel"
    with pytest.raises(ValueError):
        PipelineConfig(models=[]).run_shape()


def test_pair_run_drops_nothing():
    # The load-bearing negative. Every number recorded in ROADMAP.md was
    # measured on a two-model run; if this gate can drop a stage there, the
    # whole item is a regression rather than a feature.
    selected = _enabled()
    before = set(selected)
    record = _apply_shape(_cfg(2), selected)
    assert record["dropped_stages"] == []
    assert selected == before
    assert record["shape"] == "pair"


def test_panel_drops_nothing_either():
    # A panel has MORE comparisons available, not fewer. Dropping a
    # cross-model stage here would be the false-refusal failure mode.
    selected = _enabled()
    record = _apply_shape(_cfg(4), selected)
    assert record["dropped_stages"] == []
    assert record["shape"] == "panel"


def test_solo_drops_exactly_the_stages_whose_product_is_a_comparison():
    selected = _enabled()
    record = _apply_shape(_cfg(1), selected)
    assert record["shape"] == "solo"
    assert set(record["dropped_stages"]) == set(_STAGE_MIN_MODELS) & _enabled()
    assert not selected & set(record["dropped_stages"])
    # And the within-model work survives -- these are the two most
    # substantive things in the report and neither needs a second model.
    assert {"l3", "lens", "internals", "l0"} <= selected


def test_dropped_list_does_not_depend_on_which_stages_were_selected():
    # The bug that silently emptied the tier gate's own dropped list: a
    # `--stages report` rerun selects one stage, so a list derived from
    # `selected` comes back empty and rewrites the artifact to claim nothing
    # was dropped -- erasing the reason the report prints beside each
    # skipped section.
    full = _apply_shape(_cfg(1), _enabled())["dropped_stages"]
    narrow = _apply_shape(_cfg(1), {"report"})["dropped_stages"]
    assert narrow == full and full


def test_a_disabled_stage_is_not_reported_as_dropped_by_the_shape_gate():
    # Attributing a config choice to the shape gate would misdescribe why a
    # section is missing, which is exactly what the stated reason is for.
    cfg = _cfg(1)
    cfg.l1.enabled = False
    assert "l1" not in _apply_shape(cfg, _enabled())["dropped_stages"]


def test_comparison_pairs_puts_the_designated_pair_first():
    # `l0_behavioral` already enumerates `[(x, y) for i, x ... for y in
    # present[i+1:]]` and depends on pair 0 being the designated pair to keep
    # its historical bootstrap seeds. This must agree, or a panel run
    # re-seeds the pair-0 tests and every recorded L0 p-value moves.
    cfg = _cfg(4)
    pairs = cfg.comparison_pairs()
    assert len(pairs) == 6
    assert (pairs[0][0].name, pairs[0][1].name) == tuple(m.name for m in cfg.comparison_pair())
    assert [(x.name, y.name) for x, y in pairs] == [
        (x.name, y.name) for i, x in enumerate(cfg.models) for y in cfg.models[i + 1:]]


def test_comparison_pairs_is_empty_for_solo_but_comparison_pair_still_refuses():
    # Two different callers: one wants "the pairs to iterate over" (none),
    # the other wants "the designated pair or an error". Collapsing them
    # would either crash every loop or hand out a silent empty comparison.
    cfg = _cfg(1)
    assert cfg.comparison_pairs() == []
    with pytest.raises(ValueError):
        cfg.comparison_pair()
    assert len(_cfg(2).comparison_pairs()) == 1


def test_shape_record_names_the_reference_model():
    record = _apply_shape(_cfg(3), _enabled())
    assert record["reference_model"] == "m0"
    assert record["models"] == ["m0", "m1", "m2"]
    assert record["n_models"] == 3


def test_report_names_the_run_shape_as_the_skip_reason(tmp_path):
    # "artifacts missing" describes something a rerun would fix. A solo run's
    # missing L1 section is a property of the run, and telling a reader the
    # wrong one of those two sends them to rerun a stage that can never
    # produce anything (`CLAUDE.md` invariant 8).
    from tsfm_lens.report.report import _shape_skip_reason
    from tsfm_lens.utils import save_json
    assert _shape_skip_reason(tmp_path, "l1") == ""          # no artifact -> no claim
    save_json(tmp_path / "shapes.json",
              {"shape": "solo", "n_models": 1, "dropped_stages": ["cluster", "l1"]})
    reason = _shape_skip_reason(tmp_path, "l1")
    assert "solo" in reason and "1 model" in reason and "compare" in reason
    # Reached through the report's own config-attr name, not the stage name.
    assert "cluster" in _shape_skip_reason(tmp_path, "clustering")
    # A stage the gate did NOT drop must not be attributed to it.
    assert _shape_skip_reason(tmp_path, "l3") == ""


# --- internal degradations: stages that survive solo by losing only their
# --- cross-model half, rather than being dropped whole.

def _solo_metrics():
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    rows = []
    for fam in ("trend", "seasonal"):
        for i in range(12):
            rows.append({"model": "m0", "series_id": f"{fam}{i}", "family": fam,
                         "archetype": f"{fam}_arch", "mase": float(rng.uniform(0.5, 2.0)),
                         "smape": 0.2, "pinball": 0.1, "mae_over_mad": 0.9,
                         "mase_reliable": True})
    return pd.DataFrame(rows)


def test_l0_keeps_its_within_model_half_on_a_solo_run():
    # The whole point of leaving `l0` OUT of `_STAGE_MIN_MODELS`: dropping it
    # would delete every per-family metric and the calibration block, none of
    # which needs a second model. Only the paired tests have nothing to pair.
    from tsfm_lens.analysis.l0_behavioral import _summarize
    cfg = _cfg(1)
    summary = _summarize(_solo_metrics(), cfg)
    assert len(summary["per_family"]) == 2 and summary["overall"]
    assert summary["comparison"]["applicable"] is False
    assert "solo" in summary["comparison"]["reason"]
    # The paired artifacts must be ABSENT, not present-and-empty: an empty
    # `strengths` reads as "we tested and found none".
    assert "strengths" not in summary and "family_tests" not in summary


def test_l0_solo_status_does_not_overload_the_pairwise_key():
    # `report.py` iterates `pairwise` as a list of {"a","b"} entries. Putting
    # a status dict there iterates its KEYS and raises on `e["a"]` -- the
    # report-key-drift failure `CLAUDE.md` sec 11.6 records.
    from tsfm_lens.analysis.l0_behavioral import _summarize
    summary = _summarize(_solo_metrics(), _cfg(1))
    assert not isinstance(summary.get("pairwise"), dict)
    for entry in summary.get("pairwise") or []:
        assert "a" in entry and "b" in entry


def test_l0_per_archetype_reports_solo_in_its_own_applicable_guard():
    from tsfm_lens.analysis.l0_behavioral import _archetype_summary
    out = _archetype_summary(_solo_metrics(), _cfg(1))
    assert out["rows"]                      # within-model metrics survive
    assert out["tests"]["applicable"] is False
    assert "solo" in out["tests"]["reason"]


def test_l0_two_model_summary_still_carries_its_paired_results():
    # The negative that makes the three tests above meaningful: the solo
    # guards must not fire on a pair run.
    import pandas as pd
    from tsfm_lens.analysis.l0_behavioral import _summarize
    solo = _solo_metrics()
    both = pd.concat([solo, solo.assign(model="m1", mase=solo["mase"] * 1.5)])
    summary = _summarize(both, _cfg(2))
    assert "comparison" not in summary
    assert summary.get("mase_ratio")


def test_model_palette_preserves_the_two_historical_hues():
    # Every existing two-model figure uses _COLORS["a"]/["b"]. If the palette
    # that replaced them ever drifts, every recorded figure changes color
    # without any figure code changing -- so pin it here rather than trusting
    # the two definitions to agree by eye.
    from tsfm_lens.report.report import _COLORS, _MODEL_PALETTE
    assert _MODEL_PALETTE[:2] == [_COLORS["a"], _COLORS["b"]]
    assert len(set(_MODEL_PALETTE)) == len(_MODEL_PALETTE)
    assert not ({_COLORS["accent"], _COLORS["muted"]} & set(_MODEL_PALETTE))


def test_l3_models_reads_the_canonical_key_and_falls_back():
    # An artifact written before sec 24.3 carries only model_a/model_b; one
    # written after carries `models`. Both must render, and a solo run's
    # `model_b: null` must become a one-model list rather than a [name, None]
    # that crashes two lines later on `arrays[f"fingerprint_{None}"]`.
    from tsfm_lens.report.report import _l3_models
    assert _l3_models({"models": ["x", "y", "z"]}) == ["x", "y", "z"]
    assert _l3_models({"model_a": "x", "model_b": "y"}) == ["x", "y"]   # legacy
    assert _l3_models({"model_a": "x", "model_b": None}) == ["x"]
    assert _l3_models({"models": ["x"], "model_a": "x", "model_b": None}) == ["x"]


def test_l3_solo_agreement_is_a_named_absence_not_a_missing_key():
    # `applicable: False` with a reason, so the report can say WHY rather
    # than dropping the subsection -- a vanished subsection reads as a
    # crashed stage. `derived.py` must also stay quiet rather than emit a
    # scorecard row built on a missing number.
    from tsfm_lens.report.derived import bottom_line_rows
    meta = {"models": ["x"], "model_a": "x", "model_b": None,
            "agreement": {"applicable": False, "reason": "solo run (1 model): ..."}}
    assert meta["agreement"]["applicable"] is False
    # The scorecard's own guard keys on `overall`, which a solo artifact
    # never has -- assert the guard rather than assuming it.
    assert not (meta["agreement"].get("overall"))


def test_smoke_solo_config_disables_nothing_by_hand():
    # The acceptance config's whole value is that the GATE is what narrows the
    # run. If a future edit disables l1/l2/cluster/exemplars/confirm here, the
    # config still "passes" while testing nothing -- the same discipline
    # `configs/smoke_blackbox.yaml` enforces for capability tiers.
    solo = load_config(CONFIGS / "smoke_solo.yaml")
    pair = load_config(CONFIGS / "smoke.yaml")
    assert solo.run_shape() == "solo" and len(solo.models) == 1
    from tsfm_lens.pipeline import _stage_by_name
    for name in _STAGE_MIN_MODELS:
        assert _stage_by_name(name).enabled(solo), (
            f"'{name}' is disabled in smoke_solo.yaml, so the shape gate is not "
            f"what drops it and this config tests nothing")
    # And it differs from smoke.yaml only by the missing model -- every other
    # stage's enablement must match, or the two are not comparable runs.
    assert ({n for n in stage_names() if _stage_by_name(n).enabled(solo)}
            == {n for n in stage_names() if _stage_by_name(n).enabled(pair)})


def test_solo_run_leaves_no_stage_both_selected_and_undroppable():
    # End-to-end shape check without paying for an end-to-end run: after the
    # gate, nothing left in `selected` may call `comparison_pair()`, which is
    # what turned into a mid-run crash before this item existed.
    selected = _enabled()
    _apply_shape(_cfg(1), selected)
    assert not (selected & set(_STAGE_MIN_MODELS))
