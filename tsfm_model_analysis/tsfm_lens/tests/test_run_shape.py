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
