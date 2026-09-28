"""`sae.targets: auto` must resolve to layers this run ACTUALLY extracted.

`layer_screen` and `extract` do not see the same set of layers, and that is by
design on both sides: since `ROADMAP.md` sec 15 A1 the screen runs its own
stride-1 extraction into a throwaway store so its choice is fair to every
block, while the analysis store holds only `capture_layer_stride` of them. So
under any stride > 1 the screen can legitimately pick a layer that is not in
the store the SAE trains from -- and passing it through raised `KeyError` deep
inside `store.load`, after every earlier stage of a multi-hour run had already
completed (`CLAUDE.md` sec 11.40).

These are synthetic: `_default_targets` reads exactly two things from a store
(`layers(model)`) and three from a config, so a stub pins the resolution logic
without a checkpoint, a zarr store, or a pipeline run.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.train import _default_targets, _screen_ranked_captured


@dataclass
class _Model:
    name: str
    capture_layer_stride: int = 2


class _Cfg:
    def __init__(self, run_dir: Path, models: list):
        self._run_dir, self.models = run_dir, models

    def run_dir(self) -> Path:
        return self._run_dir


class _Store:
    """Only `layers()` is reached -- nothing here loads an array."""

    def __init__(self, by_model: dict):
        self._by_model = by_model

    def layers(self, model: str) -> list:
        return list(self._by_model.get(model, []))


def _write_selection(run_dir: Path, selection: dict) -> None:
    (run_dir / "layer_screen").mkdir(parents=True, exist_ok=True)
    (run_dir / "layer_screen" / "selection.json").write_text(
        json.dumps(selection), encoding="utf-8")


def test_every_selected_layer_captured_is_passed_through_untouched(tmp_path, caplog):
    """The stride-1 case, which is every run recorded before this fix: the
    screen's pick and the store agree, and resolution must be a bit-for-bit
    no-op. Pinned separately so the substitution path cannot start firing on
    runs it has nothing to fix."""
    _write_selection(tmp_path, {"m": {
        "selected": ["b.2", "b.6"],
        "layers": ["b.0", "b.2", "b.4", "b.6"],
        "score_per_layer": [0.1, 0.9, 0.2, 0.8]}})
    store = _Store({"m": ["b.0", "b.2", "b.4", "b.6"]})
    with caplog.at_level(logging.WARNING, logger="tsfm_lens"):
        resolved = _default_targets(_Cfg(tmp_path, [_Model("m", 1)]), store)
    assert resolved == [{"model": "m", "layer": "b.2"}, {"model": "m", "layer": "b.6"}]
    assert "Substituting" not in caplog.text, (
        "nothing was missing; the substitution path must stay silent")


def test_an_uncaptured_pick_is_substituted_and_the_count_is_preserved(tmp_path, caplog):
    """The bug: `b.5` exists for the screen (which scores every block) and not
    in a stride-2 store. Resolution must yield the same NUMBER of targets --
    dropping one would silently shrink the SAE stage instead of failing."""
    _write_selection(tmp_path, {"m": {
        "selected": ["b.2", "b.5"],
        "layers": [f"b.{i}" for i in range(8)],
        "score_per_layer": [0.1, 0.2, 0.7, 0.4, 0.3, 0.9, 0.6, 0.5]}})
    store = _Store({"m": ["b.0", "b.2", "b.4", "b.6"]})
    with caplog.at_level(logging.WARNING, logger="tsfm_lens"):
        resolved = _default_targets(_Cfg(tmp_path, [_Model("m", 2)]), store)
    layers = [t["layer"] for t in resolved]
    assert len(layers) == 2, layers
    assert "b.5" not in layers, "an uncaptured layer must never reach store.load"
    assert set(layers) <= {"b.0", "b.2", "b.4", "b.6"}, layers
    assert "b.5" in caplog.text and "capture_layer_stride" in caplog.text, (
        "the substitution must name what was dropped and the knob that caused it")


def test_the_substitute_comes_from_the_screens_ranking_not_from_position(tmp_path):
    """Load-bearing negative. `b.6` is the last captured layer -- the arbitrary
    fallback this whole resolution path exists to replace -- but the screen
    scores `b.4` highest among captured ones. Picking by position would pass
    every other assertion in this file."""
    _write_selection(tmp_path, {"m": {
        "selected": ["b.5"],
        "layers": [f"b.{i}" for i in range(8)],
        "score_per_layer": [0.1, 0.2, 0.3, 0.4, 0.95, 0.9, 0.15, 0.5]}})
    store = _Store({"m": ["b.0", "b.2", "b.4", "b.6"]})
    resolved = _default_targets(_Cfg(tmp_path, [_Model("m", 2)]), store)
    assert resolved == [{"model": "m", "layer": "b.4"}], resolved


def test_a_model_whose_every_pick_is_uncaptured_still_gets_targets(tmp_path):
    """The degenerate case (an odd-only selection under an even-only stride).
    Returning nothing here would drop the model from the SAE stage entirely,
    which reads in the report as "this model has no SAE" rather than as a
    configuration problem."""
    _write_selection(tmp_path, {"m": {
        "selected": ["b.1", "b.3"],
        "layers": [f"b.{i}" for i in range(6)],
        "score_per_layer": [0.1, 0.8, 0.2, 0.9, 0.7, 0.3]}})
    store = _Store({"m": ["b.0", "b.2", "b.4"]})
    resolved = _default_targets(_Cfg(tmp_path, [_Model("m", 2)]), store)
    assert [t["layer"] for t in resolved] == ["b.4", "b.2"], resolved


def test_ranking_puts_captured_but_unscored_layers_last(tmp_path):
    """A store can hold a layer the screen never scored (a screening run that
    failed for one block, a hand-edited selection). Those must sort after
    every scored layer rather than crashing a dict lookup or silently winning
    on a missing key."""
    sel = {"layers": ["b.0", "b.2"], "score_per_layer": [0.2, 0.9]}
    assert _screen_ranked_captured(sel, ["b.0", "b.2", "b.9"]) == ["b.2", "b.0", "b.9"]
    assert _screen_ranked_captured({}, ["b.1", "b.0"]) == ["b.0", "b.1"], (
        "with no scores at all, fall back to a stable order rather than raising")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
