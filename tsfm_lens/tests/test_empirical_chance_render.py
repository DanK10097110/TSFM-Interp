"""Clears-vs-chance is read against the empirical expectation, in the log and the report.

`chance_expected_cells` is the nominal `0.05 x cells`; `empirical_chance` is the
leave-one-draw-out estimate. The fixture plants known numbers across three
models and includes the confusable cases on purpose: a scored target WITHOUT
`empirical_chance` (whose nominal value must not be shown as chance) and a
withheld target (listed with its reason, not as 0).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens.report import sae_concepts as SC  # noqa: E402
from tsfm_lens.sae import ablation_run as AR  # noqa: E402


def _art(n_clear, exp, nominal, with_emp=True):
    art = {"withheld": False, "n_clearing_cells": n_clear,
           "chance_expected_cells": nominal,
           "clearing_cells_over_chance_ratio": n_clear / nominal,
           "excess_over_chance": n_clear - nominal}
    if with_emp:
        art["empirical_chance"] = {"rule": "lodo", "n_cells": 100,
                                   "expected_cells": exp, "rate": exp / 100,
                                   "per_channel": {}}
    return art


PLANTED = {
    ("modelA", "blocks_1"): _art(10, 4.0, 5.0),
    ("modelA", "blocks_2"): _art(7, 2.5, 5.0),
    ("modelB", "blocks_1"): _art(20, 6.5, 8.0),
    ("modelB", "blocks_9"): _art(99, 0.0, 7.77, with_emp=False),
    ("modelC", "blocks_3"): {"withheld": True, "reason": "reach probe flat",
                             "reach": {"reason": "reach probe flat"}},
}


def _write(tmp_path, planted=PLANTED):
    for (m, layer), art in planted.items():
        d = tmp_path / "sae" / m
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{layer}_ablation.json").write_text(json.dumps(art), encoding="utf-8")
    return tmp_path


def test_totals_equal_the_planted_sums(tmp_path):
    html = SC.empirical_chance_block(_write(tmp_path))
    assert "<th>overall total</th><th>37</th><th>13.00</th><th>2.8462x</th><th>18.00</th>" in html
    assert "<th>model total</th><th>17</th><th>6.50</th><th>2.6154x</th><th>10.00</th>" in html
    assert "<th>model total</th><th>20</th><th>6.50</th><th>3.0769x</th><th>8.00</th>" in html
    assert "99" not in html.split("overall total")[1].split("</tr>")[0]


def test_target_without_empirical_chance_shows_the_reason_not_the_nominal(tmp_path):
    html = SC.empirical_chance_block(_write(tmp_path))
    row = [r for r in html.split("<tr>") if "blocks_9" in r][0]
    assert "empirical chance not measured; set sae.ablation_empirical_chance: true" in row
    assert "7.77" not in html and "<td>99</td>" not in html


def test_withheld_target_is_listed_with_its_reason(tmp_path):
    html = SC.empirical_chance_block(_write(tmp_path))
    row = [r for r in html.split("<tr>") if "blocks_3" in r][0]
    assert "withheld: reach probe flat" in row
    assert "<td>0</td>" not in row


def test_no_empirical_chance_anywhere_renders_only_the_stated_reason(tmp_path):
    html = SC.empirical_chance_block(_write(tmp_path, {("m", "b_1"): _art(5, 0, 3.0, with_emp=False)}))
    assert "not measured; set <code>sae.ablation_empirical_chance: true</code>" in html
    assert "<table" not in html and "3.00" not in html


def test_caption_states_evidence_class_and_caveat(tmp_path):
    html = SC.empirical_chance_block(_write(tmp_path))
    assert "Descriptive evidence class" in html
    assert "size-mismatched null" in html


def test_log_line_uses_the_empirical_ratio_when_present():
    line = AR.clearing_log_line("modelA", "blocks.1", _art(10, 4.0, 5.0))
    assert "2.50x empirical chance" in line and "4.00 expected by empirical chance" in line
    assert "leave-one-draw-out" in line and "nominal 5%: 5.00 expected, 2.00x" in line


def test_log_line_without_empirical_chance_is_labelled_nominal():
    line = AR.clearing_log_line("m", "b", _art(10, 0, 5.0, with_emp=False))
    assert "nominal 5%" in line and "empirical" not in line


class _Hub:
    """Hands back a placeholder adapter; the battery itself is stubbed."""

    def get(self, model):
        return object()


class _Sae:
    """A checkpoint stand-in that only needs to survive `.to(device)`."""

    def to(self, device):
        return self


def _run_target(monkeypatch, tmp_path, result):
    """Drive the real `run_ablation_target` with the battery itself stubbed."""
    import numpy as np

    class _Data:
        n = 2
        meta = None

    ckpt = AR.checkpoint_path(tmp_path, "m", "blocks.1")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"")
    monkeypatch.setattr(AR, "load_sae_checkpoint", lambda path: _Sae())
    monkeypatch.setattr(AR, "load_ground_truth_table",
                        lambda path: (_ for _ in ()).throw(RuntimeError("no gt")))
    monkeypatch.setattr(AR, "feature_ablation_fingerprints", lambda *a, **k: result)
    cfg = type("C", (), {"sae": type("S", (), {"ablation_empirical_chance": True})(),
                         "data": type("D", (), {"path": "x"})()})()
    return AR.run_ablation_target(cfg, tmp_path, _Hub(), _Data(), None, "cpu", "m", "blocks.1",
                                  candidates=[{"feature": 0, "rules": []}],
                                  activations=np.zeros((2, 3)))


def test_run_ablation_target_logs_the_empirical_ratio(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="tsfm_lens"):
        out = _run_target(monkeypatch, tmp_path, _art(10, 4.0, 5.0))
    msgs = [r.getMessage() for r in caplog.records if "clearing cells" in r.getMessage()]
    assert len(msgs) == 1 and "2.50x empirical chance" in msgs[0]
    assert "nominal 5%" in msgs[0] and "leave-one-draw-out" in msgs[0]
    assert out["chance_expected_cells"] == 5.0 and out["clearing_cells_over_chance_ratio"] == 2.0


def test_run_ablation_target_without_empirical_logs_nominal(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="tsfm_lens"):
        _run_target(monkeypatch, tmp_path, _art(10, 0, 5.0, with_emp=False))
    msgs = [r.getMessage() for r in caplog.records if "clearing cells" in r.getMessage()]
    assert len(msgs) == 1 and "nominal 5%" in msgs[0] and "empirical" not in msgs[0]
