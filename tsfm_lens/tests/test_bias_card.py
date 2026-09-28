"""Per-model bias-card tests (ROADMAP.md §7 Phase 3, bullet 5).

Runnable directly (`python tests/test_bias_card.py`) or via pytest. Uses
small synthetic sweep dicts matching `param_sweep_*.json`'s real schema
(`run_parameter_sweep.py`) rather than any live model or GPU -- the
comparison/aggregation logic is pure, and the HTML render is exercised with
tiny fixture data, not a real checkpoint's sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.bias_card import (
    build_bias_card,
    compare_at_point,
    render_bias_card_html,
    summarize_param_sweep,
)


def _stat(value: float, half_width: float) -> dict:
    return {"value": value, "lo": value - half_width, "hi": value + half_width, "n": 10}


def test_compare_at_point_favors_lower_ci():
    a, b = _stat(0.2, 0.02), _stat(0.5, 0.02)  # non-overlapping, a clearly lower
    assert compare_at_point(a, b, "A", "B") == "A"
    assert compare_at_point(b, a, "B", "A") == "A"
    overlapping_a, overlapping_b = _stat(0.3, 0.1), _stat(0.35, 0.1)
    assert compare_at_point(overlapping_a, overlapping_b, "A", "B") is None
    print("compare_at_point test passed")


def _make_sweep_json(param: str, values: list, favor_a_at: set) -> dict:
    """A's MASE is low (favored) at `favor_a_at` values, high (B favored)
    elsewhere -- CIs never overlap, so every point is confidently decided."""
    models = {"A": {}, "B": {}}
    for v in values:
        label = f"{param}={v:g}"
        if v in favor_a_at:
            models["A"][label] = _stat(0.1, 0.01)
            models["B"][label] = _stat(0.5, 0.01)
        else:
            models["A"][label] = _stat(0.5, 0.01)
            models["B"][label] = _stat(0.1, 0.01)
    return {"param": param, "values": values, "models": models}


def test_summarize_param_sweep_plurality_and_anomaly():
    # A favored at 4 of 5 points; B favored (the anomaly) at value=16.
    sweep = _make_sweep_json("seasonal_period", [4, 8, 16, 32, 64], favor_a_at={4, 8, 32, 64})
    summary = summarize_param_sweep(sweep)
    assert summary["plurality_favored"] == "A"
    assert summary["favored_counts"] == {"A": 4, "B": 1}
    assert summary["n_confident_points"] == 5
    by_value = {p["value"]: p for p in summary["points"]}
    assert by_value[16]["favored"] == "B"
    assert by_value[16]["anomalous"] is True
    for v in (4, 8, 32, 64):
        assert by_value[v]["anomalous"] is False
    print("summarize_param_sweep plurality/anomaly test passed")


def test_summarize_param_sweep_requires_two_models():
    sweep = _make_sweep_json("noise_scale", [0.1, 0.2], favor_a_at={0.1})
    sweep["models"]["C"] = {"noise_scale=0.1": _stat(0.3, 0.01), "noise_scale=0.2": _stat(0.3, 0.01)}
    with pytest.raises(ValueError):
        summarize_param_sweep(sweep)
    print("two-model guard test passed")


def test_summarize_param_sweep_handles_no_confident_points():
    values = [1.0, 2.0]
    models = {"A": {f"noise_scale={v:g}": _stat(0.3, 0.2) for v in values},
             "B": {f"noise_scale={v:g}": _stat(0.32, 0.2) for v in values}}
    summary = summarize_param_sweep({"param": "noise_scale", "values": values, "models": models})
    assert summary["plurality_favored"] is None
    assert summary["favored_counts"] == {}
    assert all(p["favored"] is None and p["anomalous"] is False for p in summary["points"])
    print("no-confident-points guard test passed")


def test_build_bias_card_aggregates_across_sweeps_and_attaches_caveats():
    sweep1 = _make_sweep_json("seasonal_period", [4, 8, 16], favor_a_at={4, 8, 16})
    sweep2 = _make_sweep_json("intermittency_rate", [0.0, 0.4, 0.8], favor_a_at={0.0})
    s1 = summarize_param_sweep(sweep1)
    s2 = summarize_param_sweep(sweep2)
    card = build_bias_card([s1, s2], caveats={"intermittency_rate": "scale term is unreliable here"})

    assert set(card["models"]) == {"A", "B"}
    a_text = " ".join(card["cards"]["A"])
    b_text = " ".join(card["cards"]["B"])
    assert "seasonal_period sweep at 3/3" in a_text
    assert "Anomaly in the intermittency_rate sweep" in a_text  # A is the plurality-breaker at rate=0.0? check below
    assert "Caveat on the intermittency_rate sweep: scale term is unreliable here" in a_text
    assert "Caveat on the intermittency_rate sweep: scale term is unreliable here" in b_text
    print("build_bias_card aggregation/caveat test passed")


def test_render_bias_card_html_smoke(tmp_path):
    sweep = _make_sweep_json("seasonal_period", [4, 8, 16], favor_a_at={4, 16})
    summary = summarize_param_sweep(sweep)
    card = build_bias_card([summary])
    out = render_bias_card_html(card, [summary], {"seasonal_period": sweep}, tmp_path / "bias_card.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Per-model bias card" in text
    assert "A" in text and "B" in text
    assert "Anomaly" in text
    print("render_bias_card_html smoke test passed")


if __name__ == "__main__":
    import tempfile
    test_compare_at_point_favors_lower_ci()
    test_summarize_param_sweep_plurality_and_anomaly()
    test_summarize_param_sweep_requires_two_models()
    test_summarize_param_sweep_handles_no_confident_points()
    test_build_bias_card_aggregates_across_sweeps_and_attaches_caveats()
    with tempfile.TemporaryDirectory() as d:
        test_render_bias_card_html_smoke(Path(d))
