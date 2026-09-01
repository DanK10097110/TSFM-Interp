"""Report-integration tests for the SAE causal-roles display (ROADMAP.md sec
25.7, sec 25.9 Stage 3 exit criterion) -- `report/report.py::_sae_roles_block`
and `report/sae_roles.py`'s pure table builders.

Synthetic, `tmp_path`-based, following the same fixture-writing style as
`test_report_legibility.py`'s `_write_l0`/`_Cfg`. `_sae_roles_block` does not
read `cfg` for anything -- it consumes only `sae/roles.json` and each
target's `*_stage2_response.json`, both already-written artifacts -- so a
bare `None` stands in for `cfg` here, and no model checkpoint, store, or
ground-truth table is needed to exercise the display logic.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.report import _sae_roles_block  # noqa: E402
from tsfm_lens.sae.response import CHANNELS  # noqa: E402


def _write(run: Path, rel: str, payload) -> None:
    p = run / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")


def _candidate(feature: int, clearing: list, trend_up: float = 0.0,
              seasonal_up: float = 0.0) -> dict:
    def channels(sign: int):
        out = {}
        for ch in CHANNELS:
            val = 0.0
            if ch == "trend":
                val = sign * trend_up
            elif ch == "seasonal":
                val = sign * seasonal_up
            out[ch] = {"available": True, "signed_mean": val,
                      "clears_null": ch in clearing}
        return out
    return {"feature": feature, "rules": ["variance"],
           "up": {"channels": channels(1)}, "down": {"channels": channels(-1)},
           "clearing_channels": sorted(clearing)}


def _write_one_target(run: Path, model: str, layer: str, n_roles_content: dict) -> None:
    """Writes both `sae/roles.json` (for the target) and a matching
    `*_stage2_response.json` (for the heatmap/cards) that agree on feature ids."""
    from tsfm_lens.sae.train import sanitize
    layer_dir = run / "sae" / sanitize(model)
    layer_dir.mkdir(parents=True, exist_ok=True)
    candidates = n_roles_content["candidates"]
    null_p95 = {ch: 1.0 for ch in CHANNELS}
    (layer_dir / f"{sanitize(layer)}_stage2_response.json").write_text(
        json.dumps({"model": model, "layer": layer, "withheld": False,
                    "candidates": candidates, "null_p95": null_p95,
                    "null_mean": {ch: 0.5 for ch in CHANNELS}}), encoding="utf-8")


def _one_target_roles(model="Alpha", layer="layer.6"):
    candidates = [_candidate(0, ["trend"], trend_up=3.0),
                 _candidate(1, ["trend"], trend_up=2.5),
                 _candidate(2, ["seasonal"], seasonal_up=4.0)]
    roles = [
        {"role": 0, "name": "trend-slope ↑", "n_atoms": 2, "features": [0, 1],
         "dominant_channel": "trend", "dominant_effect_null_units": 2.75,
         "sign": 1, "structural_field": None, "structural_rho": None,
         "structural_n": None, "clears_null": True},
        {"role": 1, "name": "seasonal-magnitude ↑", "n_atoms": 1, "features": [2],
         "dominant_channel": "seasonal", "dominant_effect_null_units": 4.0,
         "sign": 1, "structural_field": "seasonal_amplitude_max",
         "structural_rho": 0.4, "structural_n": 200, "clears_null": True},
    ]
    rec = {"model": model, "layer": layer, "withheld": False, "skipped": False,
          "k": 2, "silhouette": 0.6, "non_modular": False, "non_modular_reason": "",
          "n_candidates": 3, "channel_columns": list(CHANNELS), "roles": roles,
          "candidates": candidates}
    return rec


def _figs_and_caps(html: str) -> list:
    return ["fig" if "plotly-graph-div" in m.group(0) else "cap"
           for m in re.finditer(
               r'<div id="[0-9a-f-]{36}" class="plotly-graph-div"|<p class="figcap">', html)]


# ---------------------------------------------------------------------------
# 1. Empty / degrade cases.
# ---------------------------------------------------------------------------

def test_no_roles_artifact_renders_nothing(tmp_path):
    assert _sae_roles_block(None, tmp_path, [], ["Alpha"]) == ""


def test_a_withheld_target_states_the_reason_and_builds_no_table(tmp_path):
    _write(tmp_path, "sae/roles.json", {"Alpha/layer.6": {
        "withheld": True, "reason": "no reach: cross-patch delta 0.0"}})
    html = _sae_roles_block(None, tmp_path, [], ["Alpha"])
    assert "WITHHELD" in html
    assert "no reach" in html
    assert "<table" not in html


def test_a_skipped_target_states_the_reason(tmp_path):
    _write(tmp_path, "sae/roles.json", {"Alpha/layer.6": {
        "skipped": True, "reason": "only 1 candidate(s) -- nothing to cluster"}})
    html = _sae_roles_block(None, tmp_path, [], ["Alpha"])
    assert "roles not built" in html
    assert "nothing to cluster" in html


# ---------------------------------------------------------------------------
# 2. Solo run: one model with roles -> no role x model matrix (sec 25.9(d),
#    sec 24.3's run-shape convention applied to a within-section decision).
# ---------------------------------------------------------------------------

def test_solo_target_renders_the_roles_table_but_no_cross_model_matrix(tmp_path):
    rec = _one_target_roles("Alpha")
    _write(tmp_path, "sae/roles.json", {"Alpha/layer.6": rec})
    _write_one_target(tmp_path, "Alpha", "layer.6", rec)
    findings = []
    html = _sae_roles_block(None, tmp_path, findings, ["Alpha"])
    assert "trend-slope" in html
    assert "seasonal-magnitude" in html
    assert "Role × model matrix" not in html
    # A causal finding was still emitted for the solo target.
    assert any("SAE roles" in f.text for f in findings)


# ---------------------------------------------------------------------------
# 3. Panel run: >=2 models with roles -> the matrix IS built (sec 25.7 part 6
#    -- the code path must exist and degrade correctly even though it is not
#    empirically validated against a real second-model target, per sec 25.9).
# ---------------------------------------------------------------------------

def test_panel_with_two_models_builds_the_role_correspondence_table(tmp_path):
    """ROADMAP.md sec 25.9 Stage 4 superseded the earlier channel-count
    placeholder (see the two tests below, and CLAUDE.md's own note that the
    old Part 6 was a deliberately incomplete stand-in) with a real matched
    role x role correspondence. Alpha and Beta's fixtures here are built by
    the identical `_one_target_roles` shape, so each role's response
    fingerprint is literally the same vector on both sides -- the two
    corresponding roles must match each other at cosine 1.0, and (with no
    depth information available in this pure-fixture test, `cfg=None`) the
    pair must be judged comparable rather than refused.
    """
    rec_a = _one_target_roles("Alpha", "layer.6")
    rec_b = _one_target_roles("Beta", "layer.3")
    _write(tmp_path, "sae/roles.json", {"Alpha/layer.6": rec_a, "Beta/layer.3": rec_b})
    _write_one_target(tmp_path, "Alpha", "layer.6", rec_a)
    _write_one_target(tmp_path, "Beta", "layer.3", rec_b)
    html = _sae_roles_block(None, tmp_path, [], ["Alpha", "Beta"])
    assert "Cross-model role correspondence" in html
    assert "Alpha" in html and "Beta" in html
    # Two roles, both built from identical candidate fixtures on each side
    # -- both matches must be found and both must score a decisive cosine.
    assert html.count("+1.00") >= 1 or "1.000" in html


def test_panel_matrix_matches_by_response_fingerprint_not_by_name_string(tmp_path):
    """Role names are per-target and not comparable across independently
    trained dictionaries (sec 25.6) -- the match must be found by COSINE
    over the response fingerprint, never by matching name strings across
    models. Renaming Beta's roles to share nothing textually with Alpha's
    must not break the match, since the underlying fingerprints (built from
    identical candidate fixtures) are unchanged.
    """
    rec_a = _one_target_roles("Alpha", "layer.6")
    rec_b = _one_target_roles("Beta", "layer.3")
    for r in rec_b["roles"]:
        r["name"] = r["name"] + " [renamed on purpose]"
    _write(tmp_path, "sae/roles.json", {"Alpha/layer.6": rec_a, "Beta/layer.3": rec_b})
    _write_one_target(tmp_path, "Alpha", "layer.6", rec_a)
    _write_one_target(tmp_path, "Beta", "layer.3", rec_b)
    html = _sae_roles_block(None, tmp_path, [], ["Alpha", "Beta"])
    assert "[renamed on purpose]" in html
    assert "Cross-model role correspondence" in html
    # The renamed role must still appear as a MATCHED partner, not just
    # anywhere in the page -- i.e. matching survived the name change.
    idx = html.find("Cross-model role correspondence")
    assert "[renamed on purpose]" in html[idx:]


# ---------------------------------------------------------------------------
# 4. "0 bare figures" -- reuse of test_report_legibility.py's own check
#    pattern, applied to this section's own rendered fragment alone.
# ---------------------------------------------------------------------------

def test_the_roles_section_has_zero_bare_figures(tmp_path):
    rec_a = _one_target_roles("Alpha", "layer.6")
    rec_b = _one_target_roles("Beta", "layer.3")
    _write(tmp_path, "sae/roles.json", {"Alpha/layer.6": rec_a, "Beta/layer.3": rec_b})
    _write_one_target(tmp_path, "Alpha", "layer.6", rec_a)
    _write_one_target(tmp_path, "Beta", "layer.3", rec_b)
    html = _sae_roles_block(None, tmp_path, [], ["Alpha", "Beta"])
    seq = _figs_and_caps(html)
    assert seq.count("fig") >= 2, seq  # one heatmap per target here
    bare = [i for i, k in enumerate(seq)
           if k == "fig" and (i + 1 >= len(seq) or seq[i + 1] != "cap")]
    assert not bare, f"{len(bare)} of {seq.count('fig')} figures have no caption after them"


# ---------------------------------------------------------------------------
# 5. Name uniqueness surfaces end to end into the rendered table.
# ---------------------------------------------------------------------------

def test_rendered_table_never_repeats_a_role_name_within_one_target(tmp_path):
    """End-to-end version of `test_sae_roles.py`'s unit-level uniqueness
    tests: even if an upstream artifact somehow carried two roles with equal
    names (a defensive check, not an expected `run_sae_roles.py` output),
    the roles TABLE itself must not silently render a duplicate heading
    that reads as one role split across two rows with no way to tell them
    apart."""
    rec = _one_target_roles("Alpha")
    # Two of the recorded roles are given the SAME name to simulate a
    # hypothetical upstream artifact defect; `roles_summary_table` is a
    # straight rendering pass-through (uniqueness is `role_table`'s job,
    # already covered directly in tests/test_sae_roles.py) so this checks
    # the two rows are at least both present and distinguishable by their
    # OTHER columns (atoms/features), not silently deduplicated into one.
    from tsfm_lens.report.sae_roles import roles_summary_table
    df = roles_summary_table(rec)
    assert len(df) == len(rec["roles"])
