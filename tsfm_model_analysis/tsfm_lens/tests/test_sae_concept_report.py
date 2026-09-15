"""Report-integration tests for `report/sae_concepts.py::sae_concepts_block`
(ROADMAP.md sec 30.4, Stage 4, 2026-09-11).

Synthetic with planted answers, matching this repo's own discipline (sec
2.4, and `tests/test_sae_capability_report.py`'s own header): a fixture read
off a real run would make these pass for whatever that run happened to
contain, which is the one thing a reduction/renderer test must not do. Each
test builds a minimal `run_dir` with exactly the artifacts
(`sae/concepts.json`, `sae/transfer.json`, `sae/<model>/<layer>_ablation.
json`) `sae_concepts_block` and its four `derived.py` reductions read, and
asserts on the rendered HTML string directly (sec 11.48: verify rendered
output, not the mechanism in isolation).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.response import CHANNELS  # noqa: E402
from tsfm_lens.sae.train import sanitize  # noqa: E402
from tsfm_lens.utils import save_json  # noqa: E402
from tsfm_lens.report.sae_concepts import sae_concepts_block  # noqa: E402


def _fake_cfg(models=("Alpha", "Beta"), weights=(0.5, 0.3, 0.2),
             max_cards=24, misfit_gap=0.3):
    return SimpleNamespace(
        models=[SimpleNamespace(name=m) for m in models],
        sae=SimpleNamespace(interest_weights=weights,
                            concept_cards_max=max_cards,
                            concept_misfit_cosine_gap=misfit_gap))


def _concept(concept_id, features, n_members, n_clearing, profile,
            name="strong trend riser", centroid_cosine_mean=0.8,
            description=None, description_generated=False):
    return {
        "concept": concept_id, "features": list(features),
        "n_members": n_members, "n_members_clearing": n_clearing,
        "centroid_null_units": {p["channel"]: p["signed_null_units"] for p in profile},
        "profile": profile, "dominant_channel": profile[0]["channel"] if profile else None,
        "name": name, "centroid_cosine_mean": centroid_cosine_mean, "misfits": [],
        "description": description, "description_generated": description_generated,
    }


def _target_rec(model, layer, concepts, withheld=False):
    return {
        "model": model, "layer": layer, "withheld": withheld,
        "n_candidates": sum(c["n_members"] for c in concepts) if concepts else 0,
        "k": len(concepts), "silhouette": 0.45, "non_modular": False,
        "reason": "", "channel_columns": list(CHANNELS), "concepts": concepts,
    }


def _write_concepts(run_dir, targets):
    save_json(Path(run_dir) / "sae" / "concepts.json",
              {"schema_version": 1, "space": "ablation",
               "supersedes": "roles.json", "targets": targets})


def _write_transfer(run_dir, reach, matrix, k_top_series=20, n_null_draws=200,
                    stratum_field="archetype|family"):
    save_json(Path(run_dir) / "sae" / "transfer.json", {
        "schema_version": 1, "k_top_series": k_top_series,
        "n_null_draws": n_null_draws, "stratum_field": stratum_field,
        "pairs": [], "reach": reach, "matrix": matrix, "universality": {}})


def _write_ablation(run_dir, model, layer, candidates):
    path = (Path(run_dir) / "sae" / sanitize(model)
            / f"{sanitize(layer)}_ablation.json")
    save_json(path, {"candidates": candidates})


def _candidate(feature, channel_effects: dict, scorable=True):
    """One ablation-battery candidate record, in the real schema
    `sae/concepts.py::ablation_vector` reads: `channels[ch] = {"null_p95",
    "signed_effect"}`, `signed_effect / null_p95` giving that channel's
    null-unit value. `channel_effects` maps channel name -> null-unit value
    (p95 pinned at 1.0 so signed_effect IS the null-unit value directly)."""
    channels = {ch: {"null_p95": 1.0, "signed_effect": float(v)}
               for ch, v in channel_effects.items()}
    return {
        "feature": feature, "scorable": scorable,
        "n_channels_clearing": sum(1 for v in channel_effects.values() if abs(v) >= 1.0),
        "channels": channels,
        "forecasts": [{"series_id": "s0", "activation": 1.0,
                       "context": [1.0, 2.0, 3.0], "target": [4.0, 5.0],
                       "with_feature": [4.1, 5.1], "without_feature": [3.9, 4.9]}]
                     if scorable else [],
    }


# ---------------------------------------------------------------------------
# 1. Absent-artifact degradation -- the section must not appear at all.
# ---------------------------------------------------------------------------

def test_returns_empty_string_when_concepts_json_absent(tmp_path):
    (tmp_path / "sae").mkdir()
    out = sae_concepts_block(_fake_cfg(), tmp_path, [], ["Alpha", "Beta"])
    assert out == ""


def test_returns_empty_string_when_every_target_withheld(tmp_path):
    """`concepts.json` exists but carries no usable concept -- `cards.empty`
    must still degrade to `""`, not a section with empty headers."""
    _write_concepts(tmp_path, {
        "Alpha/layer.0": _target_rec("Alpha", "layer.0", [], withheld=True)})
    out = sae_concepts_block(_fake_cfg(), tmp_path, [], ["Alpha", "Beta"])
    assert out == ""


# ---------------------------------------------------------------------------
# 2. Transfer three-state rendering (sec 11.37): not measured / measured.
# ---------------------------------------------------------------------------

def test_transfer_not_measured_renders_third_state_not_a_table(tmp_path):
    """No `sae/transfer.json` at all -- the universality table and heatmap
    must not render, and the block must say so explicitly rather than
    showing an empty-looking table (which would read as "zero universal
    concepts", a measured claim this run never made)."""
    profile = [{"channel": "trend", "signed_null_units": 2.5, "n_members_clearing": 2}]
    concepts = [_concept(0, [0, 1, 2], 3, 2, profile)]
    _write_concepts(tmp_path, {"Alpha/layer.0": _target_rec("Alpha", "layer.0", concepts)})
    findings = []
    out = sae_concepts_block(_fake_cfg(models=("Alpha",)), tmp_path, findings, ["Alpha"])
    assert out != ""
    assert "not measured" in out
    assert "Concept universality" not in out
    # The card itself must still render even though transfer is unmeasured.
    assert "strong trend riser" in out
    assert not any("universality" in f.text for f in findings)


def test_transfer_measured_renders_universality_table_heatmap_and_findings(tmp_path):
    """With both artifacts present, block 1 renders the universality table,
    a plotted heatmap (wrapped through `_frag`, so it carries a Plotly div),
    and appends a "SAE concept universality" finding with the real counts."""
    profile_a = [{"channel": "trend", "signed_null_units": 2.5, "n_members_clearing": 2}]
    profile_b = [{"channel": "seasonal", "signed_null_units": 3.0, "n_members_clearing": 1}]
    concepts_a = [_concept(0, [0, 1, 2], 3, 2, profile_a, name="trend riser")]
    concepts_b = [_concept(0, [0], 1, 1, profile_b, name="seasonal mover")]
    _write_concepts(tmp_path, {
        "Alpha/layer.0": _target_rec("Alpha", "layer.0", concepts_a),
        "Beta/layer.0": _target_rec("Beta", "layer.0", concepts_b)})
    _write_transfer(
        tmp_path,
        reach=[{"src": "Alpha/layer.0", "concept": 0, "dst_model": "Beta", "reciprocal": True}],
        matrix={"Alpha": {"Beta": 1.0}, "Beta": {"Alpha": 0.0}})
    findings = []
    out = sae_concepts_block(_fake_cfg(), tmp_path, findings, ["Alpha", "Beta"])
    assert "Concept universality" in out
    assert "plotly" in out.lower()
    uni_findings = [f for f in findings if "universality" in f.text]
    assert len(uni_findings) == 1
    assert "2 concepts" in uni_findings[0].text or "of 2 concepts" in uni_findings[0].text
    # Alpha's concept reaches its one other model (Beta) -> universal;
    # Beta's concept isn't in `reach` at all for Alpha -> model-specific.
    assert "1 universal" in uni_findings[0].text
    assert "1 model-specific" in uni_findings[0].text


# ---------------------------------------------------------------------------
# 3. Misfit three states (sec 11.37): not measured / measured-zero / found.
# ---------------------------------------------------------------------------

def test_misfits_zero_checked_when_no_ablation_artifact_on_disk(tmp_path):
    """`concepts.json` names a target, but its ablation artifact was never
    written -- `n_targets_checked` is `0` (not `None`: the only way to reach
    this renderer's misfit block at all is through a non-empty `cards`,
    which already requires at least one non-withheld target, so the
    "concepts.json itself is absent/empty" state `misfit_table` also
    disambiguates is unreachable from here -- covered directly at the
    `derived.misfit_table` unit level instead). This must render as "no
    target's ablation artifact could be loaded", never as "zero misfits
    found" (a real, different, measured claim)."""
    profile = [{"channel": "trend", "signed_null_units": 2.5, "n_members_clearing": 2}]
    concepts = [_concept(0, [0, 1], 2, 2, profile, centroid_cosine_mean=0.9)]
    _write_concepts(tmp_path, {"Alpha/layer.0": _target_rec("Alpha", "layer.0", concepts)})
    # Deliberately no `_write_ablation` call -- the target's own artifact
    # was never written to disk.
    out = sae_concepts_block(_fake_cfg(models=("Alpha",)), tmp_path, [], ["Alpha"])
    assert "Misfit detection" in out
    section = out.split("Misfits")[-1]
    assert "no target's ablation artifact could be loaded" in section
    assert "zero misfits found" not in section


def test_misfits_measured_zero_renders_distinctly_from_not_measured(tmp_path):
    """The ablation artifact exists and every member sits within its
    concept's own cohesion -- a real, measured "no misfits" verdict, which
    must name the number of targets actually checked."""
    profile = [{"channel": "trend", "signed_null_units": 2.5, "n_members_clearing": 2}]
    # Two members whose own ablation vector is nearly identical to the
    # centroid (both fire on `trend` only) -- cosine ~1.0, comfortably above
    # any `within_cosine_mean - gap` threshold.
    concepts = [_concept(0, [0, 1], 2, 2, profile, centroid_cosine_mean=0.999)]
    _write_concepts(tmp_path, {"Alpha/layer.0": _target_rec("Alpha", "layer.0", concepts)})

    # Both members fire on `trend` only, same direction as the centroid --
    # cosine 1.0, comfortably above `centroid_cosine_mean(0.999) - gap(0.3)`.
    _write_ablation(tmp_path, "Alpha", "layer.0",
                    [_candidate(0, {"trend": 3.0}), _candidate(1, {"trend": 3.0})])
    out = sae_concepts_block(_fake_cfg(models=("Alpha",)), tmp_path, [], ["Alpha"])
    section = out.split("Misfits")[-1]
    assert "zero misfits found" in section
    assert "not measured" not in section
    assert "1</b>" in section or ">1<" in section  # 1 target checked


def test_misfits_found_renders_own_channels_beside_concept_channels(tmp_path):
    """A member whose own ablation fingerprint diverges from its concept's
    centroid must render as a misfit, with its OWN top channels shown
    beside the concept's -- the whole point of the section is the
    divergence, per `sae/misfits.py`'s own docstring."""
    concept_profile = [{"channel": "trend", "signed_null_units": 3.0, "n_members_clearing": 1}]
    concepts = [_concept(0, [0, 1], 2, 1, concept_profile, centroid_cosine_mean=0.95)]
    _write_concepts(tmp_path, {"Alpha/layer.0": _target_rec("Alpha", "layer.0", concepts)})

    # Feature 0 matches the concept's own trend direction (cosine 1.0);
    # feature 1 fires almost entirely on a DIFFERENT channel (seasonal) --
    # its own ablation vector is orthogonal to the centroid (cosine 0.0),
    # far below `centroid_cosine_mean(0.95) - default_gap(0.3) = 0.65`.
    _write_ablation(tmp_path, "Alpha", "layer.0",
                    [_candidate(0, {"trend": 3.0}),
                     _candidate(1, {"seasonal": 3.0})])
    out = sae_concepts_block(_fake_cfg(models=("Alpha",)), tmp_path, [], ["Alpha"])
    section = out.split("Misfits")[-1]
    assert "misfit(s) found" in section
    assert "own fingerprint" in section
    assert "concept's fingerprint" in section
    assert "seasonal" in section  # the misfitting member's own channel
    assert "Alpha: 1 misfit" in section or "Alpha:" in section


# ---------------------------------------------------------------------------
# 4. Concept cards: capped at `concept_cards_max`, grouped by universality
#    bucket in the fixed order, and absent buckets render no header.
# ---------------------------------------------------------------------------

def test_cards_capped_at_max_and_grouped_by_present_buckets_only(tmp_path):
    """Five concepts across two models, `concept_cards_max=2` -- only the
    top 2 by interest render as cards, and since transfer is unmeasured
    here, only the "not measured" bucket header may appear (never
    "universal"/"partial"/"model-specific", which the run made no claim
    about)."""
    profiles = [
        [{"channel": "trend", "signed_null_units": 5.0, "n_members_clearing": 3}],
        [{"channel": "seasonal", "signed_null_units": 1.0, "n_members_clearing": 1}],
        [{"channel": "level", "signed_null_units": 4.0, "n_members_clearing": 2}],
    ]
    concepts_a = [_concept(i, [0, 1, 2][:n + 1], n + 1, n + 1, profiles[i],
                          name=f"concept-a-{i}")
                 for i, n in enumerate([2, 0, 1])]
    concepts_b = [_concept(0, [0], 1, 1,
                          [{"channel": "dispersion", "signed_null_units": 1.5,
                            "n_members_clearing": 1}],
                          name="concept-b-0")]
    _write_concepts(tmp_path, {
        "Alpha/layer.0": _target_rec("Alpha", "layer.0", concepts_a),
        "Beta/layer.0": _target_rec("Beta", "layer.0", concepts_b)})
    out = sae_concepts_block(_fake_cfg(max_cards=2), tmp_path, [], ["Alpha", "Beta"])
    assert "top 2 of 4 by interest" in out
    # The two strongest (causal_strength 5.0 and 4.0) must be shown; the
    # weakest (1.0) and Beta's (1.5) must not.
    assert "concept-a-0" in out
    assert "concept-a-2" in out
    assert "concept-a-1" not in out
    # No transfer measured for this run -> only "not measured" bucket header.
    assert "<h6>not measured" in out
    assert "<h6>universal" not in out
    assert "<h6>partial" not in out
    assert "<h6>model-specific" not in out


# ---------------------------------------------------------------------------
# 5. Item B (sec 32.3): the card leads with the description; the statistics
#    collapse. Covers all three description states a real run can produce.
# ---------------------------------------------------------------------------

def test_card_description_is_the_headline_and_statistics_collapse(tmp_path):
    """The description must render before the name/target line and the
    ablation sparklines, and must never be empty. ROADMAP.md sec 32.7d
    PRUNE (2026-09-15): generation is gone, so every present description is
    labelled as deterministically composed regardless of `description_
    generated` -- that field can still be `True` on an older run's stale
    `descriptions.json`, and the label must describe what the CURRENT
    pipeline does, not what produced a stale artifact (sec 32.3's own
    text, corrected in place). The statistics the card used to show
    unconditionally (member counts, n_members_clearing, interest, profile,
    transfer) must still be present, just collapsed into one `_details` per
    card -- never a second collapsing mechanism (sec 11.52's import-cycle
    constraint means `_details` is imported lazily; a second mechanism
    would duplicate it)."""
    profile = [{"channel": "trend", "signed_null_units": 2.5, "n_members_clearing": 2}]
    concepts = [
        _concept(0, [0, 1], 2, 2, profile, name="stale-flag concept",
                description="Patching this concept steepens the trend.",
                description_generated=True),
        _concept(1, [2, 3], 2, 2, profile, name="composed concept",
                description="These 2 features raise the forecast trend.",
                description_generated=False),
        _concept(2, [4, 5], 2, 2, profile, name="undescribed concept",
                description=None, description_generated=False),
    ]
    _write_concepts(tmp_path, {"Alpha/layer.0": _target_rec("Alpha", "layer.0", concepts)})
    out = sae_concepts_block(_fake_cfg(models=("Alpha",)), tmp_path, [], ["Alpha"])

    assert out.count("concept-card") == 3
    # No ablation artifact was written, so the misfits block (block 3)
    # contributes zero `<details>` here -- every one below is a card's own.
    assert out.count("<details") == 3
    assert out.count("the numbers behind this") == 3

    # No "(generated)" tag anywhere, even for the concept whose stale
    # artifact says `description_generated: True` -- the PRUNE means that
    # distinction no longer exists in the render layer.
    assert "<i>(generated)</i>" not in out
    assert out.count("<i>(composed from the measurements)</i>") == 2

    for desc_marker, tag in [
        ("Patching this concept steepens the trend.",
         "<i>(composed from the measurements)</i>"),
        ("These 2 features raise the forecast trend.",
         "<i>(composed from the measurements)</i>"),
        ("no description generated", None),
    ]:
        idx_desc = out.find(desc_marker)
        assert idx_desc != -1, desc_marker
        if tag:
            assert tag in out
        # The description must precede this card's own statistics line and
        # its own collapsing `<details>` -- i.e. it is the headline, not
        # something rendered after the numbers.
        idx_members = out.find("member(s)", idx_desc)
        idx_details = out.find("<details", idx_desc)
        assert idx_members != -1 and idx_members > idx_desc
        assert idx_details != -1 and idx_details > idx_desc

    # The statistics themselves are unchanged in substance, just relocated.
    assert out.count("member(s)") == 3
    assert out.count("clearing at least one") == 3
    assert "profile:" in out
    assert "transfer:" in out
