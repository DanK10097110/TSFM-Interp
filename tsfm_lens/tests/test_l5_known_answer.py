"""The L5 known-answer study's instruments (ROADMAP.md sec 38.3.5).

Every fixture plants a known answer and includes the confusable case on purpose:

* `concept_atoms` assigns a distributed concept's atoms by decoder match AND by
  carried effect, against an atom that carries a DIFFERENT concept's effect, an
  inert concept (assigned by cosine only), junk atoms and a dead copy;
* `build_variant_inputs`/`build_units` build V0 (one atom per side), V1 (the whole
  concept on both sides) and the matched set, with a decoy destination atom that
  has the best AUC but is not the concept;
* the rung itself, on the stub harness of `test_shared_input_agreement.py`, ablates
  WHOLE sets under V1, reads a sign-flipped destination "acts differently" and a
  destination with no effect "not scorable", never "same causal effect";
* the gate reads decoys' false "same" and partial agreement without merging it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import test_shared_input_agreement as base  # noqa: E402
from tsfm_lens.analysis import l5_known_answer as l5  # noqa: E402
from tsfm_lens.sae import shared_input_agreement as sia  # noqa: E402
from tsfm_lens.sae.transfer import auc_from_ranks, concept_scores, top_series  # noqa: E402


# ---------------------------------------------------------------------------
# 1. concept_atoms
# ---------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _manifest(dim=12):
    """Concept X (two components, effect), Y (one, effect), Z (one, INERT)."""
    e = np.eye(dim)
    comps = [
        {"id": "X#0", "concept": "X", "component": 0, "n_components": 2, "beta": 0.02, "direction_resid": e[0]},
        {"id": "X#1", "concept": "X", "component": 1, "n_components": 2, "beta": 0.02, "direction_resid": e[1]},
        {"id": "Y#0", "concept": "Y", "component": 0, "n_components": 1, "beta": 0.02, "direction_resid": e[2]},
        {"id": "Z#0", "concept": "Z", "component": 0, "n_components": 1, "beta": 0.0, "direction_resid": e[3]},
    ]
    return {"concepts": [{**c, "direction_resid": list(c["direction_resid"])} for c in comps]}


def _decoder(dim=12):
    e = np.eye(dim)
    rng = np.random.default_rng(3)
    junk = [_unit(rng.normal(size=dim) * np.r_[np.zeros(4), np.ones(dim - 4)]) for _ in range(3)]
    return np.stack([
        e[0],
        _unit(e[1] + 0.9 * e[8]),
        e[2],
        e[3],
        _unit(e[0] * 0.8 + e[2] * 0.6),
        e[0],
        *junk,
    ])


def test_concept_atoms_assigns_by_cosine_and_by_carried_effect_with_decoys():
    out = l5.concept_atoms(_manifest(), _decoder(), alive=np.array([True] * 5 + [False] + [True] * 3))
    assert out["X"]["atoms"][0] == 0
    assert 1 in out["X"]["atoms"], "an entangled atom carrying X's effect is assigned by effect"
    assert out["Y"]["atoms"] == [2]
    assert out["Z"]["atoms"] == [3], "an inert concept is assigned by decoder cosine only"
    assert 5 not in out["X"]["atoms"], "a dead atom is never assigned"
    assigned = [a for rec in out.values() for a in rec["atoms"]]
    assert len(assigned) == len(set(assigned)) or 3 in assigned
    for junk in (6, 7, 8):
        assert junk not in assigned
    assert out["X"]["main"] == 0 and out["X"]["carry"] == sorted(out["X"]["carry"], reverse=True)
    mixed = [c for c in ("X", "Y") if 4 in out[c]["atoms"]]
    assert mixed == ["X"], "an atom carrying two concepts goes to the one it carries most of"


def test_concept_atoms_without_a_carry_threshold_would_assign_junk():
    out = l5.concept_atoms(_manifest(), _decoder(), min_carry_fraction=0.0)
    assigned = {a for rec in out.values() for a in rec["atoms"]}
    assert {6, 7, 8} & assigned, "the plant is not vacuous: without the threshold junk is assigned"
    strict = l5.concept_atoms(_manifest(), _decoder())
    assert not ({6, 7, 8} & {a for rec in strict.values() for a in rec["atoms"]})


def test_control_atoms_are_alive_unplanted_and_low_carry():
    manifest, dec = _manifest(), _decoder()
    pooled = np.zeros((40, dec.shape[0]))
    pooled[:, :] = 1.0
    ctl = l5.control_atoms(manifest, dec, pooled, exclude={0, 1, 2, 3, 4}, n=2, seed=0, min_series=10)
    assert len(ctl["atoms"]) == 2 and set(ctl["atoms"]) <= {5, 6, 7, 8}
    again = l5.control_atoms(manifest, dec, pooled, exclude={0, 1, 2, 3, 4}, n=2, seed=0, min_series=10)
    assert again == ctl
    pooled[:, 6:] = 0.0
    only = l5.control_atoms(manifest, dec, pooled, exclude={0, 1, 2, 3, 4}, n=2, seed=0, min_series=10)
    assert only["atoms"] == [5], "an atom firing on too few series cannot be a control"


# ---------------------------------------------------------------------------
# 2. units: V0 / V1 / matched set
# ---------------------------------------------------------------------------

MODELS = ("A", "B")
TARGETS = {"A": "A/blocks.0", "B": "B/blocks.0"}
N_FEAT = 12


def _world():
    """A's concept = atoms [0,1,2] (main 0); B's = [4,5,6] (main 4). In B, atom 9 is a
    DECOY with the best forward AUC on A's top series but is not the concept."""
    rng = np.random.default_rng(0)
    n = 60
    trigger = np.zeros(n)
    trigger[:20] = np.linspace(2.0, 1.0, 20)
    pooled_a = rng.uniform(0, 0.05, size=(n, N_FEAT))
    pooled_b = rng.uniform(0, 0.05, size=(n, N_FEAT))
    for f in (0, 1, 2):
        pooled_a[:, f] += trigger * (1.0 - 0.05 * f)
    graded_b = trigger.copy()
    graded_b[[3, 14]] = graded_b[[14, 3]]
    for f in (4, 5, 6):
        pooled_b[:, f] += graded_b * (0.7 - 0.05 * f)
    pooled_b[:, 9] += trigger * 3.0
    pooled = {"A": pooled_a, "B": pooled_b}
    return {
        "pooled": pooled, "ranks": {m: rankdata(p, axis=0) for m, p in pooled.items()},
        "alive": {m: np.ones(N_FEAT, dtype=bool) for m in MODELS},
        "sets": {"a_shared_single": {"A": {"all": [0, 1, 2], "main": 0}, "B": {"all": [4, 5, 6], "main": 4}},
                 "e_control": {"A": {"all": [7, 8, 10], "main": 7}, "B": {"all": [7, 8, 10], "main": 7}}},
    }


def _inputs(variant, direction=("A", "B")):
    w = _world()
    return l5.build_variant_inputs(l5.VARIANT_SPECS[variant], w["sets"], direction, TARGETS, w["ranks"],
                                   w["pooled"], w["alive"], k_transfer=10)


def test_v0_hands_the_rung_one_atom_per_side_and_the_best_auc_decoy_destination():
    atlas, at, index = _inputs("V0")
    assert {(r["model"], r["feature"]) for r in atlas["rows"] if r["concept"] == 1} == {("A", 0)}
    assert at["tests"][0]["feature"] == 9, "the destination is the AUC argmax, decoy included"
    units = sia.build_units(atlas, at, dst_set="feature")
    shared = [u for u in units if index[u["concept"]][0] == "a_shared_single"][0]
    assert shared["src_features"] == [0] and shared["dst_features"] == [9]
    assert shared["dst_set_kind"] == "feature"


def test_v1_ablates_the_whole_concept_on_both_sides():
    atlas, at, index = _inputs("V1")
    units = sia.build_units(atlas, at, dst_set="concept_part")
    shared = [u for u in units if index[u["concept"]][0] == "a_shared_single"][0]
    assert shared["src_features"] == [0, 1, 2]
    assert shared["dst_features"] == [4, 5, 6] and shared["dst_set_kind"] == "concept_part"
    assert shared["dst_feature"] == 9, "U still comes from the transfer test's best feature"
    ctl = [u for u in units if index[u["concept"]][0] == "e_control"][0]
    assert ctl["dst_set_kind"] == "feature" and len(ctl["dst_features"]) == 1


def test_v0_srcset_and_v1b_variants_and_reverse_direction():
    atlas, at, index = _inputs("V0_srcset")
    shared = [u for u in sia.build_units(atlas, at) if index[u["concept"]][0] == "a_shared_single"][0]
    assert shared["src_features"] == [0, 1, 2] and shared["dst_features"] == [9]
    atlas, at, index = _inputs("V1b")
    shared = [u for u in sia.build_units(atlas, at, dst_set="matched_set")
              if index[u["concept"]][0] == "a_shared_single"][0]
    assert shared["dst_set_request"] == "matched_set"
    atlas, at, index = _inputs("V0", direction=("B", "A"))
    assert {t["src_model"] for t in at["tests"]} == {"B"} and at["tests"][0]["dst_model"] == "A"


def test_default_units_are_unchanged_by_the_new_options():
    atlas, at, _ = _inputs("V1")
    default = sia.build_units(atlas, at)
    explicit = sia.build_units(atlas, at, dst_set="feature", k_top_series=None)
    assert default == explicit
    for u in default:
        assert "transfer_k_top_series" not in u and "dst_set_request" not in u
        assert u["k_top_series"] == 10
    bigger = sia.build_units(atlas, at, k_top_series=20)
    assert all(u["k_top_series"] == 20 and u["transfer_k_top_series"] == 10 for u in bigger)
    with pytest.raises(ValueError, match="dst_set"):
        sia.build_units(atlas, at, dst_set="everything")


def test_matched_dst_set_is_top_auc_among_alive_features():
    w = _world()
    ranks, pooled = w["ranks"]["B"], w["pooled"]["B"]
    S = top_series(concept_scores(w["pooled"]["A"], [0]), 10)
    ctx = type("C", (), {"ranks": ranks, "alive_mask": np.ones(N_FEAT, dtype=bool)})()
    top3 = sia.matched_dst_set(ctx, S, 3)
    assert top3 == sorted(top3) and 9 in top3 and set(top3) <= {4, 5, 6, 9}
    dead = np.ones(N_FEAT, dtype=bool)
    dead[9] = False
    ctx.alive_mask = dead
    assert 9 not in sia.matched_dst_set(ctx, S, 3)
    assert len(sia.matched_dst_set(ctx, S, 0)) == 1


# ---------------------------------------------------------------------------
# 3. The rung: whole sets, sign flip, inert destination.
# ---------------------------------------------------------------------------

_DIM_SHAPES = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 0.5, -0.5, -1.0], [1.0, -1.0, 1.0, -1.0]])
_CONCEPT_DIMS = np.array([1.0, 0.6, 0.0])


class _ShapedAdapter(base._Adapter):
    """The base stub collapses every feature to one horizon shape, so every ablation
    is `+-` the same curve and no floor can sit below a real agreement. Here each
    residual dimension has its own horizon shape: a concept writes dims 0 and 1,
    while a distractor writes all three equally, so its shape differs."""

    def predict(self, contexts, horizon, quantiles):
        repl = base._MODULE_STATE.get(id(self.module))
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        vals = (np.zeros((len(rows), base.D_IN)) if repl is None
                else np.asarray(repl).reshape(len(rows), -1, base.D_IN).sum(axis=1))
        return {"point": (vals @ _DIM_SHAPES[:, :horizon]).astype(np.float32)}


def _set_run(monkeypatch, tmp_path, gains_a, gains_b, dst_set="feature", dst_feature=0,
             a_atlas=(0, 1, 2), b_atlas=(0, 1, 2), spy=None, n_null_directions=40):
    """Two stub models whose concept is a feature SET: `gains_*[i]` is the decode gain
    of feature `i` of the set. Concept 1 = A's atlas rows; B's atlas part is
    `b_atlas`; the transfer test names `dst_feature`."""
    pooled_a, w_dec_a = base._pooled_and_wdec(0, base.REAL_COL, real_gain=1.0)
    pooled_b, w_dec_b = base._pooled_and_wdec(0, base.REAL_COL, real_gain=1.0)
    for pooled, w_dec, gains, atl in ((pooled_a, w_dec_a, gains_a, a_atlas),
                                      (pooled_b, w_dec_b, gains_b, b_atlas)):
        for f, g in zip(atl, gains):
            pooled[:, f] = base.REAL_COL
            w_dec[f] = g * _CONCEPT_DIMS / base.D_IN
    base._MODULE_STATE.clear()
    sia.reset_caches()
    saes = {("A", "blocks_0"): base._SAE(pooled_a, w_dec_a), ("B", "blocks_0"): base._SAE(pooled_b, w_dec_b)}
    adapters = {"A": _ShapedAdapter("A"), "B": _ShapedAdapter("B")}
    data = base._Data()
    cfg = base._Cfg
    cfg.concepts.shared_input_n_null = 50
    cfg.concepts.n_null_directions = n_null_directions
    cfg.concepts.agreement_max_tests_per_pair = None
    cfg.concepts.agreement_dst_set = dst_set
    cfg.concepts.agreement_k_top_series = None
    cfg.concepts.agreement_partial_rung = True
    monkeypatch.setattr(sia, "capture_raw_tokens", base._fake_capture)
    monkeypatch.setattr(sia, "token_patch", base._fake_patch)
    monkeypatch.setattr(sia, "load_sae_checkpoint",
                        lambda path: saes[(Path(path).parent.name, Path(path).stem)])
    monkeypatch.setattr(sia, "load_all_windows",
                        lambda store, model, layer: data.contexts()[:, :base.D_IN].astype(np.float32))
    monkeypatch.setattr(sia, "reach_probe", lambda cfg, adapter, layer, data, device: {
        "reachable": True, "reason": ""})
    if spy is not None:
        real = sia.battery_for_set

        def _spy(ctx, features, *a, **k):
            spy.append((ctx.model, tuple(sorted([features] if isinstance(features, int) else features))))
            return real(ctx, features, *a, **k)
        monkeypatch.setattr(sia, "battery_for_set", _spy)
    S_a = top_series(concept_scores(pooled_a, list(a_atlas)), base.K_TOP)
    auc = float(auc_from_ranks(rankdata(pooled_b, axis=0), S_a)[dst_feature])
    atlas = {"rows": [{"model": "A", "layer": "blocks.0", "feature": f, "concept": 1} for f in a_atlas]
             + [{"model": "B", "layer": "blocks.0", "feature": f, "concept": 1} for f in b_atlas]}
    at = {"k_top_series": base.K_TOP, "tests": [
        {"concept": 1, "src_target": "A/blocks.0", "src_model": "A", "dst_target": "B/blocks.0",
         "dst_model": "B", "feature": dst_feature, "auc": auc, "reciprocal_fdr": True}]}
    return sia.run_shared_input_agreement(cfg, tmp_path, base._Hub(adapters), base._Store(
        {"A": pooled_a, "B": pooled_b}), data, "cpu", atlas, at, write=False)


def test_concept_part_ablates_the_whole_destination_set_and_the_default_does_not(monkeypatch, tmp_path):
    """Plant for 'make V1 ablate only the first feature': the spy sees every member."""
    spy: list = []
    out = _set_run(monkeypatch, tmp_path, [6.0, 6.0, 6.0], [6.0, 6.0, 6.0], dst_set="concept_part",
                   dst_feature=7, b_atlas=(0, 1, 2), spy=spy)
    t = out["tests"][0]
    assert t["dst_set_kind"] == "concept_part" and t["dst_features"] == [0, 1, 2]
    assert t["src_features"] == [0, 1, 2]
    real = [s for s in spy if s[1] in ((0, 1, 2),)]
    assert {m for m, _ in real} == {"A", "B"}, "both sides were ablated as whole sets"
    assert t["verdict"] == "same causal effect", t
    assert out["partial_agreement"]["same"] == 1 and out["tests"][0]["rung"] == "same"
    spy2: list = []
    out2 = _set_run(monkeypatch, tmp_path, [6.0, 6.0, 6.0], [6.0, 6.0, 6.0], dst_set="feature",
                    dst_feature=7, b_atlas=(0, 1, 2), spy=spy2)
    assert out2["tests"][0]["dst_features"] == [7] and out2["tests"][0]["dst_set_kind"] == "feature"
    assert out2["tests"][0]["verdict"] != "same causal effect"
    assert "partial_agreement" in out2 and "rung" in out2["tests"][0]


def test_sign_flipped_destination_set_reads_acts_differently_not_same(monkeypatch, tmp_path):
    """The (c) decoy under set ablation. Plant: scoring `abs` Spearman (a sign test
    that counts anti-correlation as agreement) makes this read 'same'."""
    out = _set_run(monkeypatch, tmp_path, [6.0, 6.0, 6.0], [-6.0, -6.0, -6.0], dst_set="concept_part",
                   dst_feature=7)
    t = out["tests"][0]
    assert t["statistic_i"]["observed"] == pytest.approx(-1.0)
    assert t["verdict"] == "acts differently", t
    assert out["partial_agreement"]["differs"] == 1


def test_destination_set_with_no_effect_is_not_scorable_never_same(monkeypatch, tmp_path):
    """The (d) decoy: the destination shares the trigger (it IS in the atlas part and
    fires on the same series) but its decode gain is 0, so its ablation moves
    nothing. Plant: treating every channel as clearing makes this read something
    other than 'not scorable'."""
    out = _set_run(monkeypatch, tmp_path, [6.0, 6.0, 6.0], [0.0, 0.0, 0.0], dst_set="concept_part",
                   dst_feature=7)
    t = out["tests"][0]
    assert t["verdict"] == "not scorable", t
    assert "B/blocks.0" in t["reason"]


def test_partial_rung_is_reported_beside_same_never_merged():
    tests = [{"verdict": v} for v in ("same causal effect", "level only", "shape only", "level only",
                                      "acts differently", "no specific agreement", "not scorable")]
    s = sia.partial_agreement_summary(tests)
    assert s["same"] == 1 and s["partial"] == 3 and s["partial_level_only"] == 2
    assert s["partial_shape_only"] == 1 and s["same_or_partial"] == 4
    assert s["differs"] == 1 and s["none"] == 1 and s["not scorable"] == 1
    assert sia.agreement_rung("level only") == "partial" != sia.agreement_rung("same causal effect")


# ---------------------------------------------------------------------------
# 4. The scorer and the gate.
# ---------------------------------------------------------------------------

SEEDS = [0, 1, 2, 3, 4]
FWD = "A->B"


def _rows(plan, variant="V0"):
    """`plan[(case, seed)] = verdict` for the A->B direction."""
    return [{"variant": variant, "seed": s, "case": c, "direction": FWD, "verdict": v}
            for (c, s), v in plan.items()]


def _plan(shared="same causal effect", opposite="acts differently", inputonly="not scorable"):
    plan = {}
    for s in SEEDS:
        plan[("a_shared_single", s)] = shared if s < 3 else "no specific agreement"
        plan[("b_shared_distributed", s)] = shared if s < 3 else "not scorable"
        plan[("c_opposite_effect", s)] = opposite
        plan[("d_input_only", s)] = inputonly
        plan[("e_control", s)] = "not scorable"
        plan[("x_pure_dispersion", s)] = "level only"
    return plan


def test_gate_passes_a_planted_adequate_variant_and_reports_exact_rates():
    summary = l5.variant_summary(_rows(_plan()), SEEDS, False, FWD)
    gate = l5.gate_variant(summary)
    assert gate["verdict"] == "ADEQUATE"
    assert gate["sensitivity"] == {"a_shared_single": 0.6, "b_shared_distributed": 0.6}
    assert gate["decoy_false_same_rate"] == 0.0 and gate["opposite_differs_rate"] == 1.0
    assert summary["cases"]["e_control"]["not_scorable_rate"] == 1.0


def test_gate_fails_on_decoys_called_same_and_on_low_sensitivity():
    """The decoy FPR is pooled over c AND d; an opposite decoy read 'same' in 1 of 10
    decoy cells is exactly at 0.10 (allowed), in 2 of 10 it fails."""
    plan = _plan()
    plan[("c_opposite_effect", 0)] = "same causal effect"
    one = l5.gate_variant(l5.variant_summary(_rows(plan), SEEDS, False, FWD))
    assert one["decoy_false_same_rate"] == pytest.approx(0.1) and one["checks"]["decoy_false_same"] is True
    plan[("d_input_only", 0)] = "same causal effect"
    two = l5.gate_variant(l5.variant_summary(_rows(plan), SEEDS, False, FWD))
    assert two["decoy_false_same_rate"] == pytest.approx(0.2)
    assert two["checks"]["decoy_false_same"] is False and two["verdict"] == "not adequate"
    low = l5.gate_variant(l5.variant_summary(
        _rows(_plan(shared="no specific agreement")), SEEDS, False, FWD))
    assert low["verdict"] == "not adequate"
    assert all(v is False for v in low["checks"]["sensitivity_shared"].values())


def test_v3_counts_partial_as_same_for_both_sensitivity_and_false_same():
    plan = _plan(shared="level only", opposite="level only")
    strict = l5.gate_variant(l5.variant_summary(_rows(plan), SEEDS, False, FWD))
    lenient = l5.variant_summary(_rows(plan), SEEDS, True, FWD)
    assert strict["sensitivity"]["a_shared_single"] == 0.0
    assert lenient["cases"]["a_shared_single"]["same_rate"] == 0.0, "partial is never merged into same"
    assert lenient["cases"]["a_shared_single"]["partial_rate"] == 0.6
    assert lenient["cases"]["a_shared_single"]["counted_same_rate"] == 0.6
    assert l5.gate_variant(lenient)["decoy_false_same_rate"] == 0.5, \
        "a decoy read 'level only' is a false same under the V3 reading"


def test_gate_is_not_evaluable_without_testable_seeds_and_records_sae_misses():
    plan = {k: v for k, v in _plan().items() if k[0] != "b_shared_distributed"}
    summary = l5.variant_summary(_rows(plan), SEEDS, False, FWD)
    assert summary["cases"]["b_shared_distributed"]["n_sae_miss"] == 5
    assert l5.gate_variant(summary)["verdict"] == "not evaluable"
    partial = {k: v for k, v in _plan().items() if k != ("a_shared_single", 4)}
    s2 = l5.variant_summary(_rows(partial), SEEDS, False, FWD)["cases"]["a_shared_single"]
    assert s2["n_testable"] == 4 and s2["n_sae_miss"] == 1
    assert s2["counted_same_rate"] == 0.75 and s2["counted_same_rate_all_seeds"] == 0.6


def test_dictionaries_that_cannot_coincide_are_measured_after_the_best_linear_map():
    rng = np.random.default_rng(0)
    n, da, db = 400, 8, 12
    latent = rng.normal(size=(n, 6))
    a_map, b_map = rng.normal(size=(6, da)), rng.normal(size=(6, db))
    act_a, act_b = latent @ a_map, latent @ b_map
    dec_a, dec_b = rng.normal(size=(30, da)), rng.normal(size=(40, db))
    shared_a, shared_b = a_map[0], b_map[0]
    out = l5.dictionary_dissimilarity(dec_a, dec_b, act_a, act_b, shared_a, shared_b, seed=0)
    assert (out["width_a"], out["width_b"]) == (8, 12) and out["dict_b"] == 40
    assert 0.0 < out["linear_cka_activations"] <= 1.0
    assert out["shared_concept_direction_cosine_after_map"] > 0.99
    assert out["mapped_b_to_a_max_abs_cosine"]["share_ge_0.9"] < 0.5
    assert out["chance_max_abs_cosine_random_directions"]["median"] > 0.0


def test_new_knobs_leave_every_existing_fingerprint_untouched_and_are_validated():
    import tempfile
    from tests.test_concept_stage import _cfg
    from tsfm_lens.manifest import resolve_config_keys
    from tsfm_lens.pipeline import _stage_by_name
    from tsfm_lens.sae.concept_stage import preflight_problem

    cfg = _cfg(tempfile.mkdtemp())
    keys = _stage_by_name("concepts").config_keys
    before = resolve_config_keys(cfg, keys)
    flat = repr(before)
    for name in ("agreement_dst_set", "agreement_k_top_series", "agreement_partial_rung"):
        assert name not in flat, f"{name} must be omitted at its default"
    cfg.concepts.agreement_dst_set = "concept_part"
    assert resolve_config_keys(cfg, keys) != before
    cfg.concepts.agreement_dst_set = "feature"
    assert resolve_config_keys(cfg, keys) == before
    cfg.sae.persist_features = True
    assert preflight_problem(cfg) is None
    cfg.concepts.agreement_dst_set = "all of them"
    assert "agreement_dst_set" in preflight_problem(cfg)
    cfg.concepts.agreement_dst_set = "matched_set"
    cfg.concepts.agreement_k_top_series = 1
    assert "agreement_k_top_series" in preflight_problem(cfg)
    cfg.concepts.agreement_k_top_series = 40
    assert preflight_problem(cfg) is None


def test_driver_config_has_two_different_architectures_and_no_absolute_paths():
    import yaml
    import run_l5_known_answer as drv
    raw = yaml.safe_load((ROOT / "configs" / "l5_known_answer.yaml").read_text(encoding="utf-8"))
    assert not str(raw["data"]["path"]).startswith("/") and not str(raw["run"]["out_dir"]).startswith("/")
    kw = [m["kwargs"] for m in raw["models"]]
    assert kw[0]["width"] != kw[1]["width"] and kw[0]["depth"] != kw[1]["depth"]
    assert kw[0]["rotate"] is False and kw[1]["rotate"] is True
    assert {k["vocabulary"] for k in kw} == {"l5"}
    cfg = drv.build_cell_config(str(ROOT / "configs" / "l5_known_answer.yaml"), "/c/public_dev", "/o", 3,
                                None, dose=2.0)
    assert cfg.run.seed == 3 and cfg.data.path == "/c/public_dev"
    assert all(m.kwargs["construction_seed"] == 3 and m.kwargs["dose"] == 2.0 for m in cfg.models)
    assert drv._targets(cfg) == {"ArchA": "ArchA/blocks.2", "ArchB": "ArchB/blocks.4"}
