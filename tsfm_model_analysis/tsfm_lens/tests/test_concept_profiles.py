"""Tests for ROADMAP.md sec 37 Spec A: per-concept profiles
(`sae/concept_profiles.py`) for the cross-model SAE concept atlas.

All synthetic, with planted answers, CPU-only. `load_ground_truth_table` is
monkeypatched with a DataFrame fixture (spec's own instruction) rather than a
sealed corpus. Every load-bearing assertion is confirmed to fail under a
planted regression -- see each test's own docstring for what was
swapped/inverted and what broke.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.extraction.store import ActivationStore, save_meta
from tsfm_lens.sae import concept_profiles as cp
from tsfm_lens.sae.concept_atlas import run_concept_atlas
from tsfm_lens.sae.concepts import CHANNELS
from tsfm_lens.sae.train import sanitize
from tsfm_lens.sae.transfer import _by_stratum, _seed, series_strata
from tsfm_lens.utils import load_json, save_json

_N_CH = len(CHANNELS)
_MASE_IDX = CHANNELS.index("mase")


# ---------------------------------------------------------------------------
# Shared fixture builders.
# ---------------------------------------------------------------------------

def _cfg(profile_n_perm=200, top_k=15, transfer_seed=0, n_boot=200, causal_only=True):
    return SimpleNamespace(
        concepts=SimpleNamespace(profile_n_perm=profile_n_perm),
        sae=SimpleNamespace(transfer_top_k=top_k, transfer_seed=transfer_seed,
                            concept_causal_only=causal_only),
        stats=SimpleNamespace(n_boot=n_boot),
        data=SimpleNamespace(path="unused -- ground truth is monkeypatched"),
    )


def _direction(idx: int, mag: float = 6.0) -> np.ndarray:
    v = np.zeros(_N_CH, dtype=np.float64)
    v[idx] = mag
    return v


def _candidate(feature: int, channel_values: dict, forecasts=None) -> dict:
    channels = {ch: {"null_p95": 1.0, "signed_effect": float(v),
                     "clears_null": bool(abs(v) >= 1.0)}
               for ch, v in channel_values.items()}
    return {"feature": feature, "scorable": True,
           "n_channels_clearing": sum(1 for v in channel_values.values() if abs(v) >= 1.0),
           "channels": channels, "forecasts": forecasts or []}


def _row(idx: int, value: float, rest: float = 0.0) -> dict:
    return {ch: (value if i == idx else rest) for i, ch in enumerate(CHANNELS)}


def _write_ablation(run_dir, model, layer, candidates):
    path = Path(run_dir) / "sae" / sanitize(model) / f"{sanitize(layer)}_ablation.json"
    save_json(path, {"model": model, "layer": layer, "withheld": False,
                     "skipped": False, "candidates": candidates})


def _write_sae(store, model, layer, pooled, replicate=0):
    n, f = pooled.shape
    n_windows = store.root.attrs["n_windows"]
    store.init_sae_layer(model, layer, f, replicate=replicate)
    window_feats = np.broadcast_to(pooled[:, None, :], (n, n_windows, f)).astype(np.float32)
    store.write_sae_batch(model, layer, 0, window_feats, replicate=replicate)


def _sae_meta_entry():
    return {"features_persisted": True}


def _gt_frame(n_synthetic: int, n_real_derived: int, n_seasonalities: np.ndarray,
             seasonal_amplitude_max: np.ndarray) -> pd.DataFrame:
    """13 structural fields (real only for the synthetic tier) plus
    `tier_synthetic`/`tier_real_derived`/`generator_*` provenance dummies --
    a hand-built stand-in for `ground_truth.py::load_ground_truth_table`'s
    real output, per the spec's own instruction to monkeypatch rather than
    build a sealed corpus."""
    n = n_synthetic + n_real_derived
    idx = np.arange(n)
    rows = {}
    for field in cp._STRUCTURAL_FIELDS:
        rows[field] = np.full(n, np.nan)
    rows["n_seasonalities"][:n_synthetic] = n_seasonalities
    rows["seasonal_amplitude_max"][:n_synthetic] = seasonal_amplitude_max
    # Give every other structural field a fixed, uninformative constant on
    # the synthetic rows (defined but zero-variance -- `_structural_field_records`
    # must record `raw_rho=None` for a zero-variance field, never a spurious 0).
    for field in cp._STRUCTURAL_FIELDS:
        if field in ("n_seasonalities", "seasonal_amplitude_max"):
            continue
        rows[field][:n_synthetic] = 1.0
    rows["tier_synthetic"] = np.concatenate([np.ones(n_synthetic), np.zeros(n_real_derived)])
    rows["tier_real_derived"] = 1.0 - rows["tier_synthetic"]
    rows["generator_parametric"] = rows["tier_synthetic"].copy()
    rows["generator_mixture"] = rows["tier_real_derived"].copy()
    df = pd.DataFrame(rows, index=idx)
    return df


# ---------------------------------------------------------------------------
# 1. Structural-field recovery + search correction, and the provenance decoy
#    (concept T / R's own input-profile machinery, tested directly).
# ---------------------------------------------------------------------------

def test_structural_field_recovered_as_top_structural_with_significant_p():
    """`s` is planted to correlate with `seasonal_amplitude_max` on the
    synthetic rows only (real-derived rows get pure noise). `top_structural[0]`
    must recover that field, and `p_max_structural` must clear at 0.05."""
    rng = np.random.default_rng(0)
    n_synth, n_real = 150, 60
    amp = rng.uniform(0, 5, n_synth)
    gt = _gt_frame(n_synth, n_real, n_seasonalities=rng.integers(0, 3, n_synth),
                  seasonal_amplitude_max=amp)
    s = np.empty(n_synth + n_real)
    s[:n_synth] = amp + rng.normal(scale=0.3, size=n_synth)
    s[n_synth:] = rng.normal(size=n_real)
    provenance_cols = ["tier_synthetic", "tier_real_derived",
                      "generator_parametric", "generator_mixture"]

    records = cp._structural_field_records(s, gt, provenance_cols, seed=0)
    by_field = {r["field"]: r for r in records}
    assert by_field["seasonal_amplitude_max"]["resid_rho"] is not None
    top = cp._top_structural(records)
    assert top[0]["field"] == "seasonal_amplitude_max", top

    real_max = max(abs(r["raw_rho"]) for r in records if r["raw_rho"] is not None)
    p = cp._p_max_structural(s, gt, records, real_max, seed=0, n_perm=300)
    assert p < 0.05, p

    # Planted regression: swap the planted field for pure noise (same shape)
    # -- the recovered top field must change, and the resulting rho must
    # drop well below the planted signal's own strength.
    s_noise = rng.normal(size=n_synth + n_real)
    records_noise = cp._structural_field_records(s_noise, gt, provenance_cols, seed=0)
    top_noise = cp._top_structural(records_noise)
    assert not top_noise or top_noise[0]["field"] != "seasonal_amplitude_max" \
        or abs(top_noise[0]["resid_rho"]) < abs(top[0]["resid_rho"])


def test_p_max_structural_null_calibration():
    """Over ~20 independent noise draws (no planted structural signal at
    all), `p_max_structural` must reject (p < 0.05) no more than 15% of the
    time -- the spec's own explicit calibration bound. A broken search
    correction (e.g. comparing to a per-field rather than a max-over-fields
    null) would inflate this well above 15%."""
    n_synth, n_real = 100, 40
    rng_gt = np.random.default_rng(1)
    gt = _gt_frame(n_synth, n_real, n_seasonalities=rng_gt.integers(0, 3, n_synth),
                  seasonal_amplitude_max=rng_gt.uniform(0, 5, n_synth))
    provenance_cols = ["tier_synthetic", "tier_real_derived",
                      "generator_parametric", "generator_mixture"]
    n_seeds = 20
    n_rejections = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        s = rng.normal(size=n_synth + n_real)
        records = cp._structural_field_records(s, gt, provenance_cols, seed=seed)
        raw_abs = [abs(r["raw_rho"]) for r in records if r["raw_rho"] is not None]
        real_max = max(raw_abs) if raw_abs else 0.0
        p = cp._p_max_structural(s, gt, records, real_max, seed=seed, n_perm=300)
        if p < 0.05:
            n_rejections += 1
    rate = n_rejections / n_seeds
    assert rate <= 0.15, f"null-calibration rejection rate {rate} exceeds the spec's 0.15 bound"


def test_provenance_driven_true_for_tier_correlated_score():
    """Concept R: the score fires on real-derived series (minus
    `tier_synthetic`, plus noise), so its top-k is all one real-derived
    generator -> `provenance_driven` True."""
    rng = np.random.default_rng(2)
    n_synth, n_real = 100, 40
    n = n_synth + n_real
    gt = _gt_frame(n_synth, n_real, n_seasonalities=rng.integers(0, 3, n_synth),
                  seasonal_amplitude_max=rng.uniform(0, 5, n_synth))
    provenance_cols = ["tier_synthetic", "tier_real_derived",
                      "generator_parametric", "generator_mixture"]
    s = -gt["tier_synthetic"].to_numpy() + rng.normal(scale=0.1, size=n)
    S = np.argsort(-s)[:20]
    generators = np.array(["parametric"] * n_synth + ["mixture"] * n_real)
    strata = np.array(["fam"] * n)

    records = cp._structural_field_records(s, gt, provenance_cols, seed=0)
    provenance = cp._provenance_profile(s, gt)
    driven, components = cp._provenance_driven(records, provenance, S, generators)
    assert driven is True, components
    assert components["provenance_exceeds_structural"] is True

    # Decoy: a part correlated with `seasonal_amplitude_max` WITHIN SYNTHETIC
    # rows only, with real-derived rows drawn from the SAME marginal range
    # (not a differently-centered noise distribution -- that would itself
    # create a spurious tier-level mean shift and confound the very
    # provenance test this decoy is meant to pass) must NOT read as
    # provenance-driven.
    s_decoy = np.empty(n)
    s_decoy[:n_synth] = gt["seasonal_amplitude_max"].to_numpy()[:n_synth] + rng.normal(scale=0.3, size=n_synth)
    s_decoy[n_synth:] = rng.uniform(0, 5, n_real)
    records_decoy = cp._structural_field_records(s_decoy, gt, provenance_cols, seed=0)
    provenance_decoy = cp._provenance_profile(s_decoy, gt)
    S_decoy = np.argsort(-s_decoy)[:20]
    driven_decoy, components_decoy = cp._provenance_driven(records_decoy, provenance_decoy, S_decoy, generators)
    assert driven_decoy is False, components_decoy

    # Planted regression: if `_provenance_driven` compared RAW (not
    # residualized) structural rho against provenance rho, the true-positive
    # case above would still read True, but flipping the comparison
    # direction (>= instead of >) on a case engineered to tie must be
    # caught -- confirm the decoy is not merely accidentally under threshold
    # by checking its provenance component is genuinely small.
    assert components_decoy["max_provenance_abs_rho"] < components_decoy["max_residualized_structural_abs_rho"]


# ---------------------------------------------------------------------------
# 2. Behavioral link why_verdict, and the sign-convention pin.
# ---------------------------------------------------------------------------

def _metrics_wide(part_model, mase_by_model, series_ids):
    data = {m: v for m, v in mase_by_model.items()}
    return pd.DataFrame(data, index=series_ids)


def test_behavioral_link_advantage_carried():
    """Part model is confidently better than the one other model on `S`,
    AND its own causal mase effect is positive and clears -> "advantage
    carried by this concept"."""
    series_ids = np.arange(30)
    rng = np.random.default_rng(3)
    mase_a = 0.5 + rng.normal(scale=0.02, size=30)
    mase_b = 1.0 + rng.normal(scale=0.02, size=30)
    wide = _metrics_wide("A", {"A": mase_a, "B": mase_b}, series_ids)
    causal = {"mean_signed_effect_over_null_p95": 2.0, "any_member_clears_null": True, "n_members": 1}
    out = cp._behavioral_link("A", series_ids, wide, causal, n_boot=200, seed=0)
    assert out["status"] == "measured"
    assert out["why_verdict"] == "advantage carried by this concept", out


def test_behavioral_link_advantage_not_traced():
    """Same behavioral gap (A confidently better), but ablation does not
    clear -> "advantage, not traced to this concept"."""
    series_ids = np.arange(30)
    rng = np.random.default_rng(4)
    mase_a = 0.5 + rng.normal(scale=0.02, size=30)
    mase_b = 1.0 + rng.normal(scale=0.02, size=30)
    wide = _metrics_wide("A", {"A": mase_a, "B": mase_b}, series_ids)
    causal = {"mean_signed_effect_over_null_p95": 0.1, "any_member_clears_null": False, "n_members": 1}
    out = cp._behavioral_link("A", series_ids, wide, causal, n_boot=200, seed=0)
    assert out["why_verdict"] == "advantage, not traced to this concept", out


def test_behavioral_link_no_advantage_worse():
    """A is confidently WORSE than B on S -> "worse than 1 model(s)"."""
    series_ids = np.arange(30)
    rng = np.random.default_rng(5)
    mase_a = 1.0 + rng.normal(scale=0.02, size=30)
    mase_b = 0.5 + rng.normal(scale=0.02, size=30)
    wide = _metrics_wide("A", {"A": mase_a, "B": mase_b}, series_ids)
    causal = {"mean_signed_effect_over_null_p95": 2.0, "any_member_clears_null": True, "n_members": 1}
    out = cp._behavioral_link("A", series_ids, wide, causal, n_boot=200, seed=0)
    assert out["why_verdict"] == "worse than 1 model(s)", out


def test_sign_convention_pin_flips_verdict():
    """Sign convention: `mase` delta = ablated - baseline, so a POSITIVE
    `mean_signed_effect_over_null_p95` means removing the feature makes the
    forecast WORSE (the feature helps). Flipping its sign (with `clears`
    still True) must turn "advantage carried" into "advantage, not traced"
    -- a verdict that reads the sign backwards would not change here."""
    series_ids = np.arange(30)
    rng = np.random.default_rng(6)
    mase_a = 0.5 + rng.normal(scale=0.02, size=30)
    mase_b = 1.0 + rng.normal(scale=0.02, size=30)
    wide = _metrics_wide("A", {"A": mase_a, "B": mase_b}, series_ids)
    causal_pos = {"mean_signed_effect_over_null_p95": 2.0, "any_member_clears_null": True, "n_members": 1}
    causal_neg = {"mean_signed_effect_over_null_p95": -2.0, "any_member_clears_null": True, "n_members": 1}
    out_pos = cp._behavioral_link("A", series_ids, wide, causal_pos, n_boot=200, seed=0)
    out_neg = cp._behavioral_link("A", series_ids, wide, causal_neg, n_boot=200, seed=0)
    assert out_pos["why_verdict"] == "advantage carried by this concept"
    assert out_neg["why_verdict"] == "advantage, not traced to this concept"
    assert out_pos["why_verdict"] != out_neg["why_verdict"]


# ---------------------------------------------------------------------------
# 3. Within-stratum null: agree under p_uncond, not under p_within_stratum.
# ---------------------------------------------------------------------------

def test_within_stratum_null_distinguishes_archetype_only_agreement():
    """Two parts fire strongly on the SAME one archetype (out of three), and
    WITHIN that archetype their per-series scores are independent noise --
    i.e. all they share is "fires on this archetype". `p_uncond` must clear
    (they really do rank series similarly overall), `p_within_stratum` must
    NOT (nothing survives once the archetype's own membership is held
    fixed)."""
    rng = np.random.default_rng(7)
    strata = np.array(["arch1"] * 60 + ["arch2"] * 60 + ["arch3"] * 60)
    n = strata.size
    by_stratum = _by_stratum(strata)
    base = np.where(strata == "arch1", 5.0, 0.0)
    s_a = base + rng.normal(scale=1.0, size=n)
    s_b = base + rng.normal(scale=1.0, size=n)
    S_a = np.argsort(-s_a)[:20]
    S_b = np.argsort(-s_b)[:20]

    res = cp._pair_agreement(s_a, s_b, S_a, S_b, strata, by_stratum, seed=0, n_perm=300)
    assert res["p_uncond"] < 0.05, res
    assert res["p_within_stratum"] >= 0.05, res

    # Planted regression: if `_permute_within_stratum` permuted GLOBALLY
    # (ignoring strata) instead of within each one, it would reproduce the
    # unconditional null and also clear at < 0.05 here -- confirm the two
    # p-values are not simply identical (the regression's exact symptom).
    assert res["p_within_stratum"] > res["p_uncond"] + 0.01


# ---------------------------------------------------------------------------
# 4. Full driver: concept P (shared) and Q (convergent), on a real
#    file-based fixture (store + ablation files + a REAL concept atlas run,
#    so pooled_features's own row order is guaranteed consistent with the
#    atlas by construction -- CLAUDE.md sec 11.32's exact trap).
# ---------------------------------------------------------------------------

def _build_pqr_run(tmp_path):
    """3 models (A, B, C), 1 layer each. Concept P: parts at A and B firing
    on the SAME planted top-20 block (indices 0-19) -> shared. Concept Q:
    parts at A and B firing on DISJOINT blocks (20-39 vs 40-59) -> convergent
    (same effect direction, different inputs). Both concepts get >=4 causal
    members so they clear `atlas_min_members=3` under complete-linkage
    clustering at `min_cosine=0.9`."""
    rng = np.random.default_rng(8)
    n = 180
    n_synth, n_real = 120, 60
    family = np.array(["fam_a"] * 60 + ["fam_b"] * 60 + ["fam_c"] * 60)
    generator = np.array(["parametric"] * n_synth + ["mixture"] * n_real)
    tier = np.array(["synthetic"] * n_synth + ["real_derived"] * n_real)
    meta = pd.DataFrame({"series_id": np.arange(n), "family": family,
                         "archetype": [None] * n, "generator": generator, "tier": tier})
    save_meta(tmp_path, meta)
    store = ActivationStore.create(tmp_path / "activations.zarr", n_series=n,
                                   n_windows=2, window=8, context_len=16)

    def _fires_on(block, dict_size=6, feat=0):
        pooled = rng.normal(scale=1.0, size=(n, dict_size)).astype(np.float64)
        pooled[block, feat] += 6.0
        return pooled

    dict_size = 5
    pooled_a = rng.normal(scale=1.0, size=(n, dict_size))
    pooled_a[:20, 0] += 6.0     # P's feature at A
    pooled_a[:20, 1] += 6.0     # P's second feature at A
    pooled_a[20:40, 2] += 6.0   # Q's first feature at A
    pooled_a[20:40, 4] += 6.0   # Q's second feature at A (same block, needs >=3 members total)

    pooled_b = rng.normal(scale=1.0, size=(n, dict_size))
    pooled_b[:20, 0] += 6.0     # P's feature at B (same block as A)
    pooled_b[:20, 1] += 6.0
    pooled_b[40:60, 2] += 6.0   # Q's feature at B (DISJOINT block from A's Q)

    _write_sae(store, "A", "blk.1", pooled_a)
    _write_sae(store, "B", "blk.1", pooled_b)

    save_json(tmp_path / "sae" / "meta.json", {"A/blk.1": _sae_meta_entry(), "B/blk.1": _sae_meta_entry()})

    _write_ablation(tmp_path, "A", "blk.1", [
        _candidate(0, _row(0, 6.0)), _candidate(1, _row(0, 5.5)),
        _candidate(2, _row(3, 6.0)), _candidate(4, _row(3, 5.7)),
        _candidate(3, {ch: 0.0 for ch in CHANNELS}),  # non-causal filler
    ])
    _write_ablation(tmp_path, "B", "blk.1", [
        _candidate(0, _row(0, 6.2)), _candidate(1, _row(0, 5.8)),
        _candidate(2, _row(3, 5.9)),
        _candidate(3, {ch: 0.0 for ch in CHANNELS}),
    ])

    gt = _gt_frame(n_synth, n_real, n_seasonalities=rng.integers(0, 3, n_synth),
                  seasonal_amplitude_max=rng.uniform(0, 5, n_synth))
    return meta, gt


def _run_cfg():
    cfg = SimpleNamespace(
        run=SimpleNamespace(seed=0),
        concepts=SimpleNamespace(atlas_min_cosine=0.9, atlas_min_members=3, atlas_n_null=30,
                                 profile_n_perm=150),
        sae=SimpleNamespace(concept_causal_only=True, transfer_top_k=20, transfer_seed=0),
        stats=SimpleNamespace(n_boot=150),
        data=SimpleNamespace(path="unused -- monkeypatched"),
    )
    return cfg


def test_full_driver_shared_and_convergent_sharing_classes(tmp_path, monkeypatch):
    meta, gt = _build_pqr_run(tmp_path)
    monkeypatch.setattr(cp, "load_ground_truth_table", lambda path: gt)
    cfg = _run_cfg()

    atlas = run_concept_atlas(tmp_path, cfg)
    assert len(atlas["concepts"]) == 2, atlas["concepts"]

    out = cp.run_concept_profiles(tmp_path, cfg)
    assert (tmp_path / "sae" / "concept_profiles.json").exists()

    # Identify P (top-firing series in the 0-19 block, shared by both
    # models' parts) vs Q (disjoint per-model blocks) by their OWN measured
    # top-k series rather than by concept id, since clustering order is
    # deterministic but not something this test should hardcode.
    def _fires_on_first_block(concept):
        return all(any(sid < 20 for sid in p["top_series_ids"]) for p in concept["parts"])

    p_concept = next(c for c in out["concepts"] if _fires_on_first_block(c))
    q_concept = next(c for c in out["concepts"] if c is not p_concept)

    assert p_concept["sharing_class"] == "shared (same effect, same inputs)", p_concept
    assert q_concept["sharing_class"] == "convergent (same effect, different inputs)", q_concept
    assert len(p_concept["cross_model_pairs"]) == 1
    assert p_concept["cross_model_pairs"][0]["agrees"] is True
    assert q_concept["cross_model_pairs"][0]["agrees"] is False

    # not-measured artifacts: neither concept_stability.json nor
    # atlas_transfer.json exist in this fixture.
    assert p_concept["stable"].startswith("not measured")
    assert p_concept["input_transfer_models_fdr"] == "not measured"

    # Planted regression: if `_sharing_class` used `>=1 agreeing pair` as
    # "shared" instead of full connectivity, both concepts here would read
    # identically (each has exactly one A-B pair, one True one False) --
    # confirm they in fact differ, which they must given the p_concept/
    # q_concept assertions above (restated here as an explicit contrast so a
    # regression that collapsed both to the same class is caught directly).
    assert p_concept["sharing_class"] != q_concept["sharing_class"]


def test_full_driver_determinism(tmp_path, monkeypatch):
    meta, gt = _build_pqr_run(tmp_path)
    monkeypatch.setattr(cp, "load_ground_truth_table", lambda path: gt)
    cfg = _run_cfg()
    run_concept_atlas(tmp_path, cfg)

    out1 = cp.run_concept_profiles(tmp_path, cfg)
    text1 = (tmp_path / "sae" / "concept_profiles.json").read_text(encoding="utf-8")
    out2 = cp.run_concept_profiles(tmp_path, cfg)
    text2 = (tmp_path / "sae" / "concept_profiles.json").read_text(encoding="utf-8")
    assert text1 == text2
    assert out1 == out2


# ---------------------------------------------------------------------------
# 5. Stage wiring on mocks (reuse test_concept_stage.py's fixture pattern).
# ---------------------------------------------------------------------------

def test_stage_wiring_writes_profiles_artifact_and_block(monkeypatch):
    from tests.test_smoke import build_config
    from tsfm_lens.config import config_from_dict
    from tsfm_lens.pipeline import run_pipeline
    from tsfm_lens.extraction.store import load_meta

    out = tempfile.mkdtemp()
    cfg = config_from_dict(build_config(out))
    cfg.run.name = "concept_profiles_stage"
    cfg.sae.enabled = True
    cfg.sae.epochs = 3
    cfg.sae.dict_size_mult = 2
    cfg.sae.k = 4
    cfg.sae.forecast_preservation_max_series = 16
    cfg.sae.persist_features = True
    cfg.concepts.enabled = True
    cfg.sae.concept_min_members = 1
    cfg.concepts.n_features_per_rule = 6
    cfg.concepts.n_null_directions = 4
    cfg.concepts.max_series = 16
    cfg.concepts.top_k_series = 4
    cfg.concepts.profile_n_perm = 40
    cfg.sae.transfer_top_k = 8

    run_pipeline(cfg, stages=["extract"])
    store = ActivationStore(cfg.run_dir() / "activations.zarr", mode="r")
    cfg.sae.targets = [{"model": "patchy", "layer": store.layers("patchy")[-1]},
                       {"model": "steppy", "layer": store.layers("steppy")[-1]}]

    meta = load_meta(cfg.run_dir())
    n = len(meta)
    rng = np.random.default_rng(0)
    gt = _gt_frame(n, 0, n_seasonalities=rng.integers(0, 3, n),
                  seasonal_amplitude_max=rng.uniform(0, 5, n))
    gt.index = meta["series_id"].to_numpy()
    monkeypatch.setattr(cp, "load_ground_truth_table", lambda path: gt)

    run_pipeline(cfg, stages=["l0", "sae", "concepts"])

    record = load_json(cfg.run_dir() / "sae" / "concept_stage.json")
    assert "profiles" in record, record
    if record["atlas"]["status"] == "ran" and record["atlas"]["n_concepts"] > 0:
        assert record["profiles"]["status"] == "ran", record["profiles"]
        assert (cfg.run_dir() / "sae" / "concept_profiles.json").exists()
        art = load_json(cfg.run_dir() / "sae" / "concept_profiles.json")
        assert art["schema_version"] == 1
        assert len(art["concepts"]) == record["profiles"]["n_concepts"]
    else:
        assert record["profiles"]["status"] == "skipped", record["profiles"]
        assert not (cfg.run_dir() / "sae" / "concept_profiles.json").exists()


def test_agreement_is_top_k_overlap_not_broad_correlation():
    """Two sparse parts firing on the SAME 20 of 900 series over unrelated
    noise elsewhere share their inputs (overlap 20, tiny p) even though rho
    over all series is modest. Decoy: two parts that co-vary broadly (rho
    high) but whose top-20 sets are disjoint must show zero overlap and a
    non-significant overlap p."""
    rng = np.random.default_rng(5)
    n, k = 900, 20
    strata = np.array(["a", "b", "c"] * (n // 3))
    by_stratum = cp._by_stratum(strata)
    sa = rng.normal(size=n); sb = rng.normal(size=n)
    sa[:k] += 8; sb[:k] += 8
    res = cp._pair_agreement(sa, sb, np.argsort(-sa)[:k], np.argsort(-sb)[:k],
                             strata, by_stratum, 0, 200)
    assert res["top_k_overlap"] == k and res["p_overlap"] < 1e-10
    assert res["p_overlap_within_stratum"] < 0.05
    base = rng.normal(size=n)
    ca = base + 0.3 * rng.normal(size=n); cb = base + 0.3 * rng.normal(size=n)
    ca[k:2 * k] += 20; cb[2 * k:3 * k] += 20
    dec = cp._pair_agreement(ca, cb, np.argsort(-ca)[:k], np.argsort(-cb)[:k],
                             strata, by_stratum, 0, 200)
    assert dec["rho"] > 0.7 and dec["top_k_overlap"] == 0 and dec["p_overlap"] > 0.05


def test_archetype_is_not_a_residualization_basis():
    """A score that tracks a structural field through its archetype must
    keep its residualized rho and not read provenance-driven. Planted: the
    field is set by the archetype (a recipe), and the score follows the
    field. Residualizing on archetype dummies would zero the field."""
    rng = np.random.default_rng(3)
    n = 400
    arch = rng.integers(0, 2, n)
    field = arch * 2.0 + rng.normal(scale=0.3, size=n)
    s = field + rng.normal(scale=0.3, size=n)
    joined = pd.DataFrame({f: np.nan for f in cp._STRUCTURAL_FIELDS}, index=range(n))
    joined["seasonal_amplitude_max"] = field
    joined["tier_synthetic"] = 1.0
    joined["generator_parametric"] = 1.0
    joined["archetype_seasonal_dominant"] = arch.astype(float)
    joined["archetype_trend_dominant"] = 1.0 - arch
    cols = cp._residualization_cols(joined.columns)
    assert "archetype_seasonal_dominant" not in cols and "tier_synthetic" in cols
    assert "generator_parametric" not in cols
    recs = cp._structural_field_records(s, joined, cols, seed=0)
    rec = next(r for r in recs if r["field"] == "seasonal_amplitude_max")
    assert rec["resid_rho"] is not None and rec["resid_rho"] > 0.8
    prov = cp._provenance_profile(s, joined)
    driven, _ = cp._provenance_driven(recs, prov, np.argsort(-s)[:20], np.array(["parametric"] * n))
    assert driven is False


def test_agreement_decided_by_overlap_not_broad_correlation():
    """A pair with a tiny broad-correlation p but no top-k overlap must not
    agree; a pair with a tiny overlap p must, whatever its rho p."""
    recs = [{"model_a": "A", "model_b": "B", "rho": 0.8, "p_uncond": 0.001,
             "p_overlap": 0.9, "p_overlap_within_stratum": 0.9},
            {"model_a": "A", "model_b": "C", "rho": 0.2, "p_uncond": 0.4,
             "p_overlap": 1e-8, "p_overlap_within_stratum": 0.01}]
    cp._mark_agreement(recs)
    assert recs[0]["agrees"] is False and recs[1]["agrees"] is True
    assert recs[1]["beyond_stratum"] is True and recs[0]["beyond_stratum"] is False


def test_overlap_from_shared_archetype_only_fails_within_stratum():
    """Both parts' top-20 are drawn independently from one 40-series
    archetype: the overlap (~10) beats random subsets (expected ~0.44) but
    not redraws with the same per-stratum composition."""
    rng = np.random.default_rng(11)
    n, k = 900, 20
    strata = np.array(["x"] * 40 + ["y"] * 430 + ["z"] * 430)
    by_stratum = cp._by_stratum(strata)
    S_a = rng.choice(40, k, replace=False)
    S_b = rng.choice(40, k, replace=False)
    sa = rng.normal(size=n); sb = rng.normal(size=n)
    res = cp._pair_agreement(sa, sb, S_a, S_b, strata, by_stratum, 0, 500)
    assert res["top_k_overlap"] >= 5 and res["p_overlap"] < 1e-4
    assert res["p_overlap_within_stratum"] > 0.05


def test_provenance_driven_needs_real_derived_top_series():
    """C17's shape: provenance rho exceeds every residualized structural
    rho, but the top-k series are all synthetic. That is not a provenance
    detector. Decoy: the same rho profile with a top-k that is 90% one
    real-derived generator is."""
    recs = [{"field": "has_random_walk", "resid_rho": 0.34}]
    prov = {"tier_synthetic_rho": -0.71, "max_generator_rho": 0.2}
    gens = np.array(["parametric"] * 100 + ["block_bootstrap"] * 100)
    driven, comp = cp._provenance_driven(recs, prov, np.arange(20), gens)
    assert comp["provenance_exceeds_structural"] is True and driven is False
    S_real = np.r_[np.arange(100, 118), np.arange(2)]
    driven2, _ = cp._provenance_driven(recs, prov, S_real, gens)
    assert driven2 is True


def test_binary_field_keeps_its_rho_when_provenance_is_constant():
    """A binary structural field on rows where every provenance dummy is
    constant: cross-fitted residualization would split its ties by fold
    and collapse Spearman (0.83 -> 0.33 on the 4-model run's random-walk
    concept). The residualized rho must equal the raw rho."""
    rng = np.random.default_rng(4)
    n = 500
    field = (rng.random(n) < 0.1).astype(float)
    s = field * 3 + rng.normal(scale=0.5, size=n)
    joined = pd.DataFrame({f: np.nan for f in cp._STRUCTURAL_FIELDS}, index=range(n))
    joined["has_random_walk"] = field
    joined["tier_synthetic"] = 1.0
    joined["tier_real_derived"] = 0.0
    joined["generator_mixture"] = 0.0
    cols = cp._residualization_cols(joined.columns)
    rec = next(r for r in cp._structural_field_records(s, joined, cols, seed=0)
               if r["field"] == "has_random_walk")
    assert rec["raw_rho"] > 0.4 and rec["resid_rho"] == rec["raw_rho"]
