"""Tests for `analysis/model_similarity.py` (ROADMAP.md sec 37, cmp-B spec).

Synthetic throughout, with a planted, known answer per test and (where the
spec asks for one) a decoy that could fool a naive implementation. Every
load-bearing assertion below was confirmed to fail under a planted
regression -- see each test's own docstring for what was broken and how,
and the final report for the exact pytest evidence.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import config_from_dict  # noqa: E402
from tsfm_lens.utils import save_json  # noqa: E402
from tsfm_lens.analysis import model_similarity as ms  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------

def _cfg(out_dir: Path, model_names: list, seed: int = 0, n_boot: int = 200,
        data: dict | None = None):
    raw = {
        "run": {"name": "run", "out_dir": str(out_dir), "seed": seed},
        "data": data or {"source": "smoke", "context_len": 64, "horizon": 16,
                        "smoke_series_per_family": 6},
        "models": [{"name": n, "adapter": "mock_patch"} for n in model_names],
        "stats": {"enabled": True, "n_boot": n_boot, "ci": 0.95},
    }
    return config_from_dict(raw)


def _write_l0(run_dir: Path, mase_by_model: dict, series_ids=None, family="f") -> list:
    n = len(next(iter(mase_by_model.values())))
    series_ids = series_ids or [f"s{i}" for i in range(n)]
    rows = []
    for model, vals in mase_by_model.items():
        for sid, v in zip(series_ids, vals):
            rows.append({"model": model, "series_id": sid, "family": family,
                        "archetype": None, "generator": "synthetic",
                        "mase": float(v), "mase_reliable": True,
                        "mae_over_mad": float(v), "smape": 0.1, "pinball": 0.1})
    out_dir = run_dir / "l0"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_dir / "metrics.parquet")
    return series_ids


def _write_json(run_dir: Path, rel_path: str, obj: dict) -> None:
    save_json(run_dir / rel_path, obj)


# ---------------------------------------------------------------------------
# 1. error_agreement -- recovers a planted correlation, and is pair-local.
# ---------------------------------------------------------------------------

def test_error_agreement_matches_scipy_and_is_pair_local(tmp_path):
    """Plant: swap the numerator sign in the Spearman call (`-log_a` instead
    of `log_a`). That flips the recovered value's sign, failing the exact
    match against scipy's own computation on the same arrays -- confirmed
    below by reverting the swap and re-running.
    """
    mase_a = np.array([1.0, 2.0, 3.0, 2.5, 4.0, 5.0, 3.5, 6.0, 7.0, 2.2])
    mase_b = np.array([1.1, 2.3, 2.8, 2.6, 4.4, 4.9, 3.9, 6.5, 6.8, 2.0])
    mase_c = np.array([9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0, 0.5])  # unrelated to A/B

    run_dir = tmp_path / "run3"
    series_ids = _write_l0(run_dir, {"A": mase_a, "B": mase_b, "C": mase_c})
    cfg = _cfg(tmp_path, ["A", "B", "C"])

    expected = spearmanr(np.log(mase_a), np.log(mase_b)).statistic
    rec_ab = ms._error_agreement(run_dir, cfg, "A", "B", None,
                                 "not measured: skip", 200, 0)
    assert rec_ab["status"] == "measured"
    assert rec_ab["value"] == pytest.approx(expected, abs=1e-9)

    # Pair-locality: adding a 4th model's rows must not move (A, B)'s value.
    mase_d = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    run_dir4 = tmp_path / "run4"
    _write_l0(run_dir4, {"A": mase_a, "B": mase_b, "C": mase_c, "D": mase_d},
             series_ids=series_ids)
    cfg4 = _cfg(tmp_path, ["A", "B", "C", "D"])
    rec_ab_4 = ms._error_agreement(run_dir4, cfg4, "A", "B", None,
                                   "not measured: skip", 200, 0)
    assert rec_ab_4["value"] == rec_ab["value"]


# ---------------------------------------------------------------------------
# 2. Partial error agreement -- removes a planted shared difficulty; a decoy
#    with a genuine shared residual beyond that covariate keeps a positive
#    partial.
# ---------------------------------------------------------------------------

def test_partial_correlation_removes_shared_difficulty_but_not_a_genuine_residual(tmp_path):
    """`z` (mocked as the "seasonal-naive difficulty") drives A and B alone;
    `w`, a SEPARATE shared factor uncorrelated with `z`, drives A and C.
    Controlling for `z` should collapse the A/B partial toward 0 while
    leaving the A/C partial clearly positive (the decoy).

    Plant: use the RAW (non-residualized) correlation `r_ab` as the return
    value instead of the partial formula. That leaves the A/B partial equal
    to its own high raw value (not near 0), which fails the load-bearing
    assertion below (confirmed by reverting the plant and re-running).
    """
    rng = np.random.default_rng(0)
    n = 60
    z = rng.uniform(1.0, 20.0, n)
    w = rng.uniform(1.0, 20.0, n)  # independent of z
    noise = lambda scale=0.03: np.exp(rng.normal(0, scale, n))

    mase_a = z * w * noise()
    mase_b = z * noise()
    mase_c = w * noise()

    series_ids = [f"s{i}" for i in range(n)]
    run_dir = tmp_path / "run"
    _write_l0(run_dir, {"A": mase_a, "B": mase_b, "C": mase_c}, series_ids=series_ids)
    cfg = _cfg(tmp_path, ["A", "B", "C"])

    naive_by_id = dict(zip(series_ids, z.tolist()))

    rec_ab = ms._error_agreement(run_dir, cfg, "A", "B", naive_by_id, None, 300, 1)
    rec_ac = ms._error_agreement(run_dir, cfg, "A", "C", naive_by_id, None, 300, 1)

    raw_ab = rec_ab["value"]
    raw_ac = rec_ac["value"]
    assert raw_ab > 0.6 and raw_ac > 0.6  # both raw agreements are high

    partial_ab = rec_ab["detail"]["partial"]
    partial_ac = rec_ac["detail"]["partial"]
    assert partial_ab["status"] == "measured"
    assert partial_ac["status"] == "measured"
    assert abs(partial_ab["value"]) < 0.35, partial_ab            # planted difficulty removed
    assert partial_ac["value"] > 0.4, partial_ac                  # decoy: genuine residual remains


def test_partial_correlation_reports_not_measured_when_corpus_cannot_load(tmp_path):
    run_dir = tmp_path / "run"
    _write_l0(run_dir, {"A": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                       "B": np.array([1.2, 2.1, 2.9, 4.4, 4.8, 6.3, 6.9, 8.5])})
    cfg = _cfg(tmp_path, ["A", "B"])
    naive_by_id, reason = ms._seasonal_naive_mase_by_series(
        config_from_dict({"run": {"name": "r", "out_dir": str(tmp_path), "seed": 0},
                          "data": {"source": "bogus_source"},
                          "models": [{"name": "A"}, {"name": "B"}]}))
    assert naive_by_id is None
    assert "could not load corpus" in reason
    rec = ms._error_agreement(run_dir, cfg, "A", "B", naive_by_id, reason, 200, 0)
    assert rec["detail"]["partial"]["status"] == "not measured"
    assert "could not load corpus" in rec["detail"]["partial"]["reason"]


# ---------------------------------------------------------------------------
# 3. cka_best -- reads either key order; CI/reference only when meta.json
#    actually carries a matching, cross-checked best_pair record.
# ---------------------------------------------------------------------------

def test_cka_best_reads_either_key_order_and_gates_ci_on_a_matching_meta_record(tmp_path):
    """Plant: drop the `elif key_ba in z.files` branch (only ever try the
    forward key). The (B, C) case below -- stored ONLY under the reversed
    key `cka_window__C__B` -- then reports `status: not measured`, failing
    the assertion that it is measured with value 0.7 (confirmed below).
    """
    run_dir = tmp_path / "run"
    (run_dir / "l1").mkdir(parents=True)

    mat_ab = np.zeros((3, 3), dtype=np.float32)
    mat_ab[1, 2] = 0.9
    mat_ac = np.zeros((3, 3), dtype=np.float32)
    mat_ac[0, 0] = 0.5
    mat_cb = np.zeros((3, 3), dtype=np.float32)  # stored as C-vs-B (reversed order for B,C)
    mat_cb[2, 1] = 0.7

    np.savez(run_dir / "l1" / "cka.npz",
            cka_window__A__B=mat_ab, cka_window__A__C=mat_ac, cka_window__C__B=mat_cb)

    layers = ["L0", "L1", "L2"]
    rel = [0.0, 0.5, 1.0]
    meta = {"pairs": [
        {"model_a": "A", "model_b": "B", "layers_a": layers, "layers_b": layers,
         "rel_depth_a": rel, "rel_depth_b": rel,
         "best_pair": {"layer_a": "L1", "layer_b": "L2", "cka": 0.9,
                       "ci": {"lo": 0.8, "hi": 0.95},
                       "null_ci": {"value": 0.1, "lo": 0.05, "hi": 0.15}}},
        {"model_a": "A", "model_b": "C", "layers_a": layers, "layers_b": layers,
         "rel_depth_a": rel, "rel_depth_b": rel,
         "best_pair": {"layer_a": "L0", "layer_b": "L0", "cka": 0.5,
                       "ci": None, "null_ci": None}},
        # No record at all for (B, C): l1 either wasn't asked for stats on
        # it or (as here) the record is simply absent -- exercises the
        # "no matching pair record" fallback.
    ]}
    save_json(run_dir / "l1" / "meta.json", meta)
    cfg = _cfg(tmp_path, ["A", "B", "C"])

    rec_ab = ms._cka_best(run_dir, cfg, "A", "B")
    assert rec_ab["value"] == pytest.approx(0.9)
    assert rec_ab["ci"] == [0.8, 0.95]
    assert rec_ab["detail"]["layer_a"] == "L1" and rec_ab["detail"]["layer_b"] == "L2"
    assert rec_ab["detail"]["rel_depth_basis"] == "l1_meta"
    assert rec_ab["reference"]["value"] == 0.1

    rec_ac = ms._cka_best(run_dir, cfg, "A", "C")
    assert rec_ac["value"] == pytest.approx(0.5)
    assert rec_ac["ci"] is None
    assert "stats.enabled" in rec_ac["reason"]

    rec_bc = ms._cka_best(run_dir, cfg, "B", "C")
    assert rec_bc["status"] == "measured"
    assert rec_bc["value"] == pytest.approx(0.7)          # read via the reversed key
    assert rec_bc["ci"] is None
    assert "no matching pair record" in rec_bc["reason"]
    assert rec_bc["detail"]["rel_depth_basis"] == "index_fallback"


def test_cka_best_not_measured_when_npz_missing(tmp_path):
    run_dir = tmp_path / "run"
    cfg = _cfg(tmp_path, ["A", "B"])
    rec = ms._cka_best(run_dir, cfg, "A", "B")
    assert rec["status"] == "not measured"
    assert rec["value"] is None
    assert "cka.npz" in rec["reason"]


# ---------------------------------------------------------------------------
# 4. atlas_co_membership -- planted "drawn together" and "segregated" atlases.
# ---------------------------------------------------------------------------

def _atlas_rows(concept_of: dict, model_of: dict, n_rows: int) -> dict:
    rows = []
    for i in range(n_rows):
        rows.append({"model": model_of[i], "layer": "L0", "feature": i,
                    "concept": concept_of.get(i), "pc1": 0.0, "pc2": 0.0})
    return {"schema_version": 1, "rows": rows, "concepts": []}


def test_atlas_co_membership_drawn_together_gives_small_p_above(tmp_path):
    """Two small concepts both mix A and B; the wider pool is mostly a third
    model D, so a random relabelling rarely reproduces that mixing.

    Plant: swap `p_above`/`p_below` in the returned detail. The assertion
    that `p_above` (not `p_below`) is small then fails -- confirmed by
    reverting the swap.
    """
    n = 24
    model_of = {i: "D" for i in range(n)}
    for i in [0, 1]:
        model_of[i] = "A"
    for i in [2, 3]:
        model_of[i] = "B"
    for i in [4]:
        model_of[i] = "A"
    for i in [5, 6]:
        model_of[i] = "B"
    concept_of = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}  # concept 0: A,B; concept 1: A,B

    run_dir = tmp_path / "run"
    save_json(run_dir / "sae" / "concept_atlas.json", _atlas_rows(concept_of, model_of, n))
    cfg = _cfg(tmp_path, ["A", "B", "D"])

    rec = ms._atlas_co_membership(run_dir, cfg, "A", "B", n_null=500)
    assert rec["status"] == "measured"
    assert rec["detail"]["observed"] == 2
    assert rec["detail"]["p_above"] < 0.15
    assert rec["value"] > 1.0


def test_atlas_co_membership_segregated_gives_small_p_below(tmp_path):
    n = 20
    model_of = {i: ("A" if i % 2 == 0 else "B") for i in range(n)}  # balanced 10 A / 10 B
    # concept 0: 4 even indices (all model A); concept 1: 4 odd indices
    # (all model B) -- never both models in one concept, and each concept
    # is large enough (4 of 20) that "pure" is genuinely unlikely by chance.
    concept_of = {0: 0, 2: 0, 4: 0, 6: 0, 9: 1, 11: 1, 13: 1, 15: 1}

    run_dir = tmp_path / "run"
    save_json(run_dir / "sae" / "concept_atlas.json", _atlas_rows(concept_of, model_of, n))
    cfg = _cfg(tmp_path, ["A", "B"])

    rec = ms._atlas_co_membership(run_dir, cfg, "A", "B", n_null=1000)
    assert rec["status"] == "measured"
    assert rec["detail"]["observed"] == 0
    assert rec["detail"]["p_below"] < 0.05


def test_atlas_co_membership_not_measured_without_assigned_concepts(tmp_path):
    n = 6
    model_of = {i: "A" if i < 3 else "B" for i in range(n)}
    run_dir = tmp_path / "run"
    save_json(run_dir / "sae" / "concept_atlas.json", _atlas_rows({}, model_of, n))
    cfg = _cfg(tmp_path, ["A", "B"])
    rec = ms._atlas_co_membership(run_dir, cfg, "A", "B")
    assert rec["status"] == "not measured"
    assert rec["value"] is None


# ---------------------------------------------------------------------------
# 5. Kendall's W -- identical rankings concordant (W=1, small p); independent
#    random rankings are usually not significant.
# ---------------------------------------------------------------------------

def test_kendall_w_identical_rankings_give_w_one_and_a_small_floored_p():
    """Plant: drop the `(1 + ...)` floor in `_resolve`'s p formula (use
    `n_ge / n_perm` instead of `(1 + n_ge) / (1 + n_perm)`). With a fixed
    seed, no random permutation of 6 independent ranks matches the perfect
    original ordering across 4 metrics, so `n_ge == 0` and the unfloored
    formula returns EXACTLY 0.0 -- failing the `p > 0` assertion (confirmed
    by reverting the plant and re-running).
    """
    pair_keys = [f"p{i}" for i in range(6)]
    ranking = {pk: float(i + 1) for i, pk in enumerate(pair_keys)}
    ranks_by_metric = {f"m{j}": dict(ranking) for j in range(4)}
    out = ms._kendall_w(ranks_by_metric, pair_keys, n_perm=500, seed=0)
    assert out["w"] == pytest.approx(1.0)
    assert out["p"] > 0.0
    assert out["p"] < 0.05


def test_kendall_w_independent_random_rankings_usually_not_significant():
    pair_keys = [f"p{i}" for i in range(6)]
    n_sig = 0
    n_trials = 20
    for seed in range(n_trials):
        rng = np.random.default_rng(1000 + seed)
        ranks_by_metric = {}
        for j in range(5):
            perm = rng.permutation(6) + 1
            ranks_by_metric[f"m{j}"] = {pk: float(r) for pk, r in zip(pair_keys, perm)}
        out = ms._kendall_w(ranks_by_metric, pair_keys, n_perm=300, seed=seed)
        if out["p"] < 0.05:
            n_sig += 1
    assert n_sig <= n_trials // 2, f"{n_sig}/{n_trials} independent trials called significant"


def test_kendall_w_needs_complete_ranks_and_at_least_two_metrics():
    pair_keys = [f"p{i}" for i in range(4)]
    ranks_by_metric = {
        "m1": {"p0": 1.0, "p1": 2.0, "p2": 3.0, "p3": 4.0},
        "m2": {"p0": 1.0, "p1": 2.0, "p2": 3.0},  # incomplete -- missing p3
    }
    out = ms._kendall_w(ranks_by_metric, pair_keys, n_perm=100, seed=0)
    assert out["w"] is None
    assert out["metrics_used"] == ["m1"]  # m2 is excluded: its ranks don't cover p3


# ---------------------------------------------------------------------------
# 6. Contrasts -- fire on a planted large rank gap; not on a consistent pair.
# ---------------------------------------------------------------------------

def _skeleton_metrics(pair_keys: list) -> dict:
    out = {}
    for key in ms.METRIC_ORDER:
        out[key] = {"pairs": {pk: {"status": "not measured", "value": None} for pk in pair_keys}}
    return out


def test_contrasts_fire_on_a_large_rank_gap_and_not_on_a_consistent_pair():
    """Plant: change the threshold from `n_pairs - 2` to `n_pairs` (i.e.
    require the maximum possible gap). With 4 pairs the planted large-gap
    cases (gap 3) then no longer clear the bar, so the assertion that a
    contrast was found for both p1 and p4 fails (confirmed below).
    """
    pair_keys = ["p1", "p2", "p3", "p4"]
    metrics = _skeleton_metrics(pair_keys)

    def _fill(key, values):
        for pk, v in zip(pair_keys, values):
            metrics[key]["pairs"][pk] = {"status": "measured", "value": v}

    _fill("cka_best", [0.9, 0.7, 0.5, 0.3])          # ranks: p1=1, p2=2, p3=3, p4=4
    _fill("error_agreement", [0.1, 0.55, 0.85, 0.95])  # ranks: p1=4, p2=3, p3=2, p4=1

    consensus = ms._consensus(metrics, pair_keys, base_seed=0)
    contrasts = ms._contrasts(metrics, pair_keys, consensus)
    flagged_pairs = {c["pair"] for c in contrasts
                    if {c["metric_hi"], c["metric_lo"]} == {"cka_best", "error_agreement"}}
    assert "p1" in flagged_pairs
    assert "p4" in flagged_pairs
    assert "p2" not in flagged_pairs
    assert "p3" not in flagged_pairs


# ---------------------------------------------------------------------------
# 7. Missing artifacts -> "not measured", never a fabricated 0, and excluded
#    from consensus.
# ---------------------------------------------------------------------------

def test_missing_artifacts_are_not_measured_and_excluded_from_consensus(tmp_path):
    """Plant: change `_not_measured`'s `value` from `None` to `0.0`. The
    assertion that every unmeasured pair's value is `None` fails (confirmed
    below by reverting the plant).
    """
    run_dir = tmp_path / "empty_run"
    cfg = _cfg(tmp_path, ["A", "B", "C"],
              data={"source": "bogus_source"})  # corpus load will fail too
    out = ms.build_model_similarity(run_dir, cfg)
    for key in ms.METRIC_ORDER:
        for pk, rec in out["metrics"][key]["pairs"].items():
            assert rec["status"] == "not measured", (key, pk)
            assert rec["value"] is None, (key, pk)
            assert rec["reason"]
    assert out["consensus"]["kendall_w_all"]["w"] is None
    assert out["contrasts"] == []


# ---------------------------------------------------------------------------
# 8. No model or architecture names in the reduction logic.
# ---------------------------------------------------------------------------

_BANNED_NAMES = ("TimesFM", "timesfm", "Chronos", "chronos", "Sundial", "sundial",
                 "Lag-Llama", "LagLlama", "lag_llama", "generic_hf",
                 "mock_patch", "mock_step", "mock_encdec", "mock_blackbox", "mock_wave")

_LOGIC_FUNCTIONS = (
    ms._error_agreement, ms._partial_error_agreement, ms._seasonal_naive_mase_by_series,
    ms._cka_best, ms._find_pair_record, ms._stitching_gain, ms._cluster_ami,
    ms._atlas_co_membership, ms._input_transfer, ms._shared_concepts,
    ms._consensus, ms._contrasts, ms._kendall_w, ms.build_model_similarity,
)


def test_no_model_or_architecture_names_in_logic():
    """The adaptivity contract (mirrors `report/derived.py`'s own test): this
    reduction must transfer to models nobody has run yet.

    Plant (done manually against the source file for verification, not left
    in the tree): add a line `# TimesFM` inside `_cka_best`. That is
    confirmed to fail this test; see the final report for the pytest output.
    """
    src = "\n".join(inspect.getsource(fn) for fn in _LOGIC_FUNCTIONS)
    for banned in _BANNED_NAMES:
        assert banned not in src, f"{banned!r} would tie this reduction to one model"


# ---------------------------------------------------------------------------
# End-to-end smoke: write_model_similarity produces valid, loadable JSON.
# ---------------------------------------------------------------------------

def test_write_model_similarity_round_trips_through_json(tmp_path):
    run_dir = tmp_path / "run"
    _write_l0(run_dir, {"A": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                       "B": np.array([1.2, 2.1, 2.9, 4.4, 4.8, 6.3, 6.9, 8.5])})
    cfg = _cfg(tmp_path, ["A", "B"], data={"source": "bogus_source"})
    out = ms.write_model_similarity(run_dir, cfg)
    out_path = run_dir / "report" / "model_similarity.json"
    assert out_path.exists()
    reloaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert reloaded["models"] == ["A", "B"]
    assert set(reloaded["metrics"]) == set(ms.METRIC_ORDER)
    assert out["pairs"] == [{"model_a": "A", "model_b": "B", "key": "A|B"}]
