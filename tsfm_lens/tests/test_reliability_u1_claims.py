"""ROADMAP.md sec 38.4 (K4) + sec 38.2 (K2): the registrable, confirmable U1
claim type ("do a model's internals predict its per-series error beyond the
free baseline?").

Planted fixture. Two models share a latent `z` that drives both of their
errors (log MASE = 0.5 x + 1.2 z + noise, `x` driving the quantile width the
free baseline reads). The PLANTED model's frozen SAE family 0 carries `z` in
its last context window; the DECOY model's internals are independent noise, so
its internals carry no information about its own error. The SAEs are
hand-built (feature j = relu(x_j)), so the planted signal is known by
construction. Forecasts come from a fake adapter, so MASE is exactly the
planted quantity.

Three layers: registration (`build_registry` over a hand-built mini run
directory), confirm (`_confirm_reliability_u1` on a synthetic private split
with the planted answer AND the decoy beside it), and an end-to-end pass over
the real mock run that the K2 tests use (real store, real capture, real
`run_confirm`). Each load-bearing assertion names the planted regression that
must break it. No test opens a sealed private split.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_concept_causal_confirm import (_stage_cfg, dev_run)  # noqa: E402,F401
from tests.test_smoke import build_config  # noqa: E402
from tsfm_lens.analysis import confirm as confirm_mod  # noqa: E402
from tsfm_lens.analysis import hypotheses as hyp_mod  # noqa: E402
from tsfm_lens.analysis import reliability_from_internals as rfi  # noqa: E402
from tsfm_lens.analysis.hypotheses import (RELIABILITY_STAGE, build_registry,  # noqa: E402
                                           check_registry_freshness, claim_family_budget)
from tsfm_lens.analysis.l0_behavioral import _predict_all, _score  # noqa: E402
from tsfm_lens.analysis.stats import _mase_scale  # noqa: E402
from tsfm_lens.config import config_from_dict  # noqa: E402
from tsfm_lens.data import BenchmarkData  # noqa: E402
from tsfm_lens.manifest import resolve_config_keys  # noqa: E402
from tsfm_lens.pipeline import _stage_by_name, run_pipeline  # noqa: E402
from tsfm_lens.sae.ablation_run import checkpoint_path  # noqa: E402
from tsfm_lens.sae.models import TopKSAE  # noqa: E402
from tsfm_lens.sae.train import load_sae_checkpoint, save_sae  # noqa: E402
from tsfm_lens.utils import load_json, save_json  # noqa: E402

LAYER, CRYS = "blocks.1", "blocks.2"
D_IN, WINDOWS, CTX, HOR = 8, 3, 48, 8
QUANTILES = [0.1, 0.5, 0.9]
FAMILY_FEATURES = {0: [0, 1], 1: [2, 3], 2: [4]}
N_DEV, N_PRIV, N_BOOT = 500, 500, 200
MODELS = ("planted", "decoy")
DEV_JSON = "reliability_dev.json"
DECOY_NOISE_SEED = 3


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _planted_sae() -> TopKSAE:
    """An SAE whose feature j (j < 8) is exactly relu(x_j), so a family sum is a
    known function of the activations."""
    sae = TopKSAE(D_IN, 16, k=D_IN)
    with torch.no_grad():
        sae.W_enc.zero_()
        for j in range(D_IN):
            sae.W_enc[j, j] = 1.0
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    return sae


def make_split(seed: int, n: int, decoy_seed: int) -> SimpleNamespace:
    """One split with a planted answer: series, latent drivers, and per-model
    activations, offsets and quantile widths."""
    rng = np.random.default_rng(seed)
    fam = np.array(["fam_a", "fam_b", "fam_c", "fam_d"])[np.arange(n) % 4]
    t = np.arange(CTX)
    period = rng.integers(8, 16, n)
    phase = rng.uniform(0, 6.28, n)
    amp = rng.uniform(0.5, 2.0, n)
    level = rng.uniform(5.0, 10.0, n)
    contexts = (level[:, None] + amp[:, None] * np.sin(2 * np.pi * t / period[:, None] + phase[:, None])
                + 0.1 * rng.normal(size=(n, CTX)))
    targets = np.repeat(contexts[:, -1:], HOR, axis=1)
    x, z = rng.normal(size=n), rng.normal(size=n)
    scale = _mase_scale(contexts, "mean_abs_diff")
    split = SimpleNamespace(n=n, contexts=contexts, targets=targets, fam=fam, x=x, z=z,
                            scale=scale, offsets={}, widths={}, acts={})
    noise_rng = np.random.default_rng(decoy_seed)
    for name in MODELS:
        eps = rng.normal(size=n)
        sign = rng.choice([-1.0, 1.0], size=n)
        split.offsets[name] = sign * scale * np.exp(0.5 * x + 1.2 * z + 0.25 * eps)
        split.widths[name] = scale * np.exp(0.5 * x)
        a = np.abs(rng.normal(size=(n, WINDOWS, D_IN))) * 0.5 + 0.1
        a[:, -1, 0] = 4.0 + (z if name == "planted" else noise_rng.normal(size=n))
        crys = rng.normal(size=(n, WINDOWS, D_IN))
        split.acts[name] = {LAYER: torch.from_numpy(a.astype(np.float32)),
                            CRYS: torch.from_numpy(crys.astype(np.float32))}
    meta = pd.DataFrame({"series_id": [f"s{seed}_{i}" for i in range(n)], "family": fam,
                         "archetype": "none", "generator": "synthetic", "tier": "synthetic"})
    split.data = BenchmarkData(values=np.concatenate([contexts, targets], axis=1), meta=meta,
                               context_len=CTX, horizon=HOR)
    return split


class _FakeAdapter:
    """Forecasts `last value + planted offset`, with a quantile band of the
    planted width; rows are found by their context bytes, so any batching works."""

    def __init__(self, split, name: str):
        self.cfg = SimpleNamespace(batch_size=64)
        self.name, self._split = name, split
        self._row = {c.tobytes(): i for i, c in enumerate(split.contexts)}

    def ensure_loaded(self):
        return None

    def predict(self, contexts, horizon, quantiles):
        idx = np.array([self._row[c.tobytes()] for c in contexts])
        point = contexts[:, -1:] + self._split.offsets[self.name][idx][:, None] * np.ones((1, horizon))
        band = self._split.widths[self.name][idx][:, None, None] * np.array([-1.0, 0.0, 1.0])
        return {"point": point, "quantiles": point[:, :, None] + band}


class _FakeHub:
    def __init__(self, split):
        self.adapters = {m: _FakeAdapter(split, m) for m in MODELS}

    def get(self, name):
        return self.adapters[name]

    def release(self, name):
        return None


def _family_rows() -> list:
    return [{"model": m, "layer": LAYER, "feature": f, "family": fam, "concept": None}
            for m in MODELS for fam, fs in FAMILY_FEATURES.items() for f in fs]


def _dev_inputs(split) -> dict:
    """The arrays `run_reliability` reduces, built with the same functions the
    real loader uses (`baseline_features`, `family_columns`, the norm feature)."""
    sae = _planted_sae()
    c22, c22_names = rfi.context_catch22(split.contexts)
    models = {}
    for m in MODELS:
        point, quants = _predict_all(_FakeAdapter(split, m), split.contexts, HOR, QUANTILES)
        scored = _score(m, point, quants, split.contexts, split.targets, QUANTILES,
                        split.data.meta, "mean_abs_diff", 0.0)
        base, names, _ = rfi.baseline_features(point, quants, split.contexts, split.scale,
                                               c22, c22_names)
        with torch.no_grad():
            last = sae.encode(split.acts[m][LAYER][:, -1, :]).numpy().astype(np.float64)
        X, fam_ids = rfi.family_columns({LAYER: last}, _family_rows(), m, split.n)
        models[m] = {"mase": scored["mase"].to_numpy(), "baseline": base, "baseline_names": names,
                     "groups": {"sae_families": X, "crystallization_norm":
                                rfi.crystallization_norm_feature(split.acts[m][CRYS][:, -1, :])},
                     "skipped": {}, "notes": {"sae_choice": {"kind": "concept_families",
                                                             "columns": [str(f) for f in fam_ids]},
                                              "crystallization_layer": CRYS}}
    return {"run": "u1mock", "strata": split.fam, "reliable": np.ones(split.n),
            "seasonal_mase": np.full(split.n, 1.0), "models": models}


def _mini_cfg(out_dir: str):
    cfg = config_from_dict(build_config(out_dir))
    cfg.run.name = "u1mock"
    cfg.run.device = "cpu"
    cfg.l0.quantiles = list(QUANTILES)
    cfg.l0.min_scale_frac = 0.0
    cfg.stats.min_series = 20
    cfg.confirm.enabled = True
    cfg.confirm.source = "smoke"
    cfg.confirm.require_seal = False
    cfg.confirm.alpha = 0.05
    cfg.confirm.register_reliability_claims = True
    cfg.confirm.reliability_dev_json = DEV_JSON
    return cfg


@pytest.fixture(scope="module")
def world():
    """Mini run directory (SAE checkpoints, concept_families.json, a REAL dev K4
    JSON computed on the planted dev split) plus a planted private split."""
    out = tempfile.mkdtemp()
    cfg = _mini_cfg(out)
    run_dir = cfg.run_dir()
    for m in MODELS:
        save_sae(_planted_sae(), checkpoint_path(run_dir, m, LAYER))
    save_json(run_dir / "sae" / "concept_families.json",
              {"measured": True, "rows": _family_rows()})
    dev = make_split(1, N_DEV, DECOY_NOISE_SEED)
    k4 = rfi.run_reliability(_dev_inputs(dev), n_folds=3, n_repeats=2, n_boot=N_BOOT, seed=0)
    save_json(run_dir / DEV_JSON, k4)
    priv = make_split(2, N_PRIV, DECOY_NOISE_SEED + 100)
    return SimpleNamespace(cfg=cfg, run_dir=run_dir, k4=k4, dev=dev, priv=priv,
                           hub=_FakeHub(priv), out=out)


def _fresh(world, **overrides):
    """A private copy of the mini run (so tests can edit artifacts) and its cfg."""
    work = Path(tempfile.mkdtemp())
    shutil.copytree(world.run_dir, work / world.run_dir.name)
    cfg = _mini_cfg(str(work))
    for k, v in overrides.items():
        setattr(cfg.confirm, k, v)
    return cfg


def _false_positive_decoy(cfg) -> None:
    """Stage a DEV false positive: the decoy's dev gain CI is edited to exclude 0,
    so it is registered and the PRIVATE refit is what must reject it."""
    path = cfg.run_dir() / DEV_JSON
    k4 = load_json(path)
    g = k4["models"]["decoy"]["u1"]["log_mase_spearman"]["gain"]
    g["lo"], g["gain"], g["hi"] = 0.004, 0.02, 0.04
    save_json(path, k4)


def _confirm(cfg, world, registry, acts=None, errors=None, hub=None):
    return confirm_mod._confirm_reliability_u1(
        cfg, hub or world.hub, world.priv.data, cfg.run_dir(), registry,
        acts if acts is not None else {f"{m}/{layer}": a for m in MODELS
                                       for layer, a in world.priv.acts[m].items()},
        errors or {})


def _registry(cfg, decoy_false_positive=False) -> dict:
    if decoy_false_positive:
        _false_positive_decoy(cfg)
    return build_registry(cfg)


def _by_model(block) -> dict:
    return {t["model"]: t for t in block["tests"]}


# ---------------------------------------------------------------------------
# 0. The fixture itself: the planted answer is really there on dev.
# ---------------------------------------------------------------------------

def test_fixture_plants_a_gain_in_one_model_and_none_in_the_decoy(world):
    """Planted regression: a decoy whose internals carry `z` would make every
    test below meaningless; this pins the premise on the real dev K4 JSON."""
    u1 = {m: world.k4["models"][m]["u1"]["log_mase_spearman"]["gain"] for m in MODELS}
    assert u1["planted"]["lo"] > 0.05
    assert u1["decoy"]["lo"] <= 0


# ---------------------------------------------------------------------------
# 1. Registration.
# ---------------------------------------------------------------------------

def test_registration_picks_only_the_model_whose_dev_gain_ci_excludes_zero(world):
    """The registry holds exactly the planted model's claim. The decoy's dev
    record is edited to a borderline result (point gain +0.02, CI lower bound
    -0.01). Planted regression: `lo > 0` replaced by `gain > 0` (registering on
    the point estimate) also registers the decoy."""
    cfg = _fresh(world)
    path = cfg.run_dir() / DEV_JSON
    k4 = load_json(path)
    g = k4["models"]["decoy"]["u1"]["log_mase_spearman"]["gain"]
    g["gain"], g["lo"], g["hi"] = 0.02, -0.01, 0.05
    save_json(path, k4)
    reg = build_registry(cfg)
    claims = [h for h in reg["hypotheses"] if h["stage"] == RELIABILITY_STAGE]
    assert [h["id"] for h in claims] == ["reliability_u1::planted::log_mase_spearman"]
    rows = {(r["model"], r["task"]): r for r in reg["reliability_u1_candidates"]["tests"]}
    assert rows[("planted", "log_mase_spearman")]["registered"] is True
    decoy = rows[("decoy", "log_mase_spearman")]
    assert decoy["registered"] is False and "not > 0" in decoy["reason"]
    assert all(r["reason"] for r in rows.values() if not r["registered"])
    assert reg["reliability_u1_candidates"]["n_registered"] == 1


def test_registered_claim_freezes_the_full_spec_and_hashes(world):
    """Every procedure input and every artifact it reads is frozen: model, task,
    baseline list, family definitions, crystallization layer, alpha grid, folds,
    repeats, seed, strata, n_boot, dev gain and CI, and sha256 of
    concept_families.json, the SAE checkpoint and the dev JSON."""
    cfg = _fresh(world)
    run_dir = cfg.run_dir()
    h = next(h for h in build_registry(cfg)["hypotheses"] if h["stage"] == RELIABILITY_STAGE)
    spec, k4 = h["spec"], load_json(run_dir / DEV_JSON)
    assert (h["model"], h["task"]) == ("planted", "log_mase_spearman")
    assert spec["baseline_features"] == rfi.baseline_feature_names()
    assert spec["baseline_features"][:2] == ["log1p_own_width", "log10_forecast_flatness"]
    fams = spec["groups"]["sae_families"]
    assert fams["layers"] == [LAYER]
    assert {f["family"]: [m["feature"] for m in f["members"]] for f in fams["families"]} \
        == FAMILY_FEATURES
    assert list(spec["groups"]) == ["sae_families", "crystallization_norm"]
    assert spec["groups"]["crystallization_norm"]["layer"] == CRYS
    st = k4["settings"]
    assert spec["ridge_alphas"] == st["ridge_alphas"]
    assert (spec["n_folds"], spec["n_repeats"], spec["seed"], spec["n_boot"]) \
        == (st["n_folds"], st["n_repeats"], st["seed"], st["n_boot"])
    assert spec["cv_strata"] == "family"
    assert spec["model_seed"] == rfi.model_seed_for("planted", st["seed"])
    gain = k4["models"]["planted"]["u1"]["log_mase_spearman"]["gain"]
    assert (h["dev"]["gain"], h["dev"]["lo"], h["dev"]["hi"]) \
        == (gain["gain"], gain["lo"], gain["hi"])
    expected = {"sae/concept_families.json": run_dir / "sae" / "concept_families.json",
                DEV_JSON: run_dir / DEV_JSON,
                checkpoint_path(run_dir, "planted", LAYER).relative_to(run_dir).as_posix():
                    checkpoint_path(run_dir, "planted", LAYER)}
    assert set(h["artifacts"]) == set(expected)
    for rel, path in expected.items():
        assert h["artifacts"][rel] == _sha(path)
    assert h["artifact"] == DEV_JSON and h["artifact_sha256"] == _sha(run_dir / DEV_JSON)
    check_registry_freshness(cfg, {"hypotheses": [h]})
    checkpoint_path(run_dir, "planted", LAYER).write_bytes(b"retrained")
    with pytest.raises(RuntimeError, match="hash changed"):
        check_registry_freshness(cfg, {"hypotheses": [h]})


def test_dev_json_outside_the_run_dir_is_pinned_by_absolute_path(world):
    """The dev K4 JSON may live anywhere (absolute path via the CLI); its key is
    then absolute and freshness still resolves it."""
    cfg = _fresh(world)
    elsewhere = Path(tempfile.mkdtemp()) / "k4.json"
    shutil.copy(cfg.run_dir() / DEV_JSON, elsewhere)
    cfg.confirm.reliability_dev_json = str(elsewhere)
    h = next(h for h in build_registry(cfg)["hypotheses"] if h["stage"] == RELIABILITY_STAGE)
    assert h["artifact"] == str(elsewhere.resolve())
    check_registry_freshness(cfg, {"hypotheses": [h]})


def test_ids_carry_model_and_task_and_duplicates_are_refused(world, monkeypatch):
    """Ids name model AND task. Planted regression: an id without the task makes
    one model's two tasks collide, which `build_registry`'s refusal turns into a
    ValueError; the refusal itself is exercised by a hand-made duplicate."""
    cfg = _fresh(world)
    path = cfg.run_dir() / DEV_JSON
    k4 = load_json(path)
    for m in MODELS:
        k4["models"][m]["u1"]["failure_auroc"] = copy.deepcopy(
            k4["models"][m]["u1"]["log_mase_spearman"])
        k4["models"][m]["u1"]["failure_auroc"]["gain"]["lo"] = 0.01
    save_json(path, k4)
    monkeypatch.setattr(hyp_mod, "RELIABILITY_SUPPORTED_TASKS",
                        ("log_mase_spearman", "failure_auroc"))
    reg = build_registry(cfg)
    ids = [h["id"] for h in reg["hypotheses"] if h["stage"] == RELIABILITY_STAGE]
    assert sorted(ids) == ["reliability_u1::decoy::failure_auroc",
                           "reliability_u1::planted::failure_auroc",
                           "reliability_u1::planted::log_mase_spearman"]
    real = hyp_mod._reliability_u1_entries

    def _dup(run_dir, c):
        entries, summary = real(run_dir, c)
        return entries + entries[:1], summary

    monkeypatch.setattr(hyp_mod, "_reliability_u1_entries", _dup)
    with pytest.raises(ValueError, match="duplicate ids"):
        build_registry(cfg)


def test_unsupported_dev_positives_are_listed_not_registered(world, monkeypatch):
    """A dev positive that cannot be frozen is recorded with the reason, never
    registered silently: a failure_auroc gain (confirm does not refit it), and a
    model whose dev record used a lens-depth group."""
    cfg = _fresh(world)
    path = cfg.run_dir() / DEV_JSON
    k4 = load_json(path)
    k4["models"]["planted"]["u1"]["failure_auroc"] = copy.deepcopy(
        k4["models"]["planted"]["u1"]["log_mase_spearman"])
    k4["models"]["decoy"]["internal_groups"]["lens_depth"] = 2
    k4["models"]["decoy"]["u1"]["log_mase_spearman"]["gain"]["lo"] = 0.01
    save_json(path, k4)
    reg = build_registry(cfg)
    rows = {(r["model"], r["task"]): r for r in reg["reliability_u1_candidates"]["tests"]}
    assert "does not refit" in rows[("planted", "failure_auroc")]["reason"]
    assert "lens_depth" in rows[("decoy", "log_mase_spearman")]["reason"]
    assert [h["id"] for h in reg["hypotheses"] if h["stage"] == RELIABILITY_STAGE] \
        == ["reliability_u1::planted::log_mase_spearman"]


def test_flags_off_registry_and_budget_are_unchanged(world):
    """Opt-in: with the fields at their defaults there is no reliability claim,
    no new registry key, no new ledger family and no fingerprint movement; a
    half-set pair refuses loudly instead of silently registering nothing."""
    cfg = _fresh(world, register_reliability_claims=False, reliability_dev_json="")
    reg = build_registry(cfg)
    assert not [h for h in reg["hypotheses"] if h["stage"] == RELIABILITY_STAGE]
    assert set(reg) == {"hypotheses", "n_replicable", "concept_transfer_candidates",
                        "concept_knob_candidates"}
    assert RELIABILITY_STAGE not in {r["family"] for r in claim_family_budget(cfg, reg)}
    assert RELIABILITY_STAGE not in {r["family"] for r in claim_family_budget(cfg)}
    keys = _stage_by_name("register").config_keys
    off = resolve_config_keys(cfg, keys)
    assert not {k for k in off if "reliability" in k}
    on = resolve_config_keys(_fresh(world), keys)
    assert {"confirm.register_reliability_claims", "confirm.reliability_dev_json"} <= set(on)
    with pytest.raises(ValueError, match="no reliability claim would be registered"):
        build_registry(_fresh(world, register_reliability_claims=False))
    with pytest.raises(ValueError, match="reliability_dev_json is"):
        build_registry(_fresh(world, reliability_dev_json=""))


def test_reliability_holm_family_is_declared_and_checked_before_opening(world, monkeypatch):
    """The family appears in the ledger with m / n_boot as its Holm floor and a
    satisfiability verdict; `confirm` refuses to open the split on an
    unsatisfiable one. Planted regression: dropping the family's row from
    `claim_family_budget` (no ledger entry, no check)."""
    cfg = _fresh(world)
    reg = _registry(cfg, decoy_false_positive=True)
    row = next(r for r in claim_family_budget(cfg, reg) if r["family"] == RELIABILITY_STAGE)
    assert row["m"] == 2 and row["n_null"] == N_BOOT
    assert row["min_attainable_p_holm"] == 2 / N_BOOT and row["satisfiable"] is True
    pre = next(r for r in claim_family_budget(cfg) if r["family"] == RELIABILITY_STAGE)
    assert pre["m"] is None and pre["n_null"] == N_BOOT
    confirm_mod.check_concept_claims_before_opening(cfg, reg)
    small = copy.deepcopy(reg)
    for h in small["hypotheses"]:
        if h["stage"] == RELIABILITY_STAGE:
            h["spec"]["n_boot"] = 20
    bad = next(r for r in claim_family_budget(cfg, small) if r["family"] == RELIABILITY_STAGE)
    assert bad["satisfiable"] is False
    with pytest.raises(RuntimeError, match="reliability_u1.*cannot reach alpha"):
        confirm_mod.check_concept_claims_before_opening(cfg, small)
    from tsfm_lens.doctor import check_concept_claim_budget
    save_json(cfg.run_dir() / "hypotheses.json", small)
    doc = check_concept_claim_budget(cfg)
    assert doc.status == "fail" and RELIABILITY_STAGE in doc.detail


# ---------------------------------------------------------------------------
# 2. The refit procedure is the dev procedure.
# ---------------------------------------------------------------------------

def test_u1_gain_reproduces_the_dev_record_exactly(world):
    """`u1_gain` on the dev arrays returns the dev K4 record's gain, CI and
    baseline score bit-for-bit: the confirm refit IS the dev procedure."""
    inputs = _dev_inputs(world.dev)
    rec = inputs["models"]["planted"]
    mase = rec["mase"]
    internals = np.concatenate([rec["groups"]["sae_families"],
                                rec["groups"]["crystallization_norm"]], axis=1)
    st = world.k4["settings"]
    out = rfi.u1_gain(np.log(np.maximum(mase, 1e-6)), rec["baseline"], internals,
                      world.dev.fam, n_folds=st["n_folds"], n_repeats=st["n_repeats"],
                      n_boot=st["n_boot"], model_seed=rfi.model_seed_for("planted", st["seed"]),
                      alphas=st["ridge_alphas"])
    ref = world.k4["models"]["planted"]["u1"]["log_mase_spearman"]["gain"]
    for key in ("gain", "lo", "hi", "baseline", "baseline_plus_internals"):
        assert out[key] == ref[key], key
    assert 0.0 < out["p"] <= 1.0


def test_family_columns_equal_the_store_route(world):
    """The array route confirm uses and the store route dev used sum the same
    members: same columns, same family ids."""
    last = np.random.default_rng(0).random((30, 16))

    class _Store:
        def has_sae_features(self, model, layer):
            return True

        def load(self, model, layer, level="series", rows=None, space="act"):
            full = np.stack([last, last, last], axis=1)
            return full if level == "window" and rows is None else \
                (full[rows] if level == "window" else full.mean(axis=1))

    rows = _family_rows()
    X_store, ids_store = rfi.family_activation_features(_Store(), "planted", [LAYER], rows)
    X_arr, ids_arr = rfi.family_columns({LAYER: last}, rows, "planted", 30)
    assert ids_store == ids_arr == [0, 1, 2]
    np.testing.assert_allclose(X_store, X_arr)
    np.testing.assert_allclose(X_arr[:, 0], last[:, 0] + last[:, 1])


# ---------------------------------------------------------------------------
# 3. Confirmation on a synthetic private split.
# ---------------------------------------------------------------------------

def test_confirm_confirms_the_planted_model_and_not_the_decoy(world):
    """On a private split with the same planted structure, the registered
    planted claim confirms (gain CI lower bound > 0, Holm p below alpha) and the
    registered DEV FALSE POSITIVE (decoy) does not. Planted regressions: a rule
    that confirms on the dev gain, or on any positive point gain, confirms the
    decoy."""
    cfg = _fresh(world)
    reg = _registry(cfg, decoy_false_positive=True)
    block = _confirm(cfg, world, reg)
    t = _by_model(block)
    assert block["family"] == RELIABILITY_STAGE and block["status"] == "tested"
    assert block["n_registered"] == 2 and block["n_tested"] == 2
    assert block["evidence_class"] == "predictive (behavioral); not causal"
    p, d = t["planted"], t["decoy"]
    assert p["status"] == "tested" and d["status"] == "tested"
    assert p["lo"] > 0.05 and p["gain"] > p["lo"] and p["confirmed"] is True
    assert p["verdict"] == "confirmed" and p["p_holm"] < 0.05
    assert d["confirmed"] is False and d["verdict"] == "not confirmed"
    assert d["lo"] <= 0 or d["p_holm"] >= 0.05
    assert p["resample_unit"] == "series" and p["n_boot"] == N_BOOT == p["spec_n_boot"]
    assert block["n_confirmed"] == 1
    json.dumps(block)


def test_confirm_scores_with_holm_not_raw_p(world, monkeypatch):
    """Two claims with raw p 0.03 and 0.04 and a positive CI each: raw p would
    confirm both, Holm (adjusted 0.06 and 0.06) confirms neither. Planted
    regression: scoring on the raw p."""
    cfg = _fresh(world)
    reg = _registry(cfg, decoy_false_positive=True)
    fake = {"reliability_u1::planted::log_mase_spearman": 0.03,
            "reliability_u1::decoy::log_mase_spearman": 0.04}

    def _stub(cfg_, hub, private, run_dir, h, acts, errors, l0_cache):
        return {"id": h["id"], "model": h["model"], "task": h["task"], "status": "tested",
                "p": fake[h["id"]], "lo": 0.01, "gain": 0.02, "hi": 0.03}

    monkeypatch.setattr(confirm_mod, "_score_reliability_claim", _stub)
    block = _confirm(cfg, world, reg)
    assert [t["p_holm"] for t in block["tests"]] == pytest.approx([0.06, 0.06])
    assert not any(t["confirmed"] for t in block["tests"]) and block["n_confirmed"] == 0


@pytest.mark.parametrize("case", ["capture_error", "missing_acts", "changed_artifact",
                                  "l0_unavailable"])
def test_unavailable_inputs_are_not_testable_never_not_confirmed(world, case):
    """A claim whose private inputs are unavailable reads `not testable` with the
    reason, stays out of the Holm family and is not counted as a failure; the
    other claim is still tested. Planted regression: mapping unavailable to a
    tested `not confirmed`/False."""
    cfg = _fresh(world)
    reg = _registry(cfg, decoy_false_positive=True)
    acts = {f"{m}/{layer}": a for m in MODELS for layer, a in world.priv.acts[m].items()}
    errors, hub = {}, world.hub
    if case == "capture_error":
        errors = {f"planted/{LAYER}": "RuntimeError: capture blew up"}
    elif case == "missing_acts":
        acts.pop(f"planted/{CRYS}")
    elif case == "changed_artifact":
        checkpoint_path(cfg.run_dir(), "planted", LAYER).write_bytes(b"not the frozen dictionary")
    else:
        hub = _FakeHub(world.priv)
        hub.adapters["planted"].predict = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("model cannot forecast private data"))
    block = _confirm(cfg, world, reg, acts, errors, hub)
    t = _by_model(block)
    assert t["planted"]["status"] == "not testable" and t["planted"]["verdict"] == "not testable"
    assert t["planted"]["confirmed"] is False and t["planted"]["reason"]
    assert "p_holm" not in t["planted"]
    assert t["decoy"]["status"] == "tested"
    assert block["n_tested"] == 1 and block["n_not_testable"] == 1
    assert t["decoy"]["p_holm"] == t["decoy"]["p"]


def test_confirm_uses_the_frozen_sae_and_never_retrains(world, monkeypatch):
    """The private features come from the registered checkpoint, byte-for-byte
    unchanged, and no SAE is trained. Planted regression: a `train_sae` (or any
    trainer) call on private activations trips the guard and/or rewrites the
    checkpoint."""
    from tsfm_lens.sae import train as train_mod
    cfg = _fresh(world)
    reg = _registry(cfg, decoy_false_positive=True)
    before = {p: _sha(p) for p in cfg.run_dir().glob("sae/*/*.pt")}

    def _refuse(*a, **k):
        raise AssertionError("confirm must not retrain an SAE")

    for name in ("train_sae", "search_dict_size", "train_sae_replicate", "run_sae"):
        monkeypatch.setattr(train_mod, name, _refuse)
    loaded, real = [], train_mod.load_sae_checkpoint
    monkeypatch.setattr(train_mod, "load_sae_checkpoint",
                        lambda path: (loaded.append(Path(path).resolve()), real(path))[1])
    block = _confirm(cfg, world, reg)
    assert block["n_tested"] == 2
    assert {checkpoint_path(cfg.run_dir(), m, LAYER).resolve() for m in MODELS} <= set(loaded)
    assert {p: _sha(p) for p in cfg.run_dir().glob("sae/*/*.pt")} == before


def test_confirm_refits_with_the_frozen_folds_seed_and_alpha_grid(world, monkeypatch):
    """Every fold assignment uses the frozen per-model seed and every ridge fit
    the frozen alpha grid (here a non-default one). Planted regressions: a
    changed seed changes the fold-seed set; dropping the grid falls back to the
    default grid."""
    cfg = _fresh(world)
    custom = [0.5, 5.0, 50.0]
    path = cfg.run_dir() / DEV_JSON
    k4 = load_json(path)
    k4["settings"]["ridge_alphas"] = custom
    save_json(path, k4)
    reg = build_registry(cfg)
    spec = next(h for h in reg["hypotheses"] if h["stage"] == RELIABILITY_STAGE)["spec"]
    seeds, grids = [], []
    real_folds, real_ridge = rfi.stratified_folds, rfi._make_ridge
    monkeypatch.setattr(rfi, "stratified_folds",
                        lambda strata, n_folds, seed: (seeds.append(seed),
                                                       real_folds(strata, n_folds, seed))[1])
    monkeypatch.setattr(rfi, "_make_ridge",
                        lambda seed, alphas=rfi.ALPHAS: (grids.append(list(alphas)),
                                                         real_ridge(seed, alphas))[1])
    block = _confirm(cfg, world, reg)
    assert block["n_tested"] == 1
    ms = spec["model_seed"]
    expected = {rfi._seed("folds", rep, base=rfi._seed("ridge", key, base=ms))
                for rep in range(spec["n_repeats"]) for key in ("baseline", "baseline+internals")}
    assert set(seeds) == expected and len(seeds) == 2 * spec["n_repeats"]
    assert grids and all(g == custom for g in grids)


def test_replicate_adds_the_block_and_ledger_row_without_touching_other_keys(world, monkeypatch):
    """`_replicate_reliability_u1` captures only what the shared cache lacks,
    adds `reliability_u1` and its ledger row, and leaves every existing key
    byte-identical (invariant 13)."""
    cfg = _fresh(world)
    reg = _registry(cfg, decoy_false_positive=True)
    captured = []

    def _capture(cfg_, hub, private, targets):
        captured.append(list(targets))
        return ({t: world.priv.acts[t.split("/")[0]][t.split("/", 1)[1]] for t in targets}, {})

    monkeypatch.setattr(confirm_mod, "_capture_private_targets", _capture)
    before = {"status": "skipped", "reason": "no registered concept_transfer hypotheses",
              "transfer": {"status": "skipped", "tests": []}, "knob": {"status": "empty"},
              "ledger": [{"family": "concept_knob", "m": 0}]}
    snapshot = json.dumps(before, sort_keys=True)
    out = confirm_mod._replicate_reliability_u1(
        cfg, world.hub, world.priv.data, reg, before, {}, {})
    assert json.dumps(before, sort_keys=True) == snapshot
    assert captured == [sorted(f"{m}/{layer}" for m in MODELS for layer in (LAYER, CRYS))]
    assert out["transfer"] == before["transfer"] and out["knob"] == before["knob"]
    assert out["ledger"][0] == before["ledger"][0]
    row = out["ledger"][-1]
    assert row["family"] == RELIABILITY_STAGE and row["m"] == 2
    assert row["n_tested"] == 2 and row["n_not_testable"] == 0
    assert out["status"] == "tested" and "reason" not in out
    assert out["reliability_u1"]["n_tested"] == 2


# ---------------------------------------------------------------------------
# 4. End to end on the real mock run (real store, capture and run_confirm).
# ---------------------------------------------------------------------------

def _real_k4(cfg, model: str, layer: str) -> dict:
    """A REAL dev K4 JSON for the mock run, through the shipped loader, with a
    concept_families.json staged over the model's persisted SAE features."""
    run_dir = cfg.run_dir()
    run_pipeline(cfg, stages=["l0"])
    rows = [{"model": model, "layer": layer, "feature": f, "family": f % 3, "concept": None}
            for f in range(9)]
    save_json(run_dir / "sae" / "concept_families.json", {"measured": True, "rows": rows})
    inputs = rfi.load_run_inputs(run_dir)
    inputs["models"] = {model: inputs["models"][model]}
    out = rfi.run_reliability(inputs, n_folds=3, n_repeats=1, n_boot=60, seed=0)
    out["models"][model]["u1"]["log_mase_spearman"]["gain"]["lo"] = 0.01
    return out


def _real_cfg(dev_run):
    from tests.test_concept_causal_confirm import _fresh_run
    cfg = _fresh_run(dev_run)
    cfg.confirm.register_concept_claims = False
    cfg.confirm.register_reliability_claims = True
    cfg.confirm.reliability_dev_json = DEV_JSON
    cfg.stats.min_series = 20
    return cfg


def test_real_run_private_feature_builder_reproduces_the_dev_store_route(dev_run):
    """Frozen-feature check on the shipped capture path: encode the activations
    `_capture_private_targets` returns with the frozen checkpoint and the family
    columns equal what `load_run_inputs` read from the dev store (the same
    series, so the two routes must agree up to the store's float16)."""
    from tests.test_concept_causal_confirm import _hub_and_private
    cfg = _real_cfg(dev_run)
    run_dir = cfg.run_dir()
    model = cfg.sae.targets[0]["model"]
    layer = cfg.sae.targets[0]["layer"]
    k4 = _real_k4(cfg, model, layer)
    save_json(run_dir / DEV_JSON, k4)
    reg = build_registry(cfg)
    h = next(h for h in reg["hypotheses"] if h["stage"] == RELIABILITY_STAGE)
    inputs = rfi.load_run_inputs(run_dir)
    dev_X = inputs["models"][model]["groups"]["sae_families"]
    from tsfm_lens.data import load_benchmark
    from tsfm_lens.models import ModelHub
    from tsfm_lens.utils import resolve_device, resolve_dtype
    device = resolve_device(cfg.run.device)
    hub = ModelHub(cfg.models, cfg.data, device, resolve_dtype(cfg.run.dtype, device))
    dev_data = load_benchmark(cfg.data, cfg.run.seed)
    acts, errors = confirm_mod._capture_private_targets(
        cfg, hub, dev_data, confirm_mod._reliability_layers(h))
    assert not errors
    X, reason = confirm_mod._reliability_internal_columns(run_dir, h, acts, dev_data.n, device)
    assert reason == "" and X.shape[0] == dev_X.shape[0]
    q = dev_X.shape[1]
    np.testing.assert_allclose(X[:, :q], dev_X, rtol=2e-2, atol=2e-2)


def test_end_to_end_run_confirm_writes_the_block_and_flags_off_changes_nothing(dev_run):
    """Real `register` + `confirm`: with the claim on, `confirmation.json` gains
    `concept_replication.reliability_u1` (a tested claim with its frozen n_boot)
    and a ledger row; with the fields off the registry and confirmation carry no
    trace of the claim type, and with the flag on but no dev positive the
    hypotheses list is byte-identical to the flags-off one."""
    from tests.test_concept_causal_confirm import _fresh_run, _k2_cfg
    on = _real_cfg(dev_run)
    model, layer = on.sae.targets[0]["model"], on.sae.targets[0]["layer"]
    k4 = _real_k4(on, model, layer)
    save_json(on.run_dir() / DEV_JSON, k4)
    run_pipeline(on, stages=["register", "confirm"])
    conf = load_json(on.run_dir() / "confirm" / "confirmation.json")
    cr = conf["concept_replication"]
    block = cr["reliability_u1"]
    assert block["family"] == RELIABILITY_STAGE and block["n_registered"] == 1
    t = block["tests"][0]
    assert t["id"] == f"reliability_u1::{model}::log_mase_spearman"
    assert t["status"] == "tested", t.get("reason")
    assert t["n_boot"] == 60 and t["resample_unit"] == "series" and t["n"] >= 20
    assert cr["status"] == "tested" and {"transfer", "knob", "ledger"} <= set(cr)
    assert any(r["family"] == RELIABILITY_STAGE for r in cr["ledger"])
    assert conf["n_registered"] == len(load_json(on.run_dir() / "hypotheses.json")["hypotheses"])

    off = _fresh_run(dev_run)
    off.confirm.register_concept_claims = False
    run_pipeline(off, stages=["l0"])
    work = Path(tempfile.mkdtemp())
    shutil.copytree(off.run_dir(), work / off.run_dir().name)
    zero = _k2_cfg(str(work), off.run.name, off.sae.targets)
    zero.confirm.register_concept_claims = False
    zero.confirm.register_reliability_claims = True
    zero.confirm.reliability_dev_json = DEV_JSON
    run_pipeline(off, stages=["register", "confirm"])
    off_reg = load_json(off.run_dir() / "hypotheses.json")
    off_conf = load_json(off.run_dir() / "confirm" / "confirmation.json")
    assert "reliability" not in json.dumps(off_reg) and "reliability" not in json.dumps(off_conf)
    assert RELIABILITY_STAGE not in off_conf["concept_replication"]

    k4z = copy.deepcopy(k4)
    k4z["models"][model]["u1"]["log_mase_spearman"]["gain"]["lo"] = -0.01
    save_json(zero.run_dir() / DEV_JSON, k4z)
    run_pipeline(zero, stages=["register", "confirm"])
    zero_reg = load_json(zero.run_dir() / "hypotheses.json")
    zero_conf = load_json(zero.run_dir() / "confirm" / "confirmation.json")
    assert zero_reg["hypotheses"] == off_reg["hypotheses"]
    assert set(zero_reg) - set(off_reg) == {"reliability_u1_candidates"}
    for d in (zero_conf, off_conf):
        d.pop("registry_sha256")
    assert zero_conf == off_conf
