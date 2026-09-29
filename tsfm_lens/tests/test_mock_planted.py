"""The planted forecaster's analytic invariants (ROADMAP.md sec 38.1.2, K1).

`mock_planted` carries the ground truth for the whole known-answer study, so
its own properties are tested first and analytically, with no SAE and no
battery in the loop:

  (i)   patching the planted layer into itself moves the forecast by exactly
        0.0, and patching a shallower layer into it moves the forecast;
  (ii)  projecting a concept's direction out of the planted layer removes
        exactly that concept's planted forecast component (tolerance 1e-5,
        float32) on series where it fires and changes nothing where it does not;
  (iii) projecting out an input-only decoy changes the forecast by exactly 0;
  (iv)  two identical `predict()` calls are identical (CLAUDE.md sec 11.49).

The expected components are computed from the MANIFEST (`series_activation`,
`beta`, the unit-RMS shapes) and the dual basis rebuilt from the manifest's
directions in float64, never from the network's own tensors, so a head that
disagrees with its manifest is caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens.config import DataConfig, ModelConfig  # noqa: E402
from tsfm_lens.data import load_benchmark  # noqa: E402
from tsfm_lens.extraction.extract import capture_raw_tokens  # noqa: E402
from tsfm_lens.extraction.hooks import token_patch  # noqa: E402
from tsfm_lens.models import ADAPTERS, build_adapter  # noqa: E402
from tsfm_lens.models.mock_planted import _shapes, concept_table  # noqa: E402

HORIZON = 32
QUANTILES = [0.1, 0.5, 0.9]
TOL = 1e-5
PLANTED = "blocks.2"
CONTROL = "blocks.3"


def _data_cfg() -> DataConfig:
    return DataConfig(source="smoke", context_len=256, horizon=HORIZON, smoke_series_per_family=24)


def _adapter(plant_set="A", dose=1.0, seed=0):
    cfg = ModelConfig(name=plant_set, adapter="mock_planted", batch_size=64,
                      kwargs={"plant_set": plant_set, "dose": dose, "construction_seed": seed})
    a = build_adapter(cfg, _data_cfg(), torch.device("cpu"), torch.float32)
    a.ensure_loaded()
    return a


@pytest.fixture(scope="module")
def world():
    adapter = _adapter()
    data = load_benchmark(_data_cfg())
    manifest = adapter.manifest()
    rows = np.arange(data.n)
    return adapter, data, manifest, rows


def _dual(manifest):
    D = np.stack([c["direction"] for c in manifest["concepts"]], axis=1)
    return D, np.linalg.solve(D.T @ D, D.T)


def _forecast(adapter, contexts):
    torch.manual_seed(0)
    return adapter.predict(contexts, HORIZON, QUANTILES)["point"].astype(np.float64)


def _patched_forecast(adapter, contexts, replacement):
    with token_patch(adapter.module, PLANTED, adapter.token_slice, replacement):
        return _forecast(adapter, contexts)


def _project_out(adapter, manifest, contexts, k):
    """Forecast after removing concept `k`'s coefficient from the planted layer."""
    D, U = _dual(manifest)
    tokens = capture_raw_tokens(adapter, contexts, [PLANTED])[PLANTED]
    coef = torch.from_numpy(tokens.double().numpy() @ U[k])
    removed = tokens.double() - coef[..., None] * torch.from_numpy(D[:, k])
    return _patched_forecast(adapter, contexts, removed.float())


def _expected_component(manifest, contexts, rows, k):
    """The concept's planted forecast component, from the manifest alone."""
    c = manifest["concepts"][k]
    a = np.asarray(c["series_activation"])[rows]
    sd = contexts.std(axis=1) + 1e-6
    shape = _shapes(HORIZON)[c["kind"]] if c["kind"] else np.zeros(HORIZON)
    return sd[:, None] * c["beta"] * a[:, None] * shape[None, :]


def test_registered_and_tier_three_by_derivation():
    from tsfm_lens.models.mock_planted import MockPlantedAdapter
    assert ADAPTERS["mock_planted"] is MockPlantedAdapter
    assert MockPlantedAdapter.capability_tier() == 3
    assert "capability_tier" not in MockPlantedAdapter.__dict__


def test_self_patch_is_exactly_zero_and_cross_patch_is_not(world):
    """Invariant (i), including the real reach probe on the planted layer."""
    adapter, data, manifest, rows = world
    contexts = data.contexts()[rows]
    clean = _forecast(adapter, contexts)
    tokens = capture_raw_tokens(adapter, contexts, [PLANTED, "blocks.0"])
    self_delta = np.abs(_patched_forecast(adapter, contexts, tokens[PLANTED]) - clean).max()
    cross_delta = np.abs(_patched_forecast(adapter, contexts, tokens["blocks.0"]) - clean).mean()
    assert self_delta == 0.0
    assert cross_delta > 1e-4

    from tsfm_lens.analysis.response_reach import reach_probe
    from tsfm_lens.config import config_from_dict
    from tests.test_smoke import build_config
    cfg = config_from_dict(build_config("/tmp/unused"))
    cfg.data.context_len, cfg.data.horizon = 256, HORIZON
    reach = reach_probe(cfg, adapter, PLANTED, data, torch.device("cpu"), max_series=16)
    assert reach["reachable"], reach["reason"]
    assert reach["self_patch_delta"] == 0.0 and reach["cross_patch_delta"] > 0.0


def test_control_layer_after_the_planted_block_is_reachable_and_one_before_is_not(world):
    """The control layer must pass the real reach probe (self-patch exactly 0.0,
    cross-layer relative change above `min_relative_reach`); a layer BEFORE the
    planted block cannot, because the planted block computes from the input."""
    from tsfm_lens.analysis.response_reach import reach_probe
    from tsfm_lens.config import config_from_dict
    from tests.test_smoke import build_config
    adapter, data, manifest, rows = world
    cfg = config_from_dict(build_config("/tmp/unused"))
    cfg.data.context_len, cfg.data.horizon = 256, HORIZON
    after = reach_probe(cfg, adapter, CONTROL, data, torch.device("cpu"), max_series=16)
    assert after["reachable"], after["reason"]
    assert after["self_patch_delta"] == 0.0
    assert after["relative_reach"] > max(1e-3, cfg.concepts.min_relative_reach)
    before = reach_probe(cfg, adapter, "blocks.1", data, torch.device("cpu"), max_series=16)
    assert not before["reachable"]
    assert before["relative_reach"] < 1e-3


def test_mock_planted_projection_removes_exactly_the_planted_component(world):
    """Invariant (ii) for EVERY real concept: the forecast moves by minus the
    manifest's planted component where it fires, and by 0 where it does not."""
    adapter, data, manifest, rows = world
    contexts = data.contexts()[rows]
    clean = _forecast(adapter, contexts)
    real = [k for k, c in enumerate(manifest["concepts"]) if c["cls"] in
            ("shared", "convergent", "opposite", "unique")]
    n_fire_checked = 0
    for k in real:
        delta = _project_out(adapter, manifest, contexts, k) - clean
        expected = -_expected_component(manifest, contexts, rows, k)
        assert np.abs(delta - expected).max() < TOL, manifest["concepts"][k]["id"]
        fires = np.asarray(manifest["concepts"][k]["series_activation"])[rows] > 0.0
        if fires.any():
            n_fire_checked += 1
            assert np.abs(expected[fires]).max() > 10 * TOL, manifest["concepts"][k]["id"]
        if (~fires).any():
            assert np.abs(delta[~fires]).max() < TOL
    assert n_fire_checked == len(real)


def test_input_only_decoy_is_causally_inert(world):
    """Invariant (iii): removing an input-only decoy's direction moves nothing,
    though it is written into the residual on the series it fires on."""
    adapter, data, manifest, rows = world
    contexts = data.contexts()[rows]
    clean = _forecast(adapter, contexts)
    decoys = [k for k, c in enumerate(manifest["concepts"]) if c["cls"] == "decoy_input_only"]
    assert len(decoys) == 3
    D, U = _dual(manifest)
    tokens = capture_raw_tokens(adapter, contexts, [PLANTED])[PLANTED].double().numpy()
    for k in decoys:
        written = np.abs(tokens @ U[k]).max()
        assert written > 0.5, "the decoy must actually be written into the residual"
        delta = _project_out(adapter, manifest, contexts, k) - clean
        assert np.abs(delta).max() < 1e-6, manifest["concepts"][k]["id"]


def test_identical_predict_calls_are_identical(world):
    """Invariant (iv)."""
    adapter, data, manifest, rows = world
    contexts = data.contexts()[rows]
    a = adapter.predict(contexts, HORIZON, QUANTILES)
    b = adapter.predict(contexts, HORIZON, QUANTILES)
    assert np.array_equal(a["point"], b["point"])
    assert np.array_equal(a["quantiles"], b["quantiles"])


def test_directions_firing_and_dose_scaling():
    """Superposition and sparsity controls, and that dose scales real betas only."""
    a1, a4 = _adapter(dose=1.0), _adapter(dose=4.0)
    m1, m4 = a1.manifest(), a4.manifest()
    D = np.stack([c["direction"] for c in m1["concepts"]])
    cos = (D @ D.T)[np.triu_indices(len(D), 1)]
    assert cos.min() >= 0.0 and cos.max() <= 0.3
    for c1, c4 in zip(m1["concepts"], m4["concepts"]):
        assert 0.10 <= c1["firing_fraction"] <= 0.30, c1["id"]
        assert c1["direction"] == c4["direction"]
        if c1["cls"] in ("shared", "convergent", "opposite", "unique"):
            assert c4["beta"] == pytest.approx(4.0 * c1["beta"], rel=1e-9)
        else:
            assert c4["beta"] == c1["beta"]


def test_pair_vocabulary_matches_the_table():
    """A and B carry the classes the design lists; unique is A only; the opposite
    twin has the opposite sign and the same readout; a convergent one does not;
    and every effect signature belongs to exactly one class, with >= 3 items."""
    ma, mb = _adapter("A").manifest(), _adapter("B").manifest()
    ia = {c["id"]: c for c in ma["concepts"]}
    ib = {c["id"]: c for c in mb["concepts"]}
    assert not any(k.startswith("unique_") for k in ib) and any(k.startswith("unique_") for k in ia)
    for cid in ("opposite_level_1", "opposite_level_3"):
        assert ia[cid]["sign"] == -ib[cid]["sign"] != 0
        assert ia[cid]["readout_weights"] == ib[cid]["readout_weights"]
        assert ia[cid]["readout_threshold"] == ib[cid]["readout_threshold"]
    for cid in ("shared_trend_1", "shared_trend_2"):
        assert ia[cid]["readout_weights"] == ib[cid]["readout_weights"]
        assert ia[cid]["sign"] == ib[cid]["sign"]
        assert ia[cid]["direction"] != ib[cid]["direction"]
    for cid in ("convergent_trend_1", "convergent_dispersion_2"):
        assert ia[cid]["readout_weights"] != ib[cid]["readout_weights"]
        assert ia[cid]["sign"] == ib[cid]["sign"]
    signature_class = {}
    for model in (ma, mb):
        for c in model["concepts"]:
            if c["cls"] in ("shared", "convergent", "opposite", "unique"):
                signature_class.setdefault((c["kind"], c["sign"]), set()).add(c["cls"])
    assert all(len(v) == 1 for v in signature_class.values()), signature_class
    counts = {}
    for c in ma["concepts"]:
        if c["cls"] in ("shared", "convergent", "opposite", "unique"):
            counts[(c["kind"], c["sign"])] = counts.get((c["kind"], c["sign"]), 0) + 1
    assert all(n >= 3 for n in counts.values()), counts
    assert {r["cls"] for r in concept_table()} >= {"shared", "convergent", "opposite", "unique",
                                                   "decoy_input_only", "decoy_sub_null"}


def test_convergent_pairs_read_rank_independent_inputs_and_shared_pairs_identical_ones():
    """Convergent means same effect, DIFFERENT inputs: the two models' series-level
    activations of a convergent object are rank-uncorrelated (|rho| < 0.1, recorded
    in the manifest), while a shared object's are identical."""
    from scipy.stats import spearmanr
    a, b = _adapter("A"), _adapter("B")
    ma = {c["id"]: c for c in a.manifest()["concepts"]}
    mb = {c["id"]: c for c in b.manifest()["concepts"]}
    conv = [i for i, c in ma.items() if c["cls"] == "convergent"]
    assert len(conv) == 6
    for i in conv:
        rho = abs(spearmanr(ma[i]["series_activation"], mb[i]["series_activation"]).statistic)
        assert rho < 0.1, (i, rho)
        assert ma[i]["readout_weights"] != mb[i]["readout_weights"]
        assert ma[i]["partner_input_abs_spearman"] == pytest.approx(rho, abs=1e-9)
    for i, c in ma.items():
        if c["cls"] == "shared":
            assert c["series_activation"] == mb[i]["series_activation"]
