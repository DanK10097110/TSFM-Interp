"""The L5 planted pair: two different architectures, one planted vocabulary
(ROADMAP.md sec 38.3.5).

`mock_planted`'s opt-in `l5` vocabulary builds two members that differ in width,
depth, head count and (for B) a random orthogonal rotation of the residual basis,
and plants the SAME concepts in both. Its own properties are tested here,
analytically and with no SAE: the architectures really differ; a shared concept has
the same trigger and the same effect in both; the opposite-effect decoy has the same
trigger and a flipped sign; the input-only decoy has the same trigger and no effect
in B; projecting a concept's RESIDUAL-basis directions out of the planted layer
removes exactly the concept's planted forecast component in the rotated member
too (which fails if the manifest's direction is the canonical one); K1's default
adapter is unchanged byte for byte.
"""

from __future__ import annotations

import hashlib
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
from tsfm_lens.models import build_adapter  # noqa: E402
from tsfm_lens.models.mock_planted import (DOSE_UNIT_EFFECT, _shapes, concept_table_l5,  # noqa: E402
                                           rotation_matrix)

HORIZON = 32
QUANTILES = [0.1, 0.5, 0.9]
TOL = 1e-5
ARCH = {"A": dict(width=64, depth=5, heads=2, planted_block=2, rotate=False),
        "B": dict(width=96, depth=7, heads=4, planted_block=4, rotate=True)}
LAYER = {"A": "blocks.2", "B": "blocks.4"}


def _data_cfg() -> DataConfig:
    return DataConfig(source="smoke", context_len=256, horizon=HORIZON, smoke_series_per_family=24)


def _adapter(plant_set, seed=0, dose=1.0, **override):
    kwargs = {"plant_set": plant_set, "vocabulary": "l5", "construction_seed": seed, "dose": dose,
              **ARCH[plant_set], **override}
    cfg = ModelConfig(name=plant_set, adapter="mock_planted", batch_size=64, kwargs=kwargs)
    a = build_adapter(cfg, _data_cfg(), torch.device("cpu"), torch.float32)
    a.ensure_loaded()
    return a


@pytest.fixture(scope="module")
def world():
    data = load_benchmark(_data_cfg())
    adapters = {s: _adapter(s) for s in "AB"}
    return adapters, data, {s: a.manifest() for s, a in adapters.items()}


def _concept_rows(manifest, concept):
    return [c for c in manifest["concepts"] if c["concept"] == concept]


def _forecast(adapter, contexts):
    torch.manual_seed(0)
    return adapter.predict(contexts, HORIZON, QUANTILES)["point"].astype(np.float64)


def _project_out(adapter, manifest, layer, contexts, comp_ids, key="direction_resid"):
    """Forecast after removing the named components' coefficients from the planted
    layer, using the directions stored under `key` (the dual basis is rebuilt from
    them in float64)."""
    D = np.stack([c[key] for c in manifest["concepts"]], axis=1)
    U = np.linalg.solve(D.T @ D, D.T)
    ids = [c["id"] for c in manifest["concepts"]]
    tokens = capture_raw_tokens(adapter, contexts, [layer])[layer]
    removed = tokens.double().clone()
    for cid in comp_ids:
        k = ids.index(cid)
        coef = torch.from_numpy(tokens.double().numpy() @ U[k])
        removed = removed - coef[..., None] * torch.from_numpy(D[:, k])
    with token_patch(adapter.module, layer, adapter.token_slice, removed.float()):
        return _forecast(adapter, contexts)


def _expected_component(manifest, contexts, comp):
    sd = contexts.std(axis=1) + 1e-6
    shape = _shapes(HORIZON)[comp["kind"]]
    a = np.asarray(comp["series_activation"])
    return sd[:, None] * comp["beta"] * a[:, None] * shape[None, :]


def test_architectures_differ_and_the_rotation_is_orthogonal(world):
    adapters, data, manifests = world
    a, b = adapters["A"], adapters["B"]
    assert (a.dim, a.n_layers, a.n_heads) == (64, 5, 2)
    assert (b.dim, b.n_layers, b.n_heads) == (96, 7, 4)
    assert manifests["A"]["rotation_seed"] is None and manifests["B"]["rotation_seed"] is not None
    assert manifests["A"]["planted_layer"] == LAYER["A"] and manifests["B"]["planted_layer"] == LAYER["B"]
    contexts = data.contexts()[:8]
    wa = capture_raw_tokens(a, contexts, [LAYER["A"]])[LAYER["A"]].shape[-1]
    wb = capture_raw_tokens(b, contexts, [LAYER["B"]])[LAYER["B"]].shape[-1]
    assert (wa, wb) == (64, 96)
    Q = rotation_matrix(96, manifests["B"]["rotation_seed"])
    assert np.abs(Q @ Q.T - np.eye(96)).max() < 1e-10
    for c in manifests["A"]["concepts"]:
        assert c["direction_resid"] == c["direction"]
    moved = [np.linalg.norm(np.asarray(c["direction_resid"]) - np.asarray(c["direction"]))
             for c in manifests["B"]["concepts"]]
    assert min(moved) > 0.5


def test_shared_concepts_share_trigger_and_effect_and_decoys_differ_only_in_effect(world):
    adapters, data, manifests = world
    for cid in ("l5_shared_single", "l5_shared_dist", "l5_opposite", "l5_inputonly"):
        a0, b0 = (_concept_rows(manifests[s], cid)[0] for s in "AB")
        assert a0["readout_weights"] == b0["readout_weights"], cid
        assert np.array_equal(a0["series_activation"], b0["series_activation"]), cid
        assert a0["kind"] == b0["kind"]
    for cid in ("l5_shared_single", "l5_shared_dist"):
        a_rows, b_rows = (_concept_rows(manifests[s], cid) for s in "AB")
        assert {c["sign"] for c in a_rows + b_rows} == {1}
        for rows in (a_rows, b_rows):
            total = sum(c["rms_effect_on_top_series_ctx_sd"] for c in rows)
            assert total == pytest.approx(DOSE_UNIT_EFFECT, rel=1e-9)
    assert [len(_concept_rows(manifests[s], "l5_shared_dist")) for s in "AB"] == [3, 4]
    fractions = [c["firing_fraction"] for c in _concept_rows(manifests["B"], "l5_shared_dist")]
    assert fractions == sorted(fractions, reverse=True) and len(set(fractions)) == 4
    opp_a, opp_b = (_concept_rows(manifests[s], "l5_opposite")[0] for s in "AB")
    assert opp_a["beta"] > 0 > opp_b["beta"]
    assert abs(opp_a["beta"]) == pytest.approx(abs(opp_b["beta"]))
    ino_a, ino_b = (_concept_rows(manifests[s], "l5_inputonly")[0] for s in "AB")
    assert ino_a["beta"] > 0 and ino_b["beta"] == 0.0
    assert len({c["concept"] for c in manifests["A"]["concepts"]}) == len(concept_table_l5()) - 3


def test_projection_removes_exactly_the_planted_component_in_both_architectures(world):
    """Every concept with an effect, including the distributed one (all its
    components removed together) and in the ROTATED member."""
    adapters, data, manifests = world
    contexts = data.contexts()
    for s in "AB":
        clean = _forecast(adapters[s], contexts)
        for concept in ("l5_shared_single", "l5_shared_dist", "l5_opposite", "l5_pure_dispersion"):
            comps = _concept_rows(manifests[s], concept)
            delta = _project_out(adapters[s], manifests[s], LAYER[s], contexts,
                                 [c["id"] for c in comps]) - clean
            expected = -sum(_expected_component(manifests[s], contexts, c) for c in comps)
            assert np.abs(delta - expected).max() < TOL, (s, concept)
            assert np.abs(expected).max() > 10 * TOL, (s, concept)


def test_input_only_decoy_is_inert_in_b_and_causal_in_a(world):
    adapters, data, manifests = world
    contexts = data.contexts()
    for s, causal in (("A", True), ("B", False)):
        clean = _forecast(adapters[s], contexts)
        comp = _concept_rows(manifests[s], "l5_inputonly")[0]
        D = np.stack([c["direction_resid"] for c in manifests[s]["concepts"]], axis=1)
        k = [c["id"] for c in manifests[s]["concepts"]].index(comp["id"])
        tokens = capture_raw_tokens(adapters[s], contexts, [LAYER[s]])[LAYER[s]].double().numpy()
        written = np.abs(tokens @ np.linalg.solve(D.T @ D, D.T)[k]).max()
        assert written > 0.5, "the decoy must be written into the residual in both members"
        delta = _project_out(adapters[s], manifests[s], LAYER[s], contexts, [comp["id"]]) - clean
        assert (np.abs(delta).max() > 10 * TOL) if causal else (np.abs(delta).max() < 1e-6)


def test_self_patch_is_exactly_zero_and_the_wider_member_is_reachable(world):
    from tests.test_smoke import build_config
    from tsfm_lens.analysis.response_reach import reach_probe
    from tsfm_lens.config import config_from_dict
    adapters, data, manifests = world
    cfg = config_from_dict(build_config("/tmp/unused"))
    cfg.data.context_len, cfg.data.horizon = 256, HORIZON
    for s in "AB":
        reach = reach_probe(cfg, adapters[s], LAYER[s], data, torch.device("cpu"), max_series=16)
        assert reach["reachable"], (s, reach["reason"])
        assert reach["self_patch_delta"] == 0.0 and reach["cross_patch_delta"] > 0.0


def test_identical_predict_calls_are_identical(world):
    adapters, data, _ = world
    contexts = data.contexts()[:32]
    for a in adapters.values():
        p, q = a.predict(contexts, HORIZON, QUANTILES), a.predict(contexts, HORIZON, QUANTILES)
        assert np.array_equal(p["point"], q["point"]) and np.array_equal(p["quantiles"], q["quantiles"])


K1_GOLDEN = {
    "A_forecast": "7a100ce07e9842044de429de7cce962a757598e623c2e785df3bc3c4c4d419bf",
    "B_forecast": "57fed8b8a0b4a46a24b0dc2a5357d10155e98bc23083e27d87c50d391273a43f",
}


def test_k1_default_adapter_is_byte_identical_to_before_the_l5_options():
    """The opt-in kwargs default to nothing: K1's forecasts (smoke corpus, seed 3,
    dose 1) hash to the digests recorded from the pre-L5 module."""
    from tsfm_lens.models import mock_planted
    data = load_benchmark(_data_cfg())
    for s in "AB":
        mock_planted._CALIBRATION_CACHE.clear()
        cfg = ModelConfig(name=s, adapter="mock_planted", batch_size=64,
                          kwargs={"plant_set": s, "dose": 1.0, "construction_seed": 3})
        a = build_adapter(cfg, _data_cfg(), torch.device("cpu"), torch.float32)
        a.ensure_loaded()
        assert "vocabulary" not in a.manifest() and a.dim == 64 and a.n_layers == 5
        p = a.predict(data.contexts(), HORIZON, QUANTILES)
        digest = hashlib.sha256(p["point"].tobytes() + p["quantiles"].tobytes()).hexdigest()
        assert digest == K1_GOLDEN[f"{s}_forecast"], s
    mock_planted._CALIBRATION_CACHE.clear()
