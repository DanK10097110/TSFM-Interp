"""`_load_l3_patching_secondary_gold` (ROADMAP.md §6.1.1's still-open
"does `work_bend` survive the per-window-patching secondary gold" item).

Additive to the existing L3-sensitivity secondary gold
(`_load_l3_secondary_gold` in the same script) -- this is a second,
independent cross-check on the bake-off's expensive SAE-mass primary gold,
built from `l3/patching.npz`'s causal per-window restoration signal instead
of the cheaper, correlational sensitivity-fingerprint signal. All synthetic
data with a known correct answer, mirroring `test_stage0_criteria.py`'s
pattern for importing a top-level CLI script by path (`run_layer_screen_
bakeoff.py` lives at the repo root next to `run.py`, not inside the
`tsfm_lens` package).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _harness():
    spec = importlib.util.spec_from_file_location(
        "run_layer_screen_bakeoff", ROOT / "run_layer_screen_bakeoff.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_layer_screen_bakeoff"] = module
    spec.loader.exec_module(module)
    return module


def _write_patching_artifacts(run_dir: Path, model: str, layers: list,
                              restoration: np.ndarray, patched_layers: list | None = None) -> None:
    """Write a minimal `l3/patching.npz` + `l3/patching.json` pair, mirroring
    `l3_perturbation.py::run_l3`'s own output shape exactly (checked directly
    against that module rather than assumed): `restoration_{model}` has shape
    `[n_corruptions, n_layers_p]`, and `patching.json[model]["layers"]` names
    which (possibly layer_stride-subsampled) layers those columns correspond to.
    """
    out = run_dir / "l3"
    out.mkdir(parents=True, exist_ok=True)
    patched_layers = patched_layers if patched_layers is not None else layers
    np.savez(out / "patching.npz", **{f"restoration_{model}": restoration})
    (out / "patching.json").write_text(
        json.dumps({model: {"layers": patched_layers, "corruptions": ["noise", "spike"]}}),
        encoding="utf-8")


def test_averages_across_the_corruption_axis():
    """A planted, layer-varying restoration score must come back unchanged by
    averaging over corruptions -- the reduction `_load_l3_secondary_gold`
    already applies to the sensitivity fingerprint's own corruption axis."""
    m = _harness()
    layers = [f"block.{i}" for i in range(4)]
    # [n_corruptions=3, n_layers=4] -- planted so the per-layer mean is exact.
    restoration = np.array([
        [0.1, 0.5, 0.9, 0.2],
        [0.3, 0.5, 0.7, 0.4],
        [0.2, 0.5, 0.8, 0.3],
    ], dtype=np.float32)
    expected = restoration.mean(axis=0)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        _write_patching_artifacts(run_dir, "M", layers, restoration)
        got = m._load_l3_patching_secondary_gold(run_dir, "M", layers)
    assert got is not None
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_missing_artifacts_degrade_to_none():
    m = _harness()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        # Neither patching.npz nor patching.json exists (e.g. l3.patching.enabled: false).
        got = m._load_l3_patching_secondary_gold(run_dir, "M", ["block.0", "block.1"])
    assert got is None


def test_layer_stride_subset_degrades_to_none_rather_than_silently_reindexing():
    """If `l3.patching.layer_stride > 1` subsampled the patched layer set,
    the cross-check must refuse rather than guess an alignment -- exactly
    `_load_l3_secondary_gold`'s own strict-shape contract, applied here to a
    layer-NAME mismatch instead of a raw shape[0] mismatch (patching's
    layer subset is a *subset by identity*, not just a shorter array)."""
    m = _harness()
    full_layers = [f"block.{i}" for i in range(6)]
    patched_layers = full_layers[::2]  # layer_stride=2, half the layers
    restoration = np.random.default_rng(0).random((2, len(patched_layers))).astype(np.float32)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        _write_patching_artifacts(run_dir, "M", full_layers, restoration,
                                  patched_layers=patched_layers)
        got = m._load_l3_patching_secondary_gold(run_dir, "M", full_layers)
    assert got is None


def test_full_layer_stride_one_matches_exactly():
    """The intended, documented usage: `l3.patching.layer_stride: 1` patches
    every layer the primary/sensitivity golds also cover, so the patched
    layer list exactly equals the full `layers` list and the cross-check fires."""
    m = _harness()
    layers = [f"block.{i}" for i in range(5)]
    restoration = np.random.default_rng(1).random((4, len(layers))).astype(np.float32)
    expected = restoration.mean(axis=0)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        _write_patching_artifacts(run_dir, "M", layers, restoration, patched_layers=layers)
        got = m._load_l3_patching_secondary_gold(run_dir, "M", layers)
    assert got is not None
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_wrong_model_key_degrades_to_none():
    m = _harness()
    layers = [f"block.{i}" for i in range(3)]
    restoration = np.random.default_rng(2).random((2, len(layers))).astype(np.float32)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        _write_patching_artifacts(run_dir, "OtherModel", layers, restoration)
        got = m._load_l3_patching_secondary_gold(run_dir, "M", layers)
    assert got is None


def test_gold_vs_patching_gold_rho_recovers_a_planted_monotone_relationship():
    """Spearman between the primary SAE-mass gold and the patching-based
    secondary gold should recover a strong positive correlation when one is
    a planted monotone function of the other, and not spuriously find one
    from independent noise -- the same sanity bar the existing sensitivity
    cross-check's `gold_agreement` implicitly relies on."""
    from scipy.stats import spearmanr
    m = _harness()
    layers = [f"block.{i}" for i in range(8)]
    rng = np.random.default_rng(3)
    gold_score = np.array([10.0, 25.0, 40.0, 55.0, 70.0, 60.0, 45.0, 20.0])
    # Planted: patching restoration tracks gold_score's rank order plus noise.
    restoration = np.tile(gold_score / gold_score.max(), (3, 1)) + rng.normal(0, 0.01, (3, 8))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        _write_patching_artifacts(run_dir, "M", layers, restoration.astype(np.float32),
                                  patched_layers=layers)
        patching_gold = m._load_l3_patching_secondary_gold(run_dir, "M", layers)
    assert patching_gold is not None
    rho, _ = spearmanr(gold_score, patching_gold)
    assert rho > 0.9, f"expected a strong recovered correlation, got rho={rho}"

    # Independent (shuffled) noise should NOT manufacture the same strength.
    shuffled = rng.permutation(gold_score)
    rho_null, _ = spearmanr(shuffled, patching_gold)
    assert abs(rho_null) < abs(rho)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
