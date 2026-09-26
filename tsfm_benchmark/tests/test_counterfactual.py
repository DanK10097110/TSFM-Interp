"""Tests for the generator-side input counterfactuals (ROADMAP.md §37.9 P6a).

Touches no existing generator: every test drives `random_parametric` and
`regenerate()` only, so the golden-hash tests in `test_generator_extensions.py`
are unaffected by this module's existence.
"""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

# `tsfm_benchmark` is `tests/`'s grandparent (`tests -> tsfm_benchmark ->
# repo root`; unlike `tsfm_lens`, this package has no extra nesting level),
# so the repo root to prepend is `parents[2]`, not `parents[1]` (that would
# insert the `tsfm_benchmark` package directory itself, which does not make
# `import tsfm_benchmark` resolve to it). The env's editable install maps
# `tsfm_benchmark` to the MAIN checkout, and when the rest of the suite is
# collected first (alphabetically, `test_benchmark_validation.py` before this
# file) it already caches that main-checkout package in `sys.modules`, so a
# plain `sys.path` insert alone would not help (import resolves from
# `sys.modules` first, CLAUDE.md sec 11.52). Deleting the cached entries would
# "fix" this file but breaks `test_benchmark_validation.py`, which had already
# bound a reference to the pre-purge `sources` submodule for `monkeypatch` at
# collection time: a later, unrelated re-import of that submodule (via the
# now-orphaned parent) creates a second, unpatched copy, and its real
# `get_dataset_config_names` call fails against the test's placeholder
# 'fake/dataset' (measured, not guessed -- this exact failure mode reproduced
# and vanished with this fix). Extending `build_pipeline`'s own `__path__` in
# place, instead of touching `sys.modules`, adds a second search location for
# a name that is unique to the worktree (`counterfactual` does not exist in
# the main checkout, so there is no collision to order) without invalidating
# any already-imported submodule's identity.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
import tsfm_benchmark.build_pipeline as _bp  # noqa: E402

_worktree_bp_dir = str(_REPO_ROOT / "tsfm_benchmark" / "build_pipeline")
if _worktree_bp_dir not in _bp.__path__:
    _bp.__path__.append(_worktree_bp_dir)

from tsfm_benchmark.build_pipeline.counterfactual import regenerate
from tsfm_benchmark.build_pipeline.generators import parametric, random_parametric

# Archetype chosen per knob so the target component is *guaranteed* present by
# construction (the archetype's sampling range for that knob excludes zero /
# always includes the dict key), rather than searching seeds.
KNOB_ARCHETYPE = {
    "seasonal_amplitude": "seasonal_dominant",
    "anomaly_magnitude": "anomaly_heavy",
    "heteroskedastic_depth": "amplitude_modulated",
    "trend_scale": "trend_dominant",
    "intermittency_rate": "intermittent_bursts",
}


def _make(archetype: str, seed: int = 0):
    return random_parametric(seed=seed, archetypes=[archetype])


def _target_seasonal_key(sample) -> str:
    seas = sample.provenance.generator_params["sampled_params"]["seasonalities"]
    idx = max(range(len(seas)), key=lambda i: seas[i]["amplitude"])
    return f"seasonal_{int(seas[idx]['period'])}"


def test_dose_one_is_bit_identical():
    for knob, archetype in KNOB_ARCHETYPE.items():
        sample = _make(archetype, seed=3)
        cf = regenerate(sample, knob, factor=1.0)
        assert np.array_equal(cf.values, sample.values), f"{knob}: dose 1.0 not bit-identical"


def test_non_target_components_bit_identical():
    target_keys_by_knob = {
        "seasonal_amplitude": None,  # resolved per-sample below
        "anomaly_magnitude": set(),
        "heteroskedastic_depth": {"noise_envelope", "noise"},
        "trend_scale": {"trend"},
        "intermittency_rate": {"intermittency_mask"},
    }
    for knob, archetype in KNOB_ARCHETYPE.items():
        sample = _make(archetype, seed=5)
        cf = regenerate(sample, knob, factor=1.7)
        target_keys = target_keys_by_knob[knob]
        if target_keys is None:
            target_keys = {_target_seasonal_key(sample)}
        for key, vals in sample.ground_truth.components.items():
            if key in target_keys:
                continue
            assert np.allclose(vals, cf.ground_truth.components[key]), \
                f"{knob}: non-target component {key!r} moved"
        assert sample.ground_truth.changepoints == cf.ground_truth.changepoints, \
            f"{knob}: changepoint positions moved"
        if knob != "anomaly_magnitude":
            assert sample.ground_truth.anomalies == cf.ground_truth.anomalies, \
                f"{knob}: anomalies moved"


def test_noise_scale_coupling_breaks_non_target_check():
    """Load-bearing plant: `noise_scale` is deliberately excluded from `KNOBS`
    (§37.9's table) because it also scales changepoint shifts and anomaly
    magnitudes -- it is not a clean intervention. This confirms the
    non-target-component check used above actually detects that coupling when
    applied to a hand-built `noise_scale` rescaling. It never goes through
    `regenerate()` (which refuses unknown knobs by design) -- this exercises
    the CHECK's ability to catch a coupled knob, not `regenerate()` itself.
    """
    sample = _make("anomaly_heavy", seed=11)
    gp = sample.provenance.generator_params
    params = copy.deepcopy(gp["sampled_params"])
    assert params.get("n_anomalies", 0) > 0
    params["noise_scale"] = params["noise_scale"] * 1.7
    cf = parametric(length=gp["length"], seed=gp["inner_seed"], **params)
    assert sample.ground_truth.anomalies != cf.ground_truth.anomalies, (
        "noise_scale rescaling should move the anomalies field, which the "
        "non-target check above treats as non-target for every real knob -- "
        "if this assertion doesn't fire, the check would have silently "
        "accepted a coupled knob as clean"
    )


def test_target_component_scales_exactly():
    sample = _make("seasonal_dominant", seed=8)
    factor = 1.37
    cf = regenerate(sample, "seasonal_amplitude", factor)
    key = _target_seasonal_key(sample)
    original = np.asarray(sample.ground_truth.components[key])
    scaled = np.asarray(cf.ground_truth.components[key])
    assert np.allclose(scaled, factor * original)


def test_trend_zero_to_nonzero_refused():
    sample = _make("trend_dominant", seed=2)
    sample.provenance.generator_params["sampled_params"]["trend"]["scale"] = 0.0
    with pytest.raises(ValueError):
        regenerate(sample, "trend_scale", 1.5)


def test_trend_nonzero_to_zero_refused():
    """Mirror of the 0 -> nonzero case: landing on exactly 0.0 crosses the same
    `parametric()` branch (skips the trend RNG draw), so it is refused too.
    """
    sample = _make("trend_dominant", seed=2)
    assert sample.provenance.generator_params["sampled_params"]["trend"]["scale"] != 0.0
    with pytest.raises(ValueError):
        regenerate(sample, "trend_scale", 0.0)


def test_missing_component_refused():
    sample = _make("seasonal_dominant", seed=1)
    sample.provenance.generator_params["sampled_params"]["seasonalities"] = None
    with pytest.raises(ValueError):
        regenerate(sample, "seasonal_amplitude", 2.0)

    sample2 = _make("amplitude_modulated", seed=1)
    sample2.provenance.generator_params["sampled_params"]["heteroskedastic"] = None
    with pytest.raises(ValueError):
        regenerate(sample2, "heteroskedastic_depth", 2.0)

    sample3 = _make("intermittent_bursts", seed=1)
    sample3.provenance.generator_params["sampled_params"]["intermittency"] = None
    with pytest.raises(ValueError):
        regenerate(sample3, "intermittency_rate", 2.0)

    sample4 = _make("clean_low_noise", seed=1)
    sample4.provenance.generator_params["sampled_params"]["n_anomalies"] = 0
    with pytest.raises(ValueError):
        regenerate(sample4, "anomaly_magnitude", 2.0)


def test_non_random_parametric_refused():
    sample = parametric(length=64, seed=0, trend={"order": 1, "scale": 1.0})
    with pytest.raises(ValueError):
        regenerate(sample, "trend_scale", 2.0)

    # A plain `parametric` sample also lacks `sampled_params`/`inner_seed`, so
    # the check above could pass even with the generator-name guard removed
    # (it was: confirmed by reverting the guard and re-running -- 8/8 still
    # passed, because the "missing sampled_params" guard fired instead). This
    # variant keeps `sampled_params`/`inner_seed` present and changes only
    # `provenance.generator`, isolating the generator-name check specifically.
    disguised = _make("trend_dominant", seed=4)
    disguised.provenance.generator = "parametric"
    with pytest.raises(ValueError):
        regenerate(disguised, "trend_scale", 2.0)


def test_dose_one_reproduces_corrupted_sample_bit_identically():
    """Load-bearing: real `random_parametric` corpora built through
    `configs/large_run.yaml` (e.g. `benchmark_large/public_dev`) apply a
    post-generation corruption chain (`corruptions.py`'s jitter/scaling/
    time_warp/dropout) to most samples via `BenchmarkBuilder._make_one`,
    recorded as an ordered `provenance.transforms` list -- `parametric()`
    alone only rebuilds the PRE-corruption structural series. This was
    measured directly against `benchmark_large/public_dev` (not assumed):
    before `regenerate()` replayed `transforms`, all 244 of the 298
    `anomaly_magnitude`-eligible series with a nonempty `transforms` list
    failed `test_dose_one_is_bit_identical`'s own check by up to 5.74
    absolute, while the 54 with an empty list passed -- a 100% clean split.
    This test builds a sample through the real `BenchmarkBuilder` (not a
    hand-rolled corruption) so a corruption chain is present by construction,
    for every one of the four registered corruption ops at once, and checks
    the SAME identity property `test_dose_one_is_bit_identical` checks.
    """
    from tsfm_benchmark.build_pipeline.builder import BenchmarkBuilder, TaskSpec

    spec = TaskSpec(
        name="corrupted_anomaly_heavy",
        generator="random_parametric",
        count=1,
        generator_params={"length": 256, "archetypes": ["anomaly_heavy"]},
        corruptions=[
            {"op": "jitter", "sigma": 0.08},
            {"op": "scaling", "sigma": 0.12},
            {"op": "time_warp", "n_knots": 3, "strength": 0.15},
            {"op": "dropout", "rate": 0.04},
        ],
    )
    sample = BenchmarkBuilder()._make_one(spec, seed=321, epoch=0)
    ops = [t.op for t in sample.provenance.transforms]
    assert ops == ["jitter", "scaling", "time_warp", "dropout"], (
        "fixture must actually carry all four corruptions, in order, or this "
        f"test exercises no more than test_dose_one_is_bit_identical: got {ops}")

    cf = regenerate(sample, "anomaly_magnitude", 1.0)
    assert np.array_equal(cf.values, sample.values), (
        "dose 1.0 must reproduce a CORRUPTED corpus series bit-identically, "
        "not just an uncorrupted synthetic one")
