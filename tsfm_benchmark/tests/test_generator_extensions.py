from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tsfm_benchmark.build_pipeline.generators import (_ARCHETYPES,
                                                      _DEFAULT_ARCHETYPES,
                                                      parametric,
                                                      random_parametric)

NEW_ARCHETYPES = ["random_walk_drift", "intermittent_bursts",
                  "amplitude_modulated", "nonsinusoidal_seasonal"]

GOLDEN = {
    0: ("2856658d044e4c49", "a45664e176fbb71a", "clean_low_noise"),
    7: ("ded6ff4ee4d08e00", "a1d7456a6daeb78e", "noisy_chaotic"),
    123: ("13157fdc8501e113", "8b8395cd08aced50", "trend_dominant"),
}


def _short_hash(values: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def test_golden_hashes_pre_extension_outputs_unchanged():
    """Bit-exact regression: outputs captured before the extensions landed.

    The extension parameters must consume random draws only when set, and the
    default archetype pool must stay frozen, or every previously sealed
    corpus stops regenerating from its seed. These hashes were recorded on
    the pre-extension code; any mismatch means that invariant broke.
    """
    for seed, (p_hash, r_hash, archetype) in GOLDEN.items():
        p = parametric(seed=seed, length=256, trend={"order": 2, "scale": 0.5},
                       seasonalities=[{"period": 24, "amplitude": 1.0}],
                       ar_coeffs=[0.6, -0.2], noise_scale=0.15,
                       n_changepoints=2, n_anomalies=3)
        r = random_parametric(seed=seed, length=256)
        assert _short_hash(p.values) == p_hash
        assert _short_hash(r.values) == r_hash
        assert r.ground_truth.generative_params["archetype"] == archetype


def test_default_pool_excludes_new_archetypes_but_registry_has_them():
    assert set(NEW_ARCHETYPES).isdisjoint(_DEFAULT_ARCHETYPES)
    assert set(NEW_ARCHETYPES) <= set(_ARCHETYPES)
    assert len(_DEFAULT_ARCHETYPES) == 8


def test_pre_extension_generative_params_carry_no_new_keys():
    p = parametric(seed=5, length=128, seasonalities=[{"period": 16, "amplitude": 1.0}])
    assert set(p.ground_truth.generative_params) == {
        "trend", "seasonalities", "ar_coeffs", "noise_scale"}


def test_seasonal_shapes_render_and_reject_unknown():
    for shape, distinct in (("square", 2), ("sawtooth", None), ("triangle", None)):
        p = parametric(seed=1, length=240,
                       seasonalities=[{"period": 24, "amplitude": 1.0, "shape": shape}],
                       noise_scale=0.0)
        comp = np.array(p.ground_truth.components["seasonal_24"])
        assert np.abs(comp).max() <= 1.0 + 1e-9
        if distinct:
            assert len(set(np.round(comp, 6))) == distinct
    try:
        parametric(seed=1, length=64,
                   seasonalities=[{"period": 8, "shape": "wavelet"}])
        raise AssertionError("unknown shape should raise")
    except ValueError:
        pass


def test_random_walk_component_variance_grows():
    p = parametric(seed=2, length=512, random_walk_scale=2.0, noise_scale=0.0)
    walk = np.array(p.ground_truth.components["random_walk"])
    assert np.abs(np.cumsum(np.diff(walk)) - (walk[1:] - walk[0])).max() < 1e-9
    assert walk[-128:].var() > walk[:128].var()
    assert p.ground_truth.generative_params["random_walk_scale"] == 2.0


def test_intermittency_zero_fraction_tracks_rate():
    p = parametric(seed=4, length=2000, intermittency={"rate": 0.6},
                   seasonalities=[{"period": 24, "amplitude": 1.0}])
    zero_frac = float((p.values == 0).mean())
    assert 0.5 < zero_frac < 0.7
    assert "intermittency_mask" in p.ground_truth.components


def test_heteroskedastic_envelope_recorded_and_bounded():
    p = parametric(seed=6, length=512,
                   heteroskedastic={"period": 128, "depth": 0.9}, noise_scale=0.3)
    env = np.array(p.ground_truth.components["noise_envelope"])
    assert 0.05 < env.min() < 0.2 and 1.8 < env.max() < 1.95


def test_new_archetypes_generate_reproducibly_and_in_regime():
    for name in NEW_ARCHETYPES:
        a = random_parametric(seed=11, length=256, archetypes=[name])
        b = random_parametric(seed=11, length=256, archetypes=[name])
        assert (a.values == b.values).all()
        assert a.ground_truth.generative_params["archetype"] == name
    zero_fracs = [(random_parametric(seed=s, length=256,
                                     archetypes=["intermittent_bursts"]).values == 0).mean()
                  for s in range(8)]
    assert min(zero_fracs) > 0.2
    shapes = set()
    for s in range(10):
        r = random_parametric(seed=s, length=256, archetypes=["nonsinusoidal_seasonal"])
        shapes |= {sp["shape"] for sp in r.ground_truth.generative_params["seasonalities"]}
    assert shapes <= {"square", "sawtooth", "triangle"} and len(shapes) >= 2


def test_mixed_pool_draws_every_archetype():
    names = _DEFAULT_ARCHETYPES + NEW_ARCHETYPES
    seen = {random_parametric(seed=s, length=128, archetypes=names)
            .ground_truth.generative_params["archetype"] for s in range(120)}
    assert seen == set(names)
