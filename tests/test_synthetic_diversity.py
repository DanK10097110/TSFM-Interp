from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tsfm_benchmark.build_pipeline.builder import BenchmarkBuilder, TaskSpec
from tsfm_benchmark.build_pipeline.generators import _ARCHETYPES, _stable_ar_coeffs, random_parametric


def test_random_parametric_is_reproducible_given_same_seed():
    a = random_parametric(seed=7, length=64)
    b = random_parametric(seed=7, length=64)
    assert (a.values == b.values).all()
    assert a.ground_truth.generative_params["archetype"] == b.ground_truth.generative_params["archetype"]


def test_random_parametric_varies_structure_across_seeds():
    samples = [random_parametric(seed=s, length=64) for s in range(40)]
    archetypes = {s.ground_truth.generative_params["archetype"] for s in samples}
    # with 40 draws over 8 archetypes we expect to see more than just one or two
    assert len(archetypes) >= 4
    assert archetypes <= set(_ARCHETYPES)


def test_random_parametric_rejects_unknown_archetype():
    import pytest

    with pytest.raises(ValueError):
        random_parametric(seed=0, archetypes=["not_a_real_archetype"])


def test_corruption_pool_varies_chain_per_sample():
    spec = TaskSpec(
        name="diverse",
        generator="random_parametric",
        count=20,
        generator_params={"length": 64},
        corruption_pool=[
            {"op": "jitter", "sigma": [0.01, 0.08]},
            {"op": "scaling", "sigma": [0.05, 0.2]},
            {"op": "time_warp", "strength": [0.05, 0.25], "n_knots": [3, 6]},
            {"op": "dropout", "rate": [0.02, 0.1]},
        ],
        n_corruptions=(1, 3),
    )
    builder = BenchmarkBuilder()
    chains = []
    for k in range(20):
        sample = builder._make_one(spec, seed=1000 + k, epoch=0)
        chains.append(tuple(t.op for t in sample.provenance.transforms))
        assert 1 <= len(sample.provenance.transforms) <= 3

    assert len(set(chains)) > 1


def test_stable_ar_coeffs_never_produce_an_exploding_process():
    # A regression test for a real bug: independently sampling each AR
    # coefficient within a per-coefficient bound is not sufficient for
    # order >= 2 stationarity. An AR(3) draw with |coeff| <= 0.85 each
    # (well within the per-coefficient bound) previously produced values up
    # to ~1e96 on a length-512 series. _stable_ar_coeffs must never do that.
    rng = np.random.default_rng(0)
    worst = 0.0
    for order in (1, 2, 3, 4):
        for _ in range(500):
            coeffs = _stable_ar_coeffs(rng, order, max_reflection=0.85)
            n = 300
            noise = rng.normal(0, 1, size=n)
            colored = np.zeros(n)
            for i in range(n):
                past = sum(coeffs[k] * colored[i - k - 1] for k in range(order) if i - k - 1 >= 0)
                colored[i] = past + noise[i]
            worst = max(worst, float(np.abs(colored).max()))

    assert worst < 1000, f"AR recursion exploded: max|value|={worst}"


def test_random_parametric_never_explodes_across_many_seeds():
    worst = 0.0
    for seed in range(500):
        s = random_parametric(seed=seed, length=512)
        worst = max(worst, float(np.abs(s.values).max()))
    assert worst < 1000, f"random_parametric produced an exploding series: max|value|={worst}"


def test_corruption_pool_is_reproducible_given_same_seed():
    spec = TaskSpec(
        name="diverse",
        generator="random_parametric",
        count=1,
        generator_params={"length": 64},
        corruption_pool=[
            {"op": "jitter", "sigma": [0.01, 0.08]},
            {"op": "dropout", "rate": [0.02, 0.1]},
        ],
        n_corruptions=(1, 2),
    )
    builder = BenchmarkBuilder()
    a = builder._make_one(spec, seed=42, epoch=0)
    b = builder._make_one(spec, seed=42, epoch=0)
    assert (a.values == b.values).all()
    assert [t.op for t in a.provenance.transforms] == [t.op for t in b.provenance.transforms]
