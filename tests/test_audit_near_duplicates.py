from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tsfm_benchmark.build_pipeline.audit import find_near_duplicates
from tsfm_benchmark.build_pipeline.schema import GroundTruth, Provenance, TimeSeriesSample


def _sample(sample_id: str, values) -> TimeSeriesSample:
    s = TimeSeriesSample(values=np.asarray(values, dtype=float), ground_truth=GroundTruth(), provenance=Provenance(generator="test"))
    s.sample_id = sample_id
    return s


def test_finds_injected_near_duplicate_pair():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, size=200) + np.sin(np.linspace(0, 10, 200))
    samples = [_sample(f"s{i}", rng.normal(0, 1, size=200) + np.sin(np.linspace(0, 10, 200) + rng.uniform(0, 6))) for i in range(50)]
    samples.append(_sample("dup_of_s0", base + rng.normal(0, 1e-4, size=200)))
    samples[0] = _sample("s0", base)

    dups = find_near_duplicates(samples)
    pairs = {frozenset((a, b)) for a, b, _ in dups}
    assert frozenset(("s0", "dup_of_s0")) in pairs


def test_no_duplicates_returns_empty():
    rng = np.random.default_rng(1)
    samples = [_sample(f"s{i}", rng.normal(0, 1, size=200) + np.sin(np.linspace(0, 10, 200) + i)) for i in range(20)]
    assert find_near_duplicates(samples) == []


def test_single_and_empty_input_short_circuit():
    assert find_near_duplicates([]) == []
    assert find_near_duplicates([_sample("only", [1.0, 2.0, 3.0])]) == []
