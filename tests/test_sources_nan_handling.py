from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tsfm_benchmark.build_pipeline.sources import _to_series


def test_to_series_rejects_nan():
    assert _to_series([1.0, 2.0, float("nan"), 4.0]) is None


def test_to_series_rejects_inf():
    assert _to_series([1.0, 2.0, float("inf"), 4.0]) is None
    assert _to_series([1.0, 2.0, float("-inf"), 4.0]) is None


def test_to_series_accepts_clean_numeric_list():
    result = _to_series([1.0, 2, 3.5])
    assert result is not None
    assert np.isfinite(result).all()
    assert list(result) == [1.0, 2.0, 3.5]


def test_to_series_rejects_non_numeric():
    assert _to_series(["a", "b"]) is None
    assert _to_series("not a list") is None
    assert _to_series([]) is None
