"""One bounded instrument control checks the real production inference functions."""
import importlib.util
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parents[1] / 'experiments/feature_search/statistics_calibration.py'
spec = importlib.util.spec_from_file_location('statistics_calibration', path)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


def test_actual_family_known_answers_and_omitted_holm_decoy():
    """A raw-only moderate effect must not survive the complete four-test family."""
    moderate = np.zeros((64, 4))
    moderate[:, 0] = np.r_[np.full(32, -1.), np.full(32, 1.)] + .28
    result = calibration.run_family(moderate, seed=113)
    assert result['effects']['h0']['p'] < .05
    assert result['effects']['h0']['p_holm'] > .05
    assert not result['adjusted_rejections'].any()
    controls = calibration.controls()
    assert not controls['all_zero']['raw_rejections'].any()
    assert controls['clear_nonzero']['adjusted_rejections'].all()
    assert all(effect['n_seeds'] == 64 for effect in controls['clear_nonzero']['effects'].values())
