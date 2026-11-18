"""Independent finite-count controls for the actual production sign helper."""
import importlib.util
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parents[1] / 'experiments/feature_search/sign_null_calibration.py'
spec = importlib.util.spec_from_file_location('sign_null_calibration', path)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


def test_actual_two_sided_controls_and_complete_holm_sign_family():
    """Exact directional reversal and an omitted-Holm decoy verify this runner."""
    moderate = np.zeros((16, 4))
    moderate[:, 0] = np.r_[np.ones(13), -3 * np.ones(3)]
    family = calibration.run_family(moderate)
    np.testing.assert_allclose(family['effects']['h0']['p'], 1394 / 65536, atol=1e-15)
    assert family['effects']['h0']['p_holm'] > .05
    assert not any(family['adjusted_rejections'])
    known = calibration.controls()
    assert known['positive']['p'] == known['negative']['p'] == 2 / 256
    assert known['ties']['n_nonzero'] == 4
    assert known['all_ties']['value'] is None
