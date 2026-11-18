"""Known-answer phase regression and confusable period/noise/trend controls."""
import importlib.util
from pathlib import Path

import numpy as np

_path=Path(__file__).parents[1]/'experiments/feature_search/phase_sweep.py'
_spec=importlib.util.spec_from_file_location('phase_sweep',_path)
phase_sweep=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phase_sweep)


def test_harmonic_recovers_phase_amplitude_offset_slope():
    """A strong trend and nonzero offset must not masquerade as phase lag."""
    t=np.arange(128,160)
    y=1.7*np.sin(2*np.pi*t/48+.8)+3+2*(t-64)/128
    fit=phase_sweep.harmonic_fit(y,48)
    np.testing.assert_allclose([fit['amplitude'],fit['phase'],fit['offset'],fit['slope']],
                               [1.7,.8,3,2],atol=1e-10)
    assert phase_sweep.harmonic_fit(np.ones(32),48)['phase'] is None


def test_matched_regimes_and_fresh_seed_bundles():
    """Noise and trend decoys share draws without changing the sine truth."""
    x,y,meta=phase_sweep.corpus(2)
    assert len(x)==64 and len(set(m['seed'] for m in meta))==2
    assert min(m['seed'] for m in meta)==882000
    np.testing.assert_equal(y[0],y[1])
    np.testing.assert_allclose(x[1]-x[0],x[3]-x[2],atol=5e-7)
    assert np.std(x[2]-x[0])>0
    assert set(m['period'] for m in meta)==set(phase_sweep.PERIODS)


def test_primary_contrasts_known_answer_with_neighbor_and_trend_decoys():
    """Center excess and trend effects remain separate despite shared gaps."""
    gaps=np.zeros((5,8,4));gaps[:,3,:]=2;gaps[:,2,:]=1;gaps[:,4,:]=1
    gaps[:,:,[2,3]]+=3
    got=phase_sweep.primary_contrasts(gaps)
    np.testing.assert_allclose(got['gap32'],3.5)
    np.testing.assert_allclose(got['excess32_over30_34'],1)
    np.testing.assert_allclose(got['model_by_trend'],3)
