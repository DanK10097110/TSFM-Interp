"""Known-answer direct-probe checks with loud variance and nuisance decoys."""
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _module():
    """Load the standalone experiment without changing package import paths."""
    path = Path(__file__).parents[1] / 'experiments/feature_search/activation_probe.py'
    spec = importlib.util.spec_from_file_location('activation_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_finds_signal_amid_high_variance_decoy():
    """The highest-variance coordinate is unrelated to the planted label."""
    rng = np.random.default_rng(17)
    x = rng.normal(size=(900, 12))
    y = x[:, 5] + rng.normal(0, .03, len(x))
    x[:, 0] *= 1000
    parts = [slice(0, 400), slice(400, 650), slice(650, 900)]
    score = _module().compare_probes(*(x[p] for p in parts), *(y[p] for p in parts))
    assert score['neuron'] > .99
    assert score['supervised_ridge'] > .99
    assert score['rotated_neuron'] < .9


def test_generating_seed_reproducibility_and_independence():
    """A changed seed must change observations while the same seed is exact."""
    module = _module()
    x, y = module.make_series(64, 128, 3)
    repeat, labels = module.make_series(64, 128, 3)
    decoy, _ = module.make_series(64, 128, 4)
    assert np.array_equal(x, repeat) and np.array_equal(y, labels)
    assert not np.array_equal(x, decoy)
    assert module.input_features(x).shape[0] == 64


def test_probe_fit_never_reads_test_labels():
    """A test-only reverse-sign decoy cannot change fit or validation selection."""
    path = Path(__file__).parents[1] / 'experiments/feature_search/activation_analysis.py'
    sys.modules['activation_probe'] = _module()
    spec = importlib.util.spec_from_file_location('activation_analysis', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rng = np.random.default_rng(13)
    x = rng.normal(size=(180, 5)); y = 2*x[:, 2] + rng.normal(0, .01, 180)
    train=np.arange(180)<80;valid=(np.arange(180)>=80)&(np.arange(180)<130)
    a, loss, alpha=module.fit_predict(x,y,train,valid,'ridge')
    changed=y.copy();changed[130:]=-100*y[130:]
    b, loss_b, alpha_b=module.fit_predict(x,changed,train,valid,'ridge')
    assert np.array_equal(a,b)
    assert loss==loss_b and alpha==alpha_b
    assert np.mean((a[130:]-y[130:])**2)<.001
