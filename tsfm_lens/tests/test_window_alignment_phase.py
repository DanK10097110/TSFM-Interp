"""Seed duplication must not inflate the phase-window similarity evidence."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

_directory=Path(__file__).parents[1]/'experiments/feature_search'
sys.path.insert(0,str(_directory))
_spec=importlib.util.spec_from_file_location('window_alignment_phase',_directory/'window_alignment_phase.py')
phase=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(phase)
sys.path.pop(0)


def test_seed_duplication_does_not_change_kernel_null():
    """Eight copies of every series provide exactly the same null evidence."""
    rng=np.random.default_rng(4)
    x=rng.normal(size=(8,3));y=x+.3*rng.normal(size=(8,3))
    independent=phase.kernel_null(x,y,8,1,draws=99)
    duplicated=phase.kernel_null(np.repeat(x,8,axis=0),np.repeat(y,8,axis=0),8,8,draws=99)
    np.testing.assert_allclose([independent[k] for k in ('value','null_p95','null_p_plusone')],
                               [duplicated[k] for k in ('value','null_p95','null_p_plusone')],atol=1e-14)
    with pytest.raises(ValueError,match='zero-variance'):
        phase.kernel_null(np.ones((8,3)),y,8,1,draws=2)
