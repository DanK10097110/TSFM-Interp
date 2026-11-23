from contextlib import ExitStack

import numpy as np
import pytest
import torch

from research.activation_transfer.forecast_requests.horizon.run import Monitor
from research.activation_transfer.forecast_requests.horizon.summarize import scores
from tsfm_lens.steering.model import InterventionSite


def test_sustained_and_localized_targets_penalize_late_energy_differently():
    x=.05*np.array([[1.]*32,[1.]*8+[0.]*24,[1.]*8+[-1.]*24])
    s=scores(x)
    np.testing.assert_allclose(s["early_skill"],[1,1,1])
    np.testing.assert_allclose(s["late_gain"],[1,0,-1])
    np.testing.assert_allclose(s["late_energy"],[1,0,1])
    np.testing.assert_allclose(s["sustained32"],[1,.25,-2])
    np.testing.assert_allclose(s["localized32"],[-2,1,-2])


def test_monitor_rejects_a_second_long_horizon_site_invocation():
    class Toy:
        def __init__(self):
            self.module=torch.nn.Sequential(torch.nn.Identity())

        def forecast(self,x,h):
            state=torch.as_tensor(x[:,None,:])
            self.module(state)
            if h>8:
                self.module(state)
            return np.zeros((len(x),h,1))
    model=Toy()
    site=InterventionSite("0",["args",0],0,[1,4],"last")
    monitor=Monitor(model,[site])
    with ExitStack() as stack:
        monitor.install(stack)
        x=np.zeros((2,4),dtype="f4")
        monitor.call(x,8)
        with pytest.raises(RuntimeError,match="2 live invocations"):
            monitor.call(x,32)
