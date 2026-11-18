"""Known-answer task-tangent instrument with wrong-readout and no-op cutpoint decoys."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch
import pytest


def _module():
    """Load the standalone prospective tangent helpers."""
    root=Path(__file__).parents[1]/'experiments/feature_search';sys.path.insert(0,str(root));s=importlib.util.spec_from_file_location('local_tangent',root/'local_tangent.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_analytic_phase_derivative_and_fixed_normalizers():
    """The other frequency is a true zero derivative despite comparable amplitude."""
    m=_module();context,future,meta=m.make_corpus();base=meta[::9];scales=np.array([r['scale'] for r in base])
    assert np.ptp(context.reshape(64,9,128).mean(-1),axis=1).max()<1e-6
    assert np.ptp(context.reshape(64,9,128).std(-1),axis=1).max()<1e-6
    for target,other in [('phi',slice(2,4)),('psi',slice(0,2))]:
        ideal=m.analytic_derivative(base,target);d01=m.coefficient_derivative(m.centered_derivative(future,target,.01),scales);d02=m.coefficient_derivative(m.centered_derivative(future,target,.02),scales)
        assert m.relative_rms(d01,ideal)<.0001
        assert m.relative_rms(d02,d01)<.0001
        assert abs(d01[:,other]).max()<1e-10


def test_internal_tangent_correct_wrong_and_noop_cutpoints():
    """A wrong readout mixes frequencies;a called-but-unused hook fails route validation."""
    m=_module();rng=np.random.default_rng(32);base=torch.tensor(rng.normal(size=(8,1,4)),dtype=torch.float64);tangent=torch.tensor(np.column_stack([rng.normal(size=(8,2)),np.zeros((8,2))])[:,None,:],dtype=torch.float64);delta=.01
    class Model(torch.nn.Module):
        def __init__(self,wrong=False,noop=False):
            super().__init__();self.cutpoint=torch.nn.Identity();self.noop=noop;self.register_buffer('matrix',torch.eye(4,dtype=torch.float64));
            if wrong:self.matrix[0,2]=.7;self.matrix[1,3]=.7
        def forward(self,h):
            code=self.cutpoint(h);return (h if self.noop else code)@self.matrix
    for wrong in [False,True]:
        model=Model(wrong=wrong);actual=(model(base+delta*tangent)-model(base-delta*tangent))/(2*delta)
        with m.apply_full_head_patch(model,'cutpoint',base+delta*tangent):plus=model(base)
        with m.apply_full_head_patch(model,'cutpoint',base-delta*tangent):minus=model(base)
        internal=(plus-minus)/(2*delta)
        assert m.relative_rms(internal.numpy(),actual.numpy())<1e-10
        offenergy=float(np.sum(actual.numpy()[...,2:]**2))
        assert (offenergy>.1) if wrong else (offenergy==0)
    noop=Model(noop=True);actual=(noop(base+delta*tangent)-noop(base-delta*tangent))/(2*delta)
    with m.apply_full_head_patch(noop,'cutpoint',base+delta*tangent):plus=noop(base)
    with m.apply_full_head_patch(noop,'cutpoint',base-delta*tangent):minus=noop(base)
    assert m.relative_rms(((plus-minus)/(2*delta)).numpy(),actual.numpy())==1


def test_aggregate_energy_ratio_not_perseed_ratio():
    """A weak-strength seed is a decoy for unstable averaging of per-seed ratios."""
    m=_module();native=np.ones(4);probe=np.zeros(4);strength=np.array([.001,1,10,100])
    effect=m.energy_ratio_effect(native,probe,strength)
    assert abs(effect['value']-4/strength.sum())<1e-12
    assert effect['value']<.04


def test_replication_targets_and_complete_family_are_frozen():
    """The failed phi field is a scope decoy;crossmodel p belongs in the same family."""
    m=_module();context,future,meta=m.make_corpus(1659000,128,('psi',))
    assert context.shape==(640,128) and future.shape==(640,32)
    assert m.condition_map()==m.CONDITIONS
    phases=np.array([r['phi'] for r in meta]).reshape(128,5)
    assert np.ptp(phases,axis=1).max()==0
    assert all(c[1] in [None,'psi'] for c in m.condition_map(('psi',)))
    primary={'bolt_psi':dict(p=.01),'timesfm_psi':dict(p=.02)};interactions={'psi':dict(p=.04)}
    family=m.complete_family(primary,interactions,True)
    assert set(family)=={'bolt_psi','timesfm_psi','timesfm_minus_bolt_psi'}
    assert primary['bolt_psi']['p_holm']==.03
    assert primary['timesfm_psi']['p_holm']==.04
    assert interactions['psi']['p_holm']==.04
    with pytest.raises(ValueError,match='complete'):m.complete_family({'bolt_psi':dict(p=.01)},interactions,True)
