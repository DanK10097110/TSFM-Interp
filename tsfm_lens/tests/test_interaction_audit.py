"""Factorial known answers: silent hidden decoy, spurious output and useful code."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest


def _module():
    """Load the standalone CPU audit."""
    path=Path(__file__).parents[1]/'experiments/feature_search/interaction_audit.py'
    spec=importlib.util.spec_from_file_location('interaction_audit',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def _factors():
    """Return±1 factors in the mandated00,01,10,11 ordering."""
    return np.array([-1,-1,1,1]),np.array([-1,1,-1,1])


def test_hidden_decoy_and_spurious_forecast_known_answers():
    """A large unread interaction is harmless; a read spurious interaction is causal."""
    m=_module();x,y=_factors();truth=x+y
    hidden=np.stack([x,y,10*x*y],axis=-1)[None,:,:]
    assert m.factorial_energy(hidden)['interaction_share'][0]>.9
    edited=m.corner_edit(hidden)
    assert np.array_equal(hidden[0,:,:2].sum(1),edited[0,:,:2].sum(1))
    baseline=hidden[0,:,0]+hidden[0,:,1]+.1*hidden[0,:,2]
    repaired=edited[0,:,0]+edited[0,:,1]+.1*edited[0,:,2]
    assert np.mean((baseline-truth)**2)==1
    assert np.mean((repaired-truth)**2)==0
    literal=m.corner_edit(hidden,dose=1)
    harmful_literal=literal[0,:,0]+literal[0,:,1]+.1*literal[0,:,2]
    assert np.mean((harmful_literal-truth)**2)==9


def test_correct_invertible_nonlinear_code_can_be_harmed():
    """Interaction energy is not causal error blame: correct decoding needs it."""
    m=_module();x,y=_factors();hidden=np.stack([x,y*(1+.5*x)],axis=-1)[None,:,:]
    decode=lambda h:h[...,0]+h[...,1]/(1+.5*h[...,0])
    baseline=decode(hidden);truth=(x+y)[None,:]
    assert np.array_equal(baseline,truth)
    assert m.factorial_energy(hidden)['interaction'][0]>0
    assert m.factorial_energy(baseline[...,None])['interaction'][0]==0
    edited=m.corner_edit(hidden)
    assert np.mean((decode(edited)-truth)**2)>.1


def test_energy_rotation_invariant_and_cells_checked():
    """A loud common offset and row ordering cannot invent factorial interactions."""
    m=_module();rng=np.random.default_rng(4);x,y=_factors();vectors=rng.normal(size=(3,8))
    cells=(1000+ x[:,None]*vectors[0]+y[:,None]*vectors[1]+(x*y)[:,None]*vectors[2])[None,:,:]
    q,_=np.linalg.qr(rng.normal(size=(8,8)))
    a=m.factorial_energy(cells);b=m.factorial_energy(cells@q)
    for key in a:assert np.allclose(a[key],b[key],rtol=1e-10,atol=1e-10)
    meta=[dict(seed=0,split='test',cell_phi=i//2,cell_psi=i%2) for i in range(4)]
    packed,_,_=m.pack_cells(cells[0][::-1],meta[::-1]);assert np.array_equal(packed,cells)
    with pytest.raises(ValueError,match='incomplete'):m.pack_cells(cells[0][:3],meta[:3])


def test_wrong_frequency_main_mapping_is_not_factorial_interaction(tmp_path):
    """A loud valid nuisance is a decoy; only the phi-to-wrong-frequency map errs."""
    m=_module();sys.path.insert(0,str(Path(__file__).parents[1]/'experiments/feature_search'))
    t=np.arange(128,160);s32=np.sin(2*np.pi*t/32);s16=np.sin(2*np.pi*t/16);x,y=_factors()
    truth=np.concatenate([(x[:,None]*s32+3*y[:,None]*s16) for _ in range(5)])
    prediction=truth+np.tile(.5*x[:,None]*s16,(5,1));meta=[dict(seed=seed,split='test',cell_phi=i//2,cell_psi=i%2,scale=1) for seed in range(5) for i in range(4)]
    source=tmp_path/'source';source.mkdir();(source/'metadata.json').write_text(json.dumps(meta),encoding='utf-8')
    np.savez(source/'corpus.npz',clean_future=truth)
    for name in ['bolt','timesfm']:np.savez(source/f'{name}_activations.npz',prediction=prediction)
    output=tmp_path/'out';m.main_effect_frequency_audit(source,output)
    results=json.loads((output/'results.json').read_text(encoding='utf-8'))
    for name in ['bolt','timesfm']:
        assert results[name]['interaction_fraction_of_total_mse']['value']<1e-20
        effects=results[name]['main_effects']
        assert abs(effects['main_phi']['wrong_frequency_fraction_of_total_mse']['value']-1)<1e-12
        assert abs(effects['main_phi']['wrong_frequency_fraction_of_main_effect']['value']-.2)<1e-12
        assert effects['main_psi']['wrong_frequency_fraction_of_main_effect']['value']<1e-20
