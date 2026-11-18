"""Toy-only FFN gates and source-held-out selectivity controls; no native weights."""
from pathlib import Path
from types import SimpleNamespace
import builtins

import numpy as np
import pytest
import torch
from torch import nn

from experiments.feature_search import ffn_selectivity_study as study
from experiments.feature_search import ffn_selectivity_analysis as analysis
from experiments.feature_search import ffn_selectivity_run as runner
from experiments.feature_search.coherence_study import Bank


def revin(x,mu,sigma,reverse=False):
    """Toy normalizer with exactly the TimesFM call signature."""
    return x*sigma+mu if reverse else (x-mu)/sigma


class ToyBlock(nn.Module):
    """Two distinct widths make postactivation capture distinguishable from residual."""
    def __init__(self):
        super().__init__()
        self.ff0=nn.Linear(4,8)
        self.activation=nn.SiLU()
        self.ff1=nn.Linear(8,4)

    def forward(self,x):
        return x+self.ff1(self.activation(self.ff0(x)))


class ToyModel(nn.Module):
    """Small CPU graph; never instantiates a cached model."""
    def __init__(self):
        super().__init__()
        self.stem=nn.Linear(32,4)
        self.stacked_xf=nn.ModuleList([ToyBlock(),ToyBlock()])
        self.head=nn.Linear(4,32)


class ToyNative:
    """Exactly counted deterministic toy forward passes."""
    instances=[]
    def __init__(self,name,checkpoint):
        torch.manual_seed(7)
        self.name=name
        self.model=ToyModel().eval()
        self.calls=0
        self.instances.append(self)

    def predict(self,x):
        self.calls+=len(x)
        with torch.inference_mode():
            value=torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float32)
            mu=value.mean(dim=-1,keepdim=True)
            sigma=value.std(dim=-1,keepdim=True,unbiased=False).clamp_min(1e-6)
            normalized=revin(value,mu,sigma)
            h=self.model.stem(normalized.reshape(len(value),4,32))
            for block in self.model.stacked_xf:
                h=block(h)
            prediction=revin(self.model.head(h.mean(dim=1)),mu,sigma,reverse=True)
        return prediction.numpy(),None


def toy_capability():
    """Protocol geometry for the small test-only graph."""
    return dict(models={m:dict(sites=[dict(tokens=4,ffn_width=8,residual_width=4) for _ in range(2)])
                        for m in study.MODELS},source_hashes={},runtime_validated=False)


def test_equal_rms_phase_sign_variants_and_no_future_targets():
    delta,conditions=study.perturbations(study.SEED)
    assert delta.shape==(14,128)
    np.testing.assert_allclose(np.sqrt(np.mean(delta**2,axis=1)),.1,rtol=0,atol=1e-15)
    assert [sum(c['kind']==kind for c in conditions) for kind in study.TYPES]==[4,4,2,2,2]
    for i in range(0,14,2):
        np.testing.assert_array_equal(delta[i],-delta[i+1])
    assert np.count_nonzero(delta[8,:96])==0 and np.count_nonzero(delta[10,:64])==0


def test_prepare_freezes480_cells_no_sae_disjoint_groups_and_causal_hold(tmp_path,monkeypatch):
    natural=tmp_path/'natural_validation_replication4955000'
    natural.mkdir()
    study.write(natural/'protocol.json',dict(nodes=1016,checkpoints={m:dict(path='',weights={}) for m in study.MODELS}))
    study.write(natural/'execution.json',{m:dict(seconds=56.) for m in study.MODELS})
    real=[(dict(source=f'reused:{i}',kind='real_reused',domain=f'd{i//2}',seed=study.SEED+100+i,
                split='FIT' if i%2==0 else 'TEST'),study.synthetic_background(study.SEED+100+i,0)) for i in range(8)]
    monkeypatch.setattr(study,'reused_real',lambda path:(real,dict(status='reused_past_only',new_real_generalization=False)))
    monkeypatch.setattr(study,'architecture',lambda checkpoints:toy_capability())
    out=tmp_path/'ffn_selectivity_pilot4961000'
    study.prepare(out)
    p=study.read(out/'protocol.json')
    rows=study.read(out/'rows.json')
    assert p['cells_per_model']==480 and p['variants']==14 and p['hold_native'] and p['no_SAE']
    assert sum(r['split']=='FIT' for r in rows)==sum(r['split']=='TEST' for r in rows)==12
    assert len(set(r['source'] for r in rows))==24
    assert p['synthetic_seeds']==list(range(study.SEED,study.SEED+16))
    causal=study.read(out/'causal_plan.json')
    assert causal['budgets']==[1,4] and causal['native_cells_stage1']==0 and not causal['automatic_launch']
    assert not (out/'execution.json').exists() and study.read(out/'knowledge.json')['exact_findings']['native_launched'] is False


def test_reused_real_reads_only_past_roots_not_target_files(tmp_path,monkeypatch):
    bank=Bank()
    domains=('monash_electricity_hourly','m4_hourly','monash_pedestrian_counts','monash_traffic')
    rows,plans=[],[]
    for d in domains:
        for split in ('FIT','TEST'):
            i=len(rows)
            root=bank.add(np.sin(np.arange(128)/5+i))
            rows.append(dict(source=f'{d}:{i}',domain=d,split=split))
            plans.append([{}, {},dict(origin=256,levels=[[root]])])
    bank.save(tmp_path/'bank.npz')
    study.write(tmp_path/'rows.json',rows)
    study.write(tmp_path/'plans.json',plans)
    study.write(tmp_path/'protocol.json',dict(hashes={name:study.sha(tmp_path/name) for name in ('bank.npz','rows.json','plans.json')}))
    (tmp_path/'targets.npz').write_bytes(b'must never be opened')
    original=builtins.open
    def guard(path,mode='r',*args,**kwargs):
        if isinstance(path,(str,Path)) and 'targets' in str(path) and 'r' in mode:
            raise AssertionError('Future read')
        return original(path,mode,*args,**kwargs)
    monkeypatch.setattr(builtins,'open',guard)
    data,record=study.reused_real(tmp_path)
    assert len(data)==8 and not record['new_real_generalization']
    assert sum(m['split']=='FIT' for m,_ in data)==4
    assert all(x.shape==(128,) for _,x in data)


def test_tokenwise_rotation_preserves_geometry_before_separate_pooling():
    rng=np.random.default_rng(2)
    x=rng.normal(size=(6,14,3,8))
    rotated=analysis.rotate_tokens(x,4961101)
    np.testing.assert_allclose(np.sum(rotated**2,axis=-1),np.sum(x**2,axis=-1),rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(rotated.mean(axis=2),analysis.rotate_tokens(x.mean(axis=2),4961101),atol=1e-13)
    np.testing.assert_allclose(np.sum(np.mean(rotated**2,axis=2),axis=-1),np.sum(np.mean(x**2,axis=2),axis=-1),atol=1e-13)
    _,conditions=study.perturbations(1)
    assert not np.allclose(analysis.pool(rotated,conditions,'rms'),analysis.rotate_tokens(analysis.pool(x,conditions,'rms'),4961101))
    decoy=np.tile(np.array([1.,-1.])[None,None,:,None],(6,14,1,8))
    np.testing.assert_array_equal(analysis.pool(decoy,conditions,'mean'),np.zeros((6,5,8)))
    np.testing.assert_array_equal(analysis.pool(decoy,conditions,'rms'),np.ones((6,5,8)))


def test_test_source_poison_cannot_change_pca_or_fit_feature_selection():
    rng=np.random.default_rng(3)
    x=rng.normal(size=(6,14,2,8))
    fit=np.array([True]*3+[False]*3)
    poisoned=x.copy()
    poisoned[~fit]=rng.normal(size=poisoned[~fit].shape)*1e8
    axes=analysis.pca_basis(x,fit)
    other=analysis.pca_basis(poisoned,fit)
    np.testing.assert_array_equal(axes,other)
    np.testing.assert_allclose(axes@axes.T,np.eye(len(axes)),atol=1e-12)
    features=np.tile(np.eye(5,8)[None],(6,1,1))
    changed=features.copy()
    changed[~fit]=0
    first=analysis.fit_readout(features,fit,4)
    second=analysis.fit_readout(changed,fit,4)
    assert first==second
    np.testing.assert_array_equal(analysis.evaluate(features,first)['accuracy'][~fit],np.ones(3))
    assert np.all(analysis.evaluate(changed,second)['accuracy'][~fit]<1)


def test_pca_type_weights_do_not_reward_duplicate_phase_exposures():
    rng=np.random.default_rng(15)
    x=rng.normal(size=(4,10,2,5))
    fit=np.array([True,True,False,False])
    conditions=[dict(kind=kind) for kind in study.TYPES for _ in range(2)]
    axes=analysis.pca_basis(x,fit,conditions)
    duplicated=np.concatenate([x,x[:,:2]],axis=1)
    other=analysis.pca_basis(duplicated,fit,conditions+conditions[:2])
    np.testing.assert_allclose(abs(axes@other.T),np.eye(5),atol=1e-12)


def test_readout_accuracy_is_not_the_same_as_unit_type_purity():
    fit=np.array([True]*3+[False]*3)
    selective=np.tile(np.eye(5,8)[None],(6,1,1))
    state=analysis.fit_readout(selective,fit,1)
    result=analysis.evaluate(selective,state)
    np.testing.assert_allclose(result['accuracy'],.4)
    np.testing.assert_array_equal(result['purity'],np.ones(6))
    amplitude_code=np.tile(np.arange(1,6.)[None,:,None],(6,1,1))
    encoded=analysis.evaluate(amplitude_code,analysis.fit_readout(amplitude_code,fit,1))
    np.testing.assert_array_equal(encoded['accuracy'],np.ones(6))
    np.testing.assert_allclose(encoded['purity'],1/3)


def test_background_bootstrap_uses_only_heldout_source_groups():
    rows=[dict(source=f's{i}',kind='synthetic' if i<3 else 'real_reused') for i in range(4)]
    result=analysis.background_summary(np.array([1e9,1.,-1e9,3.]),rows,[1,3],n_boot=99)
    assert result['n_sources']==2 and result['source_ids']==['s1','s3']
    assert result['mean']==2 and result['ci95']==[2.,2.]


def test_postactivation_capture_exact_replay_reach_and_normalizer_restoration():
    native=ToyNative('timesfm','')
    selected=study.sites(native)
    x=study.synthetic_background(7,0).astype(np.float32)[None]
    original_revin=revin
    def invoke(value,replacement=None):
        return study.capture(native,value,selected,replacement)
    with study.locked_background(native,x,invoke) as (reference,raw):
        assert native.calls==1
        baseline,states=invoke(x)
        study.exact_gate(reference,baseline,'normalizer')
        assert states['first_ffn'].shape==(1,4,8) and states['first_residual'].shape==(1,4,4)
        for key in states:
            study.exact_gate(raw[key],states[key],key)
        for tag in selected:
            replay,_=invoke(x,(tag,states[f'{tag}_ffn']))
            study.exact_gate(baseline,replay,tag)
            changed,recorded=invoke(x,(tag,np.zeros_like(states[f'{tag}_ffn'])))
            assert study.reach_gate(baseline,changed)>1e-6
            np.testing.assert_array_equal(recorded[f'{tag}_ffn'],np.zeros((1,4,8)))
    assert revin is original_revin
    assert all(not block.ff1._forward_pre_hooks for block in native.model.stacked_xf)
    with pytest.raises(ValueError,match='exact replay'):
        study.exact_gate(baseline,baseline+1e-3,'planted mismatch')
    with pytest.raises(ValueError,match='reach'):
        study.reach_gate(baseline,baseline)


def test_worker_counts_all_native_cells_and_keeps_causal_stage_out(tmp_path,monkeypatch):
    rows=[]
    queries=[]
    for i in range(4):
        x=study.synthetic_background(20+i,0)
        delta,conditions=study.perturbations(20+i)
        queries.append(np.concatenate([x[None],x[None]+delta]).astype(np.float32))
        rows.append(dict(source=f'toy{i}',kind='synthetic',split='FIT' if i<2 else 'TEST'))
    np.savez(tmp_path/'inputs.npz',queries=np.asarray(queries))
    study.write(tmp_path/'rows.json',rows)
    study.write(tmp_path/'conditions.json',conditions)
    p=dict(sources={},capture=toy_capability()['models'],cells_per_model=80,backgrounds=4,
        checkpoints={'timesfm':dict(path='',weights={})},
        hashes={name:study.sha(tmp_path/name) for name in ('inputs.npz','rows.json','conditions.json')})
    study.write(tmp_path/'protocol.json',p)
    monkeypatch.setenv('FFN_SELECTIVITY_MONITORED','1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    for key in study.THREAD_VARS:
        monkeypatch.setenv(key,'1')
    study.worker(tmp_path,'timesfm',native_factory=ToyNative)
    record=study.read(tmp_path/'timesfm_worker.json')
    assert record['cells']==ToyNative.instances[-1].calls==80
    assert record['all_reach_passed'] and record['normalizer_identity_exact'] and record['self_replay_exact']
    with np.load(tmp_path/'timesfm_captures.npz') as data:
        assert data['first_ffn'].shape==(4,15,4,8)
        assert data['forecasts'].shape==(4,15,32)
    with pytest.raises(FileExistsError):
        study.worker(tmp_path,'timesfm',native_factory=ToyNative)


def test_native_hold_and_cellcap_are_enforced_without_model_loads(tmp_path,monkeypatch):
    def forbid_launch(*args,**kwargs):
        raise AssertionError('No subprocess permitted in hold/cap test')
    monkeypatch.setattr(runner.subprocess,'Popen',forbid_launch)
    with pytest.raises(RuntimeError,match='HOLD FFN'):
        runner.run(tmp_path/'absent')
    monkeypatch.delenv('FFN_SELECTIVITY_MONITORED',raising=False)
    with pytest.raises(RuntimeError,match='HOLD native'):
        study.worker(tmp_path,'timesfm',native_factory=ToyNative)
    study.write(tmp_path/'protocol.json',dict(cells_per_model=501))
    with pytest.raises(ValueError,match='cell cap'):
        runner.run(tmp_path,slot_granted=True)
    assert not (tmp_path/'execution.json').exists()
