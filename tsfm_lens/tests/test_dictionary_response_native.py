"""Prelaunch tests/plants for native fidelity, restoration and source accounting."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.feature_search import dictionary_response_native as run
from experiments.feature_search.dictionary_native import exact_gate
from experiments.feature_search.coherence_study import normalization


def test_budget_and_fixed_balanced_receivers():
    rows=[dict(domain=d,source=f'{d}{i}',split='test') for d in 'abcde' for i in range(4)]
    p=run.design(rows)
    assert len(p['base'])==20 and len(p['paired'])==15 and p['cells']==399
    assert all(sum(rows[i]['domain']==d for i in p['paired'])==3 for d in 'abcde')
    assert 6*20+18*20+9>400
    with pytest.raises(ValueError):run.design(rows[:-1])


def test_actual_response_not_global_offset_and_small_floor():
    pair=np.array([[2.,4.],[1.,2.]])
    identical=run.response_score(pair,pair+100,2)
    assert identical['nmse']==0
    erased=run.response_score(pair,np.zeros_like(pair),2)
    assert erased['nmse']==1
    small=run.response_score(np.zeros_like(pair),pair,2)
    assert small['nmse'] is None and small['small_response'] and small['denominator_floor']==4e-12
    assert np.isfinite(small['floored_nmse'])
    assert run.distortion(pair,pair+100,2)['history_variance_nmse']==2500


def fake_native():
    class Norm(torch.nn.Module):
        def forward(self,x,loc_scale=None):
            stats=(x.mean(),x.std()) if loc_scale is None else loc_scale
            return x,stats
    site=torch.nn.Identity()
    norm=Norm()
    class Pipeline:
        def predict_quantiles(self,x,**kwargs):
            x,_=norm(x)
            y=site(x)
            return None,y
    return SimpleNamespace(name='bolt',model=SimpleNamespace(instance_norm=norm),pipeline=Pipeline()),site


def test_independent_pipeline_replacement_and_cleanup_exception():
    native,site=fake_native();query=np.arange(6,dtype=np.float32).reshape(1,6)
    base=run.raw_predict(native,query)
    np.testing.assert_array_equal(base,query)
    np.testing.assert_array_equal(run.raw_predict(native,query,site,query+1),query+1)
    assert not site._forward_hooks
    with pytest.raises(ValueError,match='shape'):
        run.raw_predict(native,query,site,np.zeros((2,6)))
    assert not site._forward_hooks
    np.testing.assert_array_equal(run.raw_predict(native,query),base)
    with pytest.raises(ValueError,match='gate failed'):
        exact_gate(base,base+1,'planted wrong identity')


def test_normalizer_restores_on_exception():
    native,_=fake_native();original=native.model.instance_norm.forward
    with pytest.raises(RuntimeError):
        with normalization(native,np.arange(6,dtype=np.float32).reshape(1,6),'background'):
            assert native.model.instance_norm.forward!=original
            raise RuntimeError('planted worker error')
    assert native.model.instance_norm.forward==original


def test_source_mean_not_energy_weighted_and_invalid_visible():
    rows=[]
    for source,energy,error in [('a',1.,1.),('b',100.,400.)]:
        metric=dict(nmse=error/energy,floored_nmse=error/energy,small_response=False,error_energy=error,native_response_energy=energy)
        rows.append(dict(source=source,baseline={m:dict(history_variance_nmse=energy) for m in run.METHODS},response={d:{m:metric for m in run.METHODS} for d in ('sinusoid','noise')}))
    r=run.summarize(rows)['lambda0']['responses']['sinusoid']
    assert r['mean_source_nmse']==2.5 and r['ratio_of_energies']!=2.5 and r['n_sources']==2
    rows[0]['response']['sinusoid']['lambda0']=dict(rows[0]['response']['sinusoid']['lambda0'],nmse=None,small_response=True)
    r=run.summarize(rows)['lambda0']['responses']['sinusoid']
    assert r['small_response_sources']==1 and r['valid_sources']==1 and r['n_sources']==2
