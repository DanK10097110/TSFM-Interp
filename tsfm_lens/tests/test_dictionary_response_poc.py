"""Regression tests with planted normalization, weighting, pairing and gate failures."""
import numpy as np
import pytest
import torch
from experiments.feature_search import dictionary_response_poc as poc
from experiments.feature_search import dictionary_core as core
from tsfm_lens.sae.models import TopKSAE


def test_full_pair_energy_not_tiny_token_normalization():
    target=torch.tensor([[[0.,0.],[2.,2.]]])
    predicted=target+1
    actual=poc.response_loss(predicted,target,.01)
    assert actual.item()==pytest.approx(.5)
    planted=((predicted-target).square().mean(-1)/target.square().mean(-1).clamp_min(.01)).mean()
    assert abs(actual.item()-planted.item())>10
    zero=torch.zeros_like(target)
    assert poc.response_loss(zero+1,zero,.25).item()==4


def test_shared_schedule_sourceweight_and_no_heldout_fit():
    weights=core.source_weights(['a','a','a','b'])
    assert weights[:3].sum()==pytest.approx(weights[3])
    first,second=poc.schedule(weights,9),poc.schedule(weights,9)
    for a,b in zip(first,second):
        np.testing.assert_array_equal(a,b)
    assert first[0].shape==(500,64) and first[1].shape==(500,2,4)
    assert .47<np.mean(first[1].ravel()==3)<.53
    assert not .47<np.mean(np.random.default_rng(1).integers(4,size=4000)==3)<.53
    with pytest.raises(ValueError,match='leakage'):
        core.split_indices([dict(source='a',split='fit'),dict(source='a',split='test')])


def test_lambda0_branch_gradient_equivalence_and_lambda1_reach():
    torch.set_num_threads(1)
    x=torch.randn(5,8,generator=torch.Generator().manual_seed(3))
    h=torch.randn(2,2,3,8,generator=torch.Generator().manual_seed(4))
    gradients=[]
    for lam in (None,0,1):
        model=TopKSAE(8,8,3,generator=torch.Generator().manual_seed(5))
        loss=(model(x)[0]-x).square().mean()
        if lam is not None:
            rec=model(h.reshape(-1,8))[0].reshape(h.shape)
            loss=loss+lam*poc.response_loss(rec[:,0]-rec[:,1],h[:,0]-h[:,1],.01)
        loss.backward()
        gradients.append(torch.cat([p.grad.ravel() for p in model.parameters()]))
    torch.testing.assert_close(gradients[0],gradients[1],rtol=0,atol=0)
    assert not torch.allclose(gradients[0],gradients[2])


def test_center_quadratures_not_background_or_local_paths():
    signals=[dict(source='s',phase_id=i,frequency=f,zero=100*f+i) for f in (1,2) for i in range(4)]
    pairs=poc.focused_pairs(signals)
    assert len(pairs)==4
    assert all(r['minus']-r['plus']==2 for r in pairs)
    assert {r['quadrature'] for r in pairs}=={0,1}
    with pytest.raises(ValueError):
        poc.focused_pairs(signals[:-1])


def test_gate_requires_each_response_and_adequate_control():
    def result(global_nmse,sine,noise,focused):
        return {'global':{'nmse':global_nmse},'old_response':{d:{'mean_source_nmse':v} for d,v in [('sinusoid',sine),('noise',noise)]},'focused':{'mean_source_nmse':focused}}
    control=result(.02,1,1,1)
    assert poc.gate(control,result(.03,.7,.7,.9))['passed']
    for bad in [result(.06,.7,.7,.9),result(.03,.7,.9,.9),result(.03,.7,.7,1.1)]:
        assert not poc.gate(control,bad)['passed']
    assert not poc.gate(result(.1,1,1,1),result(.03,.7,.7,.9))['passed']


def test_response_identity_erasure_and_source_aggregation():
    artifact=dict(center=np.zeros(2),scale=np.array(2.),k=np.array(2),basis=np.eye(2))
    pairs=np.array([[[[1.,2.]],[[2.,5.]]]])
    actual=poc.score_pairs(pairs,artifact,[dict(source='a')])
    assert actual[0]['nmse']==0
    artifact['k']=np.array(0)
    assert poc.score_pairs(pairs,artifact,[dict(source='a')])[0]['nmse']==1
    rows=[dict(source='a',error_energy=1,target_energy=1,nmse=1)]*3+[dict(source='b',error_energy=3,target_energy=1,nmse=3)]
    assert poc.aggregate(rows,'old')['mean_source_nmse']==2
    assert np.mean([r['nmse'] for r in rows])!=2
