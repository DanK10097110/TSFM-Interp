"""Known-answer invertible output mixing versus genuinely lossy output projection."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments/feature_search'))
import forecast_recoverability as study
from mixture_readout import basis


def fixture(seed,n=32):
    rng=np.random.default_rng(seed);metadata=[];context=[];truth=[]
    t=np.arange(160)
    for s in range(n):
        a,b=rng.uniform(.5,2,2);offset=rng.uniform(-2,2);p,q=rng.uniform(-np.pi,np.pi,2);scale=np.sqrt((a*a+b*b)/2)
        split='train' if s<n//2 else 'validation' if s<3*n//4 else 'test'
        for i in range(2):
            for j in range(2):
                phi,psi=p+i*np.pi/2,q+j*np.pi/2
                row=dict(generator_seed=seed*100+s,cell_phi=i,cell_psi=j,split=split,a=a,b=b,offset=offset,scale=scale,phi=phi,psi=psi)
                y=offset+a*np.sin(2*np.pi*t/32+phi)+b*np.sin(2*np.pi*t/16+psi)
                metadata.append(row);context.append(y[:128]);truth.append(y[128:])
    return metadata,np.asarray(context),np.asarray(truth)


def forecast(targets,context,matrix):
    scaled=targets@matrix@basis(128,32).T
    return context.mean(1)[:,None]+context.std(1)[:,None]*scaled


def test_invertible_mixing_retains_linear_information_but_dropping_does_not():
    old,oc,ot=fixture(1);fresh,fc,ft=fixture(2)
    y,yy=study.semantic_targets(old),study.semantic_targets(fresh)
    train=np.array([m['split']=='train' for m in old]);val=np.array([m['split']=='validation' for m in old])
    matrix=np.array([[0,2,0,0],[-1,0,0,0],[0,0,0,-2],[0,0,1,0.]])
    native=forecast(yy,fc,matrix)
    fit=study.fit_output(forecast(y,oc,matrix),native,oc,fc,y,train,val)
    recovered=study.reconstruct(fit['fresh_prediction'],fc)
    scales=np.array([m['scale'] for m in fresh])
    loss=np.mean((recovered-ft)**2,1)/scales**2
    assert loss.mean()<.001
    assert np.mean(np.mean((native-ft)**2,1)/scales**2)>.1
    assert study.gates(loss.mean(),loss.mean())
    lossy=np.diag([1,1,0,0.])
    lost=study.fit_output(forecast(y,oc,lossy),forecast(yy,fc,lossy),oc,fc,y,train,val)
    lost_error=np.mean((study.reconstruct(lost['fresh_prediction'],fc)-ft)**2,1)/scales**2
    assert lost_error.mean()>.1
    assert not study.gates(lost_error.mean(),lost_error.mean())


def test_old_test_labels_and_fresh_rows_do_not_choose_training_parameters():
    old,oc,_=fixture(3);_,fc,_=fixture(4);y=study.semantic_targets(old)
    train=np.array([m['split']=='train' for m in old]);val=np.array([m['split']=='validation' for m in old])
    p=forecast(y,oc,np.eye(4));fresh=p[:len(fc)].copy()
    fit=study.fit_output(p,fresh,oc,fc,y,train,val)
    altered=y.copy();altered[~(train|val)]+=100
    other=study.fit_output(p,fresh,oc,fc,altered,train,val)
    np.testing.assert_array_equal(fit['fresh_prediction'],other['fresh_prediction'])
    assert fit['alpha']==other['alpha']
    expected=study.output_features(p,oc)[train].mean(0)
    np.testing.assert_allclose(fit['scaler_mean'],expected)
    # A fresh feature shift cannot alter train scaling/weights/alpha.
    shifted=study.fit_output(p,fresh+100,oc,fc,y,train,val)
    for key in study.FROZEN_KEYS:np.testing.assert_array_equal(fit[key],shifted[key])


def test_physical_row_binding_split_and_both_practical_gates():
    metadata,context,truth=fixture(5,n=4)
    row_ids,seeds=study.validate_rows(metadata,context,truth)
    assert len(set(row_ids))==16 and len(set(seeds))==4
    swapped=context.copy();swapped[[0,1]]=swapped[[1,0]]
    with pytest.raises(ValueError,match='Physical row order'):
        study.validate_rows(metadata,swapped,truth)
    altered=copy.deepcopy(metadata);altered[0]['split']='other'
    with pytest.raises(ValueError,match='one generating-seed split'):
        study.validate_rows(altered,context,truth)
    assert not study.gates(.02,-.02)
    assert not study.gates(.001,.02)
    assert study.gates(.01,.01)
