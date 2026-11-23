import numpy as np
from sklearn.metrics import average_precision_score

from research.activation_transfer.moment.crosscoder.evaluate import apply, calibrate, threshold
from research.activation_transfer.moment.crosscoder.evaluation_scores import average_precision, resample, score


def test_threshold_retains_dose_and_never_splits_tied_scores():
    s=np.array([4.,3.,3.,2.,1.,0.,-1.]);y=np.array([4.,5.,0.,0.,0.,4.,0.])
    cut,n,loss=threshold(s,y)
    assert (cut,n,loss)==(2.5,3,29.)
    scores,doses=apply(s[:,None],np.ones((1,1)),np.zeros(1),np.array([cut]))
    np.testing.assert_array_equal(scores[:,0],s)
    np.testing.assert_array_equal(doses[:,0],[4,3,3,0,0,0,0])
    # Keeping either tied score alone could seem favorable by a rowwise search.
    assert threshold(np.array([3.,3.]),np.array([2.,1.]))==(np.inf,0,5.)


def test_dual_readout_matches_independent_exhaustive_primal_validation():
    rng=np.random.default_rng(5521)
    train=13;x=rng.normal(size=(24,5))+np.arange(5)*3.
    latent=(x-x[:train].mean(0))@rng.normal(size=(5,3))
    y=np.maximum(0.,latent+rng.normal(scale=.8,size=latent.shape))
    grid=(.0002,.02,.8,10.)
    w,b,gate,fit=calibrate(x,y,fit_n=train,grid=grid)
    xc=x[:train]-x[:train].mean(0)
    train_mean=y[:train].mean(0)
    scale=np.linalg.svd(xc,compute_uv=False)[0]**2
    best=[None]*3
    for lam in grid:
        coef=np.linalg.solve(xc.T@xc+np.eye(5)*lam*scale,xc.T@(y[:train]-train_mean))
        pred=(x[train:]-x[:train].mean(0))@coef+train_mean
        for j in range(3):
            levels=np.unique(pred[:,j][pred[:,j]>0])
            cuts=np.r_[0.,(levels[:-1]+levels[1:])/2,np.inf]
            for cut in cuts:
                dose=np.where(pred[:,j]>cut,pred[:,j],0.)
                obj=(np.square(dose-y[train:,j]).sum(),int(np.sum(dose>0)),-lam)
                if best[j] is None or obj<best[j][0]:
                    best[j]=(obj,pred[:,j].copy(),dose.copy())
    raw,dose=apply(x[train:],w,b,gate)
    np.testing.assert_allclose(raw,np.stack([r[1] for r in best],axis=1),atol=1e-10)
    np.testing.assert_allclose(dose,np.stack([r[2] for r in best],axis=1),atol=1e-10)
    np.testing.assert_allclose(fit["lambda_relative"],[-r[0][2] for r in best])
    assert 1<len(set(fit["lambda_relative"]))<4


def test_weighted_ap_groups_ties_before_ranking():
    s=np.array([3.,3.,1.,-2.]);y=np.array([True,False,True,False])
    w=np.array([[1,1,1,1],[1,3,0,2],[0,0,0,5]],dtype=float)
    ap=average_precision(s,y,w)
    for i in (0,1):
        repeat=w[i].astype(int)
        np.testing.assert_allclose(ap[i],average_precision_score(np.repeat(y,repeat),np.repeat(s,repeat)))
    assert np.isnan(ap[2])


def test_macro_skill_counts_false_firings_and_weights_atoms_equally():
    y=np.array([[50.,0.],[0.,1.],[0.,0.]])
    dose=np.array([[25.,2.],[0.,0.],[0.,1.]])
    w=np.array([[1.,1.,1.],[0.,1.,2.]])
    vals,report=score(y,dose,dose,w)
    # First skill 1-625/2500=.75, second 1-(4+1+1)/1=-5.
    np.testing.assert_allclose(vals["skill"][0],-2.125)
    assert np.isnan(vals["skill"][1])
    assert report["tp"]==[1,0] and report["fp"]==[0,2] and report["fn"]==[0,1]


def test_station_date_draws_share_one_date_weight_across_stations():
    w=resample((4,3),draws=60,seed=333)
    np.testing.assert_array_equal(w[0],np.ones(12))
    mats=w[1:].reshape(-1,4,3)
    np.testing.assert_array_equal(mats.sum(axis=(1,2)),np.full(60,12))
    assert np.all(np.linalg.matrix_rank(mats)==1)
    assert (mats==0).any() and (mats>2).any()
