"""Exploratory stronger input baselines; original frozen probe results retained."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def features(x):
    """Scale-invariant shape and explicit periodicity statistics plus scale/trend."""
    t=np.linspace(-1,1,x.shape[1]);slope=x@t/(t@t)
    residual=x-x.mean(1,keepdims=True)-slope[:,None]*t
    rms=np.sqrt(np.mean(residual**2,axis=1));z=residual/np.maximum(rms[:,None],1e-8)
    psd=abs(np.fft.rfft(z,axis=1))**2;fraction=psd/np.maximum(psd.sum(1,keepdims=True),1e-8)
    acf=np.column_stack([np.mean(z[:,:-lag]*z[:,lag:],axis=1) for lag in [16,24,32,48]])
    stats=np.column_stack([acf,acf**2,fraction.max(1),fraction@np.arange(fraction.shape[1]),rms,slope,x.mean(1),np.sqrt(np.mean(np.diff(x)**2,axis=1))])
    return np.column_stack([z,fraction,stats,stats**2])


def run(root):
    """Tune ridge/RBF solely on existing validation; report exploratory heldout loss."""
    meta=json.loads((root/'metadata.json').read_text(encoding='utf-8'));data=np.load(root/'corpus.npz')
    split=np.array([m['split'] for m in meta]);dose=np.array([m['dose'] for m in meta]);x=features(data['context'])
    results={};saved={}
    for target in ['presence','period_present']:
        eligible=np.ones(len(meta),dtype=bool) if target=='presence' else dose==1
        y=dose.astype(float) if target=='presence' else np.array([m['period'] for m in meta],dtype=float)
        tr,va,te=[(split==s)&eligible for s in ['train','validation','test']]
        scaler=StandardScaler().fit(x[tr]);xx=scaler.transform(x);denom=np.var(y[tr])
        methods={'ridge':[Ridge(alpha=a) for a in [.01,1,100]],'rbf':[KernelRidge(alpha=a,kernel='rbf',gamma=g/x.shape[1]) for a in [.001,.01,.1,1] for g in [.01,.1,1,10]]}
        results[target]={}
        for method,candidates in methods.items():
            fitted=[m.fit(xx[tr],y[tr]) for m in candidates]
            losses=[np.mean((m.predict(xx[va])-y[va])**2) for m in fitted];idx=int(np.argmin(losses));chosen=fitted[idx];prediction=chosen.predict(xx[te])
            results[target][method]=dict(validation_nmse=float(losses[idx]/denom),test_nmse=float(np.mean((prediction-y[te])**2)/denom),alpha=chosen.alpha,gamma=getattr(chosen,'gamma',None))
            saved[target+'_'+method]=prediction
    pair_split=split[::2];tr,va,te=[pair_split==s for s in ['train','validation','test']]
    difference=x[1::2]-x[::2];scaler=StandardScaler().fit(difference[tr]);xx=scaler.transform(difference)
    results['translation_strong_input']={}
    for destination,layer in [('bolt','encoder.block.5'),('timesfm','stacked_xf.19')]:
        pooled=np.load(root/f'{destination}.npz')[layer].mean(1).astype(float)
        delta=pooled[1::2]-pooled[::2];pca=PCA(n_components=16,svd_solver='full').fit(delta[tr]);y=pca.transform(delta)
        denom=np.mean((y[te]-y[tr].mean(0))**2)
        candidates=[KernelRidge(alpha=a,kernel='rbf',gamma=g/xx.shape[1]) for a in [.001,.01,.1,1] for g in [.01,.1,1,10]]
        fitted=[m.fit(xx[tr],y[tr]) for m in candidates];losses=[np.mean((m.predict(xx[va])-y[va])**2) for m in fitted];chosen=fitted[int(np.argmin(losses))]
        prediction=chosen.predict(xx[te]);results['translation_strong_input'][destination]=dict(test_nmse=float(np.mean((prediction-y[te])**2)/denom),alpha=chosen.alpha,gamma=chosen.gamma)
        saved['translation_to_'+destination]=prediction
    results['evidence']='exploratory robustness after original heldout probe analysis; no practical utility claim'
    (root/'input_robustness.json').write_text(json.dumps(results,indent=2),encoding='utf-8');np.savez_compressed(root/'input_robustness_predictions.npz',**saved);print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
