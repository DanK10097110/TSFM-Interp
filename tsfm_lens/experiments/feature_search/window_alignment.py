"""Neutral window-representation screen with train-only matched-rank controls.

Rows are aligned physical windows; splits and null permutations keep complete
generating series together. Token geometry does not prove empirical locality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import dct
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

METHODS=('mean','native_concat','common32_concat','dct2_common32')


def represent(tokens, token_width, window, method):
    """Map regular contiguous native spans into chronological physical windows."""
    tokens=np.asarray(tokens)
    n,t,d=tokens.shape
    context=t*token_width
    if window<token_width or window%token_width or context%window:
        raise ValueError('unsupported physical window below native resolution or non-tiling')
    native=tokens.reshape(n,context//window,window//token_width,d)
    if method=='mean':return native.mean(2)
    if method=='native_concat':return native.reshape(n,context//window,-1)
    if method not in METHODS:raise ValueError('unknown representation')
    if 32%token_width or window%32:
        raise ValueError('common32 grid requires native tokens that tile32 steps')
    grid=tokens.reshape(n,context//32,32//token_width,d).mean(2)
    grid=grid.reshape(n,context//window,window//32,d)
    if method=='dct2_common32':grid=dct(grid,type=2,axis=2,norm='ortho')[:,:,:2]
    return grid.reshape(n,context//window,-1)


def linear_cka(x,y):
    """Biased linear CKA, explicitly diagnostic until matched-rank/null checks."""
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    x=x-x.mean(0);y=y-y.mean(0)
    a=x@x.T;b=y@y.T
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/denom) if denom>0 else None


def series_permutation(n_series,n_windows,rng):
    """Permute complete series while retaining every within-series time order."""
    return (rng.permutation(n_series)[:,None]*n_windows+np.arange(n_windows)).ravel()


def matched_rank(train,test,rank=8):
    """Fit dimensional control on training series only, preserving PCA variances."""
    train=np.asarray(train);test=np.asarray(test)
    model=PCA(n_components=rank,svd_solver='randomized',random_state=0)
    a=model.fit_transform(train);b=model.transform(test)
    variance=model.explained_variance_
    return a,b,dict(rank=rank,participation_ratio=float(variance.sum()**2/(variance@variance)),
                    explained_variance_fraction=float(model.explained_variance_ratio_.sum()))


def analyze(data_dir,root):
    """Use one fixed middle-layer pair and the existing immutable seed splits."""
    from tsfm_lens.analysis.stats import mean_ci
    metadata=json.loads((data_dir/'metadata.json').read_text(encoding='utf-8'))
    keep=np.array([m['dose']==1 for m in metadata])
    meta=[m for m in metadata if m['dose']==1]
    train=np.array([m['split']=='train' for m in meta]);test=np.array([m['split']=='test' for m in meta])
    sources={'bolt':(np.load(data_dir/'bolt.npz')['encoder.block.3'][keep],16),
             'timesfm':(np.load(data_dir/'timesfm.npz')['stacked_xf.10'][keep],32)}
    root.mkdir(parents=True,exist_ok=False)
    protocol=dict(methods=METHODS,windows=[32,64,128],unsupported={'16':'TimesFM native32resolution'},
                  pca_rank=8,pca='training series only; no whitening',
                  null='99 whole-series permutations; fixed layer pair; no best-method selection',
                  ridge_alpha=1,logistic_C=1,split='existing immutable generating-seed train/test',
                  adjacency='windows within series dependent; adjacent at W32/64; one row per series atW128',
                  evidence='exploratory descriptive geometry and held-out task decodability')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    output={}
    sequence_checks={}
    for name,(tokens,width) in sources.items():
        sequences=[represent(tokens,width,w,'native_concat').reshape(len(tokens),-1) for w in (32,64,128)]
        sequence_checks[name]=float(max(np.max(np.abs(x-sequences[0])) for x in sequences))
        if sequence_checks[name]!=0:raise AssertionError('window partition changed complete native sequence')
    sequence_cka=linear_cka(sources['bolt'][0][test].reshape(int(test.sum()),-1),
                            sources['timesfm'][0][test].reshape(int(test.sum()),-1))
    waveform=np.load(data_dir/'corpus.npz')['context'][keep]
    for window in (32,64,128):
        nw=128//window
        periods=np.repeat([m['period'] for m in meta],nw)
        phase=np.array([m['phase'] for m in meta])[:,None]+2*np.pi*(np.arange(nw)*window+window/2)[None,:]/np.array([m['period'] for m in meta])[:,None]
        phase=np.column_stack((np.sin(phase.ravel()),np.cos(phase.ravel())))
        tr=np.repeat(train,nw);te=np.repeat(test,nw)
        for method in METHODS:
            controls={};raw={};record={}
            for name in ('bolt','timesfm','input_waveform'):
                if name=='input_waveform':features=waveform.reshape(len(meta)*nw,window)
                else:
                    tokens,width=sources[name]
                    features=represent(tokens,width,window,method).reshape(len(meta)*nw,-1)
                raw[name]=features[te]
                a,b,diagnostics=matched_rank(features[tr],features[te])
                controls[name]=b
                phase_probe=make_pipeline(StandardScaler(),Ridge(alpha=1))
                phase_probe.fit(a,phase[tr]);phase_error=((phase_probe.predict(b)-phase[te])**2).sum(1)
                period_probe=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000))
                period_probe.fit(a,periods[tr]);correct=period_probe.predict(b)==periods[te]
                record[name]=dict(raw_dimension=features.shape[1],pca=diagnostics,
                    phase_vector_mse=mean_ci(phase_error.reshape(-1,nw).mean(1),n_boot=999,seed=0,unit='generating_seed'),
                    period_accuracy=mean_ci(correct.reshape(-1,nw).mean(1),n_boot=999,seed=0,unit='generating_seed'))
            rng=np.random.default_rng(0)
            observed=linear_cka(controls['bolt'],controls['timesfm'])
            null=[linear_cka(controls['bolt'],controls['timesfm'][series_permutation(int(test.sum()),nw,rng)]) for _ in range(99)]
            record.update(raw_cka=linear_cka(raw['bolt'],raw['timesfm']),matched_rank_cka=observed,
                          null_p95=float(np.quantile(null,.95)),null_mean=float(np.mean(null)),
                          null_p_plusone=float((1+np.sum(np.asarray(null)>=observed))/100),
                          n_test_series=int(test.sum()),n_windows_per_series=nw)
            record.update(complete_sequence_native_cka=sequence_cka,
                          complete_sequence_partition_max_difference=sequence_checks,
                          raw_cka_caveat='dimension and repeated adjacent windows affect biasedCKA',
                          grid_caveat='common32/DCT discard fine native variation and do not restore within-coarse-token time')
            output[f'{window}_{method}']=record
    (root/'results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();analyze(args.data_dir,args.out)
