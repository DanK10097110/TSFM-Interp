"""Prespecified phase-only replication of window information tradeoffs.

Run one physical window size per invocation to bound single-thread CPU work.
Permutation units retain both phase variants and all windows of each seed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from window_alignment import METHODS, linear_cka, matched_rank, represent, series_permutation


def centered_gram(x):
    """Precompute centered row kernels once for the series-permutation null."""
    x=np.asarray(x,dtype=float);x=x-x.mean(0)
    return x@x.T


def kernel_null(a,b,n_bundle,n_rows_per_bundle,draws=999):
    """Permute complete paired-phase seed blocks, preserving their time order."""
    ka,kb=centered_gram(a),centered_gram(b)
    denominator=np.linalg.norm(ka)*np.linalg.norm(kb)
    if denominator<=0:raise ValueError('CKA undefined for zero-variance representations')
    score=float(np.sum(ka*kb)/denominator)
    rng=np.random.default_rng(993000);null=[]
    for _ in range(draws):
        idx=series_permutation(n_bundle,n_rows_per_bundle,rng)
        null.append(float(np.einsum('ij,ij->',ka,kb[np.ix_(idx,idx)])/denominator))
    return dict(value=score,null_p95=float(np.quantile(null,.95)),
                null_p_plusone=float((1+sum(x>=score for x in null))/(draws+1)),
                n_permutations=draws,resample_unit='generating_seed_both_phasevariants_allwindows')


def protocol():
    """Return the frozen full grid and primary family before any evaluation."""
    return dict(windows=[32,64,128],methods=METHODS,pca_ranks=[8,16,32],
                primary='rank8 mean-minus-native phasevectorMSE;modelBolt/TimesFM,W32/128;Holm4',
                period='32 fixed; perioddecoding skipped because labelconstant',
                split='train96/test48seedbundles;validationunused;2phasevariants eachseed',
                ridge_alpha=1,permutations=999,seed_bootstraps=9999,
                phase='sin/cos at physicalwindowcenter;Euclidean squaredvectorerror',
                unsupported={'16':'TimesFM native32stepresolution'},
                evidence='descriptive geometry and held-out predictive; no causal or methodwinner claim',
                dependence='allphasepairs and adjacentwindows cluster by seed',
                rank_caveat='PCA rank sensitivity controls capacity; not intrinsic method quality')


def run_window(data_dir,root,window):
    """Evaluate one frozen physical-window slice without choosing configurations."""
    from tsfm_lens.analysis.stats import mean_ci
    root.mkdir(parents=True,exist_ok=True)
    frozen=protocol();pp=root/'protocol.json'
    if pp.exists():
        if json.loads(pp.read_text(encoding='utf-8'))!=json.loads(json.dumps(frozen)):
            raise ValueError('existing protocol mismatch')
    else:pp.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    metadata=json.loads((data_dir/'metadata.json').read_text(encoding='utf-8'))
    train=np.array([m['split']=='train' for m in metadata]);test=np.array([m['split']=='test' for m in metadata])
    if sum(test)!=96 or [m['seed'] for m in metadata if m['split']=='test'][::2]!=list(range(144,192)):
        raise ValueError('phasepair ordering or frozen seed split mismatch')
    n=len(metadata);nw=128//window
    tokens={'bolt':(np.load(data_dir/'bolt_activations.npz')['encoder.block.3'],16),
            'timesfm':(np.load(data_dir/'timesfm_activations.npz')['stacked_xf.10'],32)}
    waveform=np.load(data_dir/'corpus.npz')['context']
    phases=np.array([m['phase'] for m in metadata])[:,None]+2*np.pi*(np.arange(nw)*window+window/2)[None,:]/32
    target=np.column_stack((np.sin(phases.ravel()),np.cos(phases.ravel())))
    tr=np.repeat(train,nw);te=np.repeat(test,nw)
    sequence={name:vals[test].reshape(96,-1) for name,(vals,width) in tokens.items()}
    invariance={name:float(max(np.max(np.abs(represent(vals,width,w,'native_concat').reshape(n,-1)-vals.reshape(n,-1))) for w in (32,64,128)))
                for name,(vals,width) in tokens.items()}
    output={};losses={}
    for method in METHODS:
        raw={name:represent(vals,width,window,method).reshape(n*nw,-1) for name,(vals,width) in tokens.items()}
        raw['input_waveform']=waveform.reshape(n*nw,window)
        record=dict(raw_cka=kernel_null(raw['bolt'][te],raw['timesfm'][te],48,2*nw),
                    complete_sequence_native_cka=linear_cka(sequence['bolt'],sequence['timesfm']),
                    sequence_partition_max_difference=invariance, ranks={})
        for rank in (8,16,32):
            controls={};part={}
            for name,features in raw.items():
                a,b,diagnostics=matched_rank(features[tr],features[te],rank)
                controls[name]=b
                probe=make_pipeline(StandardScaler(),Ridge(alpha=1))
                probe.fit(a,target[tr]);pred=probe.predict(b)
                error=((pred-target[te])**2).sum(1).reshape(48,2*nw).mean(1)
                losses[f'{method}_{rank}_{name}']=error
                part[name]=dict(phase_vector_mse=mean_ci(error,n_boot=9999,seed=993000,unit='generating_seed'),
                                pca=diagnostics,raw_dimension=features.shape[1])
            part['matched_rank_cka']=kernel_null(controls['bolt'],controls['timesfm'],48,2*nw)
            record['ranks'][str(rank)]=part
        output[method]=record
    t=np.arange(128);design=np.column_stack((np.sin(2*np.pi*t/32),np.cos(2*np.pi*t/32),np.ones(128)))
    coefficients=np.linalg.lstsq(design,waveform[test].T,rcond=None)[0]
    recovered=np.arctan2(coefficients[1],coefficients[0])
    true_phase=np.array([m['phase'] for m in metadata if m['split']=='test'])
    oracle=float(np.mean(abs(np.angle(np.exp(1j*(recovered-true_phase))))))
    output['oracle_phase_abs_radians']=oracle
    (root/f'window{window}.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    np.savez_compressed(root/f'window{window}_losses.npz',**losses)


def summarize(root):
    """Adjust the frozen four contrasts and draw two descriptive scientific panels."""
    from tsfm_lens.experiments.feature_search.statistics import compare_effects
    results={str(w):json.loads((root/f'window{w}.json').read_text(encoding='utf-8')) for w in (32,64,128)}
    contrasts={}
    for window in (32,128):
        losses=np.load(root/f'window{window}_losses.npz')
        for name in ('bolt','timesfm'):
            contrasts[f'{name}_W{window}_mean_minus_native']= (
                losses[f'mean_8_{name}'],losses[f'native_concat_8_{name}'])
    primary=compare_effects(contrasts,np.arange(144,192),n_boot=9999,n_permutations=9999,seed=993000)
    output=dict(primary=primary,grid=results,protocol=protocol())
    (root/'results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for method in METHODS:
        axes[0].plot([32,64,128],[results[str(w)][method]['ranks']['8']['matched_rank_cka']['value'] for w in (32,64,128)],marker='o',label=method)
        axes[1].plot([32,64,128],[results[str(w)][method]['ranks']['8']['bolt']['phase_vector_mse']['value'] for w in (32,64,128)],marker='o',label=method)
    axes[0].set(title='Shared geometry (PCA rank8)',ylabel='Linear CKA')
    axes[1].set(title='Bolt phase decoding (PCA rank8)',ylabel='Held-out phase vector MSE (lower better)')
    for ax in axes:ax.set_xlabel('Physical window (steps)');ax.set_xticks([32,64,128]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('48 held-out seed bundles; paired phases; window rows change with W',fontsize=10)
    fig.savefig(root/'window_tradeoff.png',dpi=180);fig.savefig(root/'window_tradeoff.pdf');plt.close(fig)
    return primary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--window',type=int,choices=[32,64,128]);parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    print(json.dumps(summarize(args.out),indent=2)) if args.summarize else run_window(args.data_dir,args.out,args.window)
