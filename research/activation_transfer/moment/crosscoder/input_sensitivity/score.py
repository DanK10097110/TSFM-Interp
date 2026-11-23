"""Test cross-model input sensitivities of fixed shared-dictionary directions."""
import argparse
import json

import numpy as np
import torch

from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.directions import ordinary_map
from research.activation_transfer.moment.crosscoder.input_sensitivity.capture import DOMAINS, DOSES, OUT, SCRATCH


def unit(x):
    length=np.linalg.norm(x,axis=-1,keepdims=True)
    return np.divide(x,length,out=np.zeros_like(x),where=length>0)


def projection(sensitivity,axes,atom):
    """Return [history, active coordinate, ordered input pulse]."""
    return unit(axes[atom]) @ sensitivity.transpose(0,2,1)


def comparisons(source,target):
    # Seven orthonormal, zero-temporal-mean basis coefficients form an ordinary Euclidean shape vector.
    # Subtracting the mean of the seven coordinates would change the measurement with basis choice.
    s,t=source[:,:,:7],target[:,:,:7]
    su,tu=unit(s),unit(t)
    same=np.sum(su*tu,axis=2)
    k=s.shape[1]
    others=(np.sum(su*tu.sum(1,keepdims=True),axis=2)-same)/(k-1)
    sl,tl=np.sign(source[:,:,7]),np.sign(target[:,:,7])
    sign=(sl==tl)&(sl!=0)
    offsign=((sl[:,:,None]==tl[:,None,:]) & (sl[:,:,None]!=0)).sum(2)-sign
    return dict(paired=same,other=others,margin=same-others,level_sign=sign.astype(float),
                other_level_sign=offsign/(k-1),level_margin=sign-offsign/(k-1))


def bounds(x):
    finite=x[np.isfinite(x)]
    return np.quantile(finite,[.025,.975]).tolist() if len(finite) else [None,None]


def summary(measures,atoms):
    n,k=atoms.shape
    keys,columns=np.unique(atoms,return_inverse=True)
    m=len(keys)
    index=np.repeat(np.arange(n),k)*m+columns.ravel()
    mat=np.bincount(index,minlength=n*m).reshape(n,m)
    freq=mat.sum(0)
    rng=np.random.default_rng(22112445)
    rw=rng.multinomial(n,np.ones(n)/n,3000)
    fw=rng.multinomial(m,np.ones(m)/m,3000)
    count=rw@mat
    out={}
    for key,events in measures.items():
        cells=np.bincount(index,weights=events.ravel(),minlength=n*m).reshape(n,m)
        by=rw@cells
        den=(fw*count).sum(1)
        event_boot=np.divide((fw*by).sum(1),den,where=den>0,out=np.full(len(den),np.nan))
        observed=count>0
        per_draw=np.divide(by,count,out=np.zeros_like(by),where=observed)
        den=(fw*observed).sum(1)
        atom_boot=np.divide((fw*per_draw).sum(1),den,where=den>0,out=np.full(len(den),np.nan))
        per=cells.sum(0)/freq
        accepted=freq>=8
        out[key]=dict(event_mean=float(events.mean()),history95=bounds((rw@events.mean(1))/n),
                      history_atom95=bounds(event_boot),equal_atom=float(per.mean()),equal_atom_crossed95=bounds(atom_boot),
                      equal_atom_min8=float(per[accepted].mean()) if accepted.any() else None,
                      positive_events=float(np.mean(events>0)),positive_atoms=float(np.mean(per>0)))
    return dict(n_histories=n,n_events=n*k,n_distinct_atoms=m,atoms_at_least_8=int((freq>=8).sum()),statistics=out)


def quant(v):
    return np.quantile(v,[0,.1,.5,.9,1]).tolist()


def strengths(s,t):
    x,y=s[:,:,:7],t[:,:,:7]
    sn=np.linalg.norm(x,axis=2);tn=np.linalg.norm(y,axis=2)
    den=np.sqrt(np.sum(sn**2)*np.sum(tn**2));assert den>0
    dot=np.sum(x*y)
    other=(np.sum(x*y.sum(1,keepdims=True))-dot)/(x.shape[1]-1)
    return dict(source_shape_norm=quant(sn),target_shape_norm=quant(tn),zero_shape_source=int((sn==0).sum()),
                zero_shape_target=int((tn==0).sum()),source_abs_level=quant(np.abs(s[:,:,7])),target_abs_level=quant(np.abs(t[:,:,7])),
                zero_level_source=int(np.sum(s[:,:,7]==0)),zero_level_target=int(np.sum(t[:,:,7]==0)),
                hilbert_matched=float(dot/den),hilbert_all_others=float(other/den),hilbert_margin=float((dot-other)/den))


def method_data(method,seed,a,b,w):
    model,means=cd.load_artifact(method,seed)
    with torch.inference_mode():
        codes=cd.encode_views(model,means,a['clean']).numpy()
    assert np.all((codes>0).sum(1)==cd.TOPK)
    atoms=np.where(codes>0)[1].reshape(len(codes),cd.TOPK)
    with np.load(cd.CACHE/f'{method}_{seed}.npz') as f:
        ds,dt=f['decoder_small'].astype(float),f['decoder_large'].astype(float)
    dense=ds@w
    dense*=np.linalg.norm(dt,axis=1,keepdims=True)/np.linalg.norm(dense,axis=1,keepdims=True)
    np.testing.assert_allclose(np.linalg.norm(dense,axis=1),np.linalg.norm(dt,axis=1),atol=1e-8)
    return (atoms,projection(a['sensitivity'].astype(float),ds,atoms),projection(b['sensitivity'].astype(float),dt,atoms),
            projection(b['sensitivity'].astype(float),dense,atoms))


def run(amp):
    cards={};margins={}
    w=ordinary_map()
    with np.load(SCRATCH/f'small_{amp}.npz') as fa,np.load(SCRATCH/f'large_{amp}.npz') as fb:
        a,b=dict(fa),dict(fb)
    np.testing.assert_array_equal(a['ids'],b['ids']);np.testing.assert_array_equal(a['sd'],b['sd'])
    for seed in cd.SEEDS:
        for method in cd.METHODS:
            stem=f'{method}_{seed}'
            ids,source,target,dense=method_data(method,seed,a,b,w)
            flat=comparisons(source,target);control=comparisons(source,dense)
            measures={**flat,'dense':control['paired'],'dense_other':control['other'],'dense_margin':control['margin'],
                      'pair_minus_dense':flat['paired']-control['paired'], 'dense_level_sign':control['level_sign']}
            # These signatures allow independent comparison in any orthogonal re-expression of the seven-dimensional shape basis.
            np.savez_compressed(OUT/f'{stem}_{amp}.npz',ids=a['ids'],domain=a['domain'],atom=ids,
                                source=source.astype('f4'),target=target.astype('f4'),dense=dense.astype('f4'))
            report={}
            for domain in DOMAINS:
                ix=a['domain']==domain
                info=summary({k:v[ix] for k,v in measures.items()},ids[ix])
                info['response_strength']=strengths(source[ix],target[ix])
                info['dense_strength']=strengths(source[ix],dense[ix])
                report[domain]=info
                margins[(stem,domain)]=flat['margin'][ix].mean(1)
            cards[stem]=report
    contrasts={}
    for seed in cd.SEEDS:
        for method in ('joint','masked'):
            key=f'{method}_{seed}'
            contrasts[key]={}
            for domain in DOMAINS:
                d=margins[(key,domain)]-margins[(f'shuffled_{seed}',domain)]
                n=len(d);rw=np.random.default_rng(22112445).multinomial(n,np.ones(n)/n,3000)
                contrasts[key][domain]=dict(mean=float(d.mean()),paired_history95=bounds((rw@d)/n))
    result=dict(amplitude=amp,measurements=cards,paired_margin_minus_shuffled=contrasts,
                plus_minus_replay_max_abs={k:float(z['plusminus_replay_max_abs']) for k,z in [('small',a),('large',b)]})
    (OUT/f'summary_{amp}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def amplitude_check():
    report={}
    for seed in cd.SEEDS:
        for method in cd.METHODS:
            key=f'{method}_{seed}'
            with np.load(OUT/f'{key}_{DOSES[0]}.npz') as p,np.load(OUT/f'{key}_{DOSES[1]}.npz') as q:
                np.testing.assert_array_equal(p['ids'],q['ids']);np.testing.assert_array_equal(p['atom'],q['atom'])
                report[key]={}
                for domain in DOMAINS:
                    sel=p['domain']==domain; cells={}
                    for arm in ('source','target','dense'):
                        x,y=[z[arm][sel,:,:7].astype(float) for z in (p,q)]
                        agreement=np.sum(unit(x)*unit(y),axis=2)
                        cells[arm]=dict(cosine_quantiles=quant(agreement),signed_hilbert_cosine=float(np.sum(x*y)/(np.linalg.norm(x)*np.linalg.norm(y))),
                                        relative_rms=float(np.linalg.norm(x-y)/np.linalg.norm(x)))
                    report[key][domain]=cells
    (OUT/'amplitudes.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('amp',choices=('0.15','0.075','compare'))
    args=cli.parse_args();torch.set_num_threads(2)
    if args.amp=='compare':amplitude_check()
    else:run(float(args.amp))
