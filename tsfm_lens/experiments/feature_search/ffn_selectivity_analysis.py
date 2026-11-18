"""FIT-only direct-coordinate perturbation selectivity and small readouts.

Tokenwise signed deltas are rotated before separate mean/RMS pooling. Type
variants collapse within each background; uncertainty resamples backgrounds.
Classification and FIT-preferred-type response purity are distinct endpoints.
"""
from pathlib import Path
import argparse

import numpy as np
from scipy.fft import dct
from scipy.linalg import svd

from experiments.feature_search.ffn_selectivity_study import TYPES, BUDGETS, MODELS, read, write, sha, knowledge


def rotate_tokens(tokens,seed):
    """Dense structured orthogonal basis; preserves all token-delta geometry."""
    x=np.asarray(tokens,float)
    rng=np.random.default_rng(seed)
    order=rng.permutation(x.shape[-1])
    signs=rng.choice([-1.,1.],size=x.shape[-1])
    return dct(x[...,order]*signs,type=2,norm='ortho',axis=-1,workers=1)


def pca_basis(tokens,fit,conditions=None):
    """FIT source/type-balanced token covariance; apply axes without offsets."""
    x=np.asarray(tokens,float)[np.asarray(fit,bool)].reshape(-1,tokens.shape[-1])
    if len(x)<2:
        raise ValueError('Need at least two FIT token observations')
    weights=np.ones(len(x))
    if conditions is not None:
        if len(conditions)!=tokens.shape[1]:
            raise ValueError('PCA variant geometry mismatch')
        frequency={kind:sum(c['kind']==kind for c in conditions) for kind in TYPES}
        variant_weights=np.asarray([1/frequency[c['kind']] for c in conditions])
        weights=np.tile(np.repeat(variant_weights,tokens.shape[2]),int(np.sum(fit)))
    weights=weights/weights.sum()
    center=np.sum(x*weights[:,None],axis=0)
    centered=(x-center)*np.sqrt(weights[:,None])
    _,singular,axes=svd(centered,full_matrices=False,check_finite=True,lapack_driver='gesdd')
    count=int(np.sum(singular>1e-10*singular[0])) if len(singular) and singular[0]>0 else 0
    return axes[:count]


def pool(tokens,conditions,view):
    """Pool token deltas, then average variants within each source/type."""
    x=np.asarray(tokens,float)
    if x.ndim!=4 or x.shape[1]!=len(conditions) or view not in ('mean','rms'):
        raise ValueError('Need source,variant,token,coordinate geometry and declared view')
    response=abs(x.mean(axis=2)) if view=='mean' else np.sqrt(np.mean(x*x,axis=2))
    values=[]
    for kind in TYPES:
        ix=[i for i,c in enumerate(conditions) if c['kind']==kind]
        if not ix:
            raise ValueError('All five perturbation types required')
        values.append(response[:,ix].mean(axis=1))
    return np.stack(values,axis=1)


def fit_readout(features,fit,budget):
    """FIT ANOVA feature selection, scaling, class centroids and preferred types."""
    x=np.asarray(features,float)[np.asarray(fit,bool)]
    if budget not in BUDGETS:
        raise ValueError('Use frozen single/few-feature budgets')
    if x.ndim!=3 or x.shape[1]!=len(TYPES) or len(x)<2:
        raise ValueError('Need FIT background/type/coordinate features')
    if x.shape[-1]==0:
        return dict(valid=False,reason='No FIT-supported directions')
    center=x.mean(axis=(0,1))
    scale=x.std(axis=(0,1))
    active=scale>max(float(scale.max())*1e-8,1e-300)
    if not active.any():
        return dict(valid=False,reason='No FIT-active features')
    class_means=x.mean(axis=0)
    between=np.mean((class_means-center)**2,axis=0)
    within=np.mean((x-class_means[None])**2,axis=(0,1))
    floor=max(float(np.mean(x*x))*1e-12,1e-300)
    score=between/np.maximum(within,floor)
    score[~active]=-np.inf
    chosen=np.argsort(-score,kind='stable')[:min(budget,int(active.sum()))]
    centroids=(class_means[:,chosen]-center[chosen])/scale[chosen]
    return dict(valid=True,requested_budget=budget,used_budget=len(chosen),candidate_count=x.shape[-1],
        FIT_active_candidates=int(active.sum()),indices=chosen.tolist(),center=center[chosen].tolist(),
        scale=scale[chosen].tolist(),centroids=centroids.tolist(),
        preferred_types=np.argmax(class_means[:,chosen],axis=0).tolist(),
        FIT_scores=score[chosen].tolist())


def evaluate(features,state):
    """Apply frozen FIT rules; each returned score corresponds to one background."""
    x=np.asarray(features,float)
    if not state['valid']:
        return None
    ix=np.asarray(state['indices'])
    z=(x[:,:,ix]-state['center'])/state['scale']
    centroids=np.asarray(state['centroids'])
    distance=np.sum((z[:,:,None,:]-centroids[None,None,:,:])**2,axis=-1)
    predictions=np.argmin(distance,axis=-1)
    accuracy=np.mean(predictions==np.arange(len(TYPES))[None],axis=1)
    preferred=np.asarray(state['preferred_types'])
    numerator=x[:,preferred,ix]
    denominator=x[:,:,ix].sum(axis=1)
    purity=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0).mean(axis=1)
    coverage=np.divide(np.sum(x[:,:,ix]**2,axis=(1,2)),np.sum(x*x,axis=(1,2)),
                       out=np.zeros(len(x)),where=np.sum(x*x,axis=(1,2))>0)
    return dict(accuracy=accuracy,purity=purity,energy_coverage=coverage,
                active_selected_fraction=(denominator>0).mean(axis=1),predictions=predictions)


def background_summary(values,rows,indices,n_boot=1999):
    """Stratified seed/source bootstrap; variants and tokens never become samples."""
    values=np.asarray(values,float)
    ix=np.asarray(indices,int)
    if not len(ix):
        return dict(n_sources=0,mean=None,ci95=None)
    groups={kind:ix[[rows[i]['kind']==kind for i in ix]] for kind in sorted({rows[i]['kind'] for i in ix})}
    rng=np.random.default_rng(4961200)
    draws=np.zeros(n_boot)
    for members in groups.values():
        picked=members[rng.integers(len(members),size=(n_boot,len(members)))]
        draws+=values[picked].sum(axis=1)
    draws/=len(ix)
    return dict(n_sources=len(ix),source_ids=[rows[i]['source'] for i in ix],mean=float(values[ix].mean()),
                ci95=np.quantile(draws,[.025,.975]).tolist() if len(ix)>=2 else None,
                scope='Conditional on FIT selection; source-group descriptive bootstrap, no token/variant resampling')


def readout_records(tokens,rows,conditions,model,site,space):
    """Compare axes and controls using identical FIT groups and held-out sources."""
    fit=np.asarray([r['split']=='FIT' for r in rows])
    axes=pca_basis(tokens,fit,conditions)
    bases=[('native',None),('rotation4961101',4961101),('rotation4961102',4961102),
           ('rotation4961103',4961103),('PCA',axes)]
    records=[]
    for basis,transform in bases:
        projected=tokens if basis=='native' else (tokens@transform.T if basis=='PCA' else rotate_tokens(tokens,transform))
        for view in ('mean','rms'):
            features=pool(projected,conditions,view)
            for budget in BUDGETS:
                state=fit_readout(features,fit,budget)
                result=evaluate(features,state)
                item=dict(model=model,site=site,space=space,basis=basis,view=view,budget=budget,FIT_rule=state,
                    FIT_source_ids=[r['source'] for r in rows if r['split']=='FIT'],
                    basis_geometry='PCA FIT-supported subspace' if basis=='PCA' else 'full orthogonal/native basis')
                if result is None:
                    item.update(valid=False,reason=state['reason'])
                else:
                    item.update(valid=True,per_source={k:v.tolist() for k,v in result.items()},summaries={})
                    for label in ('all','synthetic','real_reused'):
                        ix=[i for i,r in enumerate(rows) if r['split']=='TEST' and (label=='all' or r['kind']==label)]
                        item['summaries'][label]={metric:background_summary(result[metric],rows,ix)
                            for metric in ('accuracy','purity','energy_coverage','active_selected_fraction')}
                records.append(item)
    return records


def analyze(out):
    """No native calls; train on FIT groups and persist all comparisons without tuning."""
    if (out/'analysis.json').exists():
        raise FileExistsError('Refuse repeated analysis')
    try:
        p=read(out/'protocol.json')
        for path,digest in p['sources'].items():
            if sha(path)!=digest:
                raise ValueError('Frozen source changed')
        for name in ('inputs.npz','rows.json','conditions.json'):
            if sha(out/name)!=p['hashes'][name]:
                raise ValueError('Frozen input changed')
        rows,conditions=read(out/'rows.json'),read(out/'conditions.json')
        fit={r['source'] for r in rows if r['split']=='FIT'}
        test={r['source'] for r in rows if r['split']=='TEST'}
        if fit&test or len(fit)+len(test)!=len(rows):
            raise ValueError('Source split leakage')
        with np.load(out/'inputs.npz',allow_pickle=False) as f:
            q=f['queries'].astype(float)
        actual_input=(q[:,1:]-q[:,0,None])[:,:,None,:]
        records=readout_records(actual_input,rows,conditions,'input','input','input')
        execution=read(out/'execution.json')
        for model in MODELS:
            worker=read(out/f'{model}_worker.json')
            if (not execution[model]['valid'] or not worker['valid'] or not worker['all_reach_passed']
                    or not worker['normalizer_identity_exact'] or not worker['self_replay_exact']
                    or worker['cells']!=p['cells_per_model'] or worker['cells']>500
                    or worker['seconds']>900 or worker['peak_rss']>3*2**30
                    or worker['protocol_sha256']!=sha(out/'protocol.json')
                    or worker['sha256']!=sha(out/f'{model}_captures.npz')):
                raise ValueError('Native gate/resource/hash failed')
            with np.load(out/f'{model}_captures.npz',allow_pickle=False) as f:
                if f['source_ids'].tolist()!=[r['source'] for r in rows]:
                    raise ValueError('Capture source alignment failed')
                for site in ('first','half'):
                    for space in ('ffn','residual'):
                        raw=f[f'{site}_{space}'].astype(float)
                        delta=raw[:,1:]-raw[:,0,None]
                        records.extend(readout_records(delta,rows,conditions,model,site,space))
        lookup={(r['model'],r['site'],r['space'],r['view'],r['basis'],r['budget']):r for r in records}
        comparisons=[]
        for r in records:
            if not r['valid'] or r['basis']!='native' or r['model']=='input':
                continue
            controls=[lookup[(r['model'],r['site'],r['space'],r['view'],b,r['budget'])]
                      for b in ('rotation4961101','rotation4961102','rotation4961103','PCA')]
            controls.append(lookup[('input','input','input',r['view'],'native',r['budget'])])
            for control in controls:
                if not control['valid']:
                    comparisons.append(dict(model=r['model'],site=r['site'],space=r['space'],view=r['view'],
                        budget=r['budget'],control=control['basis'],valid=False,reason=control['reason']))
                    continue
                for metric in ('accuracy','purity'):
                    delta=np.asarray(r['per_source'][metric])-control['per_source'][metric]
                    for group in ('all','synthetic','real_reused'):
                        ix=[i for i,row in enumerate(rows) if row['split']=='TEST' and (group=='all' or row['kind']==group)]
                        comparisons.append(dict(model=r['model'],site=r['site'],space=r['space'],view=r['view'],
                            budget=r['budget'],control='actual_input' if control['model']=='input' else control['basis'],
                            metric=metric,group=group,valid=True,**background_summary(delta,rows,ix)))
        result=dict(label=p['label'],no_SAE=True,records=records,comparisons=comparisons,
            gates=dict(source_disjoint=True,FIT_only_fitting=True,native_identity_and_reach=True,group_bootstrap=True),
            interpretation='Response purity for FIT-preferred perturbation types and few-feature class readout are separate descriptive endpoints. Neither establishes general monosemanticity. Reused real strata provide no new real-generalization claim.',
            causal_stage='Not executed; requires selectivity review and a new native grant')
        write(out/'analysis.json',result)
    except Exception as exc:
        knowledge(out,'analysis','failed',dict(error_type=type(exc).__name__,error=str(exc)),
                  'Inspect failed instrumentation; do not claim selectivity.')
        raise
    knowledge(out,'analysis','complete',dict(comparisons=comparisons,gates=result['gates'],analysis_sha256=sha(out/'analysis.json')),
              'Review held-out selectivity versus rotations/PCA/input. Optional causal unit-delta removal remains unlaunched.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/ffn_selectivity_pilot4961000'))
    analyze(parser.parse_args().out)
