"""Independent arithmetic audit of the frozen FFN pilot, without native calls.

Reconstruct pooling, weighted FIT PCA with a different SVD driver, FIT feature
ranking, classification, purity and source bootstrap directly from saved arrays.
The frozen analyzer and its outputs are never rewritten.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.fftpack import dct
from scipy.linalg import svd

from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.ffn_selectivity_study import knowledge

TYPES=('sin24','sin19.2','steprecent32','ramprecent64','noise')


def project(delta,basis,fit,conditions):
    """Independent basis construction; PCA sees only balanced FIT source/types."""
    if basis=='native':
        return delta
    if basis.startswith('rotation'):
        rng=np.random.default_rng(int(basis.removeprefix('rotation')))
        permutation=rng.permutation(delta.shape[-1])
        signs=rng.choice([-1.,1.],size=delta.shape[-1])
        return dct(delta[...,permutation]*signs,type=2,norm='ortho',axis=-1)
    if basis!='PCA':
        raise ValueError('Unknown basis')
    fit_data=delta[fit]
    weights=[]
    for _ in fit_data:
        for condition in conditions:
            count=sum(c['kind']==condition['kind'] for c in conditions)
            weights.extend([1/count]*delta.shape[2])
    weights=np.asarray(weights)
    weights/=weights.sum()
    matrix=fit_data.reshape(-1,delta.shape[-1])
    centered=(matrix-np.average(matrix,axis=0,weights=weights))*np.sqrt(weights[:,None])
    _,values,vectors=svd(centered,full_matrices=False,lapack_driver='gesvd')
    retained=values>values[0]*1e-10 if len(values) and values[0]>0 else np.zeros(len(values),bool)
    return delta@vectors[retained].T


def features(delta,conditions,view):
    """Pool by source/type using explicit token and variant reductions."""
    token_response=np.abs(np.sum(delta,axis=2)/delta.shape[2]) if view=='mean' else np.sqrt(np.sum(delta**2,axis=2)/delta.shape[2])
    result=[]
    for kind in TYPES:
        indices=[j for j,c in enumerate(conditions) if c['kind']==kind]
        result.append(np.sum(token_response[:,indices],axis=1)/len(indices))
    return np.stack(result,axis=1)


def verify_readout(x,fit,record):
    """Independently refit ANOVA axes and recompute per-source held-out endpoints."""
    fit_x=x[fit]
    rule=record['FIT_rule']
    center=np.mean(fit_x,axis=(0,1))
    scale=np.std(fit_x,axis=(0,1))
    active=scale>max(float(scale.max()),1e-292)*1e-8 if len(scale) else np.array([],bool)
    if not rule['valid']:
        if active.any():
            raise ValueError('Reported inactive basis has active FIT coordinates')
        return 0.
    class_mean=np.mean(fit_x,axis=0)
    between=np.mean((class_mean-center)**2,axis=0)
    within=np.mean((fit_x-class_mean[None])**2,axis=(0,1))
    scores=between/np.maximum(within,max(float(np.mean(fit_x**2))*1e-12,1e-300))
    scores[~active]=-np.inf
    chosen=np.argsort(-scores,kind='stable')[:min(record['budget'],int(active.sum()))]
    if chosen.tolist()!=rule['indices']:
        raise ValueError('FIT-only feature selection mismatch')
    preferred=np.argmax(class_mean[:,chosen],axis=0)
    if preferred.tolist()!=rule['preferred_types']:
        raise ValueError('FIT-only preferred type mismatch')
    centers=(class_mean[:,chosen]-center[chosen])/scale[chosen]
    for actual,saved in ((center[chosen],rule['center']),(scale[chosen],rule['scale']),(centers,rule['centroids'])):
        if not np.allclose(actual,saved,atol=2e-8,rtol=2e-6):
            raise ValueError('FIT centering/scaling/readout mismatch')
    normalized=(x[:,:,chosen]-center[chosen])/scale[chosen]
    distances=np.stack([np.sum((normalized-c)**2,axis=-1) for c in centers],axis=-1)
    predictions=distances.argmin(axis=-1)
    if not np.array_equal(predictions,record['per_source']['predictions']):
        raise ValueError('Source class predictions mismatch')
    accuracy=(predictions==np.arange(5)).sum(axis=1)/5
    responses=x[:,:,chosen]
    total=responses.sum(axis=1)
    purity=np.zeros((len(x),len(chosen)))
    for j,kind in enumerate(preferred):
        np.divide(responses[:,kind,j],total[:,j],out=purity[:,j],where=total[:,j]>0)
    total_energy=np.sum(x*x,axis=(1,2))
    coverage=np.divide(np.sum(responses**2,axis=(1,2)),total_energy,out=np.zeros(len(x)),where=total_energy>0)
    metrics=dict(accuracy=accuracy,purity=purity.mean(axis=1),energy_coverage=coverage,
                 active_selected_fraction=(total>0).mean(axis=1))
    error=0.
    for key,value in metrics.items():
        saved=np.asarray(record['per_source'][key])
        error=max(error,float(np.max(abs(value-saved))))
        if not np.allclose(value,saved,atol=2e-8,rtol=2e-6):
            raise ValueError('Independent source metric mismatch: '+key)
    return error


def summary(values,rows,group,draws=1999):
    """Recompute fixed-stratum TEST-source bootstrap without token observations."""
    indices=[i for i,r in enumerate(rows) if r['split']=='TEST' and (group=='all' or r['kind']==group)]
    if not indices:
        return dict(n_sources=0,mean=None,ci95=None)
    rng=np.random.default_rng(4961200)
    bootstrap=np.zeros(draws)
    for kind in sorted({rows[i]['kind'] for i in indices}):
        subset=np.asarray([i for i in indices if rows[i]['kind']==kind])
        sample=rng.integers(0,len(subset),size=(draws,len(subset)))
        bootstrap+=np.asarray(values)[subset[sample]].sum(axis=1)
    return dict(n_sources=len(indices),source_ids=[rows[i]['source'] for i in indices],
        mean=float(np.asarray(values)[indices].mean()),
        ci95=np.quantile(bootstrap/len(indices),[.025,.975]).tolist() if len(indices)>1 else None)


def check_summary(actual,saved):
    """Require correct source identities as well as paired source uncertainty."""
    for key in ('n_sources','source_ids'):
        if actual.get(key)!=saved.get(key):
            raise ValueError('Bootstrap source identity mismatch')
    for key in ('mean','ci95'):
        if actual[key] is None:
            if saved[key] is not None:
                raise ValueError('Undefined bootstrap value reported')
        elif not np.allclose(actual[key],saved[key],atol=1e-12,rtol=1e-10):
            raise ValueError('Independent bootstrap mismatch')


def audit(out):
    """Audit all frozen outputs and save a compact, explicitly exploratory record."""
    if (out/'independent_audit.json').exists():
        raise FileExistsError('Refuse repeated audit')
    try:
        p,analysis,rows,conditions=[read(out/name) for name in ('protocol.json','analysis.json','rows.json','conditions.json')]
        for path,digest in p['sources'].items():
            if sha(path)!=digest:
                raise ValueError('Frozen source changed')
        for name,digest in p['hashes'].items():
            if sha(out/name)!=digest:
                raise ValueError('Frozen artifact changed')
        if len({r['source'] for r in rows})!=len(rows):
            raise ValueError('Source leakage/duplication')
        fit=np.asarray([r['split']=='FIT' for r in rows])
        q=np.load(out/'inputs.npz',allow_pickle=False)['queries'].astype(float)
        delta_input=(q[:,1:]-q[:,0,None])[:,:,None,:]
        rms_error=float(np.max(abs(np.sqrt(np.mean(delta_input**2,axis=-1))-.1)))
        if rms_error>1e-7:
            raise ValueError('Input RMS mismatch')
        execution=read(out/'execution.json')
        captures={}
        gate_details={}
        for model in p['models']:
            worker=read(out/f'{model}_worker.json')
            if (not execution[model]['valid'] or execution[model]['returncode']!=0
                    or not worker['valid'] or not worker['normalizer_identity_exact'] or not worker['self_replay_exact']
                    or not worker['all_reach_passed'] or worker['cells']!=480
                    or worker['sha256']!=sha(out/f'{model}_captures.npz')
                    or worker['protocol_sha256']!=sha(out/'protocol.json')
                    or max(worker['peak_rss'],execution[model]['monitored_peak'])>3*2**30
                    or max(worker['seconds'],execution[model]['seconds'])>900):
                raise ValueError('Native execution/capture gate failed')
            reach=worker['reach']
            if len(reach)!=48 or {(r['background'],r['site']) for r in reach}!={(i,s) for i in range(24) for s in ('first','half')}:
                raise ValueError('Missing within-FFN reach cells')
            if not all(r['relative_change']>1e-6 for r in reach):
                raise ValueError('Within-FFN reach failed')
            captures[model]=np.load(out/f'{model}_captures.npz',allow_pickle=False)
            data=captures[model]
            if data['source_ids'].tolist()!=[r['source'] for r in rows] or data['forecasts'].shape!=(24,15,32):
                raise ValueError('Capture source/forecast geometry mismatch')
            if not np.isfinite(data['forecasts']).all():
                raise ValueError('Nonfinite forecast')
            gate_details[model]=dict(cells=worker['cells'],minimum_reach=min(r['relative_change'] for r in reach),
                seconds=execution[model]['seconds'],peak_rss=max(worker['peak_rss'],execution[model]['monitored_peak']))
        error,geometry_error,verified,summaries=0.,0.,0,0
        group_keys=sorted({(r['model'],r['site'],r['space']) for r in analysis['records']})
        for model,site,space in group_keys:
            if model=='input':
                delta=delta_input
            else:
                raw=captures[model][f'{site}_{space}'].astype(float)
                spec=p['capture'][model]['sites'][0 if site=='first' else 1]
                if raw.shape!=(24,15,spec['tokens'],spec[f'{space}_width']) or not np.isfinite(raw).all():
                    raise ValueError('Capture geometry/finite audit failed')
                delta=raw[:,1:]-raw[:,0,None]
            for basis in ('native','rotation4961101','rotation4961102','rotation4961103','PCA'):
                transformed=project(delta,basis,fit,conditions)
                if basis.startswith('rotation'):
                    difference=np.max(abs(np.sum(transformed**2,axis=-1)-np.sum(delta**2,axis=-1)))
                    relative=float(difference/max(1.,float(np.max(np.sum(delta**2,axis=-1)))))
                    geometry_error=max(geometry_error,relative)
                    if relative>1e-12:
                        raise ValueError('Actual-capture rotation geometry failed')
                for view in ('mean','rms'):
                    x=features(transformed,conditions,view)
                    records=[r for r in analysis['records'] if (r['model'],r['site'],r['space'],r['basis'],r['view'])==(model,site,space,basis,view)]
                    if len(records)!=3:
                        raise ValueError('Missing registered feature budget')
                    for record in records:
                        if record['FIT_source_ids']!=[r['source'] for r in rows if r['split']=='FIT']:
                            raise ValueError('FIT source identities changed')
                        error=max(error,verify_readout(x,fit,record))
                        verified+=1
                        if not record['valid']:
                            continue
                        for group,metrics in record['summaries'].items():
                            for metric,saved in metrics.items():
                                check_summary(summary(record['per_source'][metric],rows,group),saved)
                                summaries+=1
            print(f'Audited {model}/{site}/{space}',flush=True)
        lookup={(r['model'],r['site'],r['space'],r['view'],r['basis'],r['budget']):r for r in analysis['records']}
        for comparison in analysis['comparisons']:
            if not comparison['valid']:
                continue
            r=lookup[(comparison['model'],comparison['site'],comparison['space'],comparison['view'],'native',comparison['budget'])]
            if comparison['control']=='actual_input':
                control=lookup[('input','input','input',comparison['view'],'native',comparison['budget'])]
            else:
                control=lookup[(comparison['model'],comparison['site'],comparison['space'],comparison['view'],comparison['control'],comparison['budget'])]
            difference=np.asarray(r['per_source'][comparison['metric']])-control['per_source'][comparison['metric']]
            check_summary(summary(difference,rows,comparison['group']),comparison)
            summaries+=1
        compact=[]
        for r in analysis['records']:
            if r['space']!='ffn' or r['basis']!='native' or not r['valid']:
                continue
            for group in ('synthetic','real_reused'):
                item={k:r[k] for k in ('model','site','view','budget')}
                item.update(group=group,n_sources=r['summaries'][group]['accuracy']['n_sources'],metrics={})
                for metric in ('accuracy','purity'):
                    native=np.asarray(r['per_source'][metric])
                    controls={basis:lookup[(r['model'],r['site'],'ffn',r['view'],basis,r['budget'])]
                              for basis in ('rotation4961101','rotation4961102','rotation4961103','PCA')}
                    actual=lookup[('input','input','input',r['view'],'native',r['budget'])]
                    residual=lookup[(r['model'],r['site'],'residual',r['view'],'native',r['budget'])]
                    rotation=np.mean([c['per_source'][metric] for b,c in controls.items() if b.startswith('rotation')],axis=0)
                    item['metrics'][metric]=dict(native=summary(native,rows,group),controls={
                        name:dict(score=summary(values,rows,group),paired_native_minus_control=summary(native-values,rows,group))
                        for name,values in (('rotation_average',rotation),('PCA',np.asarray(controls['PCA']['per_source'][metric])),
                            ('actual_input',np.asarray(actual['per_source'][metric])),('residual_native',np.asarray(residual['per_source'][metric])))})
                compact.append(item)
        result=dict(valid=True,protocol_sha256=sha(out/'protocol.json'),analysis_sha256=sha(out/'analysis.json'),
            auditor_sha256=sha(__file__),records_independently_recomputed=verified,source_summaries_and_comparisons_recomputed=summaries,
            maximum_metric_error=error,maximum_rotation_relative_energy_error=geometry_error,input_RMS_error=rms_error,
            native_gates=gate_details,native_slot_released=True,stage2_launched=False,
            audit_scope='Independent pooling, weighted FIT PCA using gesvd instead of gesdd, FIT ANOVA/readout, predictions, purity, source bootstrap, source/hash/shape/resource gates; native self-replay/reach rely on frozen worker measurements, not rerun.',
            limitations=['Small exploratory source cohorts; bootstrap conditional on FIT selection','Multiple layers/views/budgets, no confirmatory multiplicity claim',
                'Structured rotations are not Haar draws','PCA supported directions have a different candidate count',
                'Background-locked normalization makes perturbed passes controlled counterfactuals','Reused real sources support no new real generalization',
                'Perturbation selectivity is not general monosemanticity; no SAE; no selected-unit causal evidence'])
        write(out/'independent_audit.json',result)
        write(out/'compact_record.json',dict(scope='Descriptive perturbation selectivity; all registered native FFN sites/views/budgets, separate source strata',
            rotation_average='Post-analysis compact paired summary across all three frozen rotations, not best TEST rotation',
            records=compact,audit=result,stage2='Not launched; review pilot first'))
    except Exception as exc:
        knowledge(out,'independent_audit','failed',dict(error_type=type(exc).__name__,error=str(exc)),
                  'Resolve recorded audit failure without launching native or stage2.')
        raise
    knowledge(out,'independent_audit','complete',dict(audit=result,compact_record_sha256=sha(out/'compact_record.json'),
        compact_records=compact),'Report pilot selectivity with paired controls and limitations. Stage2 remains on hold.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/ffn_selectivity_pilot4961000'))
    audit(parser.parse_args().out)
