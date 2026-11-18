"""Directional sign-test sensitivity alongside the unchanged mean-based claims.

Exact binomial signs concern P(seed contrast>0 | nonzero)>.5, a different
estimand from the mean sign-flip test. Ties and missing phases are explicit.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from experiments.feature_search.mixture_interference import input_interference
from experiments.feature_search.mixture_replication import _validate_compositions,load_protocol
from tsfm_lens.analysis.stats import holm


def sign_sensitivity(contrasts):
    """Retain all four claims; report exact one-sided signs and effect distributions."""
    if set(contrasts)!=set(load_protocol()['claims']):
        raise ValueError('Sign sensitivity requires the complete frozen four-claim family')
    results={}
    for name,values in contrasts.items():
        values=np.asarray(values,dtype=float)
        if values.ndim!=1 or not len(values):
            raise ValueError('Need one vector of generating-seed contrasts per claim')
        finite=np.isfinite(values);values=values[finite]
        positive,negative=int(np.sum(values>0)),int(np.sum(values<0));ties=int(np.sum(values==0))
        nonzero=positive+negative
        p=1. if nonzero==0 else float(binomtest(positive,nonzero,p=.5,alternative='greater').pvalue)
        results[name]=dict(p_sign=p,n_seeds=len(values),excluded_seeds=int((~finite).sum()),positive=positive,negative=negative,ties=ties,
            fraction_positive_all=None if not len(values) else positive/len(values),fraction_positive_nonzero=None if not nonzero else positive/nonzero,
            distribution=None if not len(values) else dict(mean=float(values.mean()),minimum=float(values.min()),maximum=float(values.max()),
                standard_deviation=float(values.std()),quantiles=dict(zip(['p05','p25','p50','p75','p95'],map(float,np.quantile(values,[.05,.25,.5,.75,.95]))))),
            estimand='P(positive seed contrast | nonzero)>.5;exact one-sided binomial,not the mean estimand')
    adjusted=holm({name:value['p_sign'] for name,value in results.items()})
    for name,value in results.items():value['p_sign_holm4']=adjusted[name]
    return results


def seed_contrasts(predictions,truth,metadata):
    """Reconstruct exactly the four frozen contrasts from native forecast artifacts."""
    seeds=_validate_compositions(truth,metadata,'condition')
    mixed=np.array([row['condition']=='mixed' for row in metadata]);rows=[row for row,keep in zip(metadata,mixed) if keep]
    group=np.array([row['seed'] for row in metadata]);condition=np.array([row['condition'] for row in metadata]);scale=np.array([row['scale'] for row in metadata])
    errors,phase={},{ }
    for model,prediction in predictions.items():
        loss=np.mean((prediction-truth)**2,axis=1)/scale**2
        errors[model]={c:np.array([loss[(group==s)&(condition==c)].mean() for s in seeds]) for c in ('mixed','pure_phi','pure_psi')}
        for target in ('phi','psi'):phase[model,target]=input_interference(prediction[mixed],truth[mixed],rows,target)['metrics']['other_phase_change_radians']
    contrasts={f'mixed_{target}_untouched_phase_timesfm_minus_bolt':phase['timesfm',target]-phase['bolt',target] for target in ('phi','psi')}
    gap=errors['timesfm']['mixed']-errors['bolt']['mixed'];pure=.5*((errors['timesfm']['pure_phi']-errors['bolt']['pure_phi'])+(errors['timesfm']['pure_psi']-errors['bolt']['pure_psi']))
    contrasts.update(mixed_scaled_mse_timesfm_minus_bolt=gap,model_by_composition_interaction=gap-pure)
    return seeds,contrasts


def run(source,out):
    """Save sensitivity and empirical contrast arrays without changing primary results."""
    out.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'));truth=np.load(source/'corpus.npz')['clean_future']
    predictions={name:np.load(source/f'{name}_predictions.npz')['prediction'] for name in ('bolt','timesfm')}
    seeds,contrasts=seed_contrasts(predictions,truth,metadata);primary=json.loads((source/'results.json').read_text(encoding='utf-8'))['primary']
    sensitivity=sign_sensitivity(contrasts)
    report={name:dict(primary_mean_inference=primary[name],directional_sensitivity=sensitivity[name],unit=load_protocol()['claims'][name]['unit']) for name in contrasts}
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');np.savez_compressed(out/'seed_contrasts.npz',seeds=seeds,**contrasts)
    print(json.dumps(report,indent=2,allow_nan=False),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.source,args.out)
