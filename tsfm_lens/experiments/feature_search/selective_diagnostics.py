"""Marginal closure, known-injection fidelity and dominant-branch controls.

Whole-state closure can look good simply because one oscillation dominates. Scale
its error also by the actual non-dominant hidden contribution and keep this
separate from future loss and counterfactual-response accuracy.
"""
from pathlib import Path
import argparse
import numpy as np
from experiments.feature_search.coherence_study import read,write
from experiments.feature_search.selective_analysis import resolve,key


def run(out):
    """Derive independent interpretability axes from saved intervention artifacts."""
    rows=read(out/'rows.json');plans=read(out/'plans.json');protocol=read(out/'protocol.json');analysis=read(out/'analysis.json');x=np.load(out/'inputs.npz')['history'];y=np.load(out/'targets.npz')['future'];geometry=[];responses=[];dominance=[]
    for model in ['bolt','timesfm']:
        states=np.load(out/f'{model}_states.npz');cuts=read(out/f'{model}_worker.json')['cuts'];forecasts=read(out/f'{model}_forecasts.json');lookup={(r['row'],key(r)):r for r in forecasts};native=('native',0,'none','none',0.)
        for r in read(out/f'{model}_geometry.json'):
            i=r['row'];c=cuts.index(r['cut']);d=r['depth'];zero=states[f'{i}_zero'][c].astype(float);full=states[f'{i}_full'][c].astype(float);parts=states[f'{i}_depth{d}'][c].astype(float)
            w=np.ones(d+1)
            if r['weight']=='confidence':w[:-1]=[a['confidence'] for a in plans[i]['details'][:d]]
            if r['weight']=='learned':w[:-1]=protocol['learned_weights'][model][str(d)][r['cut']]['weights'][0];w[-1]=protocol['learned_weights'][model][str(d)][r['cut']]['weights'][1]
            composed=zero+np.sum((parts-zero)*w[:,None,None],axis=0);error=float(np.mean((composed-full)**2));total=float(np.mean((full-zero)**2));remaining=float(np.mean((full-parts[0])**2));identifiable=remaining>1e-10*max(total,1e-12)
            geometry.append(dict(r,model=model,source=rows[i]['source'],domain=rows[i]['domain'],case=rows[i]['case'],split=rows[i]['split'],remaining_fraction=remaining/max(total,1e-12),marginal_closure_nmse=error/remaining if identifiable else None,marginal_identifiable=identifiable))
        for i,row in enumerate(rows):
            if row.get('case')!='real_injected' or row['split']!='test':continue
            b=row['base_index'];expected=row['injected_amplitude']*np.sin(2*np.pi*np.arange(32)/row['injected_period']+row['injected_phase'])
            choices={'native':native}
            for policy in analysis['policies']:
                if policy['model']==model and policy['domain']==row['domain']:choices[policy['policy']]=tuple(policy['selected'])
            for policy,config in choices.items():
                q=np.asarray(resolve(config,i,lookup,plans)['forecast']);base=np.asarray(resolve(config,b,lookup,plans)['forecast']);delta=q-base
                responses.append(dict(model=model,source=row['source'],domain=row['domain'],policy=policy,config=list(config),response_nmse=float(np.mean((delta-expected)**2)/np.mean(expected**2)),gain=float(np.mean(delta*expected)/np.mean(expected**2)),clean_loss=float(np.mean((base-y[b])**2)/np.var(x[b].astype(float))),injected_loss=float(np.mean((q-y[i])**2)/np.var(x[i].astype(float)))))
        if (out/f'{model}_dominant.json').exists():
            for r in read(out/f'{model}_dominant.json'):
                i=r['row'];pred=np.asarray(r['forecast']);base=np.asarray(lookup[i,native]['forecast']);scale=np.var(x[i].astype(float))
                dominance.append(dict(model=model,source=rows[i]['source'],domain=rows[i]['domain'],case=rows[i]['case'],split=rows[i]['split'],cut=r['cut'],closure_nmse=r['closure_nmse'],loss=float(np.mean((pred-y[i])**2)/scale),native_fidelity=float(np.mean((pred-base)**2)/scale)))
    tails=[]
    for model in ['bolt','timesfm']:
        grouped={}
        for r in read(out/f'{model}_stability.json'):
            if rows[r['row']]['split']!='test' or r['cut']=='forecast':continue
            group=(rows[r['row']]['domain'],r['cut'],r['policy'],r['direction'],r['epsilon'])
            grouped.setdefault(group,[]).append(r)
        for (domain,cut,policy,direction,epsilon),values in grouped.items():
            record=dict(model=model,domain=domain,cut=cut,policy=policy,direction=direction,epsilon=epsilon,contexts=len(values),sources=len({rows[r['row']]['source'] for r in values}))
            for metric in ['sensitivity','curvature_ratio','relative_change']:
                a=np.asarray([r[metric] for r in values]);record[metric+'_p95']=float(np.quantile(a,.95));record[metric+'_max']=float(a.max())
            record['curvature_over_half_fraction']=float(np.mean([r['curvature_ratio']>.5 for r in values]));tails.append(record)
    write(out/'local_tail_summary.json',tails)
    write(out/'marginal_geometry.json',geometry);write(out/'injection_responses.json',responses);write(out/'dominance_scores.json',dominance)
    print(dict(geometry=len(geometry),responses=len(responses),dominance=len(dominance)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
