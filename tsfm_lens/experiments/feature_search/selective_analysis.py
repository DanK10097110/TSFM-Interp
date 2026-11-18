"""Held-source-out policies and independent hidden geometry for selective trees.

A good forecast is not treated as reconstruction of the natural hidden state.
FIT future labels select forecast policies; geometry weights use no future labels.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.optimize import minimize
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.coherence_analysis import estimate


def key(r):
    """A complete intervention identity, without source-dependent accepted depth."""
    return (r['method'],r['depth'],r['cut'],r['weight'],r['blend'])


def source_summary(values,metadata,indices):
    """Average dependent cases within a generating seed before inference."""
    groups={}
    for i in indices:groups.setdefault(metadata[i]['source'],[]).append(float(values[i]))
    return estimate([np.mean(v) for v in groups.values()])


def fit_geometry(out):
    """Learn two bounded weights on FIT hidden states only, with equal source mass."""
    rows=read(out/'rows.json');plans=read(out/'plans.json');result={}
    for model in ['bolt','timesfm']:
        states=np.load(out/f'{model}_states.npz');cuts=read(out/f'{model}_worker.json')['cuts'];result[model]={}
        for depth in [1,2,3]:
            result[model][str(depth)]={}
            for c,cut in enumerate(cuts):
                gram=np.eye(2)*.001;rhs=np.ones(2)*.001;n=0
                for i,row in enumerate(rows):
                    if row['split']!='fit' or len(plans[i]['levels'])<=depth:continue
                    h=states[f'{i}_full'][c].astype(float);z=states[f'{i}_zero'][c].astype(float);leaves=states[f'{i}_depth{depth}'][c].astype(float)
                    features=np.stack([(leaves[:-1]-z).sum(0).ravel(),(leaves[-1]-z).ravel()],axis=1);target=(h-z).ravel()
                    scale=max(float(target@target),1e-12)*sum(r['source']==row['source'] for r in rows)
                    gram+=features.T@features/scale;rhs+=features.T@target/scale;n+=1
                fit=minimize(lambda w:float(w@gram@w-2*w@rhs),np.ones(2),jac=lambda w:2*(gram@w-rhs),bounds=[(0,2),(0,2)],method='L-BFGS-B')
                if not fit.success:raise ValueError('Geometry fit failed')
                result[model][str(depth)][cut]=dict(weights=fit.x.tolist(),fit_rows=n,gram_condition=float(np.linalg.cond(gram)))
    write(out/'geometry_weights.json',result);return result


def resolve(config,row,lookup,plans):
    """Apply a requested depth with past-only early-stop fallback; zero means native."""
    method,depth,cut,weight,blend=config
    actual=min(depth,len(plans[row]['levels'])-1)
    if actual==0:return lookup[row,('native',0,'none','none',0.)]
    return lookup[row,(method,actual,cut,weight,blend)]


def choose_config(configs,losses,metadata,fit_ids):
    """Restrict every score used for policy selection to the explicit FIT sources."""
    if any(metadata[i]['split']!='fit' for i in fit_ids):raise ValueError('Selection includes TEST labels')
    return min(configs,key=lambda c:(source_summary(losses[c],metadata,fit_ids)['mean'],int(isinstance(c,tuple) and c[0]!='native'),c[1] if isinstance(c,tuple) else 0,str(c)))


def analyze(out):
    """Score future quality, suffix fidelity, hidden closure and local stability separately."""
    execution=read(out/'execution.json')
    if set(execution)!={'bolt','timesfm'} or not all(r['valid'] for r in execution.values()):raise ValueError('Native run incomplete/invalid')
    rows=read(out/'rows.json');plans=read(out/'plans.json');x=np.load(out/'inputs.npz')['history'];y=np.load(out/'targets.npz')['future'];scale=np.var(x.astype(float),axis=1)
    all_measurements=[];policies=[];curves=[];geometry_summary=[];stability_summary=[];selected_rows=[]
    for model in ['bolt','timesfm']:
        worker=read(out/f'{model}_worker.json')
        for name,digest in worker['hashes'].items():
            if sha(out/name)!=digest:raise ValueError('Native artifact binding changed')
        forecasts=read(out/f'{model}_forecasts.json');lookup={(r['row'],key(r)):r for r in forecasts};cuts=worker['cuts'];weights=['equal','confidence']+(['learned'] if 'learned_weights' in read(out/'protocol.json') else [])
        native=('native',0,'none','none',0.);configurations=[native]
        outputs=[(m,d,'none','none',1.) for m in ['output_free','output_locked','hybrid'] for d in [1,2,3]]
        fusions=[('fusion',d,c,w,b) for d in [1,2,3] for c in cuts for w in weights for b in [.5,1.]]
        configurations+=outputs+fusions;losses={};fidelity={};pred={}
        nativepred=np.asarray([lookup[i,native]['forecast'] for i in range(len(rows))])
        for config in configurations:
            pred[config]=np.asarray([resolve(config,i,lookup,plans)['forecast'] for i in range(len(rows))])
            losses[config]=np.mean((pred[config]-y)**2,axis=1)/scale
            fidelity[config]=np.mean((pred[config]-nativepred)**2,axis=1)/scale
            for i,row in enumerate(rows):all_measurements.append(dict(model=model,row=i,source=row['source'],domain=row['domain'],case=row['case'],split=row['split'],config=list(config),loss=float(losses[config][i]),native_fidelity=float(fidelity[config][i]),actual_depth=min(config[1],len(plans[i]['levels'])-1)))
        geom=read(out/f'{model}_geometry.json');glookup={(r['row'],r['depth'],r['cut'],r['weight']):r['closure_nmse'] for r in geom}
        for domain in sorted({r['domain'] for r in rows}):
            fit_ids=[i for i,r in enumerate(rows) if r['domain']==domain and r['split']=='fit'];test_ids=[i for i,r in enumerate(rows) if r['domain']==domain and r['split']=='test']
            choices={'output_selected':choose_config([native]+outputs,losses,rows,fit_ids),'fusion_selected':choose_config([native]+fusions,losses,rows,fit_ids)}
            closure={c:np.asarray([glookup.get((i,min(c[1],len(plans[i]['levels'])-1),c[2],c[3]),0.)*c[4]**2 for i in range(len(rows))]) for c in fusions}
            choices['geometry_selected']=choose_config([c for c in fusions if c[4]==1.],closure,rows,fit_ids)
            modified={c:v.copy() for c,v in losses.items()}
            for c in modified:modified[c][test_ids]=np.random.default_rng(49).normal(size=len(test_ids))*1e6
            assert choices['fusion_selected']==choose_config([native]+fusions,modified,rows,fit_ids)
            for policy,chosen in choices.items():
                gain=losses[native]-losses[chosen];summary=source_summary(gain,rows,test_ids)
                random_values=[];random_valid=[]
                if chosen[0]=='fusion':
                    rc=('random',)+chosen[1:]
                    for i in test_ids:
                        record=resolve(rc,i,lookup,plans);random_values.append(np.mean((np.asarray(record['forecast'])-y[i])**2)/scale[i]);random_valid.append(record.get('valid',True))
                row=dict(model=model,domain=domain,policy=policy,selected=list(chosen),test_loss=source_summary(losses[chosen],rows,test_ids),gain_vs_native=summary,native_fidelity=source_summary(fidelity[chosen],rows,test_ids))
                if random_values:
                    comparison=np.zeros(len(rows));comparison[test_ids]=np.asarray(random_values)-losses[chosen][test_ids]
                    row.update(random_advantage=source_summary(comparison,rows,test_ids),random_valid_fraction=float(np.mean(random_valid)))
                policies.append(row)
                for i in test_ids:selected_rows.append(dict(model=model,domain=domain,policy=policy,source=rows[i]['source'],case=rows[i]['case'],row=i,selected=list(chosen),loss=float(losses[chosen][i]),native_loss=float(losses[native][i])))
            for config in [native]+outputs:
                curves.append(dict(model=model,domain=domain,config=list(config),**source_summary(losses[config],rows,test_ids)))
            for cut in cuts:
                for depth in [1,2,3]:
                    for weight in weights:
                        ids=[i for i in test_ids if (i,depth,cut,weight) in glookup]
                        if ids:
                            v=np.zeros(len(rows));v[ids]=[glookup[i,depth,cut,weight] for i in ids]
                            geometry_summary.append(dict(model=model,domain=domain,cut=cut,depth=depth,weight=weight,**source_summary(v,rows,ids)))
        stable=read(out/f'{model}_stability.json')
        for domain in sorted({r['domain'] for r in rows}):
            for cut in cuts+['forecast']:
                for direction in ['sinusoid','noise']:
                    for policy in ['native','locked']:
                        for epsilon in [.001,.01,.1]:
                            selected=[r for r in stable if rows[r['row']]['domain']==domain and rows[r['row']]['split']=='test' and r['cut']==cut and r['direction']==direction and r['policy']==policy and r['epsilon']==epsilon]
                            if not selected:continue
                            rec=dict(model=model,domain=domain,cut=cut,direction=direction,policy=policy,epsilon=epsilon)
                            for metric in ['sensitivity','curvature_ratio','relative_change']:
                                groups={}
                                for r in selected:groups.setdefault(rows[r['row']]['source'],[]).append(r[metric])
                                rec[metric]=float(np.mean([np.mean(v) for v in groups.values()]))
                            stability_summary.append(rec)
    write(out/'measurements.json',all_measurements);write(out/'selected_rows.json',selected_rows)
    summary=dict(policies=policies,curves=curves,geometry=geometry_summary,stability=stability_summary,accepted_depths=[dict(row=i,source=r['source'],domain=r['domain'],case=r['case'],split=r['split'],depth=len(plans[i]['levels'])-1) for i,r in enumerate(rows)],source_sha256=sha(__file__))
    cases=[]
    groups={}
    for r in all_measurements:
        if r['split']=='test':groups.setdefault((r['model'],r['domain'],r['case'],tuple(r['config'])),[]).append(r)
    for (model,domain,case,config),group in groups.items():
        cases.append(dict(model=model,domain=domain,case=case,config=list(config),n_sources=len({r['source'] for r in group}),loss=float(np.mean([r['loss'] for r in group])),native_fidelity=float(np.mean([r['native_fidelity'] for r in group]))))
    summary['cases']=cases
    write(out/'analysis.json',summary)
    if read(out/'protocol.json')['stage']=='pilot':fit_geometry(out)
    for p in policies:
        if p['policy']=='fusion_selected' or p['domain']=='synthetic':print(p['model'],p['domain'],p['policy'],p['selected'],'gain',p['gain_vs_native']['mean'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);analyze(p.parse_args().out)
