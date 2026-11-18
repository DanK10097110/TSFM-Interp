"""Independent saved-array audit and isolated non-native end-to-end instrumentation."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import numpy as np
from experiments.feature_search.coherence_study import read, write, sha


def audit(out, results=False):
    """Reconstruct old fits and fresh predictions without calling the fitted estimators."""
    p=read(out/'protocol.json');checks=0;maximum=0.
    def close(a,b):
        """Track exact floating arithmetic disagreement and reject a changed estimand."""
        nonlocal checks,maximum
        maximum=max(maximum,float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
        np.testing.assert_allclose(a,b,atol=2e-11,rtol=2e-11);checks+=1
    for file,digest in p['source_hashes'].items():
        assert sha(file)==digest,file
    for file,digest in p['hashes'].items():
        assert sha(out/file)==digest,file
    old=Path(p['old'])
    for file,digest in p['old_hashes'].items(): assert sha(old/file)==digest,file
    oldraw=np.load(old/'raw_arithmetic.npz');oldgroups=read(old/'groups.json')
    predictors=read(out/'frozen_predictors.json');phase=np.array([[1,0],[0,1],[-1,0],[0,-1]])
    j=np.array([[0,-1],[1,0]]);centers=[128/24,128/19.2]
    for row in predictors:
        f=row['center'];fi=centers.index(f);t=np.arange(32);b=np.stack([np.sin(2*np.pi*f*t/128),np.cos(2*np.pi*f*t/128)],axis=1)
        gs=[g for g in oldgroups if g['frequency']==f]
        ys=[oldraw[f'{row["model"]}_{row["policy"]}_{sid}_{fi}_delta']/gs[sid]['amplitude'] for sid in range(16)]
        odd=sum(np.stack([(y[0]-y[2])/2,(y[1]-y[3])/2],axis=1) for y in ys)/16
        h=np.linalg.solve(b.T@b,b.T@odd)
        scalar=float(np.sum(b*odd)/np.sum(b*b))
        r=float(np.sum((b@j)*odd)/np.sum(b*b))
        expected=dict(zero=np.zeros((2,2)),identity=np.eye(2),scalar_bounded=np.clip(scalar,0,1)*np.eye(2),
                      scalar_unconstrained=scalar*np.eye(2),complex_linear=scalar*np.eye(2)+r*j,full_H=h)
        for name,k in expected.items():
            close(row['matrices'][name],k)
            pred=phase@k.T@b.T
            score=sum(float(np.sum((pred-y)**2)/np.sum((phase@b.T)**2)) for y in ys)/16
            close(row['training_nmse'][name],score)
    sources=read(out/'sources.json');groups=read(out/'groups.json');inputs=np.load(out/'inputs.npy')
    assert len(sources)==16 and len(groups)==96 and inputs.shape==(400,128)
    assert {s['source'] for s in sources}.isdisjoint(p['training_sources'])
    assert len(p['primary_family'])==4 and all(r['policy']=='native' and r['period']==19.2 for r in p['primary_family'])
    for g in groups:
        t=np.arange(-128,0);f=g['frequency'];a=g['amplitude']
        nominal=np.stack([np.sin(2*np.pi*f*t/128+k*np.pi/2)*a for k in range(4)])
        observed=inputs[g['indices']].astype(float)-inputs[g['base']]
        assert np.sqrt(np.mean((nominal-observed)**2))/a<1e-5
    if results:
        raw=np.load(out/'raw_arithmetic.npz');rows=read(out/'source_rows.json')
        for row in rows:
            g=next(g for g in groups if g['source_id']==row['source_id'] and g['center']==row['center'] and g['offset']==row['offset'])
            gi=groups.index(g);key=f'{row["model"]}_{row["policy"]}_{gi}'
            forecast=np.load(out/f'{row["model"]}_{row["policy"]}_forecast.npy')
            delta=forecast[g['indices']].astype(float)-forecast[g['base']].astype(float)
            close(delta,raw[f'{key}_delta'])
            f=g['frequency'];a=g['amplitude'];t=np.arange(32)
            b=np.stack([np.sin(2*np.pi*f*t/128),np.cos(2*np.pi*f*t/128)],axis=1)
            odd=np.stack([(delta[0]-delta[2])/2,(delta[1]-delta[3])/2],axis=1)/a
            h=np.linalg.solve(b.T@b,b.T@odd)
            fitted=next(r for r in predictors if r['model']==row['model'] and r['policy']==row['policy'] and r['center']==row['center'])
            maps={k:np.asarray(v) for k,v in fitted['matrices'].items()};maps['oracle_same_source']=h
            full=maps['full_H'];theta=np.arctan2(full[1,0]-full[0,1],np.trace(full));angle=(f/g['center']-1)*theta
            maps['delay_full_H']=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])@full
            ideal=a*phase@b.T
            for name,k in maps.items():
                pred=a*phase@k.T@b.T
                close(pred,raw[f'{key}_prediction_{name}'])
                close(np.sum((pred-delta)**2)/np.sum(ideal**2),row['losses'][name])
            residual=odd-b@h;even=np.stack([delta[0]+delta[2],delta[1]+delta[3]],axis=1)
            off=np.sum(residual**2)/np.sum(b*b);ev=np.sum(even**2)/(4*a*a*np.sum(b*b))
            close(off,row['decomposition']['off_frequency_nmse_contribution'])
            close(ev,row['decomposition']['even_nmse_contribution'])
            close(off+ev,row['losses']['oracle_same_source'])
        draws=np.load(out/'bootstrap_source_indices.npy');result=read(out/'results.json')
        assert draws.shape==(16384,16)
        for comparison in result['primary']:
            selected=[r for r in rows if r['model']==comparison['model'] and r['policy']=='native' and r['center']==centers[1] and r['offset']==0]
            values=np.array([r['losses'][comparison['baseline']]-r['losses']['full_H'] for r in selected])
            close(values,comparison['source_differences']);close(values.mean(),comparison['mean'])
            close(np.quantile(values[draws].mean(1),[.00625,.99375]),comparison['family95_percentile'])
        for item in read(out/'horizon_decomposition.json'):
            if item['cohort']=='fresh':
                g=next(g for g in groups if g['source']==item['source'] and g['center']==item['center'] and g['offset']==item['offset'])
                delta=raw[f'{item["model"]}_{item["policy"]}_{groups.index(g)}_delta']
            else:
                g=next(g for g in oldgroups if g['source']==item['source'] and g['frequency']==item['center'])
                delta=oldraw[f'{item["model"]}_{item["policy"]}_{g["source_id"]}_{centers.index(item["center"])}_delta']
            lo,hi=item['start'],item['stop'];a=g['amplitude'];t=np.arange(lo,hi);f=item['center']+item['offset']
            b=np.stack([np.sin(2*np.pi*f*t/128),np.cos(2*np.pi*f*t/128)],axis=1);y=delta[:,lo:hi]
            odd=np.stack([(y[0]-y[2])/2,(y[1]-y[3])/2],axis=1)/a
            h=np.linalg.solve(b.T@b,b.T@odd);close(h,item['H'])
            d=np.column_stack([b,np.ones(hi-lo)]);joint=np.linalg.solve(d.T@d,d.T@odd)
            close(joint[:2],item['H_dc_adjusted'])
    record=dict(passed=True,checks=checks,maximum_absolute_discrepancy=maximum,
                results_checked=results,native_loaded=False,source_hashes_verified=True)
    write(out/('audit.json' if results else 'preflight_audit.json'),record)
    return record


def standin(prepared,out):
    """Exercise full analysis on a separate synthetic receiver bank; no model load."""
    from experiments.feature_search.response_phase_transfer import analyze
    out.mkdir(parents=True,exist_ok=False)
    p=read(prepared/'protocol.json')
    for file in ['protocol.json','knowledge.json',*p['hashes']]: shutil.copyfile(prepared/file,out/file)
    groups=read(out/'groups.json');phase=np.array([[1,0],[0,1],[-1,0],[0,-1]])
    execution={}
    for mi,model in enumerate(('bolt','timesfm')):
        hashes={}
        for pi,policy in enumerate(('native','locked')):
            forecasts=np.zeros((400,32),np.float32)
            for g in groups:
                a=g['amplitude'];t=np.arange(32);f=g['frequency'];sid=g['source_id']
                b=np.stack([np.sin(2*np.pi*f*t/128),np.cos(2*np.pi*f*t/128)],axis=1)
                h=np.array([[.4+.01*sid,.2],[-.3,.8+.03*mi+.02*pi]])
                y=a*phase@h.T@b.T
                y[[0,2]]+=a*a*(1+np.cos(t));y[[1,3]]+=.03*a
                forecasts[g['indices']]=y
            name=f'{model}_{policy}_forecast.npy';np.save(out/name,forecasts);hashes[name]=sha(out/name)
        write(out/f'{model}_worker.json',dict(gates={'standin_identity':0.},batch_replay_exact=True,hashes=hashes,
                                            evidence='SYNTHETIC INSTRUMENTATION ONLY; no native worker'))
        execution[model]=dict(valid=True,instrumentation_only=True)
    write(out/'execution.json',execution)
    try:
        analyze(out);record=audit(out,True)
    except Exception as error:
        k=read(out/'knowledge.json');k.update(status='failed_standin_instrumentation',evidence_class='Instrumentation only',
            failed_instrumentation_runs=[dict(error=repr(error))],next_decision='Fix instrumentation before native capture.')
        write(out/'knowledge.json',k);raise
    k=read(out/'knowledge.json');k.update(status='standin_complete_audited',evidence_class='Synthetic receiver instrumentation ONLY; no model findings',
        exact_findings=dict(audit=record,synthetic_receiver=True),next_decision='Native remains subject to coordinator slot grant.')
    write(out/'knowledge.json',k)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['preflight','audit','standin'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--prepared',type=Path)
    args=parser.parse_args()
    print(standin(args.prepared,args.out) if args.command=='standin' else audit(args.out,args.command=='audit'))
