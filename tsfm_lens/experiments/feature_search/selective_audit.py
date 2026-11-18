"""Independent saved-state and forecast arithmetic audit for selective fusion."""
from pathlib import Path
import argparse
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha


def audit(out):
    """Recompute hidden closure, local sensitivity and requested-depth future losses."""
    p=read(out/'protocol.json');rows=read(out/'rows.json');plans=read(out/'plans.json');x=np.load(out/'inputs.npz')['history'].astype(float);y=np.load(out/'targets.npz')['future'];counts=dict(geometry=0,stability=0,losses=0,selectors=0)
    for path,digest in p['sources'].items():
        if sha(path)!=digest:raise ValueError('Source changed')
    for name,digest in p['hashes'].items():
        if sha(out/name)!=digest:raise ValueError('Data changed')
    measurements=read(out/'measurements.json');selected=read(out/'selected_rows.json')
    for model in ['bolt','timesfm']:
        worker=read(out/f'{model}_worker.json');cuts=worker['cuts'];states=np.load(out/f'{model}_states.npz')
        for name,digest in worker['hashes'].items():
            if sha(out/name)!=digest:raise ValueError('Worker artifact changed')
        forecasts=read(out/f'{model}_forecasts.json');lookup={(r['row'],r['method'],r['depth'],r['cut'],r['weight'],r['blend']):np.asarray(r['forecast']) for r in forecasts}
        for r in read(out/f'{model}_geometry.json'):
            i=r['row'];d=r['depth'];c=cuts.index(r['cut']);z=states[f'{i}_zero'][c].astype(float);h=states[f'{i}_full'][c].astype(float);leaves=states[f'{i}_depth{d}'][c].astype(float)
            w=np.ones(d+1)
            if r['weight']=='confidence':w[:-1]=[a['confidence'] for a in plans[i]['details'][:d]]
            if r['weight']=='learned':
                a,b=p['learned_weights'][model][str(d)][r['cut']]['weights'];w[:-1]=a;w[-1]=b
            composed=sum(weight*state for weight,state in zip(w,leaves))+(1-sum(w))*z
            value=np.mean((composed-h)**2)/max(float(np.mean((states[f'{i}_full'][c]-states[f'{i}_zero'][c])**2)),1e-12)
            np.testing.assert_allclose(value,r['closure_nmse'],rtol=1e-9,atol=1e-10);counts['geometry']+=1
        for r in read(out/f'{model}_stability.json'):
            if r['cut']=='forecast':continue
            i=r['row'];c=cuts.index(r['cut']);h=states[f'{i}_full'][c].astype(float);z=states[f'{i}_zero'][c].astype(float);pos,neg=states[f"{i}_{r['direction']}_{r['epsilon']}_{r['policy']}"][c].astype(float)
            raw=states[f"{i}_{r['direction']}_{r['epsilon']}_{r['policy']}"][c];zero32=states[f'{i}_zero'][c]
            pos,neg=(raw-zero32).astype(float);h=(states[f'{i}_full'][c]-zero32).astype(float)
            first=np.sqrt(np.mean(((pos-neg)/2)**2));scale=max(np.sqrt(np.mean(h**2)),1e-12)
            sensitivity=first/scale/r['epsilon'];curvature=np.sqrt(np.mean((pos+neg-2*h)**2))/(2*first+1e-12)
            np.testing.assert_allclose([sensitivity,curvature],[r['sensitivity'],r['curvature_ratio']],rtol=1e-8,atol=1e-9);counts['stability']+=1
        for r in [v for v in measurements if v['model']==model]:
            i=r['row'];method,d,cut,weight,blend=r['config'];actual=min(d,len(plans[i]['levels'])-1)
            identity=(i,'native',0,'none','none',0.) if actual==0 else (i,method,actual,cut,weight,blend)
            pred=lookup[identity];base=lookup[i,'native',0,'none','none',0.]
            error=np.sum((pred-y[i])**2)/32/np.var(x[i]);fidelity=np.sum((pred-base)**2)/32/np.var(x[i])
            np.testing.assert_allclose([error,fidelity],[r['loss'],r['native_fidelity']],rtol=1e-12,atol=1e-12);counts['losses']+=1
        for r in [v for v in selected if v['model']==model]:
            match=next(v for v in measurements if v['model']==model and v['row']==r['row'] and v['config']==r['selected'])
            assert rows[r['row']]['split']=='test';np.testing.assert_allclose(match['loss'],r['loss']);counts['selectors']+=1
    if p['stage']=='replication':
        previous=read(out.parent/'selective_pilot4941000/rows.json');old=read(out.parent/'coherence_replication4932000/rows.json')
        current={r['source'] for r in rows};assert not current&{r['source'] for r in previous};assert not current&{r['source'] for r in old}
    for source in {r['source'] for r in rows}:assert len({r['split'] for r in rows if r['source']==source})==1
    result=dict(counts,passed=True,scope='Replayed saved hidden states and forecasts; forecast-local-sensitivity rows lack raw +/- forecast arrays and are instrumented diagnostics, not independently replayed here',source_sha256=sha(__file__))
    write(out/'independent_audit.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);audit(p.parse_args().out)
