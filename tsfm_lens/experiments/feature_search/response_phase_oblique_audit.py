"""Independent oblique-bank arithmetic and isolated synthetic-receiver preflight."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha


def audit(out,results=False):
    """Verify exact background/predictor reuse and replay phase arithmetic without fitting."""
    p=read(out/'protocol.json');prior=Path(p['prior']);checks=0;maximum=0.
    def close(a,b):
        """Compare independently calculated arrays and preserve the numerical tolerance record."""
        nonlocal checks,maximum
        maximum=max(maximum,float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
        np.testing.assert_allclose(a,b,atol=1e-11,rtol=1e-11);checks+=1
    for path,digest in p['source_hashes'].items():assert sha(path)==digest,path
    for path,digest in p['hashes'].items():assert sha(out/path)==digest,path
    for path,digest in p['prior_hashes'].items():assert sha(prior/path)==digest,path
    assert sha(out/'frozen_predictors.json')==sha(prior/'frozen_predictors.json')
    assert p['seconds_per_model']==300 and p['worker_cap']==3*2**30 and p['cpu_threads']==1
    inputs=np.load(out/'inputs.npy');oldinputs=np.load(prior/'inputs.npy');groups=read(out/'groups.json')
    assert inputs.shape==(144,128) and len(groups)==16
    assert read(out/'sources.json')==read(prior/'sources.json')
    angles=np.pi/8+np.arange(8)*np.pi/4
    for g in groups:
        assert np.array_equal(inputs[g['base']],oldinputs[g['cardinal_base']])
        expected=np.stack([oldinputs[g['cardinal_base']].astype(float)+g['amplitude']*np.sin(2*np.pi*np.arange(-128,0)/19.2+angle) for angle in angles]).astype(np.float32)
        assert np.array_equal(inputs[g['indices']],expected);checks+=2
        assert not any(np.array_equal(inputs[new],oldinputs[old]) for new in g['indices'] for old in g['cardinal_indices'])
    if results:
        rows=read(out/'source_rows.json');saved=np.load(out/'raw_arithmetic.npz');predictors=read(out/'frozen_predictors.json')
        draws=np.load(out/'bootstrap_source_indices.npy');assert draws.shape==(16384,16)
        t=np.arange(32);b=np.stack([np.sin(2*np.pi*t/19.2),np.cos(2*np.pi*t/19.2)],axis=1)
        for row in rows:
            model=row['model'];policy=row['policy'];sid=row['source_id'];g=groups[sid]
            new=np.load(out/f'{model}_{policy}_forecast.npy');old=np.load(prior/f'{model}_{policy}_forecast.npy')
            assert np.array_equal(new[g['base']],old[g['cardinal_base']])
            maps=next(r['matrices'] for r in predictors if r['model']==model and r['policy']==policy and r['center']==128/19.2)
            for cohort,phi,y in [('oblique',angles,new[g['indices']].astype(float)-new[g['base']].astype(float)),
                                 ('cardinal',np.arange(4)*np.pi/2,old[g['cardinal_indices']].astype(float)-old[g['cardinal_base']].astype(float))]:
                c=np.stack([np.cos(phi),np.sin(phi)],axis=1);ideal=g['amplitude']*c@b.T;denominator=np.sum(ideal**2)
                half=len(phi)//2;odd=(y[:half]-y[half:])/2;even=y[:half]+y[half:]
                for name in ('zero','scalar_bounded','complex_linear','full_H'):
                    pred=np.asarray([g['amplitude']*b@np.asarray(maps[name])@v for v in c])
                    error=np.sum((y-pred)**2,axis=1);metric=row['scores'][cohort][name]
                    prefix=f'{model}_{policy}_{sid}_{cohort}_{name}'
                    for suffix,value in [('prediction',pred),('delta',y),('ideal',ideal),('odd',odd),('even',even)]:close(value,saved[f'{prefix}_{suffix}'])
                    close(error.sum()/denominator,metric['nmse']);close(error,metric['per_angle_squared_error'])
                    close(error/np.sum(ideal**2,axis=1),metric['per_angle_nmse'])
                    close(np.sum(even**2)/(2*denominator),metric['even_nmse'])
                    close(2*np.sum((odd-(pred[:half]-pred[half:])/2)**2)/denominator,metric['odd_residual_nmse'])
        result=read(out/'results.json')
        assert result['predictors_refitted'] is False and result['background_parity_exact'] is True
        for item in result['primary']:
            selected=[r for r in rows if r['model']==item['model'] and r['policy']=='native']
            if item['contrast']=='oblique_scalar_minus_fullH':
                values=np.array([r['scores']['oblique']['scalar_bounded']['nmse']-r['scores']['oblique']['full_H']['nmse'] for r in selected])
            else:
                values=np.array([r['scores']['oblique']['full_H']['nmse']-r['scores']['cardinal']['full_H']['nmse'] for r in selected])
            close(values,item['source_differences']);close(values.mean(),item['mean'])
            close(np.quantile(values[draws].mean(1),[.00625,.99375]),item['family4_percentile'])
    record=dict(passed=True,checks=checks,maximum_absolute_discrepancy=maximum,results_checked=results,
                existing_sources=16,new_sources=0,predictors_byte_identical=True,native_loaded=False)
    write(out/('audit.json' if results else 'preflight_audit.json'),record)
    return record


def standin(prepared,out):
    """Synthetic oblique receiver exercises analysis/parity without occupying a native slot."""
    from experiments.feature_search.response_phase_oblique import analyze
    p=read(prepared/'protocol.json');prior=Path(p['prior']);out.mkdir(parents=True,exist_ok=False)
    for name in ['protocol.json','knowledge.json',*p['hashes']]:shutil.copyfile(prepared/name,out/name)
    k=read(out/'knowledge.json');k.update(status='standin_instrumentation_running',evidence_class='Synthetic receiver instrumentation ONLY; not native evidence');write(out/'knowledge.json',k)
    groups=read(out/'groups.json');phi=np.pi/8+np.arange(8)*np.pi/4;t=np.arange(32)
    b=np.stack([np.sin(2*np.pi*t/19.2),np.cos(2*np.pi*t/19.2)],axis=1);c=np.stack([np.cos(phi),np.sin(phi)],axis=1)
    execution={}
    for model in ('bolt','timesfm'):
        hashes={}
        for policy in ('native','locked'):
            old=np.load(prior/f'{model}_{policy}_forecast.npy');forecasts=np.zeros((144,32),np.float32)
            for g in groups:
                base=old[g['cardinal_base']];forecasts[g['base']]=base
                h=np.array([[.4+.01*g['source_id'],.2],[-.3,.8]])
                response=g['amplitude']*c@h.T@b.T
                response+=g['amplitude']*.2*np.stack([np.cos(3*phi),np.sin(3*phi)],axis=1)@b.T
                response+=g['amplitude']**2*np.cos(2*phi)[:,None]*np.ones((1,32))
                forecasts[g['indices']]=base+response
            name=f'{model}_{policy}_forecast.npy';np.save(out/name,forecasts);hashes[name]=sha(out/name)
        write(out/f'{model}_worker.json',dict(gates={'standin_identity':0.},batch_replay_exact=True,hashes=hashes,
            evidence='SYNTHETIC RECEIVER ONLY; no native model'))
        execution[model]=dict(valid=True,instrumentation_only=True)
    write(out/'execution.json',execution)
    try:analyze(out);record=audit(out,True)
    except Exception as error:
        k=read(out/'knowledge.json');k.update(status='failed_standin_instrumentation',failed_instrumentation_runs=[dict(error=repr(error))]);write(out/'knowledge.json',k);raise
    k=read(out/'knowledge.json');k.update(status='standin_complete_audited',exact_findings=dict(synthetic_receiver_only=True,audit=record),
        evidence_class='Synthetic instrumentation ONLY; no new native outcomes',next_decision='Hold real capture for coordinator slot grant.');write(out/'knowledge.json',k)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['preflight','audit','standin'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--prepared',type=Path);args=parser.parse_args()
    print(standin(args.prepared,args.out) if args.command=='standin' else audit(args.out,args.command=='audit'))
