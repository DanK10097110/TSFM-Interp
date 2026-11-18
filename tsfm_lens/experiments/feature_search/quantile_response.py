"""Audit whether median-only response errors hide changes in other native quantiles.

These are marginal quantiles, not coherent sampled forecast paths. Their average
is called a central quantile average, never an exact predictive mean. Existing
pilot backgrounds are reused with a crossed phase grid; no population claim.
"""
from pathlib import Path
from contextlib import nullcontext
import argparse,time,resource
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha,normalization

LEVELS=np.arange(1,10)/10


def policy_context(native,anchor,policy):
    """The shared normalizer helper always locks; native must bypass it entirely."""
    if policy=='native':return nullcontext()
    if policy=='locked':return normalization(native,anchor,'background')
    raise ValueError('Unknown normalization policy')


def summaries(delta, signal):
    """Fixed location/spread summaries plus a clearly labeled oracle quantile floor."""
    delta,signal=np.asarray(delta,float),np.asarray(signal,float)
    if delta.shape!=(32,9) or signal.shape!=(32,):raise ValueError('Expected32x9 quantile shifts')
    energy=np.mean(signal**2)
    error=np.mean((delta-signal[:,None])**2,axis=0)/energy
    median=delta[:,4];average=delta.mean(1);midpoint=(delta[:,0]+delta[:,-1])/2
    return dict(median_nmse=float(error[4]),central_quantile_average_nmse=float(np.mean((average-signal)**2)/energy),
                outer_midpoint_nmse=float(np.mean((midpoint-signal)**2)/energy),
                oracle_best_quantile_nmse=float(error.min()),quantile_nmse=error.tolist(),
                quantile_shift_disagreement=float(np.mean((delta-median[:,None])**2)/energy),
                interval_width_change_energy=float(np.mean((delta[:,-1]-delta[:,0])**2)/energy))


def prepare(source,out):
    """Freeze eight existing backgrounds and cross all with common signal phases."""
    out.mkdir(parents=True,exist_ok=False)
    p=read(source/'protocol.json');old=read(source/'signals.json');inputs=np.load(source/'inputs.npy')
    bases={s['source']:s for s in old}
    rows=[];contexts=[];anchors=[]
    for sourceid,s in bases.items():
        base=inputs[s['base']].copy();index=len(contexts);contexts.append(base);anchors.append(base)
        rows.append(dict(source=sourceid,domain=s['domain'],base=index,kind='base',scale=s['scale']))
        for period in (24.,19.2):
            for amplitude in (.1,.5,2.):
                for phase in np.arange(4)*np.pi/2:
                    added=amplitude*s['scale']*np.sin(2*np.pi*np.arange(-128,0)/period+phase)
                    contexts.append((base+added).astype(np.float32));anchors.append(base)
                    rows.append(dict(source=sourceid,domain=s['domain'],base=index,kind='addition',scale=s['scale'],
                                     period=period,amplitude=amplitude,phase=float(phase)))
    np.savez_compressed(out/'inputs.npz',contexts=np.stack(contexts),anchors=np.stack(anchors))
    write(out/'rows.json',rows)
    protocol=dict(question='Do other marginal quantiles track a component missed by the median?',
        source=str(source),source_protocol_sha256=sha(source/'protocol.json'),checkpoints=p['checkpoints'],
        code_sha256=sha(__file__),helper_hashes={name:sha(Path(__file__).with_name(name+'.py')) for name in ('coherence_study','real_background_poc','homogeneous_normalizer')},
        input_sha256=sha(out/'inputs.npz'),rows_sha256=sha(out/'rows.json'),levels=LEVELS.tolist(),
        design='8 reused independent backgrounds;4 crossed phases,2 periods,3 amplitudes; singleton calls; native/locked normalizer',
        points='Both previous Native point outputs are medians; Bolt predict_quantiles second output misleadingly named mean.',
        primary='Median versus fixed central9-quantile average and(.1+.9)/2 midpoint response NMSE; keep model/policy/period/amplitude separate; equal source weighting.',
        controls='Exact quantile repeat; exact median versus establishedNative singleton path; exact native/locked background identity; uncorrected crossingrate.',
        limits='Reused8 pilot backgrounds; no trained distributional decoder or modelcalibration claim; differences of quantiles are not quantiles of a difference; centralaverage is not an exact mean; rowwise bestquantile is only optimistic ceiling.',
        budget=dict(threads=1,seconds_per_model=600,rss_bytes=3*2**30))
    write(out/'protocol.json',protocol)
    write(out/'knowledge.json',dict(status='prepared_native_held',question=protocol['question'],design=protocol,
          exact_findings=[],next_decision='Acquire native slot, run two models sequentially, then analyze fixed summaries.'))


def native_quantiles(native,x):
    """Read native trained channels without interpolating or sorting quantiles."""
    import torch
    batch=torch.tensor(np.asarray(x,np.float32))
    with torch.inference_mode():
        if native.name=='bolt':
            q,median=native.pipeline.predict_quantiles(batch,prediction_length=32,quantile_levels=LEVELS.tolist())
            return q.numpy(),median.numpy()
        value,_,extra=native.model.decode(32,batch,torch.zeros_like(batch,dtype=torch.bool))
        if extra is not None and extra.numel():raise ValueError('Unexpected autoregressive rollout')
        levels=np.asarray(native.model.config.quantiles,float)
        if not np.allclose(levels,LEVELS,rtol=0,atol=1e-12):raise ValueError('Native quantile mapping changed')
        q=value[:,-1,:32,1:10].numpy();point=value[:,-1,:32,native.model.aridx].numpy()
        if native.model.aridx!=5:raise ValueError('Point channel is not median')
        return q,point


def worker(out,model):
    """Capture exact singleton controls on the same normalization path as each arm."""
    from experiments.feature_search.real_background_poc import Native
    p=read(out/'protocol.json');start=time.monotonic()
    if p['code_sha256']!=sha(__file__):raise ValueError('Source changed')
    for name,digest in p['helper_hashes'].items():
        if sha(Path(__file__).with_name(name+'.py'))!=digest:raise ValueError('Helper changed')
    for name in ('inputs.npz','rows.json'):
        if sha(out/name)!=p['input_sha256' if name=='inputs.npz' else 'rows_sha256']:raise ValueError('Input changed')
    checkpoint=p['checkpoints'][model]
    for name,digest in checkpoint['weights'].items():
        if sha(Path(checkpoint['path'])/name)!=digest:raise ValueError('Checkpoint changed')
    if (out/f'{model}_quantiles.npz').exists():raise ValueError('Refuse overwrite')
    native=Native(model,checkpoint['path']);bank=np.load(out/'inputs.npz');rows=read(out/'rows.json');result={}
    for policy in ('native','locked'):
        values=[]
        for i,(x,anchor) in enumerate(zip(bank['contexts'],bank['anchors'])):
            with policy_context(native,anchor[None],policy):
                q,point=native_quantiles(native,x[None]);repeat,_=native_quantiles(native,x[None]);reference,_=native.predict(x[None])
            if not np.array_equal(q,repeat):raise ValueError('Quantile repeat gate')
            if not np.array_equal(point,q[:,:,4]) or not np.array_equal(point,reference):raise ValueError('Median replay gate')
            values.append(q[0])
            if time.monotonic()-start>p['budget']['seconds_per_model'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>p['budget']['rss_bytes']:
                raise RuntimeError('Resource budget exceeded')
        result[policy]=np.stack(values)
    baseids=[i for i,r in enumerate(rows) if r['kind']=='base']
    if not np.array_equal(result['native'][baseids],result['locked'][baseids]):raise ValueError('Baseline normalization identity gate')
    if not all(np.isfinite(a).all() for a in result.values()):raise ValueError('Nonfinite quantiles')
    np.savez_compressed(out/f'{model}_quantiles.npz',**result)
    write(out/f'{model}_worker.json',dict(seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          gates=dict(repeat=0,median=0,normalization=0),sha256=sha(out/f'{model}_quantiles.npz')))
    k=read(out/'knowledge.json');k.setdefault('completed_models',[]).append(model);k['status']='native_stage_complete';write(out/'knowledge.json',k)


def analyze(out):
    """Publish all fixed summaries; no post-hoc quantile or domain selection."""
    rows=read(out/'rows.json');records=[];aggregates=[]
    for model in ('bolt','timesfm'):
        data=np.load(out/f'{model}_quantiles.npz')
        for policy in ('native','locked'):
            for i,r in enumerate(rows):
                if r['kind']=='base':continue
                signal=r['amplitude']*r['scale']*np.sin(2*np.pi*np.arange(32)/r['period']+r['phase'])
                q=data[policy][i];delta=q.astype(float)-data[policy][r['base']].astype(float)
                records.append(dict(model=model,policy=policy,**r,**summaries(delta,signal),
                    crossing_fraction=float(np.mean(np.diff(q,axis=1)<0))))
    fields=['median_nmse','central_quantile_average_nmse','outer_midpoint_nmse','oracle_best_quantile_nmse',
            'quantile_shift_disagreement','interval_width_change_energy','crossing_fraction']
    for model in ('bolt','timesfm'):
        for policy in ('native','locked'):
            for period in (24.,19.2):
                for amplitude in (.1,.5,2.):
                    selected=[r for r in records if (r['model'],r['policy'],r['period'],r['amplitude'])==(model,policy,period,amplitude)]
                    sources=sorted({r['source'] for r in selected});per_source=[]
                    for source in sources:
                        group=[r for r in selected if r['source']==source]
                        per_source.append(dict(source=source,**{key:float(np.mean([r[key] for r in group])) for key in fields}))
                    aggregates.append(dict(model=model,policy=policy,period=period,amplitude=amplitude,n_sources=len(sources),
                        **{key:float(np.mean([r[key] for r in per_source])) for key in fields},source_scores=per_source))
    write(out/'results.json',records);write(out/'summary.json',aggregates)
    protocol=read(out/'protocol.json')
    write(out/'knowledge.json',dict(status='complete',question=protocol['question'],exact_findings=aggregates,
        controls=protocol['controls'],limitations=protocol['limits'],evidence='Descriptive marginalquantile counterfactual response;8backgrounds',
        next_decision='Check whether fixed alternative location summaries improve consistently before any new distributional interpretation.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','worker','analyze'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--model',choices=['bolt','timesfm'])
    a=p.parse_args()
    if a.command=='prepare':prepare(a.source,a.out)
    elif a.command=='analyze':analyze(a.out)
    else:
        try:worker(a.out,a.model)
        except Exception as error:
            k=read(a.out/'knowledge.json');k.update(status='failed_native',error=repr(error));write(a.out/'knowledge.json',k);raise
