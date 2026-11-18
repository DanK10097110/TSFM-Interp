"""Staged selective decomposition, causal activation fusion and local stability.

Keeps old experiments immutable. Uses their exact native hooks, common-normalizer
intervention and geometry-preserving random edits. Futures never enter workers.
"""
from pathlib import Path
import argparse,contextlib,os,subprocess,sys,time,resource
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha,real_data,normalization
from experiments.feature_search.selective_tree import decompose,continuation,fuse,local_metrics


def prepare(out,stage,cache):
    """Freeze stage membership and past-only decomposition before native inference."""
    out.mkdir(parents=True,exist_ok=False);n=8 if stage=='pilot' else 32;seedbase=4941000 if stage=='pilot' else 4942000
    histories=[];targets=[];rows=[];plans=[];nodes=[]
    def add(meta,x,y):
        x=np.asarray(x,float);levels,details=decompose(x);plan=[]
        for level in levels:
            ids=[]
            for part in level:ids.append(len(nodes));nodes.append(part)
            plan.append(ids)
        rows.append(dict(meta,index=len(rows)));histories.append(x);targets.append(y);plans.append(dict(levels=plan,details=details))
    for j,seed in enumerate(range(seedbase,seedbase+n)):
        rng=np.random.default_rng(seed);t=np.arange(-128,32);phi,psi=rng.uniform(-np.pi,np.pi,2);noise=rng.normal(0,.05,len(t))
        for case,a,b in [('on_grid_far',5.,17.),('off_grid_far',5.35,17.35),('off_grid_near',5.35,6.35),('chirp',5.35,17.35)]:
            phase=2*np.pi*a*t/128+phi
            if case=='chirp':phase+=2*np.pi*2*(t/128)**2
            value=.5*np.sin(phase)+8*np.sin(2*np.pi*b*t/128+psi)+noise
            add(dict(source=f'syn:{seed}',domain='synthetic',case=case,split='fit' if j<n//2 else 'test',seed=seed),value[:128],value[128:])
    offset,count=(54,2) if stage=='pilot' else (56,8)
    for source,value in real_data(cache,offset,count):add(dict(source,case='real'),value[128:256],value[256:])
    np.savez_compressed(out/'inputs.npz',history=np.asarray(histories,np.float32),nodes=np.asarray(nodes,np.float32));np.savez_compressed(out/'targets.npz',future=np.asarray(targets))
    write(out/'rows.json',rows);write(out/'plans.json',plans)
    from experiments.feature_search import selective_tree,real_background_poc,coherence_study,composition_restoration,homogeneous_normalizer
    files=[Path(__file__)]+[Path(m.__file__) for m in [selective_tree,real_background_poc,coherence_study,composition_restoration,homogeneous_normalizer]]
    checkpoints=read(out.parent/'coherence_replication4932000/protocol.json')['checkpoints']
    protocol=dict(stage=stage,seeds=list(range(seedbase,seedbase+n)),rows=len(rows),real_sources=4*count,real_offset=offset,checkpoints=checkpoints,
       sources={str(p.resolve()):sha(p) for p in files},hashes={f:sha(out/f) for f in ['inputs.npz','targets.npz','rows.json','plans.json']},
       worker_cap=3*2**30,seconds_per_model=1200 if stage=='pilot' else 2400,cpu_threads=1,
       tree='max3 peels; stability>=.65; explained>=.15; prominence>=8; continuous frequencies2..48/128; retain residual and DC',
       layers='block outputs at first,quarter,half depth; role comparable, not architecturally identical',weights=['equal','confidence'],blend=[.5,1.],
       controls='native; independent/shared-normalizer forecast sums; classical tones plus native residual; exact self patch; matched random edit preserving receiver/edit Gram',
       stability='native and full-context-locked normalization; fixed sinusoid and random direction; symmetric RMS radii.001,.01,.1',
       selection='FIT-only per-domain choose method/depth/layer/weight/blend; preserve native candidate and all depth curves; TEST best is oracle only',
       replication_rule='Expand valid pilot if any reproducible split/fusion effect or informative failure boundary; no promotion without source-heldout comparison to native, output sum and random edit',
       inference='Exploratory source-paired bootstrap; no sealed confirmation; synthetic cases averaged within seed; real domains separate',
       provenance='Pilot real ranks54-55 reused from earlier study for instrumentation; replication ranks56-63 fresh. Real pretraining overlap unknown')
    if stage=='replication':protocol['learned_weights']=read(out.parent/'selective_pilot4941000/geometry_weights.json')
    write(out/'protocol.json',protocol);print(dict(stage=stage,rows=len(rows),nodes=len(nodes)),flush=True)


def blocks(native):
    """Native role-matched block references, preserving all token positions."""
    seq=native.model.stacked_xf if native.name=='timesfm' else native.model.encoder.block
    ids=sorted(set([0,len(seq)//4,len(seq)//2]));return {str(i):seq[i] for i in ids}


def capture(native,x,sites):
    """Capture <=4 full-token states from one native prediction call."""
    from tsfm_lens.extraction.hooks import _primary
    if len(x)>4:raise ValueError('Capture batch exceeds common-normalizer contract')
    values={key:[] for key in sites};handles=[]
    for key,module in sites.items():
        def hook(module,args,output,key=key):values[key].append(_primary(output).detach().cpu().numpy().copy())
        handles.append(module.register_forward_hook(hook))
    try:forecast=native.predict(x)[0]
    finally:
        for h in handles:h.remove()
    if any(len(v)!=1 for v in values.values()):raise ValueError('Unexpected hook execution count')
    return forecast,{k:v[0] for k,v in values.items()}


def patched(native,x,module,state):
    """Run the native receiver with one full-token block-output replacement."""
    old=native.layer;native.layer=module
    try:return native.predict(x,replacement=np.asarray(state,np.float32))[0]
    finally:native.layer=old


def worker(out,model):
    """Forecast and intervene using only frozen past inputs, never target futures."""
    from experiments.feature_search.real_background_poc import Native
    from experiments.feature_search.composition_restoration import random_edits
    started=time.monotonic();p=read(out/'protocol.json')
    for path,digest in p['sources'].items():
        if sha(path)!=digest:raise ValueError('Frozen source changed')
    for file in ['inputs.npz','rows.json','plans.json']:
        if sha(out/file)!=p['hashes'][file]:raise ValueError('Input binding changed')
    checkpoint=p['checkpoints'][model]
    for file,digest in checkpoint['weights'].items():
        if sha(Path(checkpoint['path'])/file)!=digest:raise ValueError('Weights changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    native=Native(model,checkpoint['path']);sites=blocks(native);rows=read(out/'rows.json');plans=read(out/'plans.json');inputs=np.load(out/'inputs.npz')
    results=[];geometry=[];stability=[];states={};gate=dict(self_patch=0.,normalizer_identity=0.,repeat=0.,random_invalid=0,random_total=0,closure_max=0.)
    for i,row in enumerate(rows):
        x=inputs['history'][i:i+1];sd=float(x.std());plan=plans[i]
        baseline,h=capture(native,x,sites);states[f'{i}_full']=np.stack([h[k][0] for k in sites])
        repeat=native.predict(x)[0];gate['repeat']=max(gate['repeat'],float(abs(repeat-baseline).max()))
        with normalization(native,x,'background'):normal=native.predict(x)[0]
        gate['normalizer_identity']=max(gate['normalizer_identity'],float(abs(normal-baseline).max()))
        for cut,module in sites.items():gate['self_patch']=max(gate['self_patch'],float(abs(patched(native,x,module,h[cut])-baseline).max()))
        if max(gate['self_patch'],gate['normalizer_identity'],gate['repeat'])!=0:raise ValueError('Exact identity gate failed')
        results.append(dict(row=i,method='native',depth=0,cut='none',weight='none',blend=0.,forecast=baseline[0].tolist()))
        with normalization(native,x,'background'):zero,hzero=capture(native,np.zeros_like(x),sites)
        freezero=native.predict(np.zeros_like(x))[0]
        states[f'{i}_zero']=np.stack([hzero[k][0] for k in sites])
        for depth,ids in enumerate(plan['levels'][1:],1):
            leaves=inputs['nodes'][ids];closure=float(abs(leaves.sum(0)-x[0]).max()/max(1,abs(x).max()));gate['closure_max']=max(gate['closure_max'],closure)
            free=native.predict(leaves)[0];locked=[];leaf_states={k:[] for k in sites}
            for start in range(0,len(leaves),4):
                batch=leaves[start:start+4];anchor=np.repeat(x,len(batch),axis=0)
                with normalization(native,anchor,'background'):forecast,hs=capture(native,batch,sites)
                locked.extend(forecast)
                for k in sites:leaf_states[k].extend(hs[k])
            states[f'{i}_depth{depth}']=np.stack([np.asarray(leaf_states[k]) for k in sites])
            for method,pred in [('output_free',free.sum(0)-depth*freezero[0]),('output_locked',np.sum(locked,0)-depth*zero[0]),('hybrid',continuation(plan['details'],depth)+free[-1])]:
                results.append(dict(row=i,method=method,depth=depth,cut='none',weight='none',blend=1.,forecast=pred.tolist()))
            for cut,module in sites.items():
                variants=[];metas=[]
                for weighting in ['equal','confidence']+(['learned'] if 'learned_weights' in p else []):
                    weights=[1.]*depth if weighting=='equal' else [d['confidence'] for d in plan['details'][:depth]]
                    residual_weight=1.
                    if weighting=='learned':
                        fitted=p['learned_weights'][model][str(depth)][cut]['weights'];weights=[fitted[0]]*depth;residual_weight=fitted[1]
                    proposed=fuse(leaf_states[cut],hzero[cut][0],weights+[residual_weight]);delta=proposed-h[cut][0]
                    denominator=max(float(np.mean((h[cut][0]-hzero[cut][0])**2)),1e-12)
                    geometry.append(dict(row=i,depth=depth,cut=cut,weight=weighting,closure_nmse=float(np.mean(delta**2)/denominator),edit_rms=float(np.sqrt(np.mean(delta**2)))))
                    for blend in [.5,1.]:
                        variants.append(h[cut][0]+blend*delta);metas.append(dict(row=i,method='fusion',depth=depth,cut=cut,weight=weighting,blend=blend))
                    rng=np.random.default_rng(4943000+i*100+depth*10+int(cut))
                    random,invalid,diagnostics=random_edits(h[cut],delta[None],rng,count=1,return_diagnostics=True)
                    gate['random_invalid']+=int(invalid.sum());gate['random_total']+=1
                    for blend in [.5,1.]:
                        variants.append(h[cut][0]+blend*random[0,0]);metas.append(dict(row=i,method='random',depth=depth,cut=cut,weight=weighting,blend=blend,valid=not bool(invalid[0])))
                forecasts=patched(native,np.repeat(x,len(variants),axis=0),module,variants)
                results.extend(dict(meta,forecast=forecast.tolist()) for meta,forecast in zip(metas,forecasts))
        rng=np.random.default_rng(4944000+i);directions={'sinusoid':np.sin(2*np.pi*5.35*np.arange(-128,0)/128+.3),'noise':rng.normal(size=128)}
        for direction,value in directions.items():
            value=value/np.sqrt(np.mean(value**2))*sd
            for epsilon in [.001,.01,.1]:
                query=np.concatenate([x+epsilon*value,x-epsilon*value])
                for policy in ['native','locked']:
                    ctx=contextlib.nullcontext() if policy=='native' else normalization(native,np.repeat(x,2,axis=0),'background')
                    with ctx:pred,hs=capture(native,query,sites)
                    states[f'{i}_{direction}_{epsilon}_{policy}']=np.stack([hs[k] for k in sites])
                    for cut in sites:stability.append(dict(row=i,direction=direction,epsilon=epsilon,policy=policy,cut=cut,**local_metrics(h[cut][0]-hzero[cut][0],hs[cut][0]-hzero[cut][0],hs[cut][1]-hzero[cut][0],epsilon)))
                    stability.append(dict(row=i,direction=direction,epsilon=epsilon,policy=policy,cut='forecast',**local_metrics(baseline[0],pred[0],pred[1],epsilon)))
        if i%4==0:print(model,i,'/',len(rows),time.monotonic()-started,flush=True)
        if time.monotonic()-started>p['seconds_per_model'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>p['worker_cap']:raise RuntimeError('Resource cap')
    if gate['closure_max']>2e-6:raise ValueError('FP32 input closure')
    write(out/f'{model}_forecasts.json',results);write(out/f'{model}_geometry.json',geometry);write(out/f'{model}_stability.json',stability)
    np.savez_compressed(out/f'{model}_states.npz',**states)
    write(out/f'{model}_worker.json',dict(seconds=time.monotonic()-started,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gates=gate,cuts=list(sites),
        hashes={f'{model}_{suffix}':sha(out/f'{model}_{suffix}') for suffix in ['forecasts.json','geometry.json','stability.json','states.npz']}))


def run(out):
    """Serial isolated native workers with explicit time and RSS bounds."""
    p=read(out/'protocol.json');execution={}
    for model in ['bolt','timesfm']:
        if (out/f'{model}_worker.json').exists():raise ValueError('Refuse overwrite')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        start=time.monotonic();peak=0;failure=None
        with (out/f'{model}.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.selective_study','worker','--out',str(out),'--model',model],env=env,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                try:peak=max(peak,int(next(l.split()[1] for l in Path(f'/proc/{proc.pid}/status').read_text().splitlines() if l.startswith('VmRSS:')))*1024)
                except (FileNotFoundError,StopIteration):pass
                if peak>p['worker_cap'] or time.monotonic()-start>p['seconds_per_model']+30:
                    failure='Resource cap';proc.kill();proc.wait();break
                time.sleep(.2)
        execution[model]=dict(valid=proc.returncode==0 and failure is None,returncode=proc.returncode,failure=failure,seconds=time.monotonic()-start,peak_rss=peak)
        write(out/'execution.json',execution);print(model,execution[model],flush=True)
        if not execution[model]['valid']:break


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','worker','run']);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['pilot','replication'],default='pilot');p.add_argument('--cache',type=Path,default=Path.home()/'.cache/huggingface');p.add_argument('--model',choices=['bolt','timesfm']);a=p.parse_args()
    if a.command=='prepare':prepare(a.out,a.stage,a.cache)
    elif a.command=='worker':worker(a.out,a.model)
    else:run(a.out)
