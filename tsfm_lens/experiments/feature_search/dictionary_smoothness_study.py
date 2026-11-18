"""Saved-only first-block dictionary smoothness: one seed, k8/16, no native/fit."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import os
import resource
import subprocess
import sys
import time
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search import dictionary_smoothness_math as math

THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
MEMORY=2*2**30
SECONDS=180


def knowledge(out,phase,status,findings,next_decision):
    """Persistent scientific scope, controls, limitations and all failed attempts."""
    file=out/'knowledge.json';old=read(file) if file.exists() else dict(events=[])
    event=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),phase=phase,status=status,
        question='Can frozen hard-sparse encoders induce chord roughness absent from the saved model-state geometry?',
        design='Saved response_surface_focused4954100 first-block native/locked states;16 sources,768 matched-radius paths; frozen Hume PCA/TopK/nested k8/16 seed4951000; fixed first-k PC and raw identity controls',
        exact_findings=findings,controls=['Actual input-path midpoint versus off-manifold activation chord','FP32/FP64 inference and deterministic repeat',
            'Fixed first-k PC linear projection and raw identity','Resolution proxies and1/10/100 refloor sensitivity','Independent frozen-core replay; source aggregation'],
        limitations=['Activation interpolation is off-manifold, not a newly evaluated native path','Only saved FP32 model states; FP64 changes dictionary arithmetic, not native precision',
            'Frozen residual dictionaries are not FFN neuron axes; no general monosemanticity claim','One seed and two fixed k values; no fitting or outcome-selected grid',
            'Encoding nonlinearity includes hard support selection and learned ReLU; no claim of exclusively TopK causation','Resolution floors are descriptive proxies, not rigorous forward-error bounds'],
        evidence_class='saved-array methodological geometry',next_decision=next_decision)
    old.update(event);old['events'].append(event);write(file,old)


def prepare(out):
    """Bind only completed immutable captures/fit artifacts; never touch active Hume stages."""
    if out.exists():
        raise FileExistsError('Fresh smoothness run required')
    out.mkdir(parents=True)
    try:
        response=out.parent/'response_surface_focused4954100'
        dictionaries=out.parent/'dictionary_pilot4951000'/'recovery_exit_monitor'
        rp=read(response/'protocol.json');dp=read(dictionaries/'protocol.json')
        capture_source=Path(dp['source']);cp=read(capture_source/'protocol.json')
        if cp['checkpoints']!=rp['checkpoints']:
            raise ValueError('Dictionary/model checkpoint metadata mismatch')
        for name in ('paths.json','sources.json','nodes.json','inputs.npy'):
            if sha(response/name)!=rp['hashes'][name]:
                raise ValueError('Response design binding changed')
        paths=read(response/'paths.json');sources=read(response/'sources.json');nodes=read(response/'nodes.json')
        if len(sources)!=16 or len(paths)!=768 or len({s['source'] for s in sources})!=16:
            raise ValueError('Expected16-source768-path design')
        bindings={str(response/name):sha(response/name) for name in ('protocol.json','paths.json','sources.json','nodes.json','inputs.npy','completion_manifest.json')}
        bindings[str(dictionaries/'protocol.json')]=sha(dictionaries/'protocol.json')
        bindings[str(capture_source/'protocol.json')]=sha(capture_source/'protocol.json')
        inventory={};artifacts=[]
        for model,tokens,width in (('bolt',9,512),('timesfm',4,1280)):
            worker=read(response/f'{model}_worker.json')
            if not worker['batch_replay_exact'] or any(v!=0 for v in worker['gates'].values()):
                raise ValueError('Response native capture gates failed')
            bindings[str(response/f'{model}_worker.json')]=sha(response/f'{model}_worker.json')
            for policy in ('native','locked'):
                file=response/f'{model}_{policy}_hidden_0.npy';h=np.load(file,mmap_mode='r')
                if h.shape!=(len(nodes),tokens,width) or h.dtype!=np.float32 or sha(file)!=worker['hashes'][file.name]:
                    raise ValueError('First-block state geometry/hash mismatch')
                bindings[str(file)]=sha(file)
            for method in ('pca','topk','nested'):
                for k in (8,16):
                    fit=dictionaries/f'fit_{model}_cut0_{method}_k{k}_s4951000';result=read(fit/'results.json')
                    if result['protocol_sha256']!=sha(dictionaries/'protocol.json') or result['dictionary_sha256']!=sha(fit/'dictionary.npz'):
                        raise ValueError('Frozen dictionary provenance mismatch')
                    if (result['model'],result['cut'],result['method'],result['k'],result['seed'])!=(model,'0',method,k,4951000):
                        raise ValueError('Wrong dictionary identity')
                    a=dict(np.load(fit/'dictionary.npz',allow_pickle=False))
                    expected={'center':(width,),'basis':(width,width)} if method=='pca' else {'center':(width,),'W_enc':(width,width),'W_dec':(width,width),'b_enc':(width,),'b_dec':(width,)}
                    if any(a[key].shape!=shape for key,shape in expected.items()) or int(a['k'])!=k or float(a['scale'])<=0:
                        raise ValueError('Dictionary weight shape/scaler mismatch')
                    if not all(np.isfinite(v).all() for v in a.values()):
                        raise ValueError('Nonfinite dictionary weights')
                    for name in ('dictionary.npz','results.json'):
                        bindings[str(fit/name)]=sha(fit/name)
                    artifacts.append(dict(model=model,method=method,k=k,path=str(fit/'dictionary.npz'),seed=4951000))
            inventory[model]=dict(tokens=tokens,width=width)
            pc=[dict(np.load(dictionaries/f'fit_{model}_cut0_pca_k{k}_s4951000/dictionary.npz')) for k in (8,16)]
            if not all(np.array_equal(pc[0][key],pc[1][key]) for key in ('basis','center','scale')):
                raise ValueError('PCA k8/16 bases differ; cached projection shortcut invalid')
        for key in ('dictionary_core.py','dictionary_pilot.py'):
            file=Path(__file__).with_name(key)
            if sha(file)!=dp['code_hashes'][str(file.resolve())]:
                raise ValueError('Frozen dictionary implementation changed')
            bindings[str(file.resolve())]=sha(file)
        from experiments.feature_search import dictionary_smoothness_audit
        code=[Path(__file__),Path(math.__file__),Path(dictionary_smoothness_audit.__file__)]
        protocol=dict(version=1,run=out.name,response=str(response),dictionaries=str(dictionaries),inventory=inventory,artifacts=artifacts,
            models=['bolt','timesfm'],policies=['native','locked'],cut='0',seed=4951000,k=[8,16],source_count=16,paths=768,
            methods=['identity','pca_fixed','pca_adaptive','topk','nested'],
            PCA_distinction='pca_fixed uses the first-k PC columns linearly; pca_adaptive chooses signed magnitude TopK coefficients separately for each state',
            map='g(h)=frozen decoder(frozen encoder((h-center)/scale))*scale+center; no fitting or learned parameter changes',
            actual='g(h_plus)+g(h_minus)-2*g(h_actual_input_midpoint), compared to raw model-state chord',
            isolated='g(h_plus)+g(h_minus)-2*g((h_plus+h_minus)/2); activation interpolation is off-manifold, no new native path',
            decomposition='Actual decoded chord = isolated dictionary chord +2*(g(activation midpoint)-g(actual midpoint)); retain vector closure',
            precision='Main FP64 dictionaries applied to saved FP32 states; repeat and FP32 inference controls; FP32/FP64 support differences reported',
            curvature='RMS chord divided by source background hidden RMS and input-radius squared; absolute native-unit RMS also saved',
            resolution='FP32-epsilon weighted endpoint/midpoint hidden RMS proxy; ratios refloored at1/10/100 times proxy. Separate observed FP32-vs-FP64 dictionary discrepancies.',
            statistics='Average all dependent phase/frequency paths within each source and axis/radius before summaries; n16, no path/token pseudoreplication, no significance testing',
            independent_replay='First registered path per source, all36 arms, FP32 and FP64, replay canonical frozen dictionary_core with dense decode; no training entrypoints',
            runtime=dict(threads=1,resident_bytes=MEMORY,seconds=SECONDS,includes='all saved-only primary computation, independent replay, aggregation; no native or fit'),
            bindings=bindings,code_hashes={str(file.resolve()):sha(file) for file in code})
        write(out/'protocol.json',protocol)
    except Exception as exc:
        knowledge(out,'prepare','failed',dict(error_type=type(exc).__name__,error=str(exc),native_calls=0,fit_calls=0),'Report mismatch; do not touch frozen inputs.')
        raise
    knowledge(out,'prepare','complete',dict(protocol_sha256=sha(out/'protocol.json'),inventory=inventory,artifacts=12,native_calls=0,fit_calls=0),
        'Run only monitored saved-array analysis after fixtures/plants; no native slot needed.')


def worker(out):
    """One bounded saved-only process; no native loader or dictionary fitter."""
    if os.environ.get('DICTIONARY_SMOOTHNESS_MONITORED')!='1' or any(os.environ.get(k)!='1' for k in THREADS):
        raise RuntimeError('Requires monitored one-thread saved-only process')
    started=time.monotonic();p=read(out/'protocol.json')
    for path,digest in {**p['bindings'],**p['code_hashes']}.items():
        if sha(path)!=digest:
            raise ValueError('Frozen smoothness binding changed: '+path)
    response=Path(p['response']);paths=read(response/'paths.json')
    source_ids=[r['source'] for r in paths];sample=[source_ids.index(s) for s in sorted(set(source_ids))]
    radii=np.asarray([r['actual_radius'] for r in paths])
    results=[];audits=[]
    from experiments.feature_search.dictionary_smoothness_audit import replay
    for model in p['models']:
        for policy in p['policies']:
            native=np.load(response/f'{model}_{policy}_hidden_0.npy',mmap_mode='r')
            h=np.stack([native[[r[key] for r in paths]].astype(float) for key in ('minus','plus','midpoint')])
            h=np.concatenate([h,((h[0]+h[1])/2)[None]],axis=0)
            reference=math.rms(native[[r['base'] for r in paths]],(-2,-1))
            if np.any(reference<=1e-12) or not np.isfinite(h).all():
                raise ValueError('Invalid native state reference')
            cases=[('identity',0,None)]+[(method,k,next(a for a in p['artifacts'] if a['model']==model and a['k']==k and a['method']==('pca' if method.startswith('pca') else method)))
                for method in ('pca_fixed','pca_adaptive','topk','nested') for k in (8,16)]
            for method,k,item in cases:
                if time.monotonic()-started>SECONDS or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>MEMORY:
                    raise RuntimeError('Saved-only compute cap exceeded')
                artifact=dict(np.load(item['path'])) if item else None
                g64,s64=math.evaluate(h,artifact,method,np.float64)
                m64,actual64,iso64=math.measure(h,g64,s64,reference,radii)
                g32,s32=math.evaluate(h,artifact,method,np.float32)
                m32,actual32,iso32=math.measure(h.astype(np.float32),g32,s32,reference,radii)
                metrics=dict(m64)
                metrics['actual_chord_FP32_FP64_RMS']=math.rms(actual32-actual64,(-2,-1))
                metrics['isolated_chord_FP32_FP64_RMS']=math.rms(iso32-iso64,(-2,-1))
                metrics['isolated_resolved_vs_precision']=(m64['isolated_dictionary_chord_RMS']>10*np.maximum(m64['state_FP32_resolution_proxy'],metrics['isolated_chord_FP32_FP64_RMS'])).astype(float)
                metrics['support_FP32_FP64_disagreement']=np.zeros(len(paths)) if s64 is None else np.mean(np.any(s64!=s32,axis=-1),axis=(0,2))
                for key in ('decoded_actual_chord_RMS','isolated_dictionary_chord_RMS','chord_support_turnover'):
                    metrics[key+'_FP32']=m32[key]
                name=f'{model}_{policy}_{method}_k{k}'
                audit=replay(h[:,sample],artifact,method,g64[:,sample],g32[:,sample],
                    None if s64 is None else s64[:,sample],None if s32 is None else s32[:,sample])
                audit.update(model=model,policy=policy,method=method,k=k,sampled_sources=[paths[i]['source'] for i in sample])
                audits.append(audit)
                np.savez_compressed(out/f'{name}_metrics.npz',**metrics)
                results.append(dict(model=model,policy=policy,method=method,k=k,metrics_file=f'{name}_metrics.npz',
                    sha256=sha(out/f'{name}_metrics.npz'),summaries=math.aggregate(metrics,paths)))
                print(dict(model=model,policy=policy,method=method,k=k,elapsed=time.monotonic()-started),flush=True)
                del g64,g32,s64,s32,actual64,actual32,iso64,iso32
    write(out/'results.json',dict(arms=results,native_calls=0,fit_calls=0,source_count=16))
    write(out/'independent_replay.json',dict(valid=True,arms=audits,scope=p['independent_replay']))
    write(out/'worker.json',dict(valid=True,seconds=time.monotonic()-started,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        native_calls=0,fit_calls=0,arms=len(results),protocol_sha256=sha(out/'protocol.json'),
        results_sha256=sha(out/'results.json'),replay_sha256=sha(out/'independent_replay.json')))


def run(out):
    """External wall/RSS monitor bounds saved-only computation, including replay."""
    if (out/'execution.json').exists():
        raise FileExistsError('Refuse saved-only rerun')
    started=time.monotonic();peak=0;reason=None
    env=dict(os.environ,DICTIONARY_SMOOTHNESS_MONITORED='1',CUDA_VISIBLE_DEVICES='',**{k:'1' for k in THREADS})
    try:
        with (out/'worker.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.dictionary_smoothness_study','worker','--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    try:
                        lines=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                        peak=max(peak,int(next(v.split()[1] for v in lines if v.startswith('VmRSS:')))*1024)
                    except (FileNotFoundError,StopIteration):
                        pass
                    if peak+resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>MEMORY or time.monotonic()-started>SECONDS:
                        reason='Saved-only180s/2GiB cap exceeded';break
                    time.sleep(.1)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill();proc.wait()
        worker_record=read(out/'worker.json') if (out/'worker.json').exists() else {}
        result=dict(valid=proc.returncode==0 and reason is None and worker_record.get('valid',False),returncode=proc.returncode,
            seconds=time.monotonic()-started,monitored_worker_peak=peak,reason=reason,native_calls=0,fit_calls=0)
        write(out/'execution.json',result)
        if not result['valid']:
            raise RuntimeError('Saved-only smoothness computation failed: '+str(result))
    except Exception as exc:
        knowledge(out,'compute','failed',dict(error_type=type(exc).__name__,error=str(exc),native_calls=0,fit_calls=0),
            'Preserve failed instrumentation and partial outputs; do not claim uncomputed results.')
        raise
    knowledge(out,'compute','complete',dict(execution=result,worker=worker_record),
        'Audit source aggregation and distill results with actual/off-manifold distinction; no native or fitting follow-up.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','worker','run'])
    parser.add_argument('--out',type=Path,default=Path('runs/dictionary_smoothness4965000'))
    args=parser.parse_args();globals()[args.command](args.out)
