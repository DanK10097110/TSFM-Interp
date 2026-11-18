"""Focused v2: immutable weak-signal subset, bounded capture, explicit slot-grant gate.

Sixteen inference-fresh sources retain crossed phases and measured input radii.
The full32-source plan stays parked. Amplitude paths share their analytic center;
frequency paths retain separate chord midpoints. FP32 closure is always retained.
"""
from __future__ import annotations
import argparse
import contextlib
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from experiments.feature_search.coherence_study import read, write, sha, normalization
from experiments.feature_search.selective_study import capture, patched
from experiments.feature_search.response_surface_metrics import rms


def focused_subset(inputs, nodes, paths, interactions, sources, count=16):
    """Select source order prospectively; retain original-bank IDs and FP32 input residuals."""
    if count not in (4,16,32) or len(sources)<count: raise ValueError('Need balanced source count4/16/32')
    if sorted(s['background_type'] for s in sources[:count])!=sorted(list(range(4))*(count//4)):
        raise ValueError('Unbalanced source membership')
    selected=[];pairs=[];signals={};ids=set()
    for row in paths:
        if row['source_id']>=count or row['amplitude']!=.1: continue
        r=dict(row)
        r['original_midpoint']=r['midpoint']
        if r['axis']=='amplitude':r['midpoint']=r['zero']
        r['midpoint_policy']='analytic_amplitude_center' if r['axis']=='amplitude' else 'explicit_frequency_chord'
        r['actual_radius']=rms((inputs[r['plus']].astype(float)-inputs[r['minus']])/2)/r['input_scale']
        if abs(r['actual_radius']-r['radius'])>1e-7:raise ValueError('Actual RMS radius guard failed')
        r['chord_input_rounding']=rms(inputs[r['plus']].astype(float)+inputs[r['minus']]-2*inputs[r['midpoint']].astype(float))/r['input_scale']
        if r['chord_input_rounding']>2e-7: raise ValueError('FP32 input chord residual exceeds frozen gate')
        selected.append(r);ids.update(r[k] for k in ('base','minus','zero','plus','midpoint'))
        key=(r['source_id'],r['phase_id'],r['frequency'])
        signals[key]={k:r[k] for k in ('source','source_id','base','zero','phase_id','phase','frequency','amplitude','input_scale')}
    for row in interactions:
        if row['source_id']<count and row['phase_a']==row['phase_b']:
            for key in ('a','b'):
                actual=rms(inputs[row[key]].astype(float)-inputs[row['base']])/row['input_scale']
                if abs(actual-row['radius'])>1e-7:raise ValueError('Interaction RMS radius guard failed')
            pairs.append(dict(row));ids.update(row[k] for k in ('base','a','b','ab'))
    old_ids=sorted(ids);mapping={old:new for new,old in enumerate(old_ids)}
    bank=np.asarray(inputs[old_ids],np.float32)
    remapped=[]
    for old in old_ids:
        r=dict(nodes[old],original_index=old,index=mapping[old])
        if 'root' in r:r['root']=mapping[r['root']]
        remapped.append(r)
    for records,keys in [(selected,('base','minus','zero','plus','midpoint')),(pairs,('base','a','b','ab')),
                         (signals.values(),('base','zero'))]:
        for r in records:
            for k in keys:r[k]=mapping[r[k]]
    if len(selected)!=count*48 or len(pairs)!=count*12 or len(signals)!=count*8:
        raise ValueError('Incomplete source-phase-frequency-radius design')
    return bank,remapped,selected,pairs,list(signals.values()),sources[:count]


def budget(nodes, pilot, pilot_nodes=848):
    """Conservative whole-runtime scaling, retaining startup/gate overhead and a10% buffer."""
    return {model:dict(pilot_seconds=r['seconds'],pilot_nodes=pilot_nodes,
                       projected_seconds=r['seconds']*nodes/pilot_nodes,
                       buffered_seconds=r['seconds']*nodes/pilot_nodes*1.1,
                       fits600=r['seconds']*nodes/pilot_nodes*1.1<=600)
            for model,r in pilot.items()}


def prepare(out, full, pilot):
    """Freeze the low-compute subset, implementation and statistics before any capture."""
    from experiments.feature_search import response_surface_focused_analysis
    fp=read(full/'protocol.json')
    for name,digest in fp['hashes'].items():
        if sha(full/name)!=digest:raise ValueError('Parked design changed')
    if list(full.glob('*_forecast.npy')) or list(full.glob('*_worker.json')):
        raise ValueError('Subset must be declared before parked-plan model outcomes')
    values=focused_subset(np.load(full/'inputs.npy',mmap_mode='r'),read(full/'nodes.json'),read(full/'paths.json'),
                          read(full/'interactions.json'),read(full/'sources.json'))
    bank,nodes,paths,pairs,signals,sources=values
    estimate=budget(len(bank),read(pilot/'execution.json'))
    if not all(r['fits600'] for r in estimate.values()):raise ValueError('Prespecified pilot-scaled budget exceeded')
    out.mkdir(parents=True,exist_ok=False);np.save(out/'inputs.npy',bank)
    for name,records in [('nodes',nodes),('paths',paths),('interactions',pairs),('signals',signals),('sources',sources)]:
        write(out/f'{name}.json',records)
    root=Path(__file__).resolve().parents[2]
    files=[Path(__file__),Path(response_surface_focused_analysis.__file__),root/'tests/test_response_surface_focused.py',
           root/'experiments/feature_search/response_surface_focused_plants.py']
    files += [root/'experiments/feature_search'/f'{name}.py' for name in
              ['selective_study','coherence_study','real_background_poc','homogeneous_normalizer','response_surface_metrics']]
    files.append(root/'tsfm_lens/extraction/hooks.py')
    protocol=dict(stage='focused_v2_prepared_held',question='Does weak-tone local curvature shrink at matched input radii across independent sources and crossed phases, and what interactions survive normalization control?',
        source_count=16,seeds=[s['seed'] for s in sources],source_grouping='16 independent backgrounds,4 per mixture type. All4 phases, frequencies, radii and policies remain dependent within source.',
        phases=fp['phases'],centers=fp['centers'],amplitudes=[.1],radii=fp['radii'],axes=['frequency','amplitude'],
        nodes=len(bank),local_paths=len(paths),interaction_quartets=len(pairs),signals=len(signals),context=128,horizon=32,
        subset_rule='First16 frozen sources4954000..4954015, balanced4types. No model outcomes inspected. Only amplitude.1 and diagonal4 phase pairs; no all16 interaction phase pairs.',
        midpoint='Frequency: explicit chord midpoint. Amplitude: shared analytic center, avoiding384 nearly duplicate midpoint calls. Quantized chord residual is recorded and must be<=2e-7 background SD; it is not assumed exactly zero.',
        capture='Native and background-locked normalizer/inverse on same bank, first/half block outputs only; all token positions FP32 plus32-step forecasts.',
        cuts={'bolt':['0','3'],'timesfm':['0','10']},
        controls='Exact native repeats, normalizer self-identity in forecasts AND hidden, self-patch, same-batch replay; finite checks, checkpoint/code/bank hashes; FP64 arithmetic from saved arrays.',
        numerical_resolution='Record output FP32-ULP rounding proxy plus measured input closure times measured local sensitivity.10x proxy flag is descriptive, not a rigorous forward-error/Lipschitz bound or an exclusion rule. Keep all rows and sources.',
        statistics='Source-average over dependent phase/frequency cells before reporting convergence. Paired source curvature decrease and sensitivity ratio; descriptive means/medians/extrema and source counts, no p-values or population confirmation. Preserve by-type and per-source results.',
        known_response='Incremental forecast versus known continued injected tone at the two centers, separate from hidden curvature and interaction. No real future observations.',
        resource_budget=estimate,cpu_threads=1,worker_cap=3*2**30,seconds_per_model=600,
        rejected32_budget=budget(2*len(bank),read(pilot/'execution.json')),
        execution='One native model process at a time; slot release required. Parent enforces wall/RSS caps and persists failed attempts. No resume into partial outputs.',
        capture_hold=True,grant_contract='No grant created at prepare. run AND worker require external JSON with this protocol_sha256, explicit_slot_release=true, slot1 or2, and model in models. Supply only after coordinator grant.',
        full_plan='Parked and unlaunched; this new run is a separate immutable subset/addendum.',
        prior_full=str(full.resolve()),prior_full_protocol_sha256=sha(full/'protocol.json'),
        budget_pilot=str(pilot.resolve()),budget_execution_sha256=sha(pilot/'execution.json'),
        checkpoints=fp['checkpoints'],source_hashes={str(f.resolve()):sha(f) for f in files},
        hashes={f:sha(out/f) for f in ['inputs.npy','nodes.json','paths.json','interactions.json','signals.json','sources.json']})
    write(out/'protocol.json',protocol)
    write(out/'knowledge.json',dict(status='prepared_native_capture_held',question=protocol['question'],design=protocol,
        exact_findings=dict(prepared_nodes=len(bank),local_paths=len(paths),interaction_quartets=len(pairs),
                            budget=estimate,max_chord_input_residual=max(r['chord_input_rounding'] for r in paths),model_results=None),
        controls=protocol['controls'],limitations='Synthetic weak-amplitude study only; diagonal interaction phases do not represent arbitrary relative phase. Shared amplitude midpoint retains quantified FP32 closure residue.',
        evidence_class='Frozen prospective design; no native outcomes',failed_instrumentation_runs=[],
        next_decision='HOLD. Coordinator must release a slot and supply a protocol-bound grant before capture.'))
    print(dict(run=str(out.resolve()),nodes=len(bank),budget=estimate,native_capture='HELD'),flush=True)


def validate(out, model=None):
    """Check all frozen code, arrays and optional checkpoint bytes before native loading."""
    p=read(out/'protocol.json')
    for file,digest in p['source_hashes'].items():
        if sha(file)!=digest:raise ValueError(f'Frozen source changed {file}')
    for file,digest in p['hashes'].items():
        if sha(out/file)!=digest:raise ValueError(f'Bound input changed {file}')
    if model:
        cp=p['checkpoints'][model]
        for file,digest in cp['weights'].items():
            if sha(Path(cp['path'])/file)!=digest:raise ValueError('Checkpoint changed')
    return p


def require_grant(out, grant, model):
    """Fail before importing/loading a native model if no explicit, matching slot release exists."""
    if grant is None or not Path(grant).is_file():raise PermissionError('Native capture HELD: no coordinator slot grant')
    record=read(grant)
    if record.get('protocol_sha256')!=sha(out/'protocol.json') or record.get('explicit_slot_release') is not True:
        raise PermissionError('Native capture HELD: grant does not bind this protocol and explicit release')
    if record.get('slot') not in (1,2) or model not in record.get('models',[]):
        raise PermissionError('Native capture HELD: slot/model not granted')
    return record


def sites_for(native):
    """First and half native blocks, preserving every token."""
    seq=native.model.encoder.block if native.name=='bolt' else native.model.stacked_xf
    return {str(i):seq[i] for i in (0,len(seq)//2)}


def capture_bank(out, native, p, inputs, nodes):
    """Capture frozen bank; injectable small deterministic receiver supports instrument tests."""
    started=time.monotonic();sites=sites_for(native);model=native.name
    if list(sites)!=p['cuts'][model]:raise ValueError('Native block identities changed')
    gates={k:0. for k in ['repeat_forecast','repeat_hidden','normalizer_forecast','normalizer_hidden','self_patch']}
    forecasts={};hidden={};artifacts=[]
    for policy in ('native','locked'):
        name=f'{model}_{policy}_forecast.npy';artifacts.append(name)
        forecasts[policy]=np.lib.format.open_memmap(out/name,mode='w+',dtype=np.float32,shape=(len(nodes),32))
    sources=sorted(set(r['source_id'] for r in nodes))
    for source in sources:
        ids=[r['index'] for r in nodes if r['source_id']==source]
        base=next(r['index'] for r in nodes if r['source_id']==source and r['kind']=='background')
        x=inputs[base:base+1].copy();expected,hx=capture(native,x,sites);again,ha=capture(native,x,sites)
        with normalization(native,x,'background'):locked,hl=capture(native,x,sites)
        gates['repeat_forecast']=max(gates['repeat_forecast'],float(abs(expected-again).max()))
        gates['normalizer_forecast']=max(gates['normalizer_forecast'],float(abs(expected-locked).max()))
        for cut,module in sites.items():
            gates['repeat_hidden']=max(gates['repeat_hidden'],float(abs(hx[cut]-ha[cut]).max()))
            gates['normalizer_hidden']=max(gates['normalizer_hidden'],float(abs(hx[cut]-hl[cut]).max()))
            gates['self_patch']=max(gates['self_patch'],float(abs(patched(native,x,module,hx[cut])-expected).max()))
        if any(gates.values()):raise ValueError(f'Exact identity failed {gates}')
        if not hidden:
            for policy in ('native','locked'):
                hidden[policy]={}
                for cut in sites:
                    name=f'{model}_{policy}_hidden_{cut}.npy';artifacts.append(name)
                    hidden[policy][cut]=np.lib.format.open_memmap(out/name,mode='w+',dtype=np.float32,shape=(len(nodes),)+hx[cut].shape[1:])
        for start in range(0,len(ids),4):
            batch=ids[start:start+4];query=inputs[batch].copy();anchor=np.repeat(x,len(batch),0)
            for policy in ('native','locked'):
                ctx=contextlib.nullcontext() if policy=='native' else normalization(native,anchor,'background')
                with ctx:pred,hs=capture(native,query,sites)
                if not np.isfinite(pred).all() or any(not np.isfinite(v).all() for v in hs.values()):raise ValueError('Nonfinite native output')
                forecasts[policy][batch]=pred
                for cut in sites:hidden[policy][cut][batch]=hs[cut]
            if time.monotonic()-started>p['seconds_per_model']:raise RuntimeError('Worker time cap')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>p['worker_cap']:raise RuntimeError('Worker RSS cap')
        batch=ids[:4];anchor=np.repeat(x,len(batch),0)
        for policy in ('native','locked'):
            ctx=contextlib.nullcontext() if policy=='native' else normalization(native,anchor,'background')
            with ctx:pred,hs=capture(native,inputs[batch].copy(),sites)
            if not np.array_equal(pred,forecasts[policy][batch]):raise ValueError('Same-batch forecast replay failed')
            for cut in sites:
                if not np.array_equal(hs[cut],hidden[policy][cut][batch]):raise ValueError('Same-batch hidden replay failed')
        print(dict(model=model,source=source,completed_source_count=sources.index(source)+1,total_sources=len(sources),seconds=time.monotonic()-started),flush=True)
    for a in list(forecasts.values())+[a for hs in hidden.values() for a in hs.values()]:a.flush()
    return dict(cuts=list(sites),gates=gates,batch_replay_exact=True,seconds=time.monotonic()-started,
                peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,hashes={name:sha(out/name) for name in artifacts})


def worker(out,model,grant):
    """Only granted, validated native workers may instantiate cached model weights."""
    require_grant(out,grant,model);p=validate(out,model)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    if any(out.glob(f'{model}_*.npy')):raise ValueError('Refuse overwrite partial native artifacts')
    from experiments.feature_search.real_background_poc import Native
    native=Native(model,p['checkpoints'][model]['path'])
    result=capture_bank(out,native,p,np.load(out/'inputs.npy',mmap_mode='r'),read(out/'nodes.json'))
    validate(out);write(out/f'{model}_worker.json',result)


def run(out,model,grant):
    """One monitored model process, with explicit failure distillation and immutable outputs."""
    import fcntl
    release=require_grant(out,grant,model);p=validate(out)
    with (out/'native.lock').open('a',encoding='utf-8') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execution=read(out/'execution.json') if (out/'execution.json').exists() else {}
        if model in execution or list(out.glob(f'{model}_*.npy')):raise ValueError('Refuse repeat/partial overwrite')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 NUMEXPR_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        started=time.monotonic();peak=0;failure=None
        knowledge=read(out/'knowledge.json');knowledge.update(status='native_running',next_decision=f'Run granted {model}, then audit saved arithmetic; no second process.')
        write(out/'knowledge.json',knowledge)
        with (out/f'{model}.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.response_surface_focused','worker','--out',str(out),'--model',model,'--grant',str(grant)],env=env,stdout=log,stderr=subprocess.STDOUT)
            print(dict(event='native_launch',model=model,pid=proc.pid,slot=release['slot'],run=str(out.resolve())),flush=True)
            while proc.poll() is None:
                try:
                    status=Path(f'/proc/{proc.pid}/status').read_text(encoding='utf-8')
                    peak=max(peak,int(next(l.split()[1] for l in status.splitlines() if l.startswith('VmRSS:')))*1024)
                except (FileNotFoundError,StopIteration):pass
                if peak>p['worker_cap'] or time.monotonic()-started>p['seconds_per_model']:
                    failure='Frozen resource cap exceeded';proc.kill();proc.wait();break
                time.sleep(.2)
        wf=out/f'{model}_worker.json'
        if wf.exists() and read(wf)['peak_rss']>p['worker_cap']:failure='Worker HWM exceeds cap'
        if proc.returncode!=0 and failure is None:failure=f'Native worker exit {proc.returncode}; see {model}.log'
        execution[model]=dict(valid=proc.returncode==0 and failure is None and wf.exists(),returncode=proc.returncode,
            seconds=time.monotonic()-started,peak_rss=peak,failure=failure,grant_sha256=sha(grant),slot=release['slot'])
        write(out/'execution.json',execution)
        knowledge=read(out/'knowledge.json');knowledge['execution']=execution
        if not execution[model]['valid']:
            knowledge['status']='failed_native_instrumentation';knowledge['failed_instrumentation_runs'].append(dict(model=model,**execution[model]))
            knowledge['next_decision']='Stop; preserve partial arrays and diagnose in a fresh run. No failed output qualifies for inference.'
        else:
            knowledge['status']='captured_pending_analysis';knowledge['next_decision']='Analyze and independently audit the saved valid model; wait for further slot grants.'
            knowledge.setdefault('native_gates',{})[model]=read(wf)['gates']
        write(out/'knowledge.json',knowledge);print(dict(event='native_finish',model=model,**execution[model]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','worker','analyze'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--full',type=Path,default=Path('runs/response_surface_crossed4954000'))
    parser.add_argument('--pilot',type=Path,default=Path('runs/response_surface_pilot4951000'))
    parser.add_argument('--model',choices=['bolt','timesfm']);parser.add_argument('--grant',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.out,args.full,args.pilot)
    elif args.command=='analyze':
        from experiments.feature_search.response_surface_focused_analysis import analyze
        analyze(args.out,args.model)
    else:
        if args.model is None:parser.error('--model is required for run/worker')
        (run if args.command=='run' else worker)(args.out,args.model,args.grant)
