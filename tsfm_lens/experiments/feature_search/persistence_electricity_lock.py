"""Bounded normalization-path intervention on the existing64 electricity sources.

Reuse native models and the proven joint normalization/inverse helper unchanged.
All comparisons use new singleton calls. Old batch4 artifacts are separately
audited, never treated as bit-identical via a relaxed equality gate.
"""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from experiments.feature_search.persistence_protocol import ENV,exact,finite,read,sha,write
from experiments.feature_search.postrelease_weather_lock import normalization_mode,prediction,binding,require_grant
from experiments.feature_search.postrelease_weather_prepare import utcnow


class Counted:
    """Count all calls and use TF's implicit anchor capture as an exact full repeat."""
    def __init__(self,native):
        self.native=native;self.calls=0;self.trace_expected=None;self.trace_repeats=0

    def __getattr__(self,name):
        return getattr(self.native,name)

    def predict(self,x,replacement=None):
        if x.shape!=(1,128) or x.dtype!=np.float32:raise ValueError('Raw FP32 singleton required')
        self.calls+=1
        if self.calls>300:raise ValueError('Native cell cap')
        value=self.native.predict(x,replacement=replacement)
        if self.trace_expected is not None:
            exact(value[0][0],self.trace_expected,'implicit TF ordinary full repeat')
            self.trace_repeats+=1;self.trace_expected=None
        return value


def batch_difference(new,old):
    """Quantify all differences; this diagnostic never asserts approximate equivalence."""
    new,old=finite(new).astype(float),finite(old).astype(float)
    if new.shape!=old.shape or new.ndim!=2:raise ValueError('Forecast comparison shape')
    delta=new-old
    absolute=np.max(abs(delta),axis=1)
    scale=np.maximum(1.,np.maximum(abs(new).max(axis=1),abs(old).max(axis=1)))
    return dict(bit_identical=bool(np.array_equal(new,old)),different_elements=int(np.count_nonzero(delta)),
                max_absolute=float(absolute.max()),max_relative_linf=float((absolute/scale).max()),
                pointwise_relative_max=float(np.max(abs(delta)/np.maximum(np.maximum(abs(old),abs(new)),1e-12))),
                per_source_max_absolute=absolute.tolist(),per_source_relative_linf=(absolute/scale).tolist(),
                per_source_rms=np.sqrt(np.mean(delta**2,axis=1)).tolist(),
                interpretation='Nonzero differences are NOT exact replay and NOT automatically attributed to roundoff. Primary comparison uses fresh batch1 outputs only; target-dependent loss-drift bounds are reported separately.')


def measure(native,data,plans,old,restoration_rows):
    """All64 anchors exact; both residual arms singleton; bounded repeated checks."""
    native=Counted(native);forecasts={k:[] for k in ['native','ordinary_residual','locked_full','locked_residual','ordinary_hybrid','locked_hybrid']}
    gates=dict(locked_full_anchor=0,ordinary_full_repeat=0,locked_residual_repeat=0,restoration_prediction=0,abstention=0)
    for i,plan in enumerate(plans):
        full=data['full'][i:i+1];residual=data['residual'][i:i+1];tone=data['tones'][i]
        with normalization_mode(native,full,False):
            base=prediction(native,full)
            ordinary=prediction(native,residual) if plan['accepted'] else base.copy()
        if native.name=='timesfm':native.trace_expected=base.copy()
        with normalization_mode(native,full,True):
            if native.trace_expected is not None:raise ValueError('Missing implicit TF full repeat')
            anchored=prediction(native,full);exact(anchored,base,'locked full anchor');gates['locked_full_anchor']+=1
            locked=prediction(native,residual) if plan['accepted'] else anchored.copy()
            if plan['accepted']:
                exact(prediction(native,residual),locked,'locked residual repeat');gates['locked_residual_repeat']+=1
        if native.name=='bolt' or i in restoration_rows:
            exact(prediction(native,full),base,'restored full repeat');gates['restoration_prediction']+=1
            if native.name=='bolt':gates['ordinary_full_repeat']+=1
        hybrid=finite(tone+ordinary);locked_hybrid=finite(tone+locked)
        if not plan['accepted']:
            exact(full,residual,'abstention input');exact(hybrid,base,'ordinary abstention');exact(locked_hybrid,base,'locked abstention');gates['abstention']+=1
        for key,value in [('native',base),('ordinary_residual',ordinary),('locked_full',anchored),('locked_residual',locked),('ordinary_hybrid',hybrid),('locked_hybrid',locked_hybrid)]:forecasts[key].append(value)
        print(native.name,i+1,'/',len(plans),'calls',native.calls,flush=True)
    if native.name=='timesfm':gates['ordinary_full_repeat']=native.trace_repeats
    if gates['ordinary_full_repeat']!=len(plans):raise ValueError('Incomplete exact full repeats')
    forecasts={k:finite(v) for k,v in forecasts.items()}
    gates['actual_native_calls']=native.calls
    differences={k:batch_difference(forecasts[k],old[oldname]) for k,oldname in [('native','native'),('ordinary_residual','residual'),('ordinary_hybrid','hybrid')]}
    return forecasts,gates,differences


def prepare(source,out,architecture):
    """Copy only the64 existing electricity main origins; no new data or fitted rule."""
    template=binding(architecture);p=read(source/'protocol.json')
    preflight=out.with_name(out.name+'_preflight');audit=read(preflight/'knowledge.json')['exact_findings']
    assert audit['tests_passed']==3 and audit['plants_caught']==4
    for name,digest in read(preflight/'manifest.json').items():assert sha(preflight/name)==digest
    for name,digest in p['hashes'].items():assert sha(source/name)==digest
    rows=read(source/'rows.json');plans=read(source/'plans.json')
    indices=[i for i,r in enumerate(rows) if r['domain']=='monash_electricity_hourly']
    assert len(indices)==64 and [rows[i]['rank'] for i in indices]==list(range(72,136))
    bank=np.load(source/'bank.npz');lookup=dict(zip(bank['ids128'].tolist(),bank['inputs128']))
    selected=[plans[i][2] for i in indices];assert all(s['origin']==256 for s in selected)
    full=np.stack([lookup[s['root']] for s in selected]);residual=np.stack([lookup[s['residual1']] for s in selected])
    from experiments.feature_search.selective_tree import continuation
    tones=np.stack([continuation(s['details'],min(1,len(s['details']))) for s in selected])
    flat=[dict(accepted=not s['abstained'],details=s['details'][:1],original_plan=s) for s in selected]
    accepted=sum(s['accepted'] for s in flat);assert accepted==29
    errors=[]
    for x,r,s in zip(full,residual,flat):
        t=np.arange(-128,0);tone=np.zeros(128)
        for d in s['details']:
            a,b=d['coefficients'];phase=2*np.pi*d['frequency']*t/128;tone+=a*np.sin(phase)+b*np.cos(phase)
        error=float(np.max(abs(r+tone-x)));errors.append(error)
        bound=float(np.max(abs(np.spacing(x)))+np.max(abs(np.spacing(r))))+1e-12
        assert error<=bound
        if not s['accepted']:exact(x,r,'copied abstention')
    target=np.load(source/'main_targets.npz');evaluated=np.load(source/'evaluated_forecasts.npz')
    ids=np.array([rows[i]['source'] for i in indices]);assert len(set(ids))==64
    assert np.array_equal(ids,target['source_ids'][indices]) and np.array_equal(ids,evaluated['source_ids'][indices])
    out.mkdir(parents=True,exist_ok=False);(out/'inference').mkdir();(out/'evaluation').mkdir()
    np.savez(out/'inference/inputs.npz',full=finite(full),residual=finite(residual),tones=finite(tones))
    write(out/'inference/plans.json',flat);write(out/'inference/rows.json',[rows[i] for i in indices])
    np.savez(out/'evaluation/targets.npz',future=finite(target['targets'][indices]),source_ids=ids)
    for model in ['bolt','timesfm']:
        assert read(source/'execution.json')[model]['valid']
        assert sha(source/f'{model}_forecasts.npz')==read(source/f'{model}_worker.json')['sha256']
        f=np.load(source/f'{model}_forecasts.npz');oldnative=f['bank_forecasts'][[s['root'] for s in selected]];oldres=f['bank_forecasts'][[s['residual1'] for s in selected]];oldhybrid=tones+oldres
        exact(oldnative,evaluated[model][indices,evaluated['methods'].tolist().index('native')],'old native assembly')
        exact(oldhybrid,evaluated[model][indices,evaluated['methods'].tolist().index('hybrid1')],'old hybrid assembly')
        np.savez(out/'inference'/f'{model}_reference.npz',native=oldnative,residual=oldres,hybrid=oldhybrid)
    here=Path(__file__).resolve().parent
    files=list(here.glob('persistence_electricity_lock*.py'))+list((here.parents[1]/'tests').glob('test_persistence_electricity_lock*.py'))
    sources=dict(template['sources']);sources.update({str(f.resolve()):sha(f) for f in files})
    restoration=list(range(8));calls=dict(bolt=3*64+3*accepted,timesfm=3*64+3*accepted+len(restoration))
    assert max(calls.values())<=300
    protocol=dict(created_utc=utcnow(),stage='PREPARED; HOLD native until explicit new slot grant',
       question='Does the existing TimesFM electricity-positive hybrid response depend on residual-specific normalization and its inverse?',
       cohort='All64 existing electricity main-origin sources from natural_validation_replication4955000, ranks72..135, origin256;29 accepted tones,35 abstentions. No other domain, hindcast selection or new data.',
       evidence_class='Descriptive causal-path intervention on already-seen cohort; not fresh confirmation',
       methods=['native','ordinary_hybrid','locked_hybrid'],
       intervention='Reuse unmodified native code and coherence_study.normalization: joint forward/inverse full-history background lock; ordinary context is nullcontext. Same saved FP32 full/residual inputs and fixed depth1 future tones.',
       gates='All64 exact locked-full self identities and ordinary-full repeats;29 exact locked residual repeats;35 exact abstentions; helper restoration every context. Bolt64 post-exit full repeats; TF64 internal anchor-capture repeats plus8 post-exit full repeats.',
       old_batch4='NOT an exact gate. Always save per-source max absolute, relativeLinf with max(1,absolute forecast) scale, pointwise relative floor1e-12, RMS and differing-element counts. Never call nonzero differences equivalent or assume roundoff. Current contrasts use fresh batch1 arms exclusively.',
       drift_bound='For each old/new source forecast, absolute MSE drift <=2*sqrt(old MSE)*RMS(new-old)+RMS(new-old)^2. Sum native/hybrid bounds to bound gain drift; report actual old/new gains and any sign changes without selection.',
       primary='Per model paired64-source locked hybrid versus ordinary hybrid and versus native; ordinary hybrid versus native retained to reassess old positive at batch1.',
       loss='Mean32 squared error on ALREADY own-history-standardized saved forecasts/targets. Do not renormalize. Equal source weight; all64 retained including ties/abstentions.',
       bootstrap=dict(draws=4096,seed=4957200,kind='descriptive paired64-source percentile95%; shared indices across models and contrasts'),
       partition='Same saved history-frequency sine/cos QR plane, original128-step units on32-step horizon. Unmodified future errors only; no future-selected basis or prediction modification. Exactzero abstentions.',
       limitations=['Reused cohort and old positive: no fresh confirmation or new significance claim. Electricity sources may share weather/calendar/system effects.',
                    'Joint forward/inverse intervention is normalizer-mediated response, not a training diagnosis; it changes relative dynamic range and cannot separate forward versus inverse contributions.',
                    'Native outputs are medians, not means. Geometric plane is not true-component decomposition. Cached data pretraining overlap remains unknown.'],
       source=str(source.resolve()),source_protocol_sha256=sha(source/'protocol.json'),source_data_hashes={name:sha(source/name) for name in ['bank.npz','rows.json','plans.json','main_targets.npz','evaluated_forecasts.npz']},
       architecture=str(architecture.resolve()),architecture_protocol_sha256=sha(architecture/'protocol.json'),
       checkpoints=p['checkpoints'],sources=sources,versions=template['versions'],environment=ENV,
       expected_calls=calls,max_cells=300,max_seconds=300,max_rss_bytes=3*2**30,restoration_rows=restoration,
       estimate='Weather lock144 Bolt calls23.47093s and174 TF calls37.31433s; current279/287 estimated<90s each. Hard300s/model;CPU1;RSS3GiB;one native worker at a time.',
       inference_hashes={f.name:sha(f) for f in (out/'inference').iterdir()},target_sha256=sha(out/'evaluation/targets.npz'),
       forbidden_roots=[str((out/'evaluation').resolve()),str(source.resolve())],preflight_manifest_sha256=sha(preflight/'manifest.json'))
    write(out/'protocol.json',protocol);write(out/'seal.json',dict(sha256=sha(out/'protocol.json')))
    findings=dict(status='READY; native HOLD',sources=64,accepted=29,abstentions=35,expected_calls=calls,max_input_closure=max(errors),native_launched=False,tests=3,plants=4)
    write(out/'knowledge.json',dict(question=protocol['question'],design=protocol,exact_findings=findings,controls=protocol['gates'],limitations=protocol['limitations'],evidence_class=protocol['evidence_class'],next_decision='Await explicit new native slot grant; Plato32 fresh causal next. No native launch authorized yet.'))
    write(out/'preparation_knowledge.json',read(out/'knowledge.json'));write(out/'launch_hold.json',dict(native_launched=False,status='Await explicit slot grant'))
    print(findings)


def worker(out,model):
    """Use identical cached native loading and locking architecture, without model edits."""
    require_grant(out,model);p=binding(out)
    from experiments.feature_search.persistence_study import target_guard
    for root in p['forbidden_roots']:sys.addaudithook(target_guard(root))
    for key,value in ENV.items():
        if os.environ.get(key)!=value:raise ValueError('Incorrect environment: '+key)
    started=time.monotonic()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from experiments.feature_search.real_background_poc import Native
    from experiments.feature_search.persistence_singleton import singleton_class
    cp=p['checkpoints'][model]
    for name,digest in cp['weights'].items():
        if sha(Path(cp['path'])/name)!=digest:raise ValueError('Cached weights changed')
    native=singleton_class(Native)(model,cp['path'])
    f,gates,differences=measure(native,np.load(out/'inference/inputs.npz'),read(out/'inference/plans.json'),np.load(out/'inference'/f'{model}_reference.npz'),p['restoration_rows'])
    assert gates['actual_native_calls']==p['expected_calls'][model]
    denied=0
    for root in p['forbidden_roots']:
        try:(Path(root)/'targets.npz').open('rb')
        except PermissionError:denied+=1
    assert denied==len(p['forbidden_roots']);gates['target_roots_denied']=denied
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;seconds=time.monotonic()-started
    if peak>p['max_rss_bytes'] or seconds>=p['max_seconds']:raise RuntimeError('Resource cap')
    np.savez(out/f'{model}_forecasts.npz',**f);write(out/f'{model}_batch4_differences.json',differences)
    write(out/f'{model}_worker.json',dict(valid=True,gates=gates,seconds=seconds,peak_rss_bytes=peak,threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),protocol_sha256=sha(out/'protocol.json'),forecast_sha256=sha(out/f'{model}_forecasts.npz')))


def run(out,model,grant):
    """Require a later coordinator grant; monitor300seconds and3GiB, retain failures."""
    if not grant:raise PermissionError('HOLD: explicit new coordinator slot grant required')
    import fcntl
    p=binding(out);lock=(out/'native_worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/f'{model}.log').exists():raise ValueError('Refuse attempt overwrite')
    write(out/f'{model}_slot_grant.json',dict(coordinator_grant=grant,protocol_sha256=sha(out/'protocol.json')))
    started=time.monotonic();peak=0;failure=None
    with (out/f'{model}.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.persistence_electricity_lock','worker','--out',str(out),'--model',model],env=dict(os.environ,**ENV),stdout=log,stderr=subprocess.STDOUT)
        print(dict(model=model,pid=proc.pid,run=str(out.resolve()),status='native launched'),flush=True)
        try:
            while proc.poll() is None:
                try:
                    lines=Path(f'/proc/{proc.pid}/status').read_text(encoding='utf-8').splitlines()
                    peak=max(peak,int(next(s.split()[1] for s in lines if s.startswith('VmRSS:')))*1024)
                except (FileNotFoundError,StopIteration):pass
                if peak>p['max_rss_bytes'] or time.monotonic()-started>=p['max_seconds']-.5:
                    failure='RSS cap' if peak>p['max_rss_bytes'] else 'Time cap';proc.kill();proc.wait();break
                time.sleep(.05)
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
    result=dict(valid=proc.returncode==0 and failure is None,returncode=proc.returncode,failure=failure,seconds=time.monotonic()-started,peak_rss_bytes=peak,pid=proc.pid)
    write(out/f'{model}_execution.json',result);k=read(out/'knowledge.json');k['exact_findings'].setdefault('native_execution',{})[model]=result;k['exact_findings']['native_launched']=True
    k['next_decision']='Release slot after final capture; saved-array diagnostics only.' if result['valid'] else 'Preserve failed instrument; no partial scoring or automatic relaunch.'
    write(out/'knowledge.json',k);write(out/f'{model}_knowledge.json',k);print(result,flush=True)
    if not result['valid']:raise RuntimeError('Invalid native capture')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','worker'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--model',choices=['bolt','timesfm']);parser.add_argument('--slot-grant')
    parser.add_argument('--source',type=Path,default=Path('tsfm_lens/runs/natural_validation_replication4955000'))
    parser.add_argument('--architecture',type=Path,default=Path('tsfm_lens/runs/postrelease_weather_lock202610'))
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.source,args.out,args.architecture)
    elif args.command=='run':run(args.out,args.model,args.slot_grant)
    else:worker(args.out,args.model)
