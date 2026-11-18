"""Prospective normalizer mechanism followup on the same30 weather windows.

No decomposition fitting, data expansion, tuning or target-based choice. Actual
coherence_study.normalization locks BOTH directions; ordinary calls must use a
null context. Raw FP32 singleton forecasts must exactly replay the sealed pilot.
"""
import argparse
import contextlib
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import numpy as np
from experiments.feature_search.persistence_protocol import ENV, exact, finite, read, sha, write
from experiments.feature_search.postrelease_weather_prepare import utcnow
from experiments.feature_search.coherence_study import normalization


def normalization_site(native):
    """Identify the actual forward hook/function that the existing helper replaces."""
    if native.name=='bolt':
        return native.model.instance_norm,'forward'
    return sys.modules[type(native.model).__module__],'revin'


@contextlib.contextmanager
def normalization_mode(native,anchor,locked):
    """Ordinary is genuinely null; locked must install and restore the real helper."""
    module,attribute=normalization_site(native)
    before=getattr(module,attribute)
    context=normalization(native,anchor,'background') if locked else contextlib.nullcontext()
    try:
        with context:
            active=getattr(module,attribute)!=before
            if active != locked:
                raise ValueError('Ignored or unexpected normalizer locking')
            yield
    finally:
        if getattr(module,attribute)!=before:
            raise ValueError('Normalizer not restored')


class Counted:
    """Count actual singleton calls, including the helper's internal TF anchor call."""
    def __init__(self,native):
        self.native=native;self.calls=0

    def __getattr__(self,name):
        return getattr(self.native,name)

    def predict(self,x,replacement=None):
        if x.shape!=(1,128) or x.dtype!=np.float32:
            raise ValueError('Raw FP32 singleton input required')
        self.calls+=1
        if self.calls>200:
            raise ValueError('Native cell cap exceeded')
        return self.native.predict(x,replacement=replacement)


def prediction(native,x):
    """Reject wrong shape, dtype or nonfinite output without changing native math."""
    value=finite(native.predict(x)[0])
    if value.shape!=(1,32) or value.dtype!=np.float32:
        raise ValueError('Expected raw FP32 singleton native forecast')
    return value[0]


def measure(native,data,plans,reference):
    """Recompute ordinary forecasts, intervene only on normalization, and gate exactly."""
    native=Counted(native)
    forecasts={name:[] for name in ['native','ordinary_residual','locked_full','locked_residual','ordinary_hybrid','locked_hybrid']}
    gates=dict(full_replay=0,residual_replay=0,ordinary_hybrid_replay=0,locked_full_anchor=0,
               locked_residual_repeat=0,restored_full=0,abstention=0)
    for i,plan in enumerate(plans):
        full=data['full'][i:i+1];residual=data['residual'][i:i+1];tone=data['tones'][i]
        with normalization_mode(native,full,False):
            base=prediction(native,full)
            ordinary=prediction(native,residual) if plan['accepted'] else base.copy()
        exact(base,reference['native'][i],'original full forecast replay');gates['full_replay']+=1
        exact(ordinary,reference['residual'][i],'original residual forecast replay');gates['residual_replay']+=1
        with normalization_mode(native,full,True):
            anchored=prediction(native,full)
            exact(anchored,base,'locked full anchor identity');gates['locked_full_anchor']+=1
            locked=prediction(native,residual) if plan['accepted'] else anchored.copy()
            if plan['accepted']:
                exact(prediction(native,residual),locked,'locked residual repeat');gates['locked_residual_repeat']+=1
        restored=prediction(native,full)
        exact(restored,base,'restored ordinary full');gates['restored_full']+=1
        hybrid=finite(tone+ordinary);locked_hybrid=finite(tone+locked)
        exact(hybrid,reference['hybrid'][i],'original hybrid forecast replay');gates['ordinary_hybrid_replay']+=1
        if not plan['accepted']:
            exact(residual,full,'abstained residual input')
            exact(hybrid,base,'ordinary abstention')
            exact(locked_hybrid,base,'locked abstention');gates['abstention']+=1
        for name,value in [('native',base),('ordinary_residual',ordinary),('locked_full',anchored),
                           ('locked_residual',locked),('ordinary_hybrid',hybrid),('locked_hybrid',locked_hybrid)]:
            forecasts[name].append(value)
        print(native.name,i+1,'/',len(plans),'calls',native.calls,flush=True)
    return {name:finite(value) for name,value in forecasts.items()},dict(gates,actual_native_calls=native.calls)


def binding(out):
    """Validate immutable code, input and reference forecasts without opening targets."""
    p=read(out/'protocol.json')
    if sha(out/'protocol.json')!=read(out/'seal.json')['sha256']:
        raise ValueError('Protocol changed')
    for path,digest in p['sources'].items():
        if sha(path)!=digest:raise ValueError('Frozen source changed: '+path)
    for name,digest in p['inference_hashes'].items():
        if sha(out/'inference'/name)!=digest:raise ValueError('Frozen inference artifact changed: '+name)
    return p


def prepare(source,out):
    """Reuse original arrays exactly; freeze all hypotheses and budgets before launch."""
    from experiments.feature_search.postrelease_weather_pilot import binding as primary_binding
    prior=primary_binding(source);oldp=read(source/'protocol.json')
    preflight=out.with_name(out.name+'_preflight')
    verified=read(preflight/'knowledge.json')['exact_findings']
    assert verified['tests_passed']==4 and verified['plants_caught']==5
    for name,digest in read(preflight/'manifest.json').items():
        assert sha(preflight/name)==digest
    data=np.load(source/'inference/inputs.npz');plans=read(source/'inference/plans.json')
    full=data['bank'][[r['native'] for r in plans]].copy()
    residual=data['bank'][[r['residual'] for r in plans]].copy()
    assert full.dtype==residual.dtype==np.float32 and full.shape==residual.shape==(30,128)
    past=[];errors=[]
    for i,plan in enumerate(plans):
        tone=np.zeros(128)
        for item in plan['details']:
            a,b=item['coefficients'];angle=2*np.pi*item['frequency']*np.arange(-128,0)/128
            tone+=a*np.sin(angle)+b*np.cos(angle)
        r64=data['history'][i]-tone
        np.testing.assert_allclose(r64+tone,data['history'][i],rtol=1e-12,atol=1e-12)
        error=float(np.max(abs(residual[i]+tone-full[i])))
        bound=float(np.max(abs(np.spacing(residual[i])))+np.max(abs(np.spacing(full[i]))))
        if error>bound+1e-12:raise ValueError('Residual+tone reconstruction exceeds FP32 cast budget')
        if not plan['accepted']:exact(residual[i],full[i],'abstention input')
        past.append(tone);errors.append(error)
    out.mkdir(parents=True,exist_ok=False);(out/'inference').mkdir();(out/'evaluation').mkdir()
    np.savez(out/'inference/inputs.npz',full=finite(full),residual=finite(residual),tones=finite(data['tones']),
             past_tones=finite(past),history64=finite(data['history']),scales=finite(data['scales']))
    for name in ['plans.json','rows.json']:shutil.copy2(source/'inference'/name,out/'inference'/name)
    for model in ['bolt','timesfm']:
        execution=read(source/f'{model}_execution.json');assert execution['valid']
        f=np.load(source/f'{model}_forecasts.npz');w=read(source/f'{model}_worker.json')
        assert sha(source/f'{model}_forecasts.npz')==w['forecast_sha256']
        np.savez(out/'inference'/f'{model}_reference.npz',native=f['native'],hybrid=f['hybrid'],
                 residual=f['bank_predictions'][[r['residual'] for r in plans]])
    shutil.copy2(source/'evaluation/targets.npz',out/'evaluation/targets.npz')
    assert sha(out/'evaluation/targets.npz')==oldp['target_sha256']
    here=Path(__file__).resolve().parent
    files=list(here.glob('postrelease_weather_lock*.py'))+list((here.parents[1]/'tests').glob('test_postrelease_weather_lock*.py'))
    files += [here/'postrelease_weather_posthoc.py',here/'postrelease_weather_score.py',here/'natural_gain_location.py']
    sources=dict(prior['sources']);sources.update({str(f.resolve()):sha(f) for f in files})
    accepted=sum(p['accepted'] for p in plans);expected_calls=dict(bolt=3*len(plans)+3*accepted,timesfm=4*len(plans)+3*accepted)
    assert max(expected_calls.values())<=200
    protocol=dict(created_utc=utcnow(),stage='PREPARED; native HOLD awaiting a new coordinator slot grant',
       question='Does locking residual forward/inverse normalization to full history reduce weather hybrid harm?',
       hypotheses=['Locked hybrid versus ordinary hybrid: positive mean gain supports a beneficial normalizer-mediated response in this fixed cohort.',
                   'Locked hybrid versus native: does locking recover or exceed original native forecast accuracy?'],
       data='Original30 complete windows only;8 stations with counts4/4/4/4/4/2/4/4; no replacement or new data.',
       intervention='Unmodified coherence_study.normalization(native, full_raw_FP32_singleton, background) locks BOTH forward normalization and inverse. Ordinary uses contextlib.nullcontext; never another policy string.',
       methods=['native','ordinary_hybrid','locked_hybrid'],recipe='Same saved past/future tones, FP32 residual and full contexts. No refit, altered coefficients, thresholds, scaling, location centering or selection.',
       gates='All finite FP32 native outputs; exact replay of all30 full/residual/ordinary-hybrid references; exact30 locked-full anchor identities;18 locked-residual repeats;30 restored-full identities;12 abstention identities; active-lock plumbing and restoration verified.',
       reconstruction='Float64 original residual+tone rtol=atol1e-12; FP32 full/residual cast closure bounded by sum of maximum one-ULP spacings +1e-12. Forecast equality gates remain bit-exact.',
       primary='Two descriptive paired contrasts per model: ordinary_hybrid loss minus locked_hybrid loss; native loss minus locked_hybrid loss. Positive favors locking. No significance gate or multiplicity-adjusted confirmatory claim.',
       loss=oldp['loss'],stations=oldp['stations'],windows_by_station=oldp['windows_by_station'],
       bootstrap=oldp['uncertainty'],projection='Same saved past-frequency QR tone plane as natural_gain_location.py; error and update-energy partitions only, unchanged targets. Equal station weights; exactzero abstentions, no undefined plane estimates.',
       interpretation='Normalizer-mediated response of these frozen median forecasters, not training diagnosis. Locking can change internal input/output scales and relative dynamic range; plane partitions are geometric, not true-component causal labels.',
       limits=['Descriptive8-station cohort with shared-weather dependence; reused outcomes, no new confirmation.',
               'Chronological observation holdout does not imply unseen geography/distribution.',
               'Forward and inverse are changed together; this intervention does not distinguish their separate causal contributions.'],
       source=str(source.resolve()),primary_protocol_sha256=sha(source/'protocol.json'),
       source_forecast_sha256={m:sha(source/f'{m}_forecasts.npz') for m in ['bolt','timesfm']},
       checkpoints=prior['checkpoints'],sources=sources,versions=prior['versions'],environment=ENV,
       expected_calls=expected_calls,max_cells=200,max_seconds=300,max_rss_bytes=3*2**30,
       estimate='Previous49-input repeat pilot took18.5145s Bolt/25.2244s TF for98calls incl startup. Current144/174 planned calls estimated<90s/model; hard300s cap, single worker at a time.',
       inference_hashes={f.name:sha(f) for f in (out/'inference').iterdir()},target_sha256=sha(out/'evaluation/targets.npz'),
       forbidden_roots=[str((out/'evaluation').resolve()),str(source.resolve()),str((source.parent/'postrelease_weather_202610').resolve())],
       preflight_manifest_sha256=sha(preflight/'manifest.json'))
    write(out/'protocol.json',protocol);write(out/'seal.json',dict(sha256=sha(out/'protocol.json')))
    findings=dict(status='prepared; native HOLD',windows=30,accepted=accepted,abstentions=30-accepted,
                  expected_native_calls=expected_calls,max_input_reconstruction_abs=max(errors),native_launched=False,
                  tests_passed=4,plants_caught=5)
    write(out/'knowledge.json',dict(question=protocol['question'],design=protocol,exact_findings=findings,
          controls=protocol['gates'],limitations=protocol['limits']+[protocol['interpretation']],
          evidence_class='prospective mechanism intervention protocol; no native outcomes yet',
          next_decision='Ready after saved-input replay; HOLD until explicit slot grant. No downloads or tuning.'))
    write(out/'preparation_knowledge.json',read(out/'knowledge.json'));write(out/'launch_hold.json',dict(native_launched=False,status='Await explicit coordinator slot grant'))
    print(findings)


def require_grant(out,model):
    """No native launch or direct worker entry without a fresh protocol-bound grant."""
    file=out/f'{model}_slot_grant.json'
    if not file.exists():raise PermissionError('HOLD: no coordinator slot grant')
    grant=read(file)
    if not grant.get('coordinator_grant') or grant.get('protocol_sha256')!=sha(out/'protocol.json'):
        raise PermissionError('HOLD: stale or invalid grant')


def worker(out,model):
    """Target-isolated cached native worker; no changes to existing native sources."""
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
    data=np.load(out/'inference/inputs.npz');reference=np.load(out/'inference'/f'{model}_reference.npz')
    forecasts,gates=measure(native,data,read(out/'inference/plans.json'),reference)
    assert gates['actual_native_calls']==p['expected_calls'][model]
    denied=0
    for root in p['forbidden_roots']:
        try:(Path(root)/'targets.npz').open('rb')
        except PermissionError:denied+=1
    assert denied==len(p['forbidden_roots']);gates['target_roots_denied']=denied
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;seconds=time.monotonic()-started
    if peak>p['max_rss_bytes'] or seconds>=p['max_seconds']:raise RuntimeError('Resource cap')
    np.savez(out/f'{model}_forecasts.npz',**forecasts)
    write(out/f'{model}_worker.json',dict(valid=True,gates=gates,seconds=seconds,peak_rss_bytes=peak,
          threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),protocol_sha256=sha(out/'protocol.json'),
          forecast_sha256=sha(out/f'{model}_forecasts.npz')))


def run(out,model,grant):
    """At most one supervised worker; preserve every attempt and distill its outcome."""
    if not grant:raise PermissionError('HOLD: coordinator slot grant required')
    import fcntl
    p=binding(out);lock=(out/'native_worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/f'{model}.log').exists():raise ValueError('Refuse native attempt overwrite')
    write(out/f'{model}_slot_grant.json',dict(coordinator_grant=grant,protocol_sha256=sha(out/'protocol.json')))
    started=time.monotonic();peak=0;failure=None
    with (out/f'{model}.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.postrelease_weather_lock','worker',
              '--out',str(out),'--model',model],env=dict(os.environ,**ENV),stdout=log,stderr=subprocess.STDOUT)
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
    result=dict(valid=proc.returncode==0 and failure is None,returncode=proc.returncode,failure=failure,
                seconds=time.monotonic()-started,peak_rss_bytes=peak,pid=proc.pid)
    write(out/f'{model}_execution.json',result)
    k=read(out/'knowledge.json');k['exact_findings'].setdefault('native_execution',{})[model]=result;k['exact_findings']['native_launched']=True
    k['next_decision']='Release slot after final capture; saved-array analysis only.' if result['valid'] else 'Preserve failed instrumentation; no partial scoring or ungated relaunch.'
    write(out/'knowledge.json',k);write(out/f'{model}_knowledge.json',k);print(result,flush=True)
    if not result['valid']:raise RuntimeError('Native run invalid; preserve failed attempt')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','worker'])
    parser.add_argument('--source',type=Path,default=Path('tsfm_lens/runs/postrelease_weather_pilot202610'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--model',choices=['bolt','timesfm'])
    parser.add_argument('--slot-grant');args=parser.parse_args()
    if args.command=='prepare':prepare(args.source,args.out)
    elif args.command=='run':run(args.out,args.model,args.slot_grant)
    else:worker(args.out,args.model)
