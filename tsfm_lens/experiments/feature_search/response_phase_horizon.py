"""Bolt-only direct64 preparation/capture; no native action without a new slot grant.

Uses the existing OLD16/FRESH16 backgrounds, two amplitudes and two input periods.
Every64 forecast must match a separately requested32 prefix in identical batches,
with exactly one model forward per request. No horizon beyond the cached head.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.response_surface_focused import validate,require_grant
from experiments.feature_search.response_surface_crossed import wave

INPUT_PERIODS=(19.2,24.)
AMPLITUDES=(.1,2.)
OUTPUT_PERIODS=(16.,18.,19.2,20.,22.,24.,26.,28.,32.)


def static_capability(checkpoint,implementation):
    """Read configuration/installed source only; dynamic numerical parity remains pending."""
    config=read(checkpoint/'config.json');source=implementation.read_text(encoding='utf-8')
    head=config.get('chronos_config',{}).get('prediction_length')
    if head!=64 or config.get('architectures')!=['ChronosBoltModelForForecasting']:
        raise ValueError('Cached checkpoint is not the required direct64 Bolt head')
    fragments=['out_dim=self.num_quantiles * self.chronos_config.prediction_length',
               'prediction: torch.Tensor = self.model(context=context_tensor).quantile_preds',
               'remaining -= prediction.shape[-1]','while remaining > 0:',
               'return torch.cat(predictions, dim=-1)[..., :prediction_length]']
    if not all(fragment in source for fragment in fragments):raise ValueError('Installed direct-head path needs review')
    return dict(static_supported=True,declared_direct_head=64,autoregression_required_for64=False,
        implementation=str(implementation.resolve()),implementation_sha256=sha(implementation),
        config_sha256=sha(checkpoint/'config.json'),dynamic_prefix_parity='PENDING coordinator-granted native gate; no native model loaded',
        evidence='Configuration plus installed source: fixed64 quantile head, one initial forward, autoregression only when remaining>0. Static evidence alone does not establish bitwise parity.')


def check_pair(full,prefix,calls,head):
    """Exact same-batch prefix and single-forward checks; unsupported capability fails closed."""
    if head!=64 or calls!=[1,1]:raise ValueError('Direct64 capability gate: head/call count mismatch')
    if full.shape!=(len(prefix),64) or prefix.shape[1:]!=(32,):raise ValueError('Direct64 forecast shape mismatch')
    if not np.isfinite(full).all() or not np.isfinite(prefix).all():raise ValueError('Nonfinite direct-head forecast')
    if not np.array_equal(full[:,:32],prefix):raise ValueError('Direct64 prefix32 exact parity failed')


class DirectBolt:
    """Cached Bolt with explicit one-forward instrumentation and bounded horizon requests."""
    def __init__(self,checkpoint):
        from experiments.feature_search.real_background_poc import Native
        self.native=Native('bolt',checkpoint)
        self.head=self.native.pipeline.model_prediction_length
        if self.head!=64:raise ValueError('Native checkpoint direct head is not64')

    def predict_pair(self,x):
        """Compare64 versus32 on exactly the same tensor and batch geometry."""
        import torch
        if len(x)>4:raise ValueError('Frozen maximum batch4')
        batch=torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float32);predictions=[];calls=[]
        for horizon in (64,32):
            observed=[]
            hook=self.native.model.register_forward_hook(lambda *args:observed.append(1))
            try:
                with torch.inference_mode():
                    _,point=self.native.pipeline.predict_quantiles(batch,prediction_length=horizon,quantile_levels=[.5],limit_prediction_length=True)
                predictions.append(point.detach().cpu().numpy().copy());calls.append(len(observed))
            finally:hook.remove()
        check_pair(predictions[0],predictions[1],calls,self.head)
        return predictions[0],predictions[1]


def prepare(out,old,test,oblique):
    """Condition on audited oblique benefit, inspect direct head, freeze544 inputs and gates."""
    ob=read(oblique/'knowledge.json');audit=read(oblique/'audit.json');out.mkdir(parents=True,exist_ok=False)
    question='Does weak-versus-strong output-frequency distortion transfer over the direct64 horizon and sharpen the finite-window frequency comparison?'
    initial=dict(status='preparing_no_native',question=question,exact_findings=None,controls='Direct64 capability and exact prefix gates',
        limitations='Existing sources reused; no native grant and no new model evidence.',evidence_class='Preparation only',failed_instrumentation_runs=[],next_decision='Check conditional authorization and static head capability.')
    write(out/'knowledge.json',initial)
    gain=next(r for r in read(oblique/'results.json')['primary'] if r['model']=='bolt' and r['contrast']=='oblique_scalar_minus_fullH')
    if ob['status']!='complete_audited' or not audit['passed'] or gain['family4_percentile'][0]<=0:
        initial.update(status='conditional_preparation_stopped',exact_findings=dict(oblique_gain=gain),next_decision='No native or further preparation; oblique condition did not qualify.');write(out/'knowledge.json',initial);return
    previous=validate(test);checkpoint=Path(previous['checkpoints']['bolt']['path'])
    implementation=Path(sys.executable).resolve().parents[1]/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/chronos/chronos_bolt.py'
    try:capability=static_capability(checkpoint,implementation)
    except Exception as error:
        initial.update(status='capability_failure_stopped',failed_instrumentation_runs=[dict(stage='static_capability',error=repr(error))],next_decision='STOP; no native permitted for unsupported direct head.');write(out/'knowledge.json',initial);return
    oldrun=Path(read(old/'protocol.json')['prior']);inputs=[];nodes=[];groups=[];sources=[];references=[];mask=[];prior_files=[]
    worst=0.;closure=0.
    for cohort,bankdir,groupfile in [('FIT',oldrun,old/'groups.json'),('TEST',test,test/'groups.json')]:
        bank=np.load(bankdir/'inputs.npy');forecast=np.load(bankdir/'bolt_native_forecast.npy');original=read(groupfile)
        metadata=read(bankdir/'sources.json');oldnodes=read(bankdir/'nodes.json')
        for local,s in enumerate(metadata):
            sid=len(sources);sources.append(dict(s,cohort=cohort,cohort_source_id=local))
            cells=[g for g in original if g['source_id']==local and g.get('offset',0.)==0.]
            if len(cells)!=2:raise ValueError('Two center cells required per source')
            first=cells[0];x=bank[first['base']].copy();scale=first['input_scale'];base=len(inputs)
            ids=[n['index'] for n in oldnodes if n['source_id']==local]
            def add(value,kind,reference=None,**details):
                """Append one input with optional exact same-geometry historical prefix reference."""
                index=len(inputs);inputs.append(np.asarray(value,np.float32));nodes.append(dict(index=index,source_id=sid,source=s['source'],cohort=cohort,kind=kind,**details))
                mask.append(reference is not None);references.append(np.full(32,np.nan,np.float32) if reference is None else forecast[reference].copy())
                if reference is not None:
                    pos=ids.index(reference)
                    if min(4,len(ids)-4*(pos//4))!=4:raise ValueError('Historical weak/reference batch geometry is not4')
                return index
            add(x,'background',first['base'])
            for period in INPUT_PERIODS:
                g=next(g for g in cells if abs(g['frequency']-128/period)<1e-12)
                for amplitude in AMPLITUDES:
                    indices=[];physical=amplitude*scale
                    for phase in range(4):
                        nominal=x.astype(float)+physical*wave(128/period,phase*np.pi/2)
                        value=bank[g['indices'][phase]].copy() if amplitude==.1 else nominal.astype(np.float32)
                        worst=max(worst,float(np.sqrt(np.mean((value.astype(float)-nominal)**2))/physical))
                        indices.append(add(value,'signal',g['indices'][phase] if amplitude==.1 else None,root=base,period=period,amplitude=amplitude,phase_id=phase))
                    observed=np.stack([inputs[i].astype(float)-x for i in indices])
                    closure=max(closure,float(np.sqrt(np.mean((observed[:2]+observed[2:])**2))/physical))
                    groups.append(dict(cohort=cohort,source=s['source'],source_id=sid,cohort_source_id=local,background_type=s['background_type'],base=base,indices=indices,
                        input_period=period,amplitude=amplitude,physical_amplitude=physical,input_scale=scale))
        prior_files += [bankdir/n for n in ['inputs.npy','nodes.json','sources.json','bolt_native_forecast.npy','bolt_worker.json','execution.json','protocol.json']]+[groupfile]
    if len(inputs)!=544 or worst>1e-5 or closure>2e-5:raise ValueError('Input geometry/closure gate failed')
    np.save(out/'inputs.npy',np.stack(inputs));np.save(out/'historical_prefix32.npy',np.stack(references));np.save(out/'historical_reference_mask.npy',np.asarray(mask))
    for name,value in [('nodes',nodes),('groups',groups),('sources',sources),('static_capability',capability)]:write(out/f'{name}.json',value)
    root=Path(__file__).resolve().parents[2]
    files=[Path(__file__),root/'experiments/feature_search/response_phase_horizon_analysis.py',root/'experiments/feature_search/response_phase_horizon_audit.py',
           root/'experiments/feature_search/response_phase_horizon_plants.py',root/'tests/test_response_phase_horizon.py',implementation,checkpoint/'config.json']
    source_hashes=dict(previous['source_hashes']);source_hashes.update({str(f.resolve()):sha(f) for f in files})
    names=['inputs.npy','historical_prefix32.npy','historical_reference_mask.npy','nodes.json','groups.json','sources.json','static_capability.json']
    seconds=read(oblique/'execution.json')['bolt']['seconds'];estimate=1.1*(20+seconds*544/144*2)
    if estimate>=300:raise ValueError('Conservative doubled-prediction estimate exceeds cap')
    protocol=dict(question=question,stage='prepared_native_held',models=['bolt'],policies=['native'],nodes=544,context=128,horizon=64,analysis_horizons=[32,64],
        cohorts={'FIT':16,'TEST':16},new_sources=0,amplitudes=list(AMPLITUDES),input_periods=list(INPUT_PERIODS),phases=(np.arange(4)*np.pi/2).tolist(),output_period_candidates=list(OUTPUT_PERIODS),
        conditional_authorization=dict(oblique_run=str(oblique.resolve()),oblique_gain=gain,audit_sha256=sha(oblique/'audit.json')),
        capability=capability,
        controls='Require declared head64; limit_prediction_length=True; exactly one model forward for every64 AND32 request; bit-exact prefix parity on identical batches<=4; exact repeated64/32 background; exact same-batch replay; all288 historical weak/base32 forecasts match at batch4 geometry. Save both64 and separately requested32 arrays. Any mismatch stops capability and invalidates run, no autoregressive rescue.',
        selection='Separate OLD-source LOSO selection among fixed9 output periods per amplitude/input period/analysis horizon/variant. Tone-only2x2 and matched tone+DC+lineartrend4x2 maps; same-input map and OLD mean horizonx2 waveform controls. Freeze all OLD-selected maps before TEST scoring. No TEST selection.',
        primary='19.2input tone-only, TEST same-frequency minus OLD-selected-frequency phase-pooled NMSE, four cells amplitude.1/2 x horizon32/64. Paired16TEST-source balanced-type16384 bootstrap draws seed4954600; nominal95 and family4 Bonferroni98.75 percentile CIs, descriptive small-sample coverage; no threshold/oscillator claim.',
        interactions='Descriptive source-paired weak-minus-strong selected-map gain at each horizon and64-minus32 gain at each amplitude.24h and nuisance variants descriptive; retain all FIT-train/CV and TEST curves, source rows, rank/conditioning and even residual.',
        limitations='Same OLD16 and previously examined TEST16 synthetic sources, native normalization only. Strong amplitude changes normalization as well as signal dominance.32prefix reuses identical direct64 forecast; neither horizon is a new independent source. Four cardinal phases: weak32 interpolation has external oblique support, strong/64 phase behavior is not separately validated. Response prediction is not forecast improvement; no training-prior or oscillator/patch-causality attribution.',
        cpu_threads=1,worker_cap=3*2**30,seconds_per_model=300,estimated_seconds=estimate,
        budget_formula='1.1*(20s allowance + measured oblique Bolt elapsed *544/144 *2 for64/32 requests); conservative estimate, not guarantee',
        capture_hold=True,grant_contract='No native until new coordinator grant bound to this protocol. Bolt only. Gate failure stops; no extra horizon/grid or TimesFM.',
        checkpoints={'bolt':previous['checkpoints']['bolt']},source_hashes=source_hashes,hashes={n:sha(out/n) for n in names},
        prior_hashes={str(f.resolve()):sha(f) for f in prior_files},input_gate=dict(nominal_rms_over_peak_max=worst,antipodal_rms_over_peak_max=closure),historical_reference_rows=int(sum(mask)))
    write(out/'protocol.json',protocol);initial.update(status='prepared_native_capture_held',design=protocol,
        exact_findings=dict(static_capability=capability,nodes=544,historical_reference_rows=int(sum(mask)),input_gate=protocol['input_gate'],estimated_seconds=estimate,native_results=None),
        controls=protocol['controls'],limitations=protocol['limitations'],next_decision='HOLD native; require new coordinator slot grant. Dynamic prefix parity remains pending and must pass before any scientific claim.')
    write(out/'knowledge.json',initial)


def capture_bank(out,native,p):
    """Bounded native capture; no outputs qualify unless every direct-head gate is exact."""
    inputs=np.load(out/'inputs.npy');nodes=read(out/'nodes.json');reference=np.load(out/'historical_prefix32.npy');mask=np.load(out/'historical_reference_mask.npy')
    full=np.lib.format.open_memmap(out/'bolt_native_forecast64.npy',mode='w+',dtype=np.float32,shape=(544,64))
    prefix=np.lib.format.open_memmap(out/'bolt_native_forecast32.npy',mode='w+',dtype=np.float32,shape=(544,32))
    started=time.monotonic()
    for source in range(32):
        ids=[n['index'] for n in nodes if n['source_id']==source];base=ids[0]
        a,b=native.predict_pair(inputs[base:base+1]);c,d=native.predict_pair(inputs[base:base+1])
        if not np.array_equal(a,c) or not np.array_equal(b,d):raise ValueError('Exact repeated background gate failed')
        for start in range(0,len(ids),4):
            batch=ids[start:start+4];a,b=native.predict_pair(inputs[batch])
            if not np.array_equal(b[mask[batch]],reference[batch][mask[batch]]):raise ValueError('Historical weak/base prefix parity failed')
            full[batch]=a;prefix[batch]=b
            if time.monotonic()-started>p['seconds_per_model'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>p['worker_cap']:raise RuntimeError('Frozen resource cap exceeded')
        a,b=native.predict_pair(inputs[ids[:4]])
        if not np.array_equal(a,full[ids[:4]]) or not np.array_equal(b,prefix[ids[:4]]):raise ValueError('Same-batch replay failed')
        print(dict(model='bolt',source=source,seconds=time.monotonic()-started),flush=True)
    full.flush();prefix.flush()
    return dict(gates=dict(direct_head64=True,single_forward_per_request=True,prefix32_exact=True,background_repeat_exact=True,historical_prefix_exact=True,batch_replay_exact=True),
        seconds=time.monotonic()-started,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        hashes={n:sha(out/n) for n in ['bolt_native_forecast64.npy','bolt_native_forecast32.npy']})


def worker(out,grant):
    """Only a new protocol-bound Bolt slot grant may instantiate native weights."""
    require_grant(out,grant,'bolt');p=validate(out,'bolt')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):raise ValueError('CPU one thread required')
    if list(out.glob('bolt_native_forecast*.npy')):raise ValueError('Refuse partial overwrite')
    native=DirectBolt(p['checkpoints']['bolt']['path']);result=capture_bank(out,native,p);validate(out);write(out/'bolt_worker.json',result)


def run(out,grant):
    """Monitor one bounded worker, preserving capability failures without rescue attempts."""
    import fcntl
    release=require_grant(out,grant,'bolt');p=validate(out)
    with (out/'native.lock').open('a',encoding='utf-8') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'execution.json').exists() or list(out.glob('bolt_native_forecast*.npy')):raise ValueError('Refuse native repeat')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        k=read(out/'knowledge.json');k['status']='native_running';write(out/'knowledge.json',k)
        started=time.monotonic();peak=0;failure=None
        with (out/'bolt.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.response_phase_horizon','worker','--out',str(out),'--grant',str(grant)],env=env,stdout=log,stderr=subprocess.STDOUT)
            print(dict(event='native_launch',model='bolt',pid=proc.pid,slot=release['slot'],run=str(out.resolve())),flush=True)
            while proc.poll() is None:
                try:
                    status=Path(f'/proc/{proc.pid}/status').read_text(encoding='utf-8');peak=max(peak,int(next(l.split()[1] for l in status.splitlines() if l.startswith('VmRSS:')))*1024)
                except (FileNotFoundError,StopIteration):pass
                if peak>p['worker_cap'] or time.monotonic()-started>p['seconds_per_model']:
                    failure='Frozen resource cap exceeded';proc.kill();proc.wait();break
                time.sleep(.2)
        wf=out/'bolt_worker.json'
        if wf.exists() and read(wf)['peak_rss']>p['worker_cap']:failure='Worker HWM exceeds cap'
        if proc.returncode!=0 and failure is None:failure=f'Native/capability failure, exit{proc.returncode}; see bolt.log'
        result=dict(valid=proc.returncode==0 and failure is None and wf.exists(),returncode=proc.returncode,seconds=time.monotonic()-started,peak_rss=peak,failure=failure,slot=release['slot'],grant_sha256=sha(grant))
        write(out/'execution.json',dict(bolt=result));k=read(out/'knowledge.json');k['execution']={'bolt':result}
        k['status']='captured_pending_analysis' if result['valid'] else 'failed_native_capability_or_resource'
        if not result['valid']:k['failed_instrumentation_runs'].append(result)
        k['next_decision']='Release slot; only valid capture may proceed to OLD-only fit freeze then TEST analysis. Stop on any capability failure.';write(out/'knowledge.json',k)
        print(dict(event='native_finish',model='bolt',**result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','worker','fit','evaluate'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--grant',type=Path)
    parser.add_argument('--old',type=Path,default=Path('runs/response_phase4954200'));parser.add_argument('--test',type=Path,default=Path('runs/response_phase_transfer4954300'))
    parser.add_argument('--oblique',type=Path,default=Path('runs/response_phase_oblique4954500'));args=parser.parse_args()
    if args.command=='prepare':prepare(args.out,args.old,args.test,args.oblique)
    elif args.command in ('run','worker'):(run if args.command=='run' else worker)(args.out,args.grant)
    else:
        from experiments.feature_search.response_phase_horizon_analysis import fit,evaluate
        try:(fit if args.command=='fit' else evaluate)(args.out)
        except Exception as error:
            k=read(args.out/'knowledge.json');k['status']='failed_analysis_instrumentation';k['failed_instrumentation_runs'].append(dict(stage=args.command,error=repr(error)));write(args.out/'knowledge.json',k);raise
