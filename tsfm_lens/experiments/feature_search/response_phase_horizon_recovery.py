"""Single outcome-blind recovery after controlled original-geometry replay.

The original protocol and544 inputs remain byte-identical. A separately hashed
amendment replaces ONLY mixed-geometry historical exactness with raw diagnostic
differences and algebraic bounds. Same-query64/32 equality remains bit-exact.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import resource
import shutil
import subprocess
import sys
import time
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.response_surface_focused import validate,require_grant
from experiments.feature_search.response_phase_horizon import check_pair
from experiments.feature_search.response_phase_replay_preflight import DiagnosticBolt,array_hash


def mse_change_bound(old_error,perturbation,energy):
    """Cauchy bound for a fixed prediction: |newNMSE-oldNMSE| <=2sqrt(oldNMSE)u+u²."""
    e=np.asarray(old_error,float);d=np.asarray(perturbation,float)
    if energy<=0:raise ValueError('Positive fixed ideal energy required')
    return float((2*np.linalg.norm(e)*np.linalg.norm(d)+np.sum(d*d))/energy)


def response_bounds(current,reference,groups):
    """Measured32 response differences and triangle bounds, without fitting/scoring a model."""
    delta=np.asarray(current,float)-np.asarray(reference,float);rows=[]
    for g in groups:
        if g['amplitude']!=.1:continue
        response_delta=delta[g['indices']]-delta[g['base']]
        triangle=float(np.linalg.norm(np.linalg.norm(delta[g['indices']],axis=1)+np.linalg.norm(delta[g['base']])))
        t=np.arange(32);ideal=np.stack([g['physical_amplitude']*np.sin(2*np.pi*t/g['input_period']+k*np.pi/2) for k in range(4)])
        energy=float(np.sum(ideal**2));norm=float(np.linalg.norm(response_delta))
        rows.append(dict(cohort=g['cohort'],source=g['source'],source_id=g['source_id'],input_period=g['input_period'],
            response_delta_l2=norm,triangle_l2_bound=triangle,ideal_energy=energy,u=norm/np.sqrt(energy),u_triangle_bound=triangle/np.sqrt(energy)))
    return delta,rows


def validate_recovery(out,checkpoint=False):
    """Check original protocol plus the immutable authorized amendment and diagnostic evidence."""
    p=validate(out,'bolt' if checkpoint else None);a=read(out/'amendment.json')
    if sha(out/'protocol.json')!=a['parent_protocol_sha256']:raise ValueError('Original protocol changed')
    parent=Path(a['parent']);preflight=Path(a['preflight'])
    if sha(parent/'protocol.json')!=a['parent_protocol_sha256']:raise ValueError('Parent protocol changed')
    for name,digest in a['parent_files'].items():
        if sha(parent/name)!=digest:raise ValueError('Original failed run altered')
    for name,digest in a['preflight_files'].items():
        if sha(preflight/name)!=digest:raise ValueError('Eligibility evidence changed')
    for file,digest in a['source_hashes'].items():
        if sha(file)!=digest:raise ValueError('Recovery source changed')
    for name in p['hashes']:
        if sha(out/name)!=sha(parent/name):raise ValueError('Recovery is not identical frozen input bank')
    if not read(preflight/'diagnostic.json')['decision']['eligible'] or not read(preflight/'audit.json')['passed']:raise ValueError('No eligible geometry evidence')
    return p


def require_recovery_grant(out,grant):
    """The new grant must bind BOTH the old protocol and the explicit amendment."""
    record=require_grant(out,grant,'bolt')
    if record.get('amendment_sha256')!=sha(out/'amendment.json'):raise PermissionError('Grant must bind recovery amendment')
    return record


def prepare(out,parent,preflight):
    """Freeze the permitted comparator amendment before scientific capture or TEST scoring."""
    decision=read(preflight/'diagnostic.json')['decision'];audit=read(preflight/'audit.json');p=validate(parent)
    if not decision['eligible'] or not audit['passed']:raise ValueError('Replay did not isolate geometry; STOP')
    if (parent/'fit_freeze.json').exists() or (parent/'results.json').exists():raise ValueError('Parent has scientific outcomes before amendment')
    out.mkdir(parents=True,exist_ok=False)
    for name in ['protocol.json',*p['hashes']]:shutil.copyfile(parent/name,out/name)
    root=Path(__file__).resolve().parents[2]
    files=[Path(__file__),root/'experiments/feature_search/response_phase_horizon_recovery_audit.py',
           root/'experiments/feature_search/response_phase_horizon_recovery_plants.py',root/'tests/test_response_phase_horizon_recovery.py',
           root/'experiments/feature_search/response_phase_replay_preflight.py']
    evidence=['protocol.json','cases.json','queries.npz','diagnostic.json','audit.json','execution.json']
    evidence += [f.name for f in preflight.glob('*_raw.npz')]+[f.name for f in preflight.glob('*_calls.json')]
    amendment=dict(parent=str(parent.resolve()),preflight=str(preflight.resolve()),parent_protocol_sha256=sha(parent/'protocol.json'),
        question=p['question'],outcome_blind=True,scientific_capture_started=False,frequency_fits_started=False,test_performance_evaluated=False,
        unchanged='All544 inputs, source/FIT/TEST membership, amplitudes, periods, phases, candidate grid, OLD-only sourceCV, metrics, bootstrap, native model, one-thread3GiB300s caps and current64/current32 exact prefix requirement.',
        replaced_clause='Original historical weak/base prefix exactness compared different row positions/batch contents. Remove that cross-geometry equality as a validity gate; preserve and measure every historical difference instead.',
        mandatory_gates=['direct_head64','single_forward_per_request','prefix32_exact','background_repeat_exact','batch_replay_exact','original_geometry_preflight_rechecked'],
        historical_diagnostic='For all288 historical weak/base references save current32-old32 arrays, per-row exact/maximum/L2 statistics, input hashes and actual query batches. For each weak phase group, d=response_current-response_historical; ||d|| bounded by source-wise triangle inequality. No silent tolerance or pass/fail threshold on mixed geometry.',
        error_bound='For any FIXED predictor P and unchanged ideal signal energy E: |NMSE(current,P)-NMSE(historical,P)| <= (2*||historical-P||*||d||+||d||²)/E =2sqrt(NMSE_historical)u+u². Save measured u and conservative triangle-u. After OLD fits are frozen, audit actual score changes against these bounds for every32h weak candidate in both cohorts; these bounds do not claim selected-period invariance.',
        preflight_decision=decision,preflight_audit=audit,
        initial_recheck='Before544-node capture, replay all4 original and mixed queries; require exact original historical32, exact current64/current32, and exact agreement with saved preflight32/64 for each geometry. Any failure stops this one child attempt.',
        persistence='Raw query32/64/count/difference files are saved BEFORE capability checks, including failed calls; completion mask persisted. Original missing failed delta is not reconstructed or erased.',
        limitations='Four controlled cases isolate the geometry effect locally, not architecture-wide. Original failed numerical values were not saved. Mixed geometry roundoff may affect very small score/selection differences; report bounds and ambiguity. No further native retry/grid.',
        source_hashes={str(f.resolve()):sha(f) for f in files},preflight_files={n:sha(preflight/n) for n in evidence},
        parent_files={n:sha(parent/n) for n in ['protocol.json','knowledge.json','execution.json','bolt.log','failure_bookkeeping_audit.json','bolt_native_forecast32.npy','bolt_native_forecast64.npy']})
    write(out/'amendment.json',amendment)
    write(out/'knowledge.json',dict(status='amendment_frozen_before_capture',question=p['question'],design=p,amendment=amendment,
        exact_findings=dict(preflight=decision,native_results=None),controls=amendment['mandatory_gates'],limitations=amendment['limitations'],
        evidence_class='One authorized comparator-instrumentation recovery; no scientific outcomes yet',failed_instrumentation_runs=[],
        original_failure_preserved=dict(run=str(parent.resolve()),missing_numeric_delta=True),next_decision='Run one granted recovery child only after tests/plants; preserve raw values before every gate.'))


class Recorder:
    """Save raw queries and predictions before strict checks so a new failure remains auditable."""
    def __init__(self,out,native):self.out=out;self.native=native;self.records=[]
    def pair(self,query,label):
        """No numerical failure can discard already-produced32/64 diagnostic outputs."""
        full,prefix,calls=self.native.pair(query)
        file=f'{label}.npz';np.savez_compressed(self.out/file,query=query,current64=full,current32=prefix,prefix_difference=full[:,:32].astype(float)-prefix.astype(float))
        self.records.append(dict(label=label,file=file,query_hash=array_hash(query),shape=list(query.shape),calls=calls,raw_sha256=sha(self.out/file)))
        write(self.out/'query_records.json',self.records)
        check_pair(full,prefix,calls,self.native.native.pipeline.model_prediction_length)
        return full,prefix


def capture(out,native,p):
    """Single full-bank capture with exact current geometry gates and measured old differences."""
    a=read(out/'amendment.json');preflight=Path(a['preflight']);queries=np.load(preflight/'queries.npz');recorder=Recorder(out,native)
    for case in read(preflight/'cases.json'):
        key=case['key'];prior=np.load(preflight/f'{key}_raw.npz')
        for geometry in ('original','mixed'):
            full,prefix=recorder.pair(queries[f'{key}_{geometry}_query'],f'gate_{key}_{geometry}')
            if not np.array_equal(prefix,prior[f'{geometry}_32']) or not np.array_equal(full,prior[f'{geometry}_64']):raise ValueError('Preflight geometry replay changed in recovery')
            if geometry=='original' and not np.array_equal(prefix,queries[f'{key}_historical32']):raise ValueError('Original-geometry historical replay failed')
    inputs=np.load(out/'inputs.npy');nodes=read(out/'nodes.json');reference=np.load(out/'historical_prefix32.npy');mask=np.load(out/'historical_reference_mask.npy')
    full=np.lib.format.open_memmap(out/'bolt_native_forecast64.npy',mode='w+',dtype=np.float32,shape=(544,64))
    prefix=np.lib.format.open_memmap(out/'bolt_native_forecast32.npy',mode='w+',dtype=np.float32,shape=(544,32));filled=np.zeros(544,bool)
    started=time.monotonic()
    for sid in range(32):
        ids=[r['index'] for r in nodes if r['source_id']==sid];base=ids[0]
        x,y=recorder.pair(inputs[base:base+1],f's{sid}_base');xx,yy=recorder.pair(inputs[base:base+1],f's{sid}_base_repeat')
        if not np.array_equal(x,xx) or not np.array_equal(y,yy):raise ValueError('Background repeat failed')
        for start in range(0,len(ids),4):
            batch=ids[start:start+4];x,y=recorder.pair(inputs[batch],f's{sid}_batch{start//4}')
            full[batch]=x;prefix[batch]=y;filled[batch]=True;full.flush();prefix.flush();np.save(out/'filled_mask.npy',filled)
            if time.monotonic()-started>p['seconds_per_model'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>p['worker_cap']:raise RuntimeError('Frozen resource cap exceeded')
        x,y=recorder.pair(inputs[ids[:4]],f's{sid}_batch0_replay')
        if not np.array_equal(x,full[ids[:4]]) or not np.array_equal(y,prefix[ids[:4]]):raise ValueError('Same-batch replay failed')
        print(dict(model='bolt',source=sid,seconds=time.monotonic()-started),flush=True)
    difference,bounds=response_bounds(prefix,reference,read(out/'groups.json'));np.save(out/'historical_delta32.npy',difference)
    diagnostic=dict(references=int(mask.sum()),exact_rows=int(np.sum(np.all(difference[mask]==0,axis=1))),
        max_abs=float(np.max(abs(difference[mask]))),rms=float(np.sqrt(np.mean(difference[mask]**2))),
        per_row=[dict(index=int(i),max_abs=float(np.max(abs(difference[i]))),l2=float(np.linalg.norm(difference[i])),exact=bool(np.all(difference[i]==0))) for i in np.flatnonzero(mask)],
        response_bounds=bounds,not_a_tolerance_gate=True,formula=a['error_bound'])
    write(out/'historical_diagnostic.json',diagnostic)
    return dict(gates={key:True for key in a['mandatory_gates']},historical_mixed_geometry_diagnostic=dict(max_abs=diagnostic['max_abs'],exact_rows=diagnostic['exact_rows'],total_rows=int(mask.sum())),
        amendment_sha256=sha(out/'amendment.json'),seconds=time.monotonic()-started,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        hashes={name:sha(out/name) for name in ['bolt_native_forecast64.npy','bolt_native_forecast32.npy','filled_mask.npy','historical_delta32.npy','historical_diagnostic.json','query_records.json']})


def worker(out,grant):
    """Require the new amendment-bound grant before loading one native model."""
    require_recovery_grant(out,grant);p=validate_recovery(out,True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):raise ValueError('CPU1thread required')
    if list(out.glob('gate_*.npz')) or list(out.glob('bolt_native_forecast*.npy')):raise ValueError('Refuse recovery repeat')
    native=DiagnosticBolt(p['checkpoints']['bolt']['path']);result=capture(out,native,p);validate_recovery(out);write(out/'bolt_worker.json',result)


def run(out,grant):
    """One300s/3GiB child attempt; failures remain terminal and raw arrays remain saved."""
    import fcntl
    release=require_recovery_grant(out,grant);p=validate_recovery(out)
    with (out/'native.lock').open('a',encoding='utf-8') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'execution.json').exists() or list(out.glob('gate_*.npz')):raise ValueError('One recovery attempt only')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        start=time.monotonic();peak=0;failure=None;k=read(out/'knowledge.json');k['status']='native_recovery_running';write(out/'knowledge.json',k)
        with (out/'bolt.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.response_phase_horizon_recovery','worker','--out',str(out),'--grant',str(grant)],env=env,stdout=log,stderr=subprocess.STDOUT)
            print(dict(event='native_launch',stage='one_recovery_child',pid=proc.pid,slot=release['slot'],run=str(out.resolve())),flush=True)
            while proc.poll() is None:
                try:
                    status=Path(f'/proc/{proc.pid}/status').read_text(encoding='utf-8');peak=max(peak,int(next(l.split()[1] for l in status.splitlines() if l.startswith('VmRSS:')))*1024)
                except (FileNotFoundError,StopIteration):pass
                if peak>p['worker_cap'] or time.monotonic()-start>p['seconds_per_model']:
                    failure='Frozen recovery resource cap';proc.kill();proc.wait();break
                time.sleep(.2)
        wf=out/'bolt_worker.json'
        if wf.exists() and read(wf)['peak_rss']>p['worker_cap']:failure='Worker peak exceeds cap'
        if proc.returncode!=0 and failure is None:failure=f'Native recovery exited{proc.returncode}; raw arrays/log preserved'
        result=dict(valid=proc.returncode==0 and failure is None and wf.exists(),returncode=proc.returncode,seconds=time.monotonic()-start,peak_rss=peak,failure=failure,slot=release['slot'],amendment_sha256=sha(out/'amendment.json'))
        write(out/'execution.json',dict(bolt=result));k=read(out/'knowledge.json');k['execution']={'bolt':result};k['status']='captured_pending_amended_audit' if result['valid'] else 'recovery_failed_branch_closed'
        if not result['valid']:k['failed_instrumentation_runs'].append(result)
        k['next_decision']='Release slot. No further native branch. Valid capture permits frozen OLD-only fits then TEST and independent arithmetic/bounds audit.' if result['valid'] else 'STOP. No fits, no retries or grid.'
        write(out/'knowledge.json',k);print(dict(event='native_finish',stage='one_recovery_child',**result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','worker','fit','evaluate'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--parent',type=Path,default=Path('runs/response_phase_horizon4954600'))
    parser.add_argument('--preflight',type=Path,default=Path('runs/response_phase_horizon_replay4954700'));parser.add_argument('--grant',type=Path);args=parser.parse_args()
    if args.command=='prepare':prepare(args.out,args.parent,args.preflight)
    elif args.command in ('run','worker'):(run if args.command=='run' else worker)(args.out,args.grant)
    else:
        validate_recovery(args.out)
        from experiments.feature_search.response_phase_horizon_analysis import fit,evaluate
        try:(fit if args.command=='fit' else evaluate)(args.out)
        except Exception as error:
            k=read(args.out/'knowledge.json');k['status']='recovery_analysis_failed';k['failed_instrumentation_runs'].append(dict(stage=args.command,error=repr(error)));write(args.out/'knowledge.json',k);raise
