"""One bounded original-batch replay diagnostic before any horizon recovery amendment.

Four cases are fixed before inference: weak19.2 phase0 at source0 and background
at source1, separately OLD FIT and transfer TEST. All raw queries,32/64 outputs,
historical targets and numerical differences are saved before a gate decision.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.response_surface_focused import validate,require_grant


def array_hash(x):
    """Bind dtype, shape and ordered contiguous query bytes, not just selected history."""
    x=np.ascontiguousarray(x)
    return hashlib.sha256(str((x.dtype.str,x.shape)).encode()+x.tobytes()).hexdigest()


def original_batch(nodes,source,index):
    """Replay the frozen workers' per-source list order and successive chunks of four."""
    ids=[r['index'] for r in nodes if r['source_id']==source]
    position=ids.index(index);batch=ids[4*(position//4):4*(position//4)+4]
    return batch,position%4


def differences(a,b):
    """Exact and algebraic difference statistics; no tolerance-based eligibility."""
    difference=np.asarray(a,float)-np.asarray(b,float)
    return dict(exact=bool(np.array_equal(a,b)),max_abs=float(np.max(abs(difference))),
                rms=float(np.sqrt(np.mean(difference**2))),l2=float(np.linalg.norm(difference)),
                unequal_entries=int(np.count_nonzero(difference)))


def classify(records):
    """Recovery is conditional on exact same-geometry replay and reproduced geometry effects."""
    good=len(records)==4 and all(r['same_geometry_exact'] and r['current_prefix_exact'] and r['single_forward'] and r['repeat_exact'] and r['target_query_identical'] for r in records)
    changed=any(not r['mixed_target_vs_historical']['exact'] for r in records)
    return dict(eligible=bool(good and changed),same_geometry_all_exact=bool(good),geometry_effect_reproduced=bool(changed),
                reason='Exact original geometry replays; only changed query geometry changes selected-row outputs in the fixed cases.' if good and changed else 'Stop: same-geometry/prefix/repeat failure, missing cases, or original mismatch not reproduced; cause not isolated.')


def prepare(out,parent):
    """Freeze representative original/mixed batches without opening model weights."""
    p=validate(parent);out.mkdir(parents=True,exist_ok=False)
    parent_inputs=np.load(parent/'inputs.npy');parent_nodes=read(parent/'nodes.json');parent_groups=read(parent/'groups.json')
    root=Path(__file__).resolve().parents[2];old=root/'runs/response_phase4954200';fit=Path(read(old/'protocol.json')['prior']);test=root/'runs/response_phase_transfer4954300'
    cases=[];arrays={};bindings={str((parent/n).resolve()):sha(parent/n) for n in ['protocol.json','inputs.npy','nodes.json','groups.json','knowledge.json','bolt.log','failure_bookkeeping_audit.json']}
    for role,run,groupfile in [('FIT',fit,old/'groups.json'),('TEST',test,test/'groups.json')]:
        inputs=np.load(run/'inputs.npy');forecast=np.load(run/'bolt_native_forecast.npy');nodes=read(run/'nodes.json');groups=read(groupfile)
        for sid,kind in [(0,'weak19.2_phase0'),(1,'base')]:
            g=next(g for g in groups if g['source_id']==sid and abs(g['frequency']-128/19.2)<1e-12 and g.get('offset',0)==0)
            newg=next(g for g in parent_groups if g['cohort']==role and g['cohort_source_id']==sid and g['input_period']==19.2 and g['amplitude']==.1)
            old_index=g['indices'][0] if kind!='base' else g['base'];new_index=newg['indices'][0] if kind!='base' else newg['base']
            old_ids,old_row=original_batch(nodes,sid,old_index);new_ids,new_row=original_batch(parent_nodes,newg['source_id'],new_index)
            old_query=inputs[old_ids].copy();mixed_query=parent_inputs[new_ids].copy()
            if not np.array_equal(old_query[old_row],mixed_query[new_row]):raise ValueError('Selected history differs')
            key=f'{role}_{kind}';arrays[f'{key}_original_query']=old_query;arrays[f'{key}_mixed_query']=mixed_query
            arrays[f'{key}_historical32']=forecast[old_ids].copy()
            cases.append(dict(key=key,cohort=role,source=g['source'],source_id=sid,kind=kind,original_ids=old_ids,original_row=old_row,
                mixed_ids=new_ids,mixed_row=new_row,original_query_hash=array_hash(old_query),mixed_query_hash=array_hash(mixed_query),target_query_hash=array_hash(old_query[old_row])))
        for file in [run/'inputs.npy',run/'nodes.json',run/'bolt_native_forecast.npy',run/'bolt_worker.json',groupfile]:bindings[str(file.resolve())]=sha(file)
    np.savez_compressed(out/'queries.npz',**arrays);write(out/'cases.json',cases)
    files=[Path(__file__),root/'tests/test_response_phase_replay_preflight.py',root/'experiments/feature_search/response_phase_replay_preflight_plants.py']
    sources=dict(p['source_hashes']);sources.update({str(f.resolve()):sha(f) for f in files})
    protocol=dict(question='Does exact original batch membership/order/row-position replay isolate the failed historical comparator to changed batch geometry?',
        parent=str(parent.resolve()),parent_protocol_sha256=sha(parent/'protocol.json'),cases=cases,
        ordering='Exactly frozen focused/transfer workers: filter nodes by source preserving list order; chunk every4. Preserve entire original query and selected row, not merely batch size.',
        sequence=['original','mixed','original_repeat','mixed_repeat'],
        gates='All original32 rows must equal saved historical32 bit-for-bit; repeat originals/mixed bit-for-bit; every current64 prefix equals current32; one forward per horizon request. At least one mixed selected-row mismatch must reproduce. No tolerance. Otherwise stop with no recovery child.',
        inference='No frequency fitting or TEST performance evaluation. TEST cases are instrumentation-only replay comparisons. Four cases cannot establish an architecture-wide cause.',
        persistence='Save every raw32/64 forecast, call counts, query hashes and difference array before deciding eligibility. Original failed batch values remain unavailable.',
        cpu_threads=1,worker_cap=3*2**30,seconds_per_model=60,checkpoints=p['checkpoints'],source_hashes=sources,
        hashes={n:sha(out/n) for n in ['queries.npz','cases.json']},prior_bindings=bindings)
    write(out/'protocol.json',protocol);write(out/'knowledge.json',dict(status='prepared_replay_diagnostic',question=protocol['question'],design=protocol,
        exact_findings=None,controls=protocol['gates'],limitations=protocol['inference'],evidence_class='Comparator instrumentation diagnostic only',
        failed_instrumentation_runs=[],next_decision='Run once under granted slot2; original-geometry failure or unisolated cause stops branch. No frequency metrics.'))


class DiagnosticBolt:
    """Same pipeline path as the failed worker, returning raw values before any gate."""
    def __init__(self,checkpoint):
        from experiments.feature_search.real_background_poc import Native
        self.native=Native('bolt',checkpoint)
        if self.native.pipeline.model_prediction_length!=64:raise ValueError('Wrong direct head')

    def pair(self,x):
        """One64 and one32 request; all outputs survive even a subsequent parity rejection."""
        import torch
        query=torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float32);outputs=[];counts=[]
        for horizon in (64,32):
            calls=[];handle=self.native.model.register_forward_hook(lambda *args:calls.append(1))
            try:
                with torch.inference_mode():_,point=self.native.pipeline.predict_quantiles(query,prediction_length=horizon,quantile_levels=[.5],limit_prediction_length=True)
                outputs.append(point.detach().cpu().numpy().copy());counts.append(len(calls))
            finally:handle.remove()
        return outputs[0],outputs[1],counts


def worker(out,grant):
    """Persist diagnostic arrays before fail-closed eligibility, without scientific fits."""
    require_grant(out,grant,'bolt');p=validate(out,'bolt')
    for file,digest in p['prior_bindings'].items():
        if sha(file)!=digest:raise ValueError('Prior binding changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):raise ValueError('CPU1thread required')
    if (out/'diagnostic.json').exists() or list(out.glob('*_raw.npz')):raise ValueError('Refuse diagnostic repeat')
    native=DiagnosticBolt(p['checkpoints']['bolt']['path']);queries=np.load(out/'queries.npz');records=[]
    for case in p['cases']:
        key=case['key'];old=queries[f'{key}_historical32'];raw={};calls={}
        for name in p['sequence']:
            query=queries[f'{key}_{"original" if name.startswith("original") else "mixed"}_query']
            full,prefix,count=native.pair(query)
            raw[f'{name}_64']=full;raw[f'{name}_32']=prefix;calls[name]=count
            raw[f'{name}_prefix_difference']=full[:,:32].astype(float)-prefix.astype(float)
            if name.startswith('original'):raw[f'{name}_historical_difference']=prefix.astype(float)-old.astype(float)
            else:raw[f'{name}_target_historical_difference']=prefix[case['mixed_row']].astype(float)-old[case['original_row']].astype(float)
            np.savez_compressed(out/f'{key}_raw.npz',**raw)
            write(out/f'{key}_calls.json',calls)
        same=differences(raw['original_32'],old);mixed=differences(raw['mixed_32'][case['mixed_row']],old[case['original_row']])
        record=dict(case,same_geometry_exact=same['exact'] and np.array_equal(raw['original_repeat_32'],old),same_geometry_vs_historical=same,
            mixed_target_vs_historical=mixed,current_prefix_exact=all(np.array_equal(raw[f'{name}_64'][:,:32],raw[f'{name}_32']) for name in p['sequence']),
            single_forward=all(c==[1,1] for c in calls.values()),repeat_exact=all(np.array_equal(raw[f'{name}_{h}'],raw[f'{name}_repeat_{h}']) for name in ('original','mixed') for h in (32,64)),
            target_query_identical=np.array_equal(queries[f'{key}_original_query'][case['original_row']],queries[f'{key}_mixed_query'][case['mixed_row']]),
            numerical_differences={name:differences(raw[f'{name}_64'][:,:32],raw[f'{name}_32']) for name in p['sequence']},raw_sha256=sha(out/f'{key}_raw.npz'))
        records.append(record);write(out/'diagnostic.json',dict(records=records,decision=classify(records)))
        print(dict(case=key,same_geometry=record['same_geometry_vs_historical'],mixed_geometry=mixed,repeat_exact=record['repeat_exact']),flush=True)
        if not all(record[k] for k in ('same_geometry_exact','current_prefix_exact','single_forward','repeat_exact','target_query_identical')):break
    decision=classify(records);k=read(out/'knowledge.json');k.update(status='replay_eligible_for_outcome_blind_amendment' if decision['eligible'] else 'replay_failed_branch_closed',exact_findings=dict(records=records,decision=decision),next_decision='Freeze a separate amendment before any scientific fit/evaluation.' if decision['eligible'] else 'STOP. No child capture or frequency fitting; no retry loop.')
    if not decision['eligible']:k['failed_instrumentation_runs'].append(dict(stage='original_geometry_replay',decision=decision))
    write(out/'knowledge.json',k)
    write(out/'bolt_worker.json',dict(diagnostic_complete=True,eligible=decision['eligible'],peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,hashes={f.name:sha(f) for f in out.glob('*_raw.npz')}))


def run(out,grant):
    """One60-second diagnostic worker, with no automatic child launch or retry."""
    import fcntl
    release=require_grant(out,grant,'bolt');p=validate(out)
    with (out/'native.lock').open('a',encoding='utf-8') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'execution.json').exists() or list(out.glob('*_raw.npz')):raise ValueError('Refuse second diagnostic attempt')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        start=time.monotonic();peak=0;failure=None
        with (out/'bolt.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.response_phase_replay_preflight','worker','--out',str(out),'--grant',str(grant)],env=env,stdout=log,stderr=subprocess.STDOUT)
            print(dict(event='native_launch',stage='replay_preflight',pid=proc.pid,run=str(out.resolve()),slot=release['slot']),flush=True)
            while proc.poll() is None:
                try:
                    status=Path(f'/proc/{proc.pid}/status').read_text(encoding='utf-8');peak=max(peak,int(next(l.split()[1] for l in status.splitlines() if l.startswith('VmRSS:')))*1024)
                except (FileNotFoundError,StopIteration):pass
                if peak>p['worker_cap'] or time.monotonic()-start>p['seconds_per_model']:
                    failure='Diagnostic resource cap';proc.kill();proc.wait();break
                time.sleep(.2)
        wf=out/'bolt_worker.json'
        if wf.exists() and read(wf)['peak_rss']>p['worker_cap']:failure='Worker peak exceeds cap'
        valid=proc.returncode==0 and failure is None and wf.exists()
        result=dict(valid_execution=valid,eligible=valid and read(wf)['eligible'],returncode=proc.returncode,seconds=time.monotonic()-start,peak_rss=peak,failure=failure,slot=release['slot'])
        write(out/'execution.json',result);k=read(out/'knowledge.json');k['execution']=result
        if not valid:k.update(status='replay_execution_failed_branch_closed',next_decision='STOP; preserve diagnostics, no retry or child capture.');k['failed_instrumentation_runs'].append(result)
        write(out/'knowledge.json',k);print(dict(event='native_finish',stage='replay_preflight',**result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','worker'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--parent',type=Path,default=Path('runs/response_phase_horizon4954600'));parser.add_argument('--grant',type=Path);args=parser.parse_args()
    if args.command=='prepare':prepare(args.out,args.parent)
    else:(run if args.command=='run' else worker)(args.out,args.grant)
