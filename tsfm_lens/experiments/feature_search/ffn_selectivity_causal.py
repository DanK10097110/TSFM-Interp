"""Bounded Bolt-first FFN unit-delta removal, prepared separately from stage1.

FIT-selected native axes and FIT activity matching are frozen before fresh
background evaluation. No parameter fitting, SAE, future targets or native
execution is performed by prepare. The dispatcher requires a new slot grant.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import os
import time
import resource
import numpy as np

from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search import ffn_selectivity_study as stage1

SEED=4963000
TYPES=stage1.TYPES
CELL_CAP=384
MEMORY=3*2**30
SECONDS=900


def knowledge(out,phase,status,findings,next_decision):
    """Persist preparation, execution, analysis and failed instrumentation events."""
    path=out/'knowledge.json'
    previous=read(path) if path.exists() else dict(events=[])
    event=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),phase=phase,status=status,
        question='Do FIT-selected Bolt first-FFN units remove more type-specific native forecast response than FIT activity-matched random units?',
        design='Reused8 synthetic TEST backgrounds k1/k4; fresh8 synthetic backgrounds k4; five types/source, one balanced variant/type, eight fixed matched draws assigned one/source;384 Bolt cells total',
        exact_findings=findings,controls=['FIT-only raw and perturbation-delta RMS activity matching','Matched random units at equal budgets',
            'Exact background normalization/repeat/self-patch checks','Full within-FFN reach and exact selected-only patch fidelity','Fixed native parameter hash','Type-specific off-target and energy diagnostics'],
        limitations=['Reused TEST outcomes are exploratory','Fresh confirmation is limited to this chosen site/budget and synthetic generator',
            'One of eight random draws per source, not eight controls per source; bootstrap conditions on fixed draws and FIT selection',
            'Only one balanced phase/sign variant per type/source','Background-locked normalization is a controlled counterfactual',
            'Matched activity is not matched function or exact removed energy','No general monosemanticity or true-future accuracy claim'],
        evidence_class='instrumentation' if phase!='analysis' else 'within-model causal perturbation-response intervention',next_decision=next_decision)
    previous.update(event)
    previous['events'].append(event)
    write(path,previous)


def activity(raw,fit,conditions):
    """FIT-only baseline RMS and equal-type perturbation-delta RMS per native unit."""
    x=np.asarray(raw)[np.asarray(fit,bool)].astype(float)
    base=np.sqrt(np.mean(x[:,0]**2,axis=(0,1)))
    delta=x[:,1:]-x[:,0,None]
    energy=[]
    for kind in TYPES:
        indices=[i for i,c in enumerate(conditions) if c['kind']==kind]
        energy.append(np.mean(delta[:,indices]**2,axis=(0,1,2)))
    return np.stack([base,np.sqrt(np.mean(energy,axis=0))],axis=1)


def match_units(values,selected,seed=SEED+90):
    """Eight nested random sets, sampling32 nearest FIT matches within factor2."""
    values=np.asarray(values,float)
    if len(selected)!=4 or len(set(selected))!=4 or np.any(values[selected]<=0):
        raise ValueError('Need four distinct FIT-active selected units')
    rng=np.random.default_rng(seed)
    draws=np.zeros((8,4),int)
    details=[]
    for column,unit in enumerate(selected):
        valid=np.all(values>0,axis=1)
        valid[selected]=False
        distance=np.max(abs(np.log(np.maximum(values,1e-300)/values[unit])),axis=1)
        pool=np.flatnonzero(valid&(distance<=np.log(2)))
        pool=pool[np.argsort(distance[pool],kind='stable')[:32]]
        if len(pool)<8:
            raise ValueError('Insufficient FIT activity matches; no caliper relaxation')
        used=set()
        for draw in range(8):
            available=[int(u) for u in pool if u not in used and u not in draws[draw,:column]]
            if not available:
                raise ValueError('Cannot form distinct matched sets')
            chosen=int(rng.choice(available))
            draws[draw,column]=chosen
            used.add(chosen)
        details.append(dict(selected_unit=int(unit),FIT_activity=values[unit].tolist(),pool=pool.tolist(),
            matched_units=draws[:,column].tolist(),matched_activity=values[draws[:,column]].tolist(),
            maximum_log_ratio=float(distance[draws[:,column]].max())))
    return dict(selected=selected,k1=[selected[0]],draws=draws.tolist(),details=details,seed=seed,
                matching='FIT baseline RMS and equal-type delta RMS, factor2 caliper, nearest32; no selected-unit overlap; nested k1/k4;8 fixed draws')


def condition_indices(index):
    """Exactly balanced phase/sign schedules across each eight-source cohort."""
    return [index%4,4+(index+1)%4,8+index%2,10+(index+1)%2,12+index%2]


def patch_units(perturbed,background,units):
    """Subtract only the chosen postactivation unit deltas, at every token."""
    result=np.array(perturbed,copy=True)
    if result.shape!=np.shape(background) or len(set(units))!=len(units):
        raise ValueError('Patch geometry/units invalid')
    result[...,units]=np.asarray(background)[...,units]
    return result


def response_metrics(baseline,clean,patched,input_rms=.1):
    """Signed response removal, remaining energy and orthogonal forecast changes."""
    response=np.asarray(clean,float)-baseline
    removed=np.asarray(clean,float)-patched
    energy=float(np.mean(response**2))
    changed=float(np.mean(removed**2))
    valid=energy>1e-6*input_rms**2
    projection=float(np.mean(removed*response)/energy) if valid else None
    orthogonal=float(np.mean((removed-projection*response)**2)/energy) if valid else None
    return dict(response_energy=energy,removed_output_energy=changed,remaining_output_energy=float(np.mean((np.asarray(patched)-baseline)**2)),
        eligible_response=valid,signed_response_removal=projection,orthogonal_change_ratio=orthogonal,
        changed_energy_ratio=changed/energy if valid else None,
        remaining_energy_ratio=float(np.mean((np.asarray(patched)-baseline)**2)/energy) if valid else None)


def parameter_hash(model):
    """Hash parameter values before and after all interventions without updates."""
    digest=hashlib.sha256()
    for name,value in model.named_parameters():
        array=value.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype,array.shape)).encode())
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def prepare(out):
    """Freeze cohort, FIT matching, intervention schedule and budget; no model loads."""
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():
        raise FileExistsError('Refuse frozen causal preparation overwrite')
    try:
        source=out.parent/'ffn_selectivity_pilot4961000'
        p=read(source/'protocol.json')
        audited=read(source/'independent_audit.json')
        if not audited['valid'] or audited['analysis_sha256']!=sha(source/'analysis.json'):
            raise ValueError('Audited stage1 analysis required')
        for path,digest in p['sources'].items():
            if sha(path)!=digest:
                raise ValueError('Stage1 source changed')
        for name,digest in p['hashes'].items():
            if sha(source/name)!=digest:
                raise ValueError('Stage1 artifact changed')
        worker=read(source/'bolt_worker.json')
        if not worker['valid'] or worker['sha256']!=sha(source/'bolt_captures.npz'):
            raise ValueError('Stage1 Bolt capture hash failed')
        rows=read(source/'rows.json');conditions=read(source/'conditions.json')
        fit=np.asarray([r['split']=='FIT' for r in rows])
        records=read(source/'analysis.json')['records']
        rules={r['budget']:r for r in records if r['model']=='bolt' and r['site']=='first' and r['space']=='ffn' and r['basis']=='native' and r['view']=='rms'}
        selected=rules[4]['FIT_rule']['indices']
        if selected!=[1925,909,108,901] or rules[1]['FIT_rule']['indices']!=selected[:1]:
            raise ValueError('Reviewed stage1 selected units changed')
        captures=np.load(source/'bolt_captures.npz',allow_pickle=False)
        if captures['source_ids'].tolist()!=[r['source'] for r in rows]:
            raise ValueError('Stage1 source alignment failed')
        matching=match_units(activity(captures['first_ffn'],fit,conditions),selected)
        old=np.load(source/'inputs.npz',allow_pickle=False)['queries']
        reused=[i for i,r in enumerate(rows) if r['kind']=='synthetic' and r['split']=='TEST']
        if len(reused)!=8:
            raise ValueError('Exactly eight reused synthetic TEST sources required')
        new_rows=[];queries=[]
        for group in ('reused_TEST','fresh_confirmation'):
            for j in range(8):
                indices=condition_indices(j)
                if group=='reused_TEST':
                    i=reused[j];seed=rows[i]['seed'];source_id=rows[i]['source']
                    query=old[i,[0]+[k+1 for k in indices]]
                else:
                    seed=SEED+j;source_id=f'syn:{seed}'
                    x=stage1.synthetic_background(seed,j%4)
                    delta,_=stage1.perturbations(seed)
                    query=np.concatenate([x[None],x[None]+delta[indices]]).astype(np.float32)
                new_rows.append(dict(source=source_id,seed=seed,group=group,family=j%4,index=len(new_rows),
                    condition_indices=indices,conditions=[conditions[k] for k in indices],
                    control_draw=j if group=='reused_TEST' else (j+3)%8,budgets=[1,4] if group=='reused_TEST' else [4]))
                queries.append(query)
        if len({r['source'] for r in new_rows})!=16 or {r['source'] for r in new_rows[8:]}&{r['source'] for r in rows}:
            raise ValueError('Fresh source leakage')
        queries=np.asarray(queries)
        error=float(np.max(abs(np.sqrt(np.mean((queries[:,1:].astype(float)-queries[:,0,None])**2,axis=-1))-.1)))
        if error>1e-7:
            raise ValueError('Actual input RMS mismatch')
        cells=sum(9+10*len(r['budgets']) for r in new_rows)
        if cells!=CELL_CAP:
            raise ValueError('Causal cell budget mismatch')
        np.savez_compressed(out/'inputs.npz',queries=queries)
        write(out/'rows.json',new_rows);write(out/'matching.json',matching)
        execution=read(source/'execution.json')['bolt']
        budget=dict(cells=cells,reused_cells=232,fresh_cells=152,reference_seconds=execution['seconds'],reference_cells=480,
            estimated_seconds=execution['seconds']*cells/480,conservative_seconds=execution['seconds']*cells/480*3,
            cpu_threads=1,resident_cap_bytes=MEMORY,seconds_cap=SECONDS,
            breakdown_per_source='1 unlocked background +1 locked repeat +5 clean perturbations +1 exact self-patch +1 full-FFN reach +5 types*2 arms*number_of_budgets')
        write(out/'budget.json',budget)
        from experiments.feature_search import ffn_selectivity_causal_run,ffn_selectivity_causal_analysis
        paths=[Path(__file__),Path(ffn_selectivity_causal_run.__file__),Path(ffn_selectivity_causal_analysis.__file__)]
        artifact_names=['inputs.npz','rows.json','matching.json','budget.json']
        bindings=['protocol.json','analysis.json','independent_audit.json','bolt_worker.json','bolt_captures.npz','inputs.npz','rows.json','conditions.json','causal_plan.json','execution.json']
        protocol=dict(version=1,run=out.name,hold_native=True,requires_new_slot=True,model='bolt',site='first',no_SAE=True,
            native_parameters='Fixed; no optimizer/gradient fitting; parameter hash identical before/after',
            checkpoint=p['checkpoints']['bolt'],capture=p['capture']['bolt']['sites'][0],history=128,horizon=32,
            native_point_forecast='Median',normalization='Background-locked native input and inverse normalization',
            condition_types=list(TYPES),input_RMS=.1,backgrounds=16,cohort=dict(reused_TEST=8,fresh_confirmation=8),
            selected=selected,budgets=dict(reused_TEST=[1,4],fresh_confirmation=[4]),matching=matching['matching'],
            random_control_assignment='Eight fixed draws, exactly one/source within each cohort; shared across all five types and nested k1/k4. Not eight controls/source. Bootstrap conditional on draws.',
            intervention='Set selected perturbed first-FFN postactivation coordinates to that source background values at all nine tokens; leave other coordinates unchanged',
            target_types={'1':['ramprecent64'],'4':['ramprecent64','noise']},
            target_definition='Union of FIT-preferred types of selected units; not chosen using causal TEST outcomes',
            primary='Fresh k4 source-paired native-minus-matched signed response-removal fraction aggregated over ramp/noise, equal type weighting',
            secondary='Reused TEST k1/k4 exploratory; per-type response changes for all five types; off-target aggregate; all native and control values reported',
            response_removal='dot(clean-patched,clean-background)/sum((clean-background)^2); signed, may be negative or above1',
            energy='Report baseline response, removed forecast and remaining forecast energy; actual removed FFN energy and share; orthogonal change ratio',
            off_target='sin24,sin19.2,steprecent32 for k4; additionally noise for k1. Report signed removal, absolute change energy and orthogonal change; do not relabel changed output as selective removal.',
            response_floor=1e-6*.1**2,small_response='Ratios undefined at/below floor; retain absolute metrics and report source counts/exclusions; never replace sources',
            uncertainty='1999 paired source-group bootstrap; n8 fresh, n8 reused separately; no token/type/variant resampling. Primary single frozen fresh k4 contrast; all other intervals descriptive.',
            interpretation='Positive signed removal vs controls supports within-model forecast-response dependence for these stimuli; energy and off-target changes constrain selectivity; no general monosemanticity',
            gates=dict(normalizer_identity_exact=True,self_patch_exact=True,full_FFN_reach_gt=1e-6,selected_only_patch_exact=True,
                parameters_unchanged=True,finite=True,resource_caps=True),
            cells=cells,cells_per_source=dict(reused_TEST=29,fresh_confirmation=19),cpu_threads=1,memory_cap=MEMORY,seconds_cap=SECONDS,
            sources={**p['sources'],**{str(path.resolve()):sha(path) for path in paths}},
            stage1_bindings={str(source/name):sha(source/name) for name in bindings},hashes={name:sha(out/name) for name in artifact_names})
        write(out/'protocol.json',protocol)
    except Exception as exc:
        knowledge(out,'prepare','failed',dict(error_type=type(exc).__name__,error=str(exc),native_launched=False),'Resolve preparation failure; HOLD native.')
        raise
    knowledge(out,'prepare','complete',dict(protocol_sha256=sha(out/'protocol.json'),budget=budget,matching=matching,input_RMS_error=error,
        native_launched=False,native_gates_pending=True),'HOLD native until a new grant; stage2 preparation only.')
    print(dict(status='READY_HOLD_NATIVE',budget=budget),flush=True)


def worker(out,native_factory=None):
    """Run only the frozen Bolt-first causal schedule in a granted monitored worker."""
    if os.environ.get('FFN_CAUSAL_MONITORED')!='1':
        raise RuntimeError('HOLD native: monitored grant required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in stage1.THREAD_VARS):
        raise RuntimeError('Requires one-thread CPU worker')
    p=read(out/'protocol.json')
    for path,digest in p['sources'].items():
        if sha(path)!=digest:
            raise ValueError('Frozen causal source changed')
    for name,digest in p['hashes'].items():
        if sha(out/name)!=digest:
            raise ValueError('Frozen causal artifact changed')
    if (out/'bolt_causal.npz').exists():
        raise FileExistsError('Refuse native overwrite')
    for name,digest in p['checkpoint']['weights'].items():
        if sha(Path(p['checkpoint']['path'])/name)!=digest:
            raise ValueError('Checkpoint changed')
    if native_factory is None:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from experiments.feature_search.real_background_poc import Native
        native_factory=Native
    started=time.monotonic();native=native_factory('bolt',p['checkpoint']['path'])
    selected={'first':stage1.sites(native)['first']}
    before_hash=parameter_hash(native.model)
    rows=read(out/'rows.json');matching=read(out/'matching.json')
    queries=np.load(out/'inputs.npz',allow_pickle=False)['queries']
    if queries.shape!=(len(rows),6,128):
        raise ValueError('Causal query geometry mismatch')
    count=0
    def invoke(x,replacement=None):
        nonlocal count
        count+=len(x)
        if count>p['cells'] or count>CELL_CAP:
            raise RuntimeError('Native cell cap exceeded')
        if time.monotonic()-started>SECONDS or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>MEMORY:
            raise RuntimeError('Resource cap exceeded')
        return stage1.capture(native,x,selected,replacement)
    clean_forecasts=[];clean_states=[];patched_forecasts=[];patched_states=[];events=[];reach=[]
    for row,q in zip(rows,queries):
        with stage1.locked_background(native,q[:1],invoke) as (reference,original):
            baseline,states=invoke(q[:1])
            stage1.exact_gate(reference,baseline,'background normalization/repeat')
            for key,value in states.items():
                stage1.exact_gate(original[key],value,'background state repeat')
            anchor=states['first_ffn']
            if anchor.shape!=(1,p['capture']['tokens'],p['capture']['ffn_width']):
                raise ValueError('FFN geometry mismatch')
            forecasts=[baseline[0]];source_states=[anchor[0]]
            for condition,query in enumerate(q[1:]):
                clean,observed=invoke(query[None]);ffn=observed['first_ffn']
                forecasts.append(clean[0]);source_states.append(ffn[0])
                if condition==0:
                    replay,_=invoke(query[None],('first',ffn))
                    stage1.exact_gate(clean,replay,'FFN self-patch')
                    zero,_=invoke(query[None],('first',np.zeros_like(ffn)))
                    reach.append(dict(source=row['source'],relative_change=stage1.reach_gate(clean,zero)))
                for budget in row['budgets']:
                    for arm in ('selected','matched'):
                        units=matching['selected'][:budget] if arm=='selected' else matching['draws'][row['control_draw']][:budget]
                        replacement=patch_units(ffn,anchor,units)
                        pred,actual=invoke(query[None],('first',replacement))
                        stage1.exact_gate(replacement,actual['first_ffn'],'selected-only intervention')
                        delta=ffn.astype(float)-anchor
                        removed=ffn.astype(float)-replacement
                        energy=float(np.sum(delta**2));removed_energy=float(np.sum(removed**2))
                        events.append(dict(index=len(events),source_index=row['index'],source=row['source'],group=row['group'],
                            condition=condition,kind=TYPES[condition],budget=budget,arm=arm,units=units,control_draw=row['control_draw'],
                            FFN_delta_energy=energy,removed_FFN_energy=removed_energy,removed_FFN_share=removed_energy/energy if energy>0 else None))
                        patched_forecasts.append(pred[0]);patched_states.append(actual['first_ffn'][0])
            clean_forecasts.append(forecasts);clean_states.append(source_states)
    after_hash=parameter_hash(native.model)
    if before_hash!=after_hash or count!=p['cells']:
        raise ValueError('Parameters changed or cell count mismatch')
    arrays=dict(clean_forecasts=np.asarray(clean_forecasts),clean_FFN=np.asarray(clean_states),
        patched_forecasts=np.asarray(patched_forecasts),patched_FFN=np.asarray(patched_states))
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite causal outputs')
    np.savez_compressed(out/'bolt_causal.npz',**arrays,source_ids=np.asarray([r['source'] for r in rows]))
    write(out/'interventions.json',events)
    write(out/'bolt_worker.json',dict(valid=True,cells=count,seconds=time.monotonic()-started,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,reach=reach,
        background_repeat_exact=True,normalizer_identity_exact=True,self_patch_exact=True,selected_only_patch_exact=True,
        parameter_hash_before=before_hash,parameter_hash_after=after_hash,
        protocol_sha256=sha(out/'protocol.json'),arrays_sha256=sha(out/'bolt_causal.npz'),interventions_sha256=sha(out/'interventions.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','worker'])
    parser.add_argument('--out',type=Path,default=Path('runs/ffn_selectivity_causal4963000'))
    args=parser.parse_args()
    prepare(args.out) if args.command=='prepare' else worker(args.out)
