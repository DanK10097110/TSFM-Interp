"""Frozen32-source Bolt-first k4 replication with four crossed FIT controls.

Every source receives all four first-frozen matched sets, separating source
variation from the previous one-draw/source assignment. No unit/dose/depth
search. All new energy arithmetic is float64; the prior run stays unchanged.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import os
import resource
import subprocess
import sys
import time
import numpy as np
from experiments.feature_search import ffn_selectivity_causal as causal
from experiments.feature_search.ffn_selectivity_causal_audit import edit_audit

read,write,sha=causal.read,causal.write,causal.sha
SEED=4964000
COUNT=32
CELLS=1088
MEMORY=3*2**30
SECONDS=900
ARMS=('selected','control0','control1','control2','control3')


def knowledge(out,phase,status,findings,next_decision):
    """Persist every stage, including failed instrumentation, without old-run edits."""
    path=out/'knowledge.json';old=read(path) if path.exists() else dict(events=[])
    event=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),phase=phase,status=status,
        question='Does the chosen Bolt-first k4 classification/selectivity advantage imply stable causal forecast-response importance versus fixed crossed controls?',
        design='32 fresh synthetic sources4964000..4964031, fixed k4 units and ramp/noise targets, all5 types/source, first4 frozen FIT matched sets crossed with every source;1088 cells',
        exact_findings=findings,controls=['Four fixed matched sets crossed with every source','Exact repeat/self-patch and fixed parameter hash',
            'Actual selected/matched edit norms and output reach distinct from full-FFN reach','Source-paired bootstrap, no control/token resampling'],
        limitations=['Chosen model/site/k4 only; no general monosemanticity','Structured synthetic backgrounds and one balanced variant/type/source',
            'Fixed controls are not a population sample of all possible matched sets','FIT activity matching does not force equal actual intervention energy',
            'Background-locked normalization is a controlled counterfactual','Practical±.02 band is descriptive; no rescue or replacement of primary'],
        evidence_class='instrumentation' if phase!='analysis' else 'within-model causal synthetic replication',next_decision=next_decision)
    old.update(event);old['events'].append(event);write(path,old)


def bootstrap(values,ids):
    """Source-only CI with explicit exclusions and a fixed bootstrap seed."""
    indices=[i for i,v in enumerate(values) if v is not None and np.isfinite(v)]
    if not indices:
        return dict(n_sources=0,source_ids=[],excluded_sources=ids,mean=None,ci95=None)
    x=np.asarray([values[i] for i in indices],float)
    rng=np.random.default_rng(SEED+99)
    samples=x[rng.integers(len(x),size=(3999,len(x)))].mean(axis=1)
    return dict(n_sources=len(x),source_ids=[ids[i] for i in indices],excluded_sources=[s for i,s in enumerate(ids) if i not in indices],
        mean=float(x.mean()),ci95=np.quantile(samples,[.025,.975]).tolist() if len(x)>1 else None)


def crossed_contrast(selected,controls,ids):
    """Mean the four controls within each source before paired source uncertainty."""
    selected=np.asarray(selected,float);controls=np.asarray(controls,float)
    if controls.shape!=(len(selected),4):
        raise ValueError('Exactly four crossed controls required per source')
    valid=np.isfinite(selected)&np.isfinite(controls).all(axis=1)
    mean_control=controls.mean(axis=1)
    paired=selected-mean_control
    def score(values):
        return bootstrap([float(v) if ok else None for v,ok in zip(values,valid)],ids)
    result=dict(selected_absolute=score(selected),mean_control=score(mean_control),paired_primary=score(paired),
        individual_controls=[score(controls[:,j]) for j in range(4)],
        control_spread=dict(per_source_sd=[float(v) if np.isfinite(v) else None for v in np.std(controls,axis=1,ddof=1)],
            per_source_min=[float(v) if np.isfinite(v) else None for v in np.min(controls,axis=1)],
            per_source_max=[float(v) if np.isfinite(v) else None for v in np.max(controls,axis=1)],
            control_population='Only the four frozen sets; no control-as-source pseudoreplication'),
        practical_band=[-.02,.02],practical_check={})
    for name in ('selected_absolute','paired_primary'):
        interval=result[name]['ci95']
        result['practical_check'][name]=dict(CI_contained=bool(interval is not None and interval[0]>=-.02 and interval[1]<=.02),
            interpretation='Descriptive interval-contained equivalence only if true; not a separate primary success or rescue')
    return result


def prepare(out):
    """Freeze fresh inputs, exact original four controls and analysis before native."""
    if out.exists():
        raise FileExistsError('Fresh replication run required')
    out.mkdir(parents=True)
    try:
        previous=out.parent/'ffn_selectivity_causal4963000'
        p=read(previous/'protocol.json');audit=read(previous/'independent_statistics_audit.json')
        if not audit['valid'] or audit['analysis_sha256']!=sha(previous/'analysis.json'):
            raise ValueError('Completed initial independent audit required')
        if read(previous/'compact_record.json')['status']!='COMPLETE_AUDITED_SLOT2_RELEASED':
            raise ValueError('Initial pilot must be completed first')
        for path,digest in p['sources'].items():
            if sha(path)!=digest:
                raise ValueError('Original frozen code changed')
        if sha(previous/'matching.json')!=p['hashes']['matching.json']:
            raise ValueError('Original FIT controls changed')
        matching=read(previous/'matching.json')
        units=matching['selected'];controls=matching['draws'][:4]
        if units!=[1925,909,108,901] or p['target_types']['4']!=['ramprecent64','noise']:
            raise ValueError('Reviewed units/target types changed')
        if any(len(set(c))!=4 or set(c)&set(units) for c in controls):
            raise ValueError('Invalid frozen matched set')
        old_ids={r['source'] for r in read(previous/'rows.json')}
        stage1=out.parent/'ffn_selectivity_pilot4961000'
        old_ids|={r['source'] for r in read(stage1/'rows.json')}
        rows=[];queries=[]
        for i,seed in enumerate(range(SEED,SEED+COUNT)):
            source=f'syn:{seed}'
            if source in old_ids:
                raise ValueError('Fresh source reused')
            x=causal.stage1.synthetic_background(seed,i%4);delta,conditions=causal.stage1.perturbations(seed)
            indices=causal.condition_indices(i//4)
            query=np.concatenate([x[None],x[None]+delta[indices]]).astype(np.float32)
            rows.append(dict(index=i,source=source,seed=seed,family=i%4,split='TEST',conditions=[conditions[j] for j in indices],
                condition_indices=indices,control_ids=[0,1,2,3],budget=4))
            queries.append(query)
        queries=np.asarray(queries)
        error=float(np.max(abs(np.sqrt(np.mean((queries[:,1:].astype(float)-queries[:,0,None])**2,axis=-1))-.1)))
        if error>1e-7:
            raise ValueError('Input RMS mismatch')
        np.savez_compressed(out/'inputs.npz',queries=queries);write(out/'rows.json',rows)
        assignment=dict(selected=units,controls=controls,control_ids=[0,1,2,3],origin='First4 frozen activity-matched draws, no outcome-based selection',
            FIT_matching_details=matching['details'],all_sets_every_source=True)
        write(out/'controls.json',assignment)
        reference=read(previous/'execution.json')['bolt']
        budget=dict(cells=CELLS,sources=COUNT,cells_per_source=34,cell_cap=1200,reference_seconds=reference['seconds'],reference_cells=384,
            estimated_seconds=reference['seconds']*CELLS/384,conservative_seconds=reference['seconds']*CELLS/384*2,
            breakdown='1 unlocked background+1 locked repeat+5 clean+25 intervention calls+1 exact self-patch+1 full-FFN reach',
            cpu_threads=1,resident_cap_bytes=MEMORY,seconds_cap=SECONDS)
        write(out/'budget.json',budget)
        binding_names=['protocol.json','matching.json','rows.json','analysis.json','independent_statistics_audit.json','compact_record.json','execution.json']
        source_paths=[Path(__file__),Path(sys.modules[edit_audit.__module__].__file__)]
        protocol=dict(version=1,run=out.name,hold_native=True,requires_new_slot=True,model='bolt',site='first',budget=4,units=units,
            checkpoint=p['checkpoint'],capture=p['capture'],no_SAE=True,native_parameters='Fixed values; exact before/after parameter hash',
            synthetic_seeds=list(range(SEED,SEED+COUNT)),sources_count=COUNT,history=128,horizon=32,point_output='median',
            input_RMS=.1,normalization='Background-locked native normalization including inverse',types=list(causal.TYPES),
            variant_schedule='condition_indices(i//4) with family=i%4; sinusoid phase/sign variants balanced globally and within each background family',
            targets=['ramprecent64','noise'],off_targets=['sin24','sin19.2','steprecent32'],arms=list(ARMS),
            controls='Cross first4 frozen FIT activity-matched sets with EVERY source/type; no control selection after outcomes',
            intervention='Subtract chosen-unit perturbation delta by restoring background postactivation values at all first-FFN tokens',
            primary='Source-paired selected minus mean of4 controls, equal-type ramp/noise signed response removal; unchanged by practical-band outcome',
            absolute='Report selected absolute removal CI, each control and mean-control CI, source-level control spread',
            response_floor=1e-6*.1**2,ratio_policy='Ratios undefined for small clean response; require both targets for source primary; no replacement',
            precision='All response/remaining/orthogonal energy arithmetic float64 from saved float32 forecasts; fixes only new-run secondary precision',
            uncertainty='3999 paired source bootstrap,32 independent source groups; controls/types/tokens not bootstrap observations',
            practical_band=[-.02,.02],practical_rule='Describe equivalence/smallness only when the entire95% CI is within[-.02,.02], separately for selected absolute and primary difference; do not rescue/replace primary',
            secondary='All5 types, target/off-target aggregates, actual FFN edit norm/energy/share, output reach, orthogonal change, control spread',
            gates=dict(exact_normalizer_repeat=True,exact_self_patch=True,parameters_unchanged=True,selected_only_edit_exact=True,
                full_FFN_reach_relative_gt=1e-6,actual_selected_edit_and_output_reach_separate=True,finite=True),
            cells=CELLS,cell_cap=1200,cpu_threads=1,memory_cap=MEMORY,seconds_cap=SECONDS,
            no_search='No unit,k,dose,depth,model or outcome-based control search',
            sources={**p['sources'],**{str(path.resolve()):sha(path) for path in source_paths}},
            prior_bindings={str(previous/name):sha(previous/name) for name in binding_names},
            stage1_rows_binding={str(stage1/'rows.json'):sha(stage1/'rows.json')},
            hashes={name:sha(out/name) for name in ('inputs.npz','rows.json','controls.json','budget.json')})
        write(out/'protocol.json',protocol)
    except Exception as exc:
        knowledge(out,'prepare','failed',dict(error_type=type(exc).__name__,error=str(exc),native_launched=False),'Resolve preparation failure; HOLD native.')
        raise
    knowledge(out,'prepare','complete',dict(protocol_sha256=sha(out/'protocol.json'),budget=budget,input_RMS_error=error,
        controls=assignment,native_launched=False),'HOLD native until new slot grant after coordinator scheduling; no automatic expansion.')
    print(dict(status='READY_HOLD_NATIVE',budget=budget),flush=True)


def worker(out,native_factory=None):
    """Execute only1088 frozen Bolt cells and save actual edits and all forecasts."""
    if os.environ.get('FFN_CROSSED_MONITORED')!='1':
        raise RuntimeError('HOLD crossed-control native pending monitored grant')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in causal.stage1.THREAD_VARS):
        raise RuntimeError('Requires one-thread CPU worker')
    p=read(out/'protocol.json')
    for path,digest in p['sources'].items():
        if sha(path)!=digest:
            raise ValueError('Frozen replication source changed')
    for name,digest in p['hashes'].items():
        if sha(out/name)!=digest:
            raise ValueError('Frozen replication artifact changed')
    if (out/'bolt_causal.npz').exists():
        raise FileExistsError('Refuse replication overwrite')
    for name,digest in p['checkpoint']['weights'].items():
        if sha(Path(p['checkpoint']['path'])/name)!=digest:
            raise ValueError('Checkpoint changed')
    if native_factory is None:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from experiments.feature_search.real_background_poc import Native
        native_factory=Native
    started=time.monotonic();native=native_factory('bolt',p['checkpoint']['path'])
    sites={'first':causal.stage1.sites(native)['first']};before=causal.parameter_hash(native.model)
    rows=read(out/'rows.json');controls=read(out/'controls.json');q=np.load(out/'inputs.npz',allow_pickle=False)['queries']
    if q.shape!=(len(rows),6,128):
        raise ValueError('Replication query geometry invalid')
    count=0
    def invoke(x,replacement=None):
        nonlocal count
        count+=len(x)
        if count>p['cells'] or count>1200 or time.monotonic()-started>SECONDS or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>MEMORY:
            raise RuntimeError('Native cell/resource cap exceeded')
        return causal.stage1.capture(native,x,sites,replacement)
    clean_forecasts=[];clean_states=[];patched_forecasts=[];patched_states=[];events=[];reach=[]
    for row,queries in zip(rows,q):
        with causal.stage1.locked_background(native,queries[:1],invoke) as (reference,original):
            baseline,state=invoke(queries[:1]);causal.stage1.exact_gate(reference,baseline,'background repeat')
            for key in state:
                causal.stage1.exact_gate(original[key],state[key],'background state repeat')
            anchor=state['first_ffn']
            if anchor.shape!=(1,p['capture']['tokens'],p['capture']['ffn_width']):
                raise ValueError('FFN geometry invalid')
            forecasts=[baseline[0]];states=[anchor[0]]
            for c,x in enumerate(queries[1:]):
                clean,observed=invoke(x[None]);ffn=observed['first_ffn'];forecasts.append(clean[0]);states.append(ffn[0])
                if c==0:
                    replay,_=invoke(x[None],('first',ffn));causal.stage1.exact_gate(clean,replay,'FFN self-patch')
                    zero,_=invoke(x[None],('first',np.zeros_like(ffn)))
                    reach.append(dict(source=row['source'],relative_change=causal.stage1.reach_gate(clean,zero)))
                for arm,units in zip(ARMS,[controls['selected']]+controls['controls']):
                    replacement=causal.patch_units(ffn,anchor,units)
                    pred,actual=invoke(x[None],('first',replacement))
                    causal.stage1.exact_gate(replacement,actual['first_ffn'],'selected-only patch')
                    record=edit_audit(anchor,ffn,actual['first_ffn'],units,clean[0],pred[0])
                    total=float(np.sum((ffn.astype(float)-anchor)**2))
                    events.append(dict(index=len(events),source_index=row['index'],source=row['source'],condition=c,
                        kind=causal.TYPES[c],arm=arm,units=units,FFN_delta_energy=total,
                        removed_FFN_energy=record['actual_selected_edit_L2']**2,
                        removed_FFN_share=record['actual_selected_edit_L2']**2/total if total>0 else None,**record))
                    patched_forecasts.append(pred[0]);patched_states.append(actual['first_ffn'][0])
            clean_forecasts.append(forecasts);clean_states.append(states)
    after=causal.parameter_hash(native.model)
    if before!=after or count!=p['cells']:
        raise ValueError('Parameter values changed or cell count mismatch')
    arrays=dict(clean_forecasts=np.asarray(clean_forecasts),clean_FFN=np.asarray(clean_states),
        patched_forecasts=np.asarray(patched_forecasts),patched_FFN=np.asarray(patched_states))
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite replication output')
    np.savez_compressed(out/'bolt_causal.npz',**arrays,source_ids=[r['source'] for r in rows]);write(out/'interventions.json',events)
    write(out/'bolt_worker.json',dict(valid=True,cells=count,seconds=time.monotonic()-started,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,parameter_hash_before=before,parameter_hash_after=after,
        exact_repeat=True,exact_self_patch=True,exact_patch_support=True,reach=reach,
        protocol_sha256=sha(out/'protocol.json'),arrays_sha256=sha(out/'bolt_causal.npz'),events_sha256=sha(out/'interventions.json')))


def analyze(out):
    """Score all fixed crossed controls, preserve absolute CI and practical-band rule."""
    if (out/'analysis.json').exists():
        raise FileExistsError('Refuse repeated replication analysis')
    try:
        p=read(out/'protocol.json');rows=read(out/'rows.json');worker=read(out/'bolt_worker.json');execution=read(out/'execution.json')['bolt']
        for path,digest in p['sources'].items():
            if sha(path)!=digest:
                raise ValueError('Frozen source changed')
        if (not execution['valid'] or execution['returncode']!=0 or not worker['valid'] or worker['cells']!=p['cells']
                or worker['parameter_hash_before']!=worker['parameter_hash_after']
                or not all(worker[k] for k in ('exact_repeat','exact_self_patch','exact_patch_support'))
                or worker['arrays_sha256']!=sha(out/'bolt_causal.npz') or worker['events_sha256']!=sha(out/'interventions.json')
                or worker['protocol_sha256']!=sha(out/'protocol.json')
                or max(worker['peak_rss'],execution['monitored_peak'])>MEMORY or max(worker['seconds'],execution['seconds'])>SECONDS):
            raise ValueError('Replication gate/hash/resource failed')
        if len(worker['reach'])!=len(rows) or not all(r['relative_change']>1e-6 for r in worker['reach']):
            raise ValueError('Full FFN reach failed')
        data=np.load(out/'bolt_causal.npz',allow_pickle=False);events=read(out/'interventions.json');assignment=read(out/'controls.json')
        if data['source_ids'].tolist()!=[r['source'] for r in rows]:
            raise ValueError('Source alignment failed')
        expected={(i,c,arm) for i in range(len(rows)) for c in range(5) for arm in ARMS}
        if len(events)!=len(expected) or {(e['source_index'],e['condition'],e['arm']) for e in events}!=expected:
            raise ValueError('Crossed schedule incomplete')
        metrics=[]
        for event in events:
            i,c,j=event['source_index'],event['condition'],event['index']
            units=assignment['selected'] if event['arm']=='selected' else assignment['controls'][int(event['arm'][-1])]
            if event['units']!=units or event['source']!=rows[i]['source'] or event['kind']!=p['types'][c]:
                raise ValueError('Frozen intervention identity changed')
            check=edit_audit(data['clean_FFN'][i,0],data['clean_FFN'][i,c+1],data['patched_FFN'][j],units,
                data['clean_forecasts'][i,c+1],data['patched_forecasts'][j])
            for key in ('actual_selected_edit_L2','output_change_L2','output_reach_relative'):
                if not np.isclose(check[key],event[key],atol=1e-12,rtol=1e-12):
                    raise ValueError('Actual edit/output reach mismatch')
            values=causal.response_metrics(data['clean_forecasts'][i,0].astype(float),data['clean_forecasts'][i,c+1].astype(float),
                data['patched_forecasts'][j].astype(float))
            metrics.append(dict(event,**values))
        lookup={(e['source'],e['kind'],e['arm']):e for e in metrics};ids=[r['source'] for r in rows]
        names=['signed_response_removal','removed_output_energy','remaining_output_energy','orthogonal_change_ratio',
            'removed_FFN_energy','removed_FFN_share','actual_selected_edit_L2','output_reach_relative']
        summaries=[]
        for label,kinds in {**{k:[k] for k in p['types']},'target':p['targets'],'off_target':p['off_targets']}.items():
            for metric in names:
                matrix=np.full((len(rows),5),np.nan)
                for i,source in enumerate(ids):
                    for j,arm in enumerate(ARMS):
                        v=[lookup[source,kind,arm][metric] for kind in kinds]
                        if all(x is not None for x in v):
                            matrix[i,j]=np.mean(v)
                statistics=crossed_contrast(matrix[:,0],matrix[:,1:],ids)
                if metric!='signed_response_removal':
                    statistics.pop('practical_band');statistics.pop('practical_check')
                summaries.append(dict(types=label,metric=metric,
                    primary=label=='target' and metric=='signed_response_removal',
                    **statistics))
        result=dict(primary=[r for r in summaries if r['primary']][0],summaries=summaries,events=metrics,
            parameters_fixed=True,actual_edit_audit=True,scope='Only chosen Bolt-first k4; no parameter fitting or general monosemanticity',
            practical_band_policy=p['practical_rule'],protocol_sha256=sha(out/'protocol.json'))
        write(out/'analysis.json',result)
    except Exception as exc:
        knowledge(out,'analysis','failed',dict(error_type=type(exc).__name__,error=str(exc)),'Resolve saved-output instrumentation; no new native launch.')
        raise
    knowledge(out,'analysis','complete',dict(primary=result['primary'],summaries=summaries,analysis_sha256=sha(out/'analysis.json')),
        'Report unchanged primary, selected absolute CI, control spread, actual reach/energy and off-target changes; no automatic expansion.')
    return result


def run(out,slot_granted=False):
    """Launch only an explicitly granted single CPU Bolt worker with hard caps."""
    if not slot_granted:
        raise RuntimeError('HOLD replication native pending new slot grant')
    if (out/'execution.json').exists():
        raise FileExistsError('Refuse native replication rerun')
    p=read(out/'protocol.json')
    if p['model']!='bolt' or p['cells']!=CELLS or p['cells']>1200:
        raise ValueError('Frozen1088-cell Bolt-only schedule required')
    import fcntl
    result={}
    try:
        with (out/'native.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',FFN_CROSSED_MONITORED='1',
                TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='1',**{k:'1' for k in causal.stage1.THREAD_VARS})
            started=time.monotonic();peak=0;reason=None
            with (out/'bolt.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.ffn_selectivity_causal_replication',
                    'worker','--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
                try:
                    while proc.poll() is None:
                        try:
                            lines=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                            peak=max(peak,int(next(v.split()[1] for v in lines if v.startswith('VmRSS:')))*1024)
                        except (FileNotFoundError,StopIteration):
                            pass
                        if peak>MEMORY or time.monotonic()-started>SECONDS:
                            reason='Resource cap exceeded';break
                        time.sleep(.1)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            proc.kill();proc.wait()
            worker=read(out/'bolt_worker.json') if (out/'bolt_worker.json').exists() else {}
            valid=(proc.returncode==0 and reason is None and worker.get('valid',False) and worker.get('cells')==CELLS
                and worker.get('peak_rss',float('inf'))<=MEMORY and worker.get('seconds',float('inf'))<=SECONDS)
            result=dict(bolt=dict(valid=valid,returncode=proc.returncode,seconds=time.monotonic()-started,monitored_peak=peak,reason=reason,cells=worker.get('cells')))
            write(out/'execution.json',result)
            if not valid:
                raise RuntimeError('Crossed replication native failed')
    except Exception as exc:
        knowledge(out,'native_execution','failed',dict(error_type=type(exc).__name__,error=str(exc),execution=result),
            'Release slot and inspect failure; no automatic rerun.')
        raise
    knowledge(out,'native_execution','complete',result,'Release native slot immediately; analyze saved replication outputs only.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','worker','run','analyze'])
    parser.add_argument('--out',type=Path,default=Path('runs/ffn_selectivity_causal_replication4964000'))
    parser.add_argument('--slot-granted',action='store_true')
    args=parser.parse_args()
    run(args.out,args.slot_granted) if args.command=='run' else globals()[args.command](args.out)
