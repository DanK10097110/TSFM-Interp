"""Independent arithmetic audit and complete distillation for4962000.

Does not change the frozen decision rule, fit probes, or run native models.
Retains separate local-FIT, frozen-extrapolation and intervention-correctness axes.
"""
import json
import math
import numpy as np
from experiments.feature_search.comparability_conditions import DEFAULT,OLD,RANDOM,TARGETS,CONDITIONS,SEEDS,FAMILIES,read,write,sha,verify,fit_ids,knowledge
from experiments.feature_search.comparability_replication_audit import source_means,bootstrap


def audit(out):
    """Replay physical-unit prediction arithmetic, source CIs and both primary decisions."""
    p=verify(out);metadata=read(out/'rows.json');_,ids=fit_ids(metadata);rows=[metadata[i] for i in ids]
    with np.load(out/'inputs.npz') as f:x=f['history'][ids].astype(float)
    with np.load(out/'targets.npz') as f:future=f['future'][ids]
    native={}
    for m in ('bolt','timesfm'):
        with np.load(out/f'{m}_native.npz') as f:native[m]={k:f[k][ids].astype(float) for k in ('native','fusion')}
    center=np.array([math.fsum(row)/128 for row in x]);variance=np.array([math.fsum((row-m)**2)/128 for row,m in zip(x,center)]);scale=np.sqrt(variance)
    checks=[];local_losses={};source_tables=[]
    def compare(label,actual,expected):
        assert actual['n_sources']==expected['n_sources'],label
        error=max(abs(actual[k]-expected[k]) for k in ('mean','low','high'))
        assert error<=2e-12,label
        checks.append(dict(label=label,max_absolute_error=error))
    for axis,filename,resultfile in [('local','local_predictions.npz','local_results.json'),('frozen','frozen_predictions.npz','frozen_extrapolation.json')]:
        with np.load(out/filename) as f:predictions={k:f[k] for k in f.files}
        recorded=read(out/resultfile)
        for a,b,ep in TARGETS:
            prefix=f'{a}_to_{b}__{ep}__';losses={}
            observed=native[b]['native'] if ep=='forecast' else native[b]['fusion']-native[b]['native']
            for key,pred in predictions.items():
                if not key.startswith(prefix):continue
                arm=key[len(prefix):]
                if axis=='frozen' and '__' in arm:arm=arm.replace('__','_')
                physical=pred*scale[:,None]+(center[:,None] if ep=='forecast' else 0)
                values=np.array([math.fsum(d*d)/32/v for d,v in zip(physical-observed,variance)])
                losses[arm]=values
                source_tables.append(dict(axis=axis,A=a,B=b,endpoint=ep,arm=arm,source_losses=source_means(values,rows).tolist()))
            if axis=='local':local_losses[a,b,ep]=losses
            for cell in [r for r in recorded if (r['A'],r['B'],r['endpoint'])==(a,b,ep)]:
                subset=[i for i,r in enumerate(rows) if cell['case']=='ALL' or r['case']==cell['case']]
                for arm,summary in cell['losses'].items():compare(f'{axis}:{prefix}:{cell["case"]}:{arm}',bootstrap(source_means(losses[arm][subset],rows,subset)),summary)
    primary=read(out/'primary_result.json');losses=local_losses['timesfm','bolt','fusion_response'];primary_replay={}
    for family in FAMILIES:
        random=np.mean([losses[f'{family}_{seed}'] for seed in SEEDS],axis=0);gain=source_means(random-losses['trained'],rows)
        replay=bootstrap(gain,coverage=.975);stored=primary['families'][family]
        compare('primary:'+family,replay,stored['gain'])
        denominator=float(np.mean(source_means(random,rows)));relative=replay['mean']/denominator if denominator>0 else None
        assert abs(denominator-stored['random_mean_loss'])<2e-12
        assert relative is None and stored['relative_reduction'] is None or relative is not None and abs(relative-stored['relative_reduction'])<2e-12
        passed=bool(denominator>0 and replay['low']>0 and relative>=.05)
        assert passed==stored['passed'];primary_replay[family]=dict(gain=replay,relative_reduction=relative,passed=passed)
    assert all(r['passed'] for r in primary_replay.values())==primary['passed']
    plans=read(out/'plans.json')
    for cell in read(out/'behavior_correctness.json'):
        model=cell['model'];subset=[i for i,r in enumerate(rows) if cell['case']=='ALL' or r['case']==cell['case']]
        observed=native[model]['native'];fused=native[model]['fusion'];delta=fused-observed
        n=np.sum((observed-future)**2,axis=1)/32/variance;f=np.sum((fused-future)**2,axis=1)/32/variance
        energy=np.sum(delta**2,axis=1)/32/variance;dot=np.sum(delta*(future-observed),axis=1)/32/variance
        values=dict(native_loss=n,fusion_loss=f,correctness_gain=n-f,response_energy=energy,correction_dot=dot)
        for name,v in values.items():compare(f'behavior:{model}:{cell["case"]}:{name}',bootstrap(source_means(v[subset],rows,subset)),cell['metrics'][name])
        assert abs(float(np.mean(energy[subset]==0))-cell['exact_zero_response_fraction'])<1e-15
        assert sum(plans[ids[i]]['accepted'] for i in subset)==cell['tree_eligible']
        np.testing.assert_allclose(n-f,2*dot-energy,atol=1e-10,rtol=1e-10)
    v=np.linspace(.01,.03,32);ref=bootstrap(v,.975)
    plants=[]
    for name,bad in [('reversed_gain',bootstrap(-v,.975)),('wrong_offset',bootstrap(v+.01,.975)),('wrong_source_count',dict(ref,n_sources=160))]:
        caught=False
        try:compare('planted_'+name,bad,ref)
        except AssertionError:caught=True
        assert caught,name;plants.append(dict(name=name,detected=True))
    result=dict(passed=True,intervals_replayed=len(checks),maximum_absolute_error=max(c['max_absolute_error'] for c in checks),checks=checks,primary_replay=primary_replay,source_losses=source_tables,self_plants=plants,
        independence='No production loss/group/bootstrap helper used. Physical inverse, compensated sums and independently audited index-bootstrap/explicit quantile interpolation replay predictions and all decisions.',
        limitations='Arithmetic audit does not independently rerun native inference. Native correctness is bounded by exact saved replay/normalizer gates and checkpoint/source bindings.')
    write(out/'arithmetic_audit.json',result)
    return result


def render(out=DEFAULT):
    """Report the fixed primary whether positive or negative, keeping every axis visible."""
    replay=audit(out);p=read(out/'protocol.json');primary=read(out/'primary_result.json');local=read(out/'local_results.json');frozen=read(out/'frozen_extrapolation.json');behavior=read(out/'behavior_correctness.json')
    primary_cell=next(r for r in local if r['A']=='timesfm' and r['B']=='bolt' and r['endpoint']=='fusion_response' and r['case']=='ALL')
    trained=primary_cell['losses']['trained']['mean']
    lines=['# Condition-shift study4962000','',f"Frozen local-FIT primary: **{'PASS' if primary['passed'] else 'FAIL'}**. Success requires BOTH random-family contrasts to have positive97.5% paired-source interval lower bounds and at least5% lower trained mean loss than each random-family mean loss. No descriptive control or extrapolation result can rescue a failed primary.",'',
        '52 sources from4962000–4962051, five conditions each. Exactly20 newFIT sources/96 fitting rows;32 untouched TEST sources/160 rows. Four shifted-balanced FIT rows are excluded as preregistered. Source-forecast kernel recipes, readout ranks/grids, random widths and all three initialization seeds were frozen before native inference.','',
        '| Local TF→Bolt fusion response | Mean prediction loss | Trained reduction | Random−trained gain [97.5% source CI] | Pass |', '|---|---:|---:|---|---|',
        f'| Trained hidden | {trained:.10g} | — | — | — |']
    for family,r in primary['families'].items():
        gain=r['gain'];relative='unscorable' if r['relative_reduction'] is None else f"{100*r['relative_reduction']:.3f}%"
        lines.append(f"| {family}, mean loss over3 initializations | {r['random_mean_loss']:.10g} | {relative} | {gain['mean']:.10g} [{gain['low']:.10g}, {gain['high']:.10g}] | {r['passed']} |")
    lines+=['','The following condition rows are descriptive. They show whether an aggregate result is uniform; they do not change the primary.','',
        '| Condition | Local base | Local trained | Local RFF mean | Local MLP mean | Frozen trained | Frozen RFF mean | Frozen MLP mean |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    condition_summaries=[]
    for case in ['ALL']+[c[0] for c in CONDITIONS]:
        ll=next(r for r in local if (r['A'],r['B'],r['endpoint'],r['case'])==('timesfm','bolt','fusion_response',case))['losses']
        ff=next(r for r in frozen if (r['A'],r['B'],r['endpoint'],r['case'])==('timesfm','bolt','fusion_response',case))['losses']
        lm={f:float(np.mean([ll[f'{f}_{s}']['mean'] for s in SEEDS])) for f in FAMILIES};fm={f:float(np.mean([ff[f'{f}_{s}']['mean'] for s in SEEDS])) for f in FAMILIES}
        lines.append(f"| {case} | {ll['input+A_forecast']['mean']:.7g} | {ll['trained']['mean']:.7g} | {lm['rff']:.7g} | {lm['mlp_relu']:.7g} | {ff['input+A_forecast+A_hidden']['mean']:.7g} | {fm['rff']:.7g} | {fm['mlp_relu']:.7g} |")
        condition_summaries.append(dict(case=case,local_base=ll['input+A_forecast']['mean'],local_trained=ll['trained']['mean'],local_random=lm,frozen_trained=ff['input+A_forecast+A_hidden']['mean'],frozen_random=fm))
    lines+=['','Response energy and correctness are different. Correctness gain is native true-future normalized MSE minus fused MSE; it equals2×response/correction alignment minus response energy. None of these comparisons establishes semantic equivalence or cross-model causality.','',
        '| Model / condition | Response energy | Correctness gain [95% source CI] | Tree eligible / rows |', '|---|---:|---|---|']
    for r in behavior:
        g=r['metrics']['correctness_gain'];lines.append(f"| {r['model']} / {r['case']} | {r['metrics']['response_energy']['mean']:.7g} | {g['mean']:.7g} [{g['low']:.7g}, {g['high']:.7g}] | {r['tree_eligible']}/{r['rows']} |")
    lines+=['','Forecast targets are descriptive controls; all individual random initializations, fixed input-only and source-forecast baselines remain in local_results.json and frozen_extrapolation.json. Frozen extrapolation uses old coefficients and support without updates; local FIT is a separately trained matched-budget comparison.','',
        'Source bootstrap averages the five dependent conditions within each source before resampling32 sources. It conditions on the20-source FIT sample and three fixed random initializations. Readout widths and search budgets are matched; pretrained architectures/FLOPs/training data are not. These are synthetic-condition findings; no natural-data leg was run.','',
        f"Audit:11 clean prelaunch tests;29/29 isolated plants detected. A failed preflight fixture is preserved separately. Independent saved-prediction arithmetic replayed{replay['intervals_replayed']} intervals and both primary decisions; max discrepancy{replay['maximum_absolute_error']:.4g}. All native repeat, normalizer identity and first/half self-patch gates must be exactly zero.",'',
        'Native capture is finished and slot2 is released. All scoring/audit work is cached-data only. No further native work is authorized without slot coordination.']
    with (out/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    decision=('The frozen BOTH-family primary passed; interpret the condition heterogeneity and frozen-transfer differences before considering another bounded study.' if primary['passed'] else 'The frozen BOTH-family primary failed. Do not promote a general local-FIT trained-feature advantage; retain any condition-specific or single-family positives as descriptive only.')+' No further native work without explicit slot coordination; no natural expansion.'
    knowledge(out,dict(experiment='independent_arithmetic_and_distillation',status='complete',evidence_class='Prospectively frozen synthetic predictive feature-efficiency comparison plus descriptive extrapolation/correctness axes',exact_findings=dict(primary=primary,condition_summaries=condition_summaries,arithmetic_intervals=replay['intervals_replayed'],arithmetic_max_error=replay['maximum_absolute_error']),controls=['Independent physical-unit loss/source-CI replay','Both-family decision replay','Three arithmetic plants'],limitations=['Synthetic five-condition scope','Conditional on localFIT and3 random maps','No causal/equivalent-semantic conclusion'],next_decision=decision))
    k=read(out/'knowledge.json');k.update(status='complete_native_released',evidence_class='Prospectively frozen conditional synthetic predictive feature-efficiency comparison; extrapolation and correctness are separate descriptive axes.',exact_findings=dict(primary=primary,condition_summaries=condition_summaries,local_results=local,frozen_extrapolation=frozen,behavior_correctness=behavior),controls=dict(validation=read(out/'validation.json'),plant_audit=read(out/'plant_audit.json'),arithmetic_audit=dict(passed=True,intervals=replay['intervals_replayed'],max_error=replay['maximum_absolute_error']),native={m:read(out/f'{m}_worker.json') for m in ('bolt','timesfm')},execution=read(out/'execution.json')),limitations=['Five synthetic conditions only; no natural-data expansion.','Twenty-source localFIT uncertainty and random-family universality are not covered by source bootstrap.','Predicting an intervention response does not establish correctness or shared semantics.'],next_decision=decision,report='report.md',report_source_sha256=sha(__file__))
    (out/'knowledge.json').write_text(json.dumps(k,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write(out/'completion_manifest.json',dict(status='complete_native_released',native_slot2='released; no further native work authorized',hashes={f:sha(out/f) for f in ('protocol.json','primary_result.json','local_results.json','frozen_extrapolation.json','behavior_correctness.json','arithmetic_audit.json','knowledge.json','report.md','execution.json')}))
    print('Primary',primary['passed'],'independent intervals',replay['intervals_replayed'],'max error',replay['maximum_absolute_error'],flush=True)


if __name__=='__main__':render()
