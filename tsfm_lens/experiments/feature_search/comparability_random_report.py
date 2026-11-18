"""Independent arithmetic replay and balanced distillation of random-feature controls."""
import json
import numpy as np
from experiments.feature_search.comparability_random_features import DEFAULT,UP,read,write,sha,verify
from experiments.feature_search.comparability_replication_audit import bootstrap,source_means


def replay(out):
    """Replay saved predictions in physical units using the independently audited arithmetic."""
    rows=read(UP/'rows.json');results=read(out/'results.json');per_seed=read(out/'per_initialization.json')
    with np.load(UP/'inputs.npz') as f:x=f['history'].astype(float)
    with np.load(out/'predictions.npz') as f:predictions={k:f[k] for k in f.files}
    with np.load(UP/'predictions.npz') as f:original={k:f[k] for k in f.files}
    native={}
    for m in ('bolt','timesfm'):
        with np.load(UP/f'{m}_native.npz') as f:native[m]={key:f[key].astype(float) for key in ('native','fusion')}
    center=x.mean(1);scale=x.std(1);losses={};checks=[]
    def check(label,value,stored):
        assert value['n_sources']==stored['n_sources']
        error=max(abs(value[k]-stored[k]) for k in ('mean','low','high'));assert error<2e-12,label
        checks.append(dict(label=label,max_error=error))
    for row in per_seed:
        a,b,ep,family,seed=[row[k] for k in ('A','B','endpoint','family','seed')]
        prefix=f'{a}_to_{b}__{ep}';key=f'{prefix}__{family}__{seed}'
        actual=native[b]['native'] if ep=='forecast' else native[b]['fusion']-native[b]['native']
        loss={}
        for name,pred in [('random',predictions[key]),('trained',original[prefix+'__input+A_forecast+A_hidden']),('base',original[prefix+'__input+A_forecast'])]:
            physical=pred*scale[:,None]+(center[:,None] if ep=='forecast' else 0)
            loss[name]=np.square(physical-actual).sum(1)/(32*scale**2)
            check(key+':'+name,bootstrap(source_means(loss[name],rows)),row['losses'][name])
        check(key+':gain',bootstrap(source_means(loss['base']-loss['random'],rows)),row['random_gain_over_base'])
        check(key+':advantage',bootstrap(source_means(loss['random']-loss['trained'],rows)),row['trained_advantage'])
        losses[key]=loss
    for row in results:
        prefix=f"{row['A']}_to_{row['B']}__{row['endpoint']}__{row['family']}"
        values=[losses[prefix+f'__{seed}'] for seed in row['initializations']]
        averaged=np.mean([v['random'] for v in values],axis=0);base=values[0]['base'];trained=values[0]['trained']
        for name,v in [('random_loss',averaged),('base_loss',base),('trained_loss',trained)]:check(prefix+':'+name,bootstrap(source_means(v,rows)),row[name])
        check(prefix+':advantage',bootstrap(source_means(averaged-trained,rows),1-.05/6),row['trained_advantage'])
        check(prefix+':gain',bootstrap(source_means(base-averaged,rows)),row['random_gain_over_base'])
    summary=dict(passed=True,intervals=len(checks),max_absolute_error=max(c['max_error'] for c in checks),checks=checks,method='Physical-unit inverse, squared residuals, independent source grouping/bootstrap/quantile code from the completed replication audit; no production metric helper.')
    write(out/'arithmetic_replay.json',summary)
    return summary


def render(out=DEFAULT):
    """Keep generic-feature counterevidence and the bounded fusion-response positive together."""
    verify(out);audit=replay(out);rows=read(out/'results.json');seeds=read(out/'per_initialization.json')
    lines=['# Random-feature efficiency controls4953000','',
        'Exploratory post-hoc reuse of the previously evaluated64-source replication. No native calls. The protocol, random recipes and all18 old-FIT-only probes were frozen before the new TEST comparisons; this does not restore historical unseen-data status. Original primary results are unchanged.','',
        'Input-only arms had selected RBF gamma0.5/alpha0.001. Input+Aforecast uses gamma2/PCA16 for forecast and gamma0.5/PCA16 for fusion response. Every trained hidden extension selected PCA16/alpha0.001. Random maps receive the complete normalized128-step history and two normalizer fields; they share the frozen input+Aforecast blocks.','',
        'Each random family uses three fixed initializations, matching flattened width4608/5120, PCA8/16, four ridge values,20 FIT sources/96 rows and grouped inner selection. Results below average squared losses across initializations; they do not ensemble predictions or select the best seed. This matches readout/feature budgets, not pretrained architecture, FLOPs, parameter count or training data.','',
        '| Target | Family | Input+Aforecast loss | +trained hidden loss | +random loss | Trained advantage [99.17% source CI] |',
        '|---|---|---:|---:|---:|---|']
    for r in rows:
        g=r['trained_advantage']
        lines.append(f"| {r['A']}→{r['B']} {r['endpoint']} | {r['family']} | {r['base_loss']['mean']:.9g} | {r['trained_loss']['mean']:.9g} | {r['random_loss']['mean']:.9g} | {g['mean']:.9g} [{g['low']:.9g}, {g['high']:.9g}] |")
    lines += ['', 'Positive advantage means trained hidden has lower loss. The six family-by-target intervals use Bonferroni coverage but remain exploratory on this previously inspected cohort. Three fixed seeds do not quantify uncertainty across all possible random maps.','',
        'Random features explain much of the forecast-emulation gain: both random families beat the trained Bolt-hidden extension for TimesFM forecasts; for Bolt forecasts, trained TimesFM hidden beats RFF while random MLP is competitive and its difference interval includes zero. A nonsignificant difference is not an equivalence result.','',
        'The fusion-response contrast is narrower and stronger: both random families slightly harm the observable-only baseline, while trained TimesFM hidden has about7.7% lower loss than either family. This is conditional learned-feature/readout efficiency for predicting Bolt’s fixed intervention response. It does not demonstrate new information beyond input, causality, correct response semantics or equivalent model mechanisms.','',
        '| Target / family | Seed | Random loss | Trained-minus-random loss |', '|---|---:|---:|---:|']
    for r in seeds:lines.append(f"| {r['A']}→{r['B']} {r['endpoint']} / {r['family']} | {r['seed']} | {r['losses']['random']['mean']:.9g} | {-r['trained_advantage']['mean']:.9g} |")
    lines += ['',
        'Audit: original replication8 tests rejected25 isolated mutations and independently replayed60 intervals. These new controls passed8 tests and11 additional isolated mutations. Saved random predictions independently replay120 intervals; arithmetic_replay.json records the numerical discrepancy. All tests/mutations operate on isolated copies; original sources and primary artifacts remain bound by hashes.','',
        'Native follow-up remains HOLD. The proposed52-source/260-context study has20 new FIT sources and32 held-out TEST sources with anchor, weaker-amplitude, shifted-frequency, shifted-weaker and shifted-balanced conditions. Freeze old probes for the extrapolation comparison; separately refit every representation on an exactly matched20-source/96-row local FIT budget to assess feature efficiency. Prioritize fusion response; retain forecast comparisons as controls. Natural data would require a separate, larger source/domain design and must not be pooled into these synthetic claims.','',
        'Artifacts: protocol.json, probe_specs.json, per_initialization.json, results.json, plant_audit.json, arithmetic_replay.json, fit_target_provenance.json, new_condition_proposal.json and knowledge.json.']
    with (out/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    k=read(out/'knowledge.json');k['interpretation']={
        'counterevidence':'Generic untrained nonlinear features reproduce or exceed forecast-emulation gains. There is no broad trained-hidden advantage across these two families.',
        'bounded_support':'Trained TimesFM hidden retains a fusion-response prediction advantage over both random families under matched readout budgets; about7.7% lower loss.',
        'information_statement':'Deterministic hidden features introduce no information beyond the complete input. These comparisons measure supervised finite-budget feature/readout efficiency.',
        'scope':'Previously inspected synthetic TEST cohort; no causal, equivalent-semantic, universal-random-family or natural-data claim.'}
    k['controls']['independent_arithmetic_replay']=audit;k['controls']['fit_target_provenance']=read(out/'fit_target_provenance.json');k['controls']['execution']=read(out/'execution.json')
    k['next_decision']='Keep native HOLD. Review a limited new-condition study focused on fusion-response efficiency, with frozen-transfer and matched local-FIT axes, forecast targets as controls, and no pooled natural-data claim.'
    k['failed_instrumentation_runs']=[];k['report_source_sha256']=sha(__file__)
    (out/'knowledge.json').write_text(json.dumps(k,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write(out/'completion_manifest.json',dict(native_status='HOLD; zero native calls',originals_verified=True,hashes={f:sha(out/f) for f in ('protocol.json','results.json','per_initialization.json','predictions.npz','knowledge.json','report.md','arithmetic_replay.json','new_condition_proposal.json')}))
    print('Random controls report saved; independent intervals',audit['intervals'],'error',audit['max_absolute_error'])


if __name__=='__main__':render()
