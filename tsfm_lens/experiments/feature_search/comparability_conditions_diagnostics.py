"""Post-result saved-data sensitivity audit; never refits or changes the primary."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json
import resource
import shutil
import numpy as np
from experiments.feature_search.comparability_conditions import DEFAULT, read, write, sha, fit_ids, knowledge
from experiments.feature_search.comparability_replication_audit import source_means, bootstrap


def run(out=DEFAULT):
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    original = read(out / 'completion_manifest.json')
    for name, digest in original['hashes'].items():
        assert sha(out / name) == digest, name
    metadata = read(out / 'rows.json')
    _, ids = fit_ids(metadata)
    rows = [metadata[i] for i in ids]
    with np.load(out / 'inputs.npz') as data:
        variance = data['history'][ids].astype(float).var(axis=1)
    with np.load(out / 'bolt_native.npz') as data:
        response = (data['fusion'][ids].astype(float) - data['native'][ids].astype(float)) / np.sqrt(variance[:, None])
    prefix = 'timesfm_to_bolt__fusion_response__'
    with np.load(out / 'local_predictions.npz') as data:
        losses = {k[len(prefix):]: ((data[k] - response)**2).mean(axis=1) for k in data.files if k.startswith(prefix)}
    for family in ('rff', 'mlp_relu'):
        losses[family] = np.mean([v for k, v in losses.items() if k.startswith(family + '_')], axis=0)
    energy = (response**2).mean(axis=1)
    cases = list(dict.fromkeys(r['case'] for r in rows))

    def summarize(indices, physical=False):
        scale = variance if physical else np.ones(len(rows))
        selected = [rows[i] for i in indices]
        risk = {a: float(np.mean(source_means(losses[a][indices] * scale[indices], selected)))
                for a in ('input+A_forecast', 'trained', 'rff', 'mlp_relu')}
        zero = float(np.mean(source_means(energy[indices] * scale[indices], selected)))
        contrasts = {}
        for family in ('rff', 'mlp_relu'):
            gain = source_means((losses[family] - losses['trained'])[indices] * scale[indices], selected)
            contrasts[family] = dict(gain=bootstrap(gain, .975), relative_reduction=(risk[family]-risk['trained'])/risk[family])
        return dict(loss=risk, zero_response_loss=zero, loss_over_mean_response_energy={a: v/zero for a,v in risk.items()}, contrasts=contrasts)

    all_ids = np.arange(len(rows))
    result = dict(evidence_class='Post-hoc descriptive saved-data sensitivity; no selection, refitting or primary revision',
                  denominator='Primary is ratio of source-mean loss difference to source-mean random loss; never mean per-row ratios. All arms use identical history variance. Energy ratios here divide by aggregate energy, not tiny individual responses.',
                  all_normalized=summarize(all_ids), all_physical=summarize(all_ids, True),
                  response_energy_quantiles=dict(zip(('min','p05','median','p95','max'), np.quantile(energy,[0,.05,.5,.95,1]).tolist())),
                  exact_zero_responses=int(np.sum(energy == 0)), history_variance_range=[float(variance.min()),float(variance.max())],
                  by_condition={}, leave_one_condition_out={}, fitting_budget=[])
    for case in cases:
        ix=np.array([i for i,r in enumerate(rows) if r['case']==case])
        result['by_condition'][case]=summarize(ix)
        ix=np.array([i for i,r in enumerate(rows) if r['case']!=case])
        result['leave_one_condition_out'][case]=summarize(ix)
    fitrows=read(out/'local_fit_rows.json')
    counts={s:sum(r['source']==s for r in fitrows) for s in {r['source'] for r in fitrows}}
    weights=np.array([1/len(counts)/counts[r['source']] for r in fitrows])
    sw=np.sqrt(weights)
    with np.load(out/'local_probes.npz') as data:
        for spec in read(out/'local_probe_specs.json'):
            if not spec['key'].startswith(prefix):continue
            kernel=data[spec['key']+'__train_kernel']
            eigenvalues=np.linalg.eigvalsh(kernel*sw[:,None]*sw[None,:])
            result['fitting_budget'].append(dict(arm=spec['arm'],rank=spec.get('rank'),alpha=spec['alpha'],
                cv_loss=spec.get('cv_loss'),weighted_kernel_trace=float(eigenvalues.sum()),
                effective_df_including_intercept=float(1+np.sum(eigenvalues/(eigenvalues+spec['alpha'])))))
    result['budget_caveat']='Same 20 sources/96 rows, widths, rank grid and eight choices per added arm; each random family averages three separately selected losses, not an ensemble prediction. Equal grids need not yield equal effective degrees of freedom. No matched pretraining architecture/compute; intervals condition on this FIT sample and three fixed random maps.'
    primary=read(out/'primary_result.json')
    for family in ('rff','mlp_relu'):
        actual=result['all_normalized']['contrasts'][family]
        for k in ('mean','low','high'):
            assert abs(actual['gain'][k]-primary['families'][family]['gain'][k])<2e-12
        assert abs(actual['relative_reduction']-primary['families'][family]['relative_reduction'])<2e-12
    result['primary_replay_matches']=True
    result['alternative_explanations']=[
        'Task-conditioned learned feature geometry or regularization can improve finite-sample prediction without adding information beyond deterministic full inputs.',
        'The five-condition mixture and history-variance weighting affect the estimand; leave-condition-out and physical-unit results are descriptive sensitivity checks, not new primary tests.',
        'Only two random-map families and three initializations were tested; architecture, pretraining data/compute, retained variance and effective complexity are not matched.',
        'The fixed base recipe and small localFIT sample can favor particular features; conditional TEST intervals omit training-set and initialization-distribution uncertainty.',
        'Weak frozen transfer and harmful anchor fusion limit claims of generalization or correct semantics. Prediction does not establish cross-model causal equivalence.'
    ]
    write(out/'confound_diagnostics.json', result)
    shutil.copyfile(out/'knowledge.json',out/'knowledge_before_closure.json')
    knowledge(out,dict(experiment='saved_confound_audit_and_branch_closure',status='complete',question='Could denominators, small responses or fitting budgets explain the primary?',design='Saved predictions only; no fits or model calls; all sensitivity analyses post-hoc.',exact_findings=result,controls=['Replayed original primary from saved predictions','Preserved all original scientific artifacts and completion manifest'],limitations=result['alternative_explanations'],evidence_class=result['evidence_class'],next_decision='Close native branch; no broad expansion or further new-model calls.'))
    k=read(out/'knowledge.json')
    k.update(status='complete_native_branch_closed',next_decision='Native branch closed by user; no broad expansion and no further new-model calls.',confound_diagnostics='confound_diagnostics.json',alternative_explanations=result['alternative_explanations'])
    (out/'knowledge.json').write_text(json.dumps(k,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,digest in original['hashes'].items():
        if name!='knowledge.json':assert sha(out/name)==digest,name
    write(out/'closure_manifest.json',dict(status='native_branch_closed',original_completion_manifest=sha(out/'completion_manifest.json'),
        prior_knowledge=sha(out/'knowledge_before_closure.json'),hashes={f:sha(out/f) for f in ('confound_diagnostics.json','knowledge.json')},diagnostic_code_sha256=sha(__file__)))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
