"""Distill completed fresh replication without changing frozen analysis or selection."""
from experiments.feature_search.comparability_replication import DEFAULT,read,knowledge,sha
import json


def render(out=DEFAULT):
    """Report every registered contrast, observable-only challenge and behavioral limitation."""
    results=read(out/'results.json');behavior=read(out/'behavior_vs_semantics.json')
    lines=['# Comparability fresh replication 4952000', '',
        '64 fresh source seeds4952000–4952063; five dependent conditions per seed. All320 rows are TEST-only. The original20 FIT sources determine the serialized probes. No fresh outcome selects a layer, PCA rank, baseline, ridge setting, or mixture.', '',
        '| Direction / target | Input loss | Input+A hidden | Input+A forecast | Input+A forecast+A hidden | Input+B hidden | Gain vs input [98.33% source CI] | Replicates |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for r in results:
        l=r['losses'];g=r['contrasts']['primary_gain']
        lines.append(f"| {r['A']} → {r['B']} / {r['endpoint']} | {l['input']['mean']:.8g} | {l['input+A_hidden']['mean']:.8g} | {l['input+A_forecast']['mean']:.8g} | {l['input+A_forecast+A_hidden']['mean']:.8g} | {l['input+B_hidden']['mean']:.8g} | {g['mean']:.8g} [{g['low']:.8g}, {g['high']:.8g}] | {r['primary_replicates']} |")
    lines += ['', 'The primary screen requires at least5% lower prediction loss than the frozen strong input baseline, a positive Bonferroni3 source-bootstrap lower bound, and gain above the95th percentile of99 source-matched alignment shuffles. Shuffle tail ranks are diagnostics, not conditional-independence p-values.', '',
        'The following added-hidden comparison was introduced after the pilot and is explicitly exploratory. Its feature/ridge choice used only old FIT sources; the fresh TEST remains untouched by selection.', '',
        '| Direction / target | Added hidden gain beyond input+A forecast [95% source CI] | Alignment-null tail rank |', '|---|---|---:|']
    for r in results:
        g=r['contrasts']['exploratory_hidden_beyond_Aforecast']
        tail=(1+sum(v>=g['mean'] for v in r['shuffled_gains']['input+A_forecast+A_hidden']))/100
        lines.append(f"| {r['A']} → {r['B']} / {r['endpoint']} | {g['mean']:.8g} [{g['low']:.8g}, {g['high']:.8g}] | {tail:.2f} |")
    lines += ['', 'Prediction and intervention correctness are separate measurements. The response target is the recipient model’s first-block pure depth1 fusion-minus-native forecast. Its successful emulation does not establish shared component meaning, correct decomposition semantics, or cross-model causal mediation.', '',
        '| Actual native model | Native forecast loss | Fused forecast loss | Fusion gain [95% source CI] |', '|---|---:|---:|---|']
    for r in behavior:
        if r['case']=='ALL':
            g=r['fusion_gain']
            lines.append(f"| {r['model']} | {r['native_loss']['mean']:.8g} | {r['fusion_loss']['mean']:.8g} | {g['mean']:.8g} [{g['low']:.8g}, {g['high']:.8g}] |")
    lines += ['', 'All losses use the same history-only normalizer for every arm. Input controls include untruncated full-history ridge and nonlinear descriptors; the frozen pilot FIT selection chose the comparator for each target. Same-model hidden is an empirical intermediate-layer comparator, not a mathematical ceiling.', '',
        'All repeat, normalizer-identity, first-block self-patch and half-depth self-patch gates are exact. Native worker records bind source and checkpoint hashes. Workers were serial, one CPU thread/core, with900-second and3-GiB RSS limits. execution.json records the actual use.', '',
        'Uncertainty resamples64 whole seeds, never conditions or horizon steps. Results generalize only to fresh seeds of these five synthetic conditions. They do not reverse the pilot’s mixed/negative natural-real results. Conditional bootstrap omits old-FIT uncertainty. No new natural data was included.', '',
        'The failed preflight is preserved in ../comparability_replication4952000_preflight_failed/: one test assumed a seed field on real metadata. The corrected suite passed8 tests; fresh inputs, training support, probe specifications and frozen coefficients were byte-identical after correction. Pilot rotation failure and the unchanged1e-7 rotation threshold are documented in ../comparability_pilot4951000/knowledge.json.', '',
        'Persistent exact findings, failures, controls and decisions are in knowledge.json. results.json retains all99 null draws. behavior_vs_semantics.json separates outcomes by condition; predictions.npz preserves every frozen prediction.']
    passed=[r for r in results if r['primary_replicates']]
    exploratory=[r for r in results if r['contrasts']['exploratory_hidden_beyond_Aforecast']['low']>0]
    decision='No further broad native expansion is justified by these same-generator results alone. '
    if exploratory:
        decision+='At least one exploratory added-hidden contrast has a positive interval; preregister that narrow target on new conditions before any larger mechanistic or real-data study.'
    else:
        decision+='The added-hidden arm does not establish value beyond observable forecasts; prioritize cheaper observable controls rather than more hidden captures.'
    lines += ['',f'{len(passed)}/3 primary contrasts pass the frozen screen. '+decision]
    with (out/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    knowledge(out,dict(experiment='final_distillation',status='complete',question='Which predictive gains replicate, and do hidden states add beyond the other model forecast?',evidence_class='Fresh-seed predictive replication; exploratory added-hidden extension',exact_findings=dict(primary_pass_count=len(passed),primary_total=3,exploratory_positive_interval_targets=[dict(A=r['A'],B=r['B'],endpoint=r['endpoint']) for r in exploratory]),controls=['Frozen old-FIT probes','All fresh seeds TEST','Strong input and observable-only comparators','Source-matched alignment null','Exact native replay gates'],limitations=['Same synthetic generator families','No causal or semantic equivalence conclusion','Exploratory extra arm'],next_decision=decision,report='report.md',report_source_sha256=sha(__file__)))
    distilled=read(out/'knowledge.json')
    distilled.update(status='complete',design=read(out/'protocol.json'),exact_findings=results,
        behavior_vs_semantics=behavior,evidence_class='Fresh-seed frozen-probe predictive replication; added-hidden follow-up is exploratory; no cross-model causal or semantic identity claim.',
        controls=dict(validation=read(out/'validation.json'),execution=read(out/'execution.json'),native={m:read(out/f'{m}_worker.json') for m in ('bolt','timesfm')}),
        limitations=['Same five synthetic conditions; no natural-real extension.','Source-bootstrap intervals condition on old FIT training and generator.','Matched shuffles are alignment diagnostics, not conditional-independence p-values.','The input+Aforecast+Ahidden arm was added after the pilot and remains exploratory.','Prediction of a fusion response is distinct from its correctness and semantic interpretation.'],
        failed_instrumentation_runs=[dict(run='../comparability_replication4952000_preflight_failed',status='1 test failed,7 passed before native launch',cause='Test fixture assumed seed field on real metadata; fixed optional lookup;8 passed. Inputs and fitted probes unchanged.')],next_decision=decision)
    (out/'knowledge.json').write_text(json.dumps(distilled,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':render()
