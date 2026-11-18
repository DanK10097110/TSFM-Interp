"""Isolated mutation audit and independent saved-prediction arithmetic replay.

Never edit frozen sources, primary results, or prior knowledge records. Each
mutation receives its own shadow module copies and isolated pytest process.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
REPL=ROOT/'runs/comparability_replication4952000'
DEFAULT=ROOT/'runs/comparability_replication_audit4952100'
TEST='test_comparability_replication.py'


def read(path):
    """Read existing UTF-8 records without changing them."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    """Refuse to overwrite any audit artifact."""
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def sha(path):
    """Bind originals and every isolated mutant by content hash."""
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def plants():
    """Map each load-bearing assertion to an explicit compile-valid regression."""
    generator='test_fresh_generator_matches_old_five_conditions'
    permutation='test_source_matched_permutation_and_row_shuffle_plant'
    grant='test_slot_grant_required_and_bound'
    gates='test_exact_native_gates_reject_tiny_broken_replay'
    records=[]
    def add(name,test,old,new,module='comparability_replication.py'):
        records.append(dict(name=name,test=test,module=module,old=old,new=new))
    ret='    return rows,np.asarray(histories),np.asarray(futures)'
    add('generator_history_corruption',generator,ret,'    return rows,np.asarray(histories)+.01,np.asarray(futures)')
    add('generator_future_corruption',generator,ret,'    return rows,np.asarray(histories),np.asarray(futures)+.01')
    add('fresh_row_dropped',generator,ret,"    if rows[0]['seed']>=4952000: rows=rows[:-1]\n"+ret)
    add('fresh_sources_collapsed',generator,ret,"    if rows[0]['seed']>=4952000:\n        for row in rows: row['source']='collapsed'\n"+ret)
    add('fresh_test_mislabeled_fit',generator,ret,"    if rows[0]['seed']>=4952000: rows[0]['split']='fit'\n"+ret)
    retperm="    return np.array([by[mapping[r['source']],r['case']] for r in rows])"
    add('null_row_shuffle',permutation,retperm,'    return np.random.default_rng(seed).permutation(len(rows))')
    add('null_condition_rotation',permutation,retperm,"    result=np.array([by[mapping[r['source']],r['case']] for r in rows])\n    return np.roll(result.reshape(-1,5),1,axis=1).ravel()")
    add('null_identity',permutation,retperm,'    return np.arange(len(rows))')
    add('grant_absent_bypass',grant,"if not path.exists():raise RuntimeError('Native launch held: no slot grant')",'if not path.exists():return')
    add('grant_wrong_slot_accepted',grant,"grant.get('slot')!=2",'False')
    add('grant_wrong_hash_accepted',grant,"grant.get('protocol_sha256')!=sha(out/'protocol.json')",'False')
    add('grant_valid_rejected',grant,"    grant=read(path)","    grant=read(path)\n    raise RuntimeError('planted valid grant rejection')")
    for name in ('repeat','normalizer_identity','first_self','half_self'):
        add('ignored_gate_'+name,gates,"gates[k]!=0 for k in ('repeat','normalizer_identity','first_self','half_self')",f"(k!='{name}' and gates[k]!=0) for k in ('repeat','normalizer_identity','first_self','half_self')")
    add('closure_gate_disabled',gates,"if gates['input_closure']>2e-6:",'if False:')
    add('clean_gate_rejected',gates,"    if gates['input_closure']>2e-6:","    if gates['input_closure']>=0:")
    add('pca_fits_test_rows','test_frozen_kernel_train_block_ignores_fresh_features','    w = weights(rows, train)\n    cap =','    train = np.arange(len(rows))\n    w = weights(rows, train)\n    cap =','comparability_core.py')
    add('extension_deletes_hidden','test_exploratory_addition_recovers_missing_signal',"return (2*bank['input+A_forecast'][spec['base_kernel']]+bank['A_hidden'][spec['hidden_kernel']])/3","return bank['input+A_forecast'][spec['base_kernel']]")
    old="    return dict(specs[chosen],cv_loss=scores[chosen],status='exploratory old-FIT-only follow-up')"
    new="    result=dict(specs[chosen],cv_loss=scores[chosen],status='exploratory old-FIT-only follow-up')\n    if np.max(abs(y))>1e8: result['alpha']=999.\n    return result"
    add('extension_reads_test_labels','test_exploratory_addition_recovers_missing_signal',old,new)
    for name in ('torch','chronos'):
        add('import_side_effect_'+name,'test_module_import_has_no_native_side_effect',"if __name__=='__main__':main()",f"import types\nsys.modules['{name}']=types.ModuleType('{name}')\nif __name__=='__main__':main()")
    add('knowledge_erases_failure','test_knowledge_preserves_failed_instrumentation',"    data['events'].append(dict(unix_time=time.time(),**event))","    data['events']=[]\n    data['events'].append(dict(unix_time=time.time(),**event))")
    add('knowledge_reorders_history','test_knowledge_preserves_failed_instrumentation',"    data['events'].append(dict(unix_time=time.time(),**event))","    data['events'].insert(0,dict(unix_time=time.time(),**event))")
    return records


def isolated(out,plant=None):
    """Run unchanged tests against shadow copies, not mutated workspace modules."""
    name=plant['name'] if plant else 'unmodified_baseline'
    folder=out/'isolated'/name;modules=folder/'experiments/feature_search';tests=folder/'tests'
    modules.mkdir(parents=True);tests.mkdir()
    hashes={}
    for filename in ('comparability_replication.py','comparability_core.py'):
        source=ROOT/'experiments/feature_search'/filename;target=modules/filename
        text=source.read_text(encoding='utf-8')
        if plant and plant['module']==filename:
            assert text.count(plant['old'])==1,(name,text.count(plant['old']))
            text=text.replace(plant['old'],plant['new'],1)
            compile(text,str(target),'exec')
        target.write_text(text,encoding='utf-8')
        hashes[filename]=dict(original=sha(source),shadow=sha(target),changed=sha(source)!=sha(target))
    shutil.copyfile(ROOT/'tests'/TEST,tests/TEST)
    if plant:assert hashes[plant['module']]['changed']
    identity="from pathlib import Path\nimport experiments.feature_search.comparability_replication as r\nimport experiments.feature_search.comparability_core as c\nassert Path(r.__file__).resolve().is_relative_to(Path(__file__).parent)\nassert Path(c.__file__).resolve().is_relative_to(Path(__file__).parent)\n"
    (folder/'audit_identity.py').write_text(identity,encoding='utf-8')
    target=str(tests/TEST)+(('::'+plant['test']) if plant else '')
    env=dict(os.environ,PYTHONPATH=str(folder)+os.pathsep+str(ROOT),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TSFM_TEST_THREADS='1')
    result=subprocess.run([sys.executable,'-m','pytest','-p','audit_identity',target,'-q','--junitxml='+str(folder/'junit.xml')],cwd=folder,env=env,capture_output=True,text=True)
    (folder/'pytest.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    suites=ET.parse(folder/'junit.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    failures=[dict(test=case.attrib,message=failure.attrib.get('message',''),text=failure.text) for suite in suites for case in suite.findall('testcase') for failure in case.findall('failure')]
    passed=(result.returncode==1 and counts==dict(tests=1,failures=1,errors=0,skipped=0)) if plant else (result.returncode==0 and counts==dict(tests=8,failures=0,errors=0,skipped=0))
    record=dict(name=name,plant=plant,hashes=hashes,returncode=result.returncode,counts=counts,expected_detection=passed,failures=failures,log=str((folder/'pytest.log').relative_to(out)))
    write(folder/'record.json',record)
    return record


def source_means(values,rows,indices=None):
    """Independent arithmetic: explicit source accumulation with compensated summation."""
    indices=list(range(len(rows))) if indices is None else list(indices)
    grouped={}
    for index,value in zip(indices,values):grouped.setdefault(rows[index]['source'],[]).append(float(value))
    return np.array([math.fsum(grouped[s])/len(grouped[s]) for s in sorted(grouped)])


def bootstrap(values,coverage=.95):
    """Independent source-index bootstrap with explicit quantile interpolation."""
    v=np.asarray(values,float);indices=np.random.default_rng(4951000).integers(0,len(v),size=(4000,len(v)))
    draws=np.sort(v[indices].mean(axis=1))
    def quantile(q):
        index=q*(len(draws)-1);lo=int(math.floor(index));hi=int(math.ceil(index))
        return float(draws[lo]*(hi-index)+draws[hi]*(index-lo)) if hi!=lo else float(draws[lo])
    tail=(1-coverage)/2
    return dict(mean=math.fsum(v)/len(v),low=quantile(tail),high=quantile(1-tail),n_sources=len(v))


def replay(out):
    """Recompute losses in physical units, source aggregation and all60 stored intervals."""
    rows=read(REPL/'rows.json');results=read(REPL/'results.json');behaviors=read(REPL/'behavior_vs_semantics.json')
    with np.load(REPL/'inputs.npz') as f:x=f['history'].astype(float)
    with np.load(REPL/'targets.npz') as f:future=f['future']
    with np.load(REPL/'predictions.npz') as f:pred={k:f[k].copy() for k in f.files}
    saved={}
    for model in ('bolt','timesfm'):
        with np.load(REPL/f'{model}_native.npz') as f:saved[model]={k:f[k].astype(float) for k in ('native','fusion')}
    center=np.array([math.fsum(row)/len(row) for row in x])
    variance=np.array([math.fsum((row-m)**2)/len(row) for row,m in zip(x,center)])
    scale=np.sqrt(variance);checks=[];source_rows=[]
    def check(label,actual,recorded):
        assert actual['n_sources']==recorded['n_sources'],label
        error=max(abs(actual[k]-recorded[k]) for k in ('mean','low','high'))
        assert error<=2e-12,label
        checks.append(dict(label=label,max_absolute_error=error,replayed=actual))
    all_losses={}
    for result in results:
        a,b,ep=result['A'],result['B'],result['endpoint'];losses={}
        for arm,recorded in result['losses'].items():
            key=f'{a}_to_{b}__{ep}__{arm}'
            physical=pred[key]*scale[:,None]
            observed=saved[b]['native'] if ep=='forecast' else saved[b]['fusion']-saved[b]['native']
            if ep=='forecast':physical=physical+center[:,None]
            row_loss=np.array([math.fsum(residual**2)/len(residual)/var for residual,var in zip(physical-observed,variance)])
            losses[arm]=row_loss;aggregated=source_means(row_loss,rows)
            check(key,bootstrap(aggregated),recorded)
            source_rows.append(dict(key=key,source_losses=aggregated.tolist()))
        all_losses[a,b,ep]=losses
        for label,base,added in [('primary_gain','input','input+A_hidden'),('vs_Aforecast','input+A_forecast','input+A_hidden'),('exploratory_hidden_beyond_Aforecast','input+A_forecast','input+A_forecast+A_hidden')]:
            g=source_means(losses[base]-losses[added],rows)
            check(f'{a}:{b}:{ep}:{label}',bootstrap(g,1-.05/3 if label=='primary_gain' else .95),result['contrasts'][label])
    for record in behaviors:
        model,case=record['model'],record['case'];ids=[i for i,r in enumerate(rows) if case=='ALL' or r['case']==case]
        native=np.array([math.fsum((saved[model]['native'][i]-future[i])**2)/32/variance[i] for i in ids])
        fused=np.array([math.fsum((saved[model]['fusion'][i]-future[i])**2)/32/variance[i] for i in ids])
        for name,v in [('native_loss',native),('fusion_loss',fused),('fusion_gain',native-fused)]:
            check(f'{model}:{case}:{name}',bootstrap(source_means(v,rows,ids)),record[name])
    self_plants=[]
    ref=results[0];loss=all_losses[ref['A'],ref['B'],ref['endpoint']]
    gain=loss['input']-loss['input+A_hidden'];recorded=ref['contrasts']['primary_gain']
    for label,wrong in [('reversed_gain',bootstrap(source_means(-gain,rows),1-.05/3)),('row_bootstrap',bootstrap(gain,1-.05/3)),('changed_prediction_loss',bootstrap(source_means(gain+.01,rows),1-.05/3))]:
        caught=False
        try:check(label,wrong,recorded)
        except AssertionError:caught=True
        assert caught,label
        self_plants.append(dict(plant=label,detected=caught))
    write(out/'arithmetic_replay.json',dict(passed=True,intervals_replayed=len(checks),max_absolute_error=max(c['max_absolute_error'] for c in checks),checks=checks,source_losses=source_rows,self_plants=self_plants,
        independence='No production metric, grouping, bootstrap or inference function called. Physical-unit inverse and compensated sums replay saved predictions. Source-index resampling and explicit quantile interpolation replay intervals.',
        limits='Does not independently reproduce native inference or regenerate null predictions; frozen-probe construction and native gates remain upstream evidence.'))
    return read(out/'arithmetic_replay.json')


def run(out):
    """Complete the audit before permitting any new random-feature analysis."""
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    watched=[ROOT/'experiments/feature_search/comparability_replication.py',ROOT/'experiments/feature_search/comparability_core.py',ROOT/'tests'/TEST]+[REPL/f for f in ('protocol.json','results.json','predictions.npz','knowledge.json','behavior_vs_semantics.json','validation.json')]
    before={str(p.relative_to(ROOT)):sha(p) for p in watched}
    write(out/'protocol.json',dict(question='Do unchanged replication tests reject isolated load-bearing regressions, and do raw prediction arithmetic/source CIs replay independently?',test_sha256=sha(ROOT/'tests'/TEST),plants=plants(),original_hashes=before,source_sha256=sha(__file__),no_native=True,limits=dict(cpu_threads=1,address_space_bytes=3*2**30),replay_tolerance=2e-12))
    baseline=isolated(out)
    assert baseline['expected_detection'],'Isolated clean baseline failed'
    records=[]
    for plant in plants():
        record=isolated(out,plant);records.append(record)
        print(plant['name'],'detected',record['expected_detection'],flush=True)
    arithmetic=replay(out)
    after={str(p.relative_to(ROOT)):sha(p) for p in watched};assert before==after,'Frozen original changed'
    complete=all(r['expected_detection'] for r in records)
    summary=dict(passed=complete,baseline=baseline['counts'],planted=len(records),detected=sum(r['expected_detection'] for r in records),records=records,arithmetic=dict(intervals=arithmetic['intervals_replayed'],max_absolute_error=arithmetic['max_absolute_error']),originals_unchanged=True,seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(out/'audit_summary.json',summary)
    write(out/'knowledge.json',dict(question=read(out/'protocol.json')['question'],design='25 compile-valid mutations on isolated copies of frozen modules; unchanged copied tests; subprocess import-path assertions; XML distinguishes test failures from collection errors. Independent physical-unit prediction replay and source bootstrap.',exact_findings=summary,controls=['Clean isolated8-test baseline','Original hashes unchanged','Three arithmetic self-plants','No native imports/inference'],limitations=arithmetic['limits'],evidence_class='Post-result instrumentation and arithmetic audit; no new scientific outcome or stronger evidence class.',failed_instrumentation_runs=[r for r in records if not r['expected_detection']],next_decision='Proceed to freeze random-feature controls.' if complete else 'Investigate surviving plants before new controls.'))
    if not complete:raise RuntimeError('Mutation audit incomplete')
    print('Audit complete',summary['detected'],'/',summary['planted'],'intervals',arithmetic['intervals_replayed'],flush=True)


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(3*2**30,3*2**30));os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=DEFAULT);run(parser.parse_args().out)
