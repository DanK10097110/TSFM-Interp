"""Isolated load-bearing regression plants for the prepared4962000 study."""
from experiments.feature_search.comparability_conditions import ROOT,DEFAULT,read,write,sha,verify,knowledge
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def mutations():
    """One changed scientific or launch invariant per compile-valid isolated plant."""
    return [
        ('wrong_seed_namespace','test_namespace_and_waveforms','for seed in range(4962000,4962052):','for seed in range(4954000,4954052):'),
        ('wrong_shifted_frequency','test_namespace_and_waveforms',"('shifted_frequency',7.35,23.35,8.)","('shifted_frequency',8.35,23.35,8.)"),
        ('fit_test_boundary','test_namespace_and_waveforms',"split='fit' if seed<4962020 else 'test'","split='fit' if seed<4962019 else 'test'"),
        ('fit_budget100_not96','test_exact_local_budget_and_whole_sources',"not (seed<4962004 and case=='shifted_balanced')","not (seed<4962000 and case=='shifted_balanced')"),
        ('source_overlap_guard_disabled','test_exact_local_budget_and_whole_sources',"    assert not ({rows[i]['source'] for i in fit}&{rows[i]['source'] for i in test})","    pass # planted missing source-overlap guard"),
        ('either_family_rescues','test_primary_both_families_and_fixed_coverage',"passed=all(v['passed'] for v in results.values())","passed=any(v['passed'] for v in results.values())"),
        ('uncorrected_interval','test_primary_both_families_and_fixed_coverage','COVERAGE=.975','COVERAGE=.95'),
        ('row_instead_of_source_n','test_primary_both_families_and_fixed_coverage',"list(group_values(loss-trained_loss,rows,np.arange(len(rows))).values())","list(loss-trained_loss)"),
        ('minimum_reduction_removed','test_primary_five_percent_random_denominator','MIN_REDUCTION=.05','MIN_REDUCTION=0.'),
        ('wrong_reduction_denominator','test_primary_five_percent_random_denominator',"relative=gain['mean']/random_mean if random_mean>0 else None","relative=gain['mean']/float(np.mean(trained_loss)) if random_mean>0 else None"),
        ('interval_gate_removed','test_primary_interval_and_initialization_average',"random_mean>0 and gain['low']>0 and relative>=MIN_REDUCTION","random_mean>0 and relative>=MIN_REDUCTION"),
        ('first_random_seed_only','test_primary_interval_and_initialization_average',"loss=np.mean(np.asarray(random_losses[family]),axis=0)","loss=np.asarray(random_losses[family])[0]"),
        ('energy_called_correctness','test_response_energy_is_not_correctness','gain=native_loss-fusion_loss','gain=energy'),
        ('response_sign_reversed','test_response_energy_is_not_correctness','delta=fusion-native','delta=native-fusion'),
        ('missing_grant_bypassed','test_old_or_missing_slot_grant_is_rejected',"if not path.exists():raise RuntimeError('Native HOLD: new slot grant required')","if not path.exists():return"),
        ('old_grant_accepted','test_old_or_missing_slot_grant_is_rejected',"grant.get('protocol_sha256')!=sha(out/'protocol.json')","False"),
        ('invalid_slot_accepted','test_old_or_missing_slot_grant_is_rejected',"grant.get('slot') not in (1,2)","grant.get('slot') not in (1,2,3)"),
        ('failed_tests_accepted','test_readiness_binds_tests_and_plants_to_protocol',"not record['passed'] or record.get('protocol_sha256')!=sha(out/'protocol.json')","record.get('protocol_sha256')!=sha(out/'protocol.json')"),
        ('old_tests_accepted','test_readiness_binds_tests_and_plants_to_protocol',"not record['passed'] or record.get('protocol_sha256')!=sha(out/'protocol.json')","not record['passed']"),
        ('test_label_selected_hyperparameter','test_local_selection_ignores_test_labels','    pick=stable_argmin(scores)','    pick=stable_argmin(scores)\n    if np.max(abs(y))>1e8: pick=(pick+1)%len(candidates)'),
        ('rank_grid_changed','test_prepare_fixed_base_and_no_native','ranks=[8,16]','ranks=[8,32]'),
        ('fixed_sourceforecast_base_changed','test_prepare_fixed_base_and_no_native',"    return {s['arm']:s for s in p['base_specs'] if (s['A'],s['B'],s['endpoint'])==(a,b,ep)}","    result={s['arm']:s for s in p['base_specs'] if (s['A'],s['B'],s['endpoint'])==(a,b,ep)}\n    result['input+A_forecast']=dict(result['input+A_forecast'],kernel='rbf2.0+pca16')\n    return result"),
        ('alpha_grid_changed','test_prepare_fixed_base_and_no_native','alphas=list(ALPHAS)','alphas=[.01,.1,1.]'),
        ('width_budget_changed','test_prepare_fixed_base_and_no_native','widths=dict(bolt=4608,timesfm=5120)','widths=dict(bolt=4608,timesfm=1024)'),
        ('natural_expansion_added','test_prepare_fixed_base_and_no_native',"natural_leg='NONE; no natural-context expansion authorized or prepared.'","natural_leg='EXPANSION; planted unsupported natural leg'"),
        ('random_seed_grid_changed','test_prepare_fixed_base_and_no_native','initialization_seeds=list(SEEDS)','initialization_seeds=list(SEEDS)[:1]'),
        ('native_import_on_prepare','test_prepare_fixed_base_and_no_native',"if __name__=='__main__':main()","import types\nsys.modules['torch']=types.ModuleType('torch')\nif __name__=='__main__':main()"),
        ('all_rows_enter_local_fit','test_mock_pipeline_keeps_fit_and_axes_separate',"allrows=read(out/'rows.json');ids,_=fit_ids(allrows);rows=[allrows[i] for i in ids]","allrows=read(out/'rows.json');ids=np.arange(len(allrows));rows=[allrows[i] for i in ids]"),
        ('axes_conflated','test_mock_pipeline_keeps_fit_and_axes_separate',"axis='frozen_extrapolation_descriptive'","axis='local_FIT'"),
    ]


def run(out=DEFAULT):
    """Write plant-to-test evidence, preserve originals, and mark readiness without a grant."""
    verify(out)
    if not read(out/'validation.json')['passed']:raise RuntimeError('Clean tests must pass first')
    source=ROOT/'experiments/feature_search/comparability_conditions.py';test=ROOT/'tests/test_comparability_conditions.py';before={str(p):sha(p) for p in [source,test]};records=[]
    for name,node,old,new in mutations():
        folder=out/'isolated_plants'/name;pkg=folder/'experiments/feature_search';tests=folder/'tests';pkg.mkdir(parents=True);tests.mkdir()
        text=source.read_text(encoding='utf-8');assert text.count(old)==1,(name,text.count(old));mutated=text.replace(old,new,1);assert mutated!=text;compile(mutated,str(pkg/source.name),'exec')
        (pkg/source.name).write_text(mutated,encoding='utf-8');shutil.copyfile(test,tests/test.name)
        (folder/'audit_identity.py').write_text("from pathlib import Path\nfrom experiments.feature_search import comparability_conditions as r\nassert Path(r.__file__).resolve().is_relative_to(Path(__file__).parent)\n",encoding='utf-8')
        env=dict(os.environ,PYTHONPATH=str(folder)+os.pathsep+str(ROOT),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TSFM_TEST_THREADS='1')
        result=subprocess.run([sys.executable,'-m','pytest','-p','audit_identity',str(tests/test.name)+'::'+node,'-q','--junitxml='+str(folder/'junit.xml')],cwd=folder,env=env,capture_output=True,text=True)
        (folder/'pytest.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        suites=ET.parse(folder/'junit.xml').getroot().findall('testsuite');counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        detected=result.returncode==1 and counts==dict(tests=1,failures=1,errors=0,skipped=0)
        record=dict(name=name,test=node,original_text=old,mutant_text=new,mutant_sha256=sha(pkg/source.name),counts=counts,returncode=result.returncode,detected=detected,log=str((folder/'pytest.log').relative_to(out)))
        write(folder/'record.json',record);records.append(record);print(name,detected,flush=True)
    assert before=={p:sha(Path(p)) for p in before}
    summary=dict(passed=all(r['detected'] for r in records),planted=len(records),detected=sum(r['detected'] for r in records),records=records,originals_unchanged=True,protocol_sha256=sha(out/'protocol.json'))
    write(out/'plant_audit.json',summary)
    knowledge(out,dict(experiment='isolated_plants',status='passed' if summary['passed'] else 'failed',exact_findings=dict(planted=len(records),detected=summary['detected']),controls='Separate subprocesses/shadow copies; no original source edits; collection errors excluded',next_decision='Request slot only if all passed; no launch yet.'))
    if not summary['passed']:raise RuntimeError('Prelaunch plant audit failed')
    assert not (out/'slot_grant.json').exists(),'Preparation must not create a grant'
    write(out/'readiness.json',dict(status='READY_TO_REQUEST_SLOT; NATIVE_HOLD',protocol_sha256=sha(out/'protocol.json'),validation_sha256=sha(out/'validation.json'),plant_audit_sha256=sha(out/'plant_audit.json'),native_rows_captured=0))
    k=read(out/'knowledge.json');k.update(status='ready_native_hold',controls=dict(validation=read(out/'validation.json'),plant_audit=summary),next_decision='Request a NEW native slot for4962000. No inherited grant; do not launch until explicitly granted.')
    (out/'knowledge.json').write_text(__import__('json').dumps(k,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(3*2**30,3*2**30));os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});run()
