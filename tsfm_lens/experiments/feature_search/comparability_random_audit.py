"""Isolated planted-regression artifact for the random-feature follow-up."""
from experiments.feature_search.comparability_random_features import ROOT,DEFAULT,read,write,sha
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def run(out=DEFAULT):
    """Require actual assertion failures for load-bearing mutations without touching frozen code."""
    module='comparability_random_features.py';test='test_comparability_random_features.py'
    plants=[
        ('missing_normalizer','test_full_input_and_normalizer_complete',module,"x[:,234:236]","np.zeros_like(x[:,234:236])"),
        ('ignored_initialization','test_seeded_nonlinear_maps[rff]',module,'rng=np.random.default_rng(seed)','rng=np.random.default_rng(0)'),
        ('nondeterministic_initialization','test_seeded_nonlinear_maps[mlp_relu]',module,'rng=np.random.default_rng(seed)','rng=np.random.default_rng()'),
        ('linearized_mlp','test_seeded_nonlinear_maps[mlp_relu]',module,'feature=np.maximum(np.sqrt(2/z.shape[1])*(z@projection.T)+bias,0)','feature=np.sqrt(2/z.shape[1])*(z@projection.T)+bias'),
        ('test_fitted_random_map','test_random_map_fit_excludes_test[rff]',module,'    xx=full_input(x);mean=', '    train=np.arange(len(x));w=np.ones(len(x))/len(x)\n    xx=full_input(x);mean='),
        ('test_fitted_mlp','test_random_map_fit_excludes_test[mlp_relu]',module,'    xx=full_input(x);mean=', '    train=np.arange(len(x));w=np.ones(len(x))/len(x)\n    xx=full_input(x);mean='),
        ('unbounded_pca_rank','test_projection_rank_and_rotation','comparability_core.py','    keep = np.flatnonzero(vals > max(vals[-1] * 1e-10, 1e-14))[-rank:]','    keep = np.flatnonzero(vals > max(vals[-1] * 1e-10, 1e-14))'),
        ('test_selected_alpha','test_grouped_selection_signal_and_test_label_isolation',module,"    picked=stable_argmin(scores)","    picked=stable_argmin(scores)\n    if np.max(abs(y))>1e8: picked=(picked+1)%len(candidates)"),
        ('rank_budget_changed','test_prepared_budget_and_native_hold',module,'PCA_ranks=[8,16]','PCA_ranks=[8,32]'),
        ('seed_cherry_pick_budget','test_prepared_budget_and_native_hold',module,'initialization_seeds=list(SEEDS)','initialization_seeds=list(SEEDS)[:1]'),
        ('native_hold_removed','test_prepared_budget_and_native_hold',module,"native_launch='HOLD; no native entry point'","native_launch='launch'"),
    ]
    records=[];original={str(p):sha(p) for p in [ROOT/'experiments/feature_search'/module,ROOT/'experiments/feature_search/comparability_core.py',ROOT/'tests'/test]}
    for name,node,filename,old,new in plants:
        folder=out/'isolated_plants'/name;pkg=folder/'experiments/feature_search';tests=folder/'tests';pkg.mkdir(parents=True);tests.mkdir()
        for file in (module,'comparability_core.py'):
            text=(ROOT/'experiments/feature_search'/file).read_text(encoding='utf-8')
            if file==filename:
                assert text.count(old)==1;mutated=text.replace(old,new,1);assert text!=mutated;text=mutated
            compile(text,str(pkg/file),'exec');(pkg/file).write_text(text,encoding='utf-8')
        shutil.copyfile(ROOT/'tests'/test,tests/test)
        (folder/'audit_identity.py').write_text("from pathlib import Path\nfrom experiments.feature_search import comparability_random_features as r\nassert Path(r.__file__).resolve().is_relative_to(Path(__file__).parent)\n",encoding='utf-8')
        env=dict(os.environ,PYTHONPATH=str(folder)+os.pathsep+str(ROOT),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TSFM_TEST_THREADS='1')
        result=subprocess.run([sys.executable,'-m','pytest','-p','audit_identity',str(tests/test)+'::'+node,'-q','--junitxml='+str(folder/'junit.xml')],cwd=folder,env=env,capture_output=True,text=True)
        (folder/'pytest.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        suites=ET.parse(folder/'junit.xml').getroot().findall('testsuite');counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        detected=result.returncode==1 and counts==dict(tests=1,failures=1,errors=0,skipped=0)
        record=dict(name=name,test=node,module=filename,original_text=old,mutated_text=new,mutant_sha256=sha(pkg/filename),counts=counts,returncode=result.returncode,detected=detected,log=str((folder/'pytest.log').relative_to(out)))
        write(folder/'record.json',record);records.append(record)
    assert original=={p:sha(Path(p)) for p in original}
    summary=dict(passed=all(r['detected'] for r in records),planted=len(records),detected=sum(r['detected'] for r in records),originals_unchanged=True,records=records)
    write(out/'plant_audit.json',summary)
    if not summary['passed']:raise RuntimeError('Random-control mutation audit failed')
    print('Random-control mutations detected',summary['detected'],'/',len(records),flush=True)


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(3*2**30,3*2**30));os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});run()
