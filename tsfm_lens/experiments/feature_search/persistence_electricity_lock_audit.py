"""Preflight small fake-model checks; never load checkpoints or change shared code."""
import argparse
from pathlib import Path
import subprocess
import sys
from experiments.feature_search.persistence_protocol import write,sha


def audit(out):
    """Catch missing locks/repeats and falsely accepted batch4 drift on known answers."""
    out.mkdir(parents=True,exist_ok=False)
    tests='tsfm_lens/tests/test_persistence_electricity_lock.py'
    clean=subprocess.run([sys.executable,'-m','pytest','-q',tests],capture_output=True,text=True)
    (out/'tests.log').write_text(clean.stdout+clean.stderr,encoding='utf-8')
    assert clean.returncode==0 and '3 passed' in clean.stdout
    plants=[
      ('ignored_lock','persistence_electricity_lock','with normalization_mode(native,full,True):','with normalization_mode(native,full,False):','test_fresh_singletons_and_old_batch_diagnostic'),
      ('ignored_implicit_repeat','persistence_electricity_lock',"exact(value[0][0],self.trace_expected,'implicit TF ordinary full repeat')",'pass','test_exact_implicit_repeat_and_slot_hold'),
      ('silent_approximate_identity','persistence_electricity_lock','bit_identical=bool(np.array_equal(new,old))','bit_identical=bool(np.allclose(new,old))','test_old_loss_drift_bound_known_answer'),
      ('wrong_drift_bound','persistence_electricity_lock_analysis','bound=2*np.sqrt(old_loss)*rms+rms**2','bound=np.sqrt(old_loss)*rms+rms**2','test_old_loss_drift_bound_known_answer')]
    results=[]
    for name,module,old,new,test in plants:
        source=Path(__file__).with_name(module+'.py').read_text(encoding='utf-8')
        assert source.count(old)==1;mutant=source.replace(old,new);assert mutant!=source
        code='from experiments.feature_search import '+module+' as m\nimport pytest\nexec(compile('+repr(mutant)+',m.__file__,"exec"),m.__dict__)\nraise SystemExit(pytest.main(["-q",'+repr(tests+'::'+test)+']))'
        run=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        (out/f'{name}.log').write_text(run.stdout+run.stderr,encoding='utf-8')
        caught=run.returncode==1 and '1 failed' in run.stdout and '1 error' not in run.stdout
        results.append(dict(plant=name,test=test,caught=caught,returncode=run.returncode));write(out/'plants.json',results)
        k=dict(question='Can reused singleton locking distinguish its exact gates from nonexact historical batch4 replay?',
               design='Known-answer fake models using existing joint normalizer helper; in-memory mutations only.',
               exact_findings=dict(tests_passed=3,plants_completed=len(results),plants_caught=sum(r['caught'] for r in results),native_launched=False),
               controls='Exact implicit full repeat, known locked residual28 versus ordinary4, explicit nonidentity for tiny batch differences, algebraic MSE drift bound.',
               limitations='No native outcomes; existing helper inverse tests remain bound from completed weather instrument.',
               evidence_class='software verification',next_decision='Freeze same64 electricity sources only; HOLD native until explicit slot grant after Plato32 causal.')
        write(out/'knowledge.json',k)
        assert caught,(name,run.stdout,run.stderr)
    write(out/'manifest.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()});print(k['exact_findings'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);audit(parser.parse_args().out)
