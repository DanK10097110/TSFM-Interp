"""Model-free preflight for the weather normalizer intervention; real helper, fake net."""
import argparse
from pathlib import Path
import subprocess
import sys
from experiments.feature_search.persistence_protocol import write,sha


def run(out):
    """Record clean tests and isolated ignored-lock/inverse/replay/hold regressions."""
    out.mkdir(parents=True,exist_ok=False)
    tests='tsfm_lens/tests/test_postrelease_weather_lock.py'
    clean=subprocess.run([sys.executable,'-m','pytest','-q',tests],capture_output=True,text=True)
    (out/'tests.log').write_text(clean.stdout+clean.stderr,encoding='utf-8')
    assert clean.returncode==0 and '4 passed' in clean.stdout
    plants=[
      ('ignored_lock','postrelease_weather_lock',"normalization(native,anchor,'background') if locked else contextlib.nullcontext()",'contextlib.nullcontext()','test_known_mechanism_forecasts_and_call_budget'),
      ('bolt_wrong_inverse','coherence_study','def fixed(x, loc_scale=None): return original(x, reference)','def fixed(x, loc_scale=None): return original(x, reference)[0], original(x)[1]','test_real_helper_locks_forward_inverse_and_restores'),
      ('timesfm_wrong_inverse','coherence_study','def fixed(x, unused_mu, unused_sigma, reverse=False): return original(x, mu, sigma, reverse=reverse)','def fixed(x, unused_mu, unused_sigma, reverse=False): return original(x, unused_mu if reverse else mu, unused_sigma if reverse else sigma, reverse=reverse)','test_real_helper_locks_forward_inverse_and_restores'),
      ('ignored_full_replay','postrelease_weather_lock',"exact(base,reference['native'][i],'original full forecast replay')",'pass','test_full_forecast_mismatch_and_ignored_lock_fail'),
      ('bypass_slot_hold','postrelease_weather_lock',"if not grant:raise PermissionError('HOLD: coordinator slot grant required')",'if not grant:return','test_slot_hold_before_any_native_work')]
    results=[]
    for name,module,old,new,test in plants:
        file=Path(__file__).with_name(module+'.py');source=file.read_text(encoding='utf-8')
        assert source.count(old)==1;mutant=source.replace(old,new);assert mutant!=source
        code='from experiments.feature_search import '+module+' as m\nimport pytest\nexec(compile('+repr(mutant)+',m.__file__,"exec"),m.__dict__)\n'
        if module=='coherence_study':
            code+='from experiments.feature_search import postrelease_weather_lock as study\nstudy.normalization=m.normalization\n'
        code+='raise SystemExit(pytest.main(["-q",'+repr(tests+'::'+test)+']))'
        proc=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        (out/f'{name}.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
        caught=proc.returncode==1 and '1 failed' in proc.stdout and '1 error' not in proc.stdout
        results.append(dict(plant=name,test=test,caught=caught,returncode=proc.returncode))
        write(out/'plants.json',results)
        knowledge=dict(question='Can the mechanism instrument detect ignored locking, unmatched inverse, wrong full replay and unauthorized launch?',
              design='Real coherence normalizer helper applied to two tiny known-answer nonlinear fake models; no weights, data downloads or native inference.',
              exact_findings=dict(tests_passed=4,plants_completed=len(results),plants_caught=sum(r['caught'] for r in results),native_launched=False),
              controls='Full12, ordinary residual4, full-history-locked residual28; inverse and restoration independently checked; in-memory mutants never edit shared helper.',
              limitations='Instrument unit tests do not establish native anchor identity; that remains an exact gate in the future authorized worker.',
              evidence_class='software verification',next_decision='Freeze same30-window mechanism protocol; HOLD native until slot grant.')
        write(out/'knowledge.json',knowledge)
        assert caught,(name,proc.stdout,proc.stderr)
    write(out/'manifest.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(knowledge['exact_findings'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
