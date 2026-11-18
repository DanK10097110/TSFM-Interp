"""Run focused tests and isolated plants before sealing the weather forecast pilot."""
import argparse
from pathlib import Path
import subprocess
import sys
from experiments.feature_search.persistence_protocol import read, write, sha


def audit(out):
    """Each plant must produce a test assertion failure, never a collection error."""
    out.mkdir(parents=True,exist_ok=False)
    tests='tsfm_lens/tests/test_postrelease_weather_pilot.py'
    run=subprocess.run([sys.executable,'-m','pytest','-q',tests],capture_output=True,text=True)
    (out/'tests.log').write_text(run.stdout+run.stderr,encoding='utf-8')
    assert run.returncode==0 and '5 passed' in run.stdout
    plants=[
        ('wrong_season','pilot','x[-24:][np.arange(32) % 24]','x[-23:][np.arange(32) % 23]','test_fixed_tone_and_seasonal'),
        ('wrong_depth','pilot','decompose(x, depth=1)','decompose(x, depth=0)','test_fixed_tone_and_seasonal'),
        ('no_reconstruction_gate','pilot','np.testing.assert_allclose(reconstructed, x, rtol=1e-12, atol=1e-12)','pass','test_reconstruction_gate'),
        ('no_repeat_gate','worker',"exact(repeated, predictions[i:i+1], 'batch1 reversed-order repeat')",'pass','test_known_forecasts_and_gates'),
        ('wrong_hybrid','worker',"finite(data['tones'])+residual_pred","finite(data['tones'])+native_pred",'test_known_forecasts_and_gates'),
        ('no_abstention_gate','worker',"exact(forecast['hybrid'][i], forecast['native'][i], 'abstention forecast')",'pass','test_known_forecasts_and_gates'),
        ('normalize_after_pooling','score','axis=1)/scales','axis=1)/np.mean(scales)','test_station_equal_weight_normalize_first'),
        ('window_weighted_stations','score',"per_window[[r['station']==s for r in rows]].mean()","per_window.mean()",'test_station_equal_weight_normalize_first'),
        ('wrong_bootstrap_tail','score','np.quantile(draws,[.025,.975]).tolist()','[0.,0.]','test_station_equal_weight_normalize_first'),
        ('stale_grant_accepted','worker',"raise PermissionError('Stale or invalid slot grant')",'return','test_hold_and_protocol_binding'),
        ('changed_input_accepted','pilot',"raise ValueError('Frozen input changed: '+name)",'pass','test_hold_and_protocol_binding'),
    ]
    results=[]
    for name,module,old,new,test in plants:
        path=Path(__file__).with_name('postrelease_weather_'+module+'.py')
        source=path.read_text(encoding='utf-8');assert source.count(old)==1
        mutant=source.replace(old,new);assert mutant!=source
        code='from experiments.feature_search import postrelease_weather_'+module+' as m\nimport pytest\nexec(compile('+repr(mutant)+',m.__file__,"exec"),m.__dict__)\nraise SystemExit(pytest.main(["-q",'+repr(tests+'::'+test)+']))'
        run=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        (out/f'{name}.log').write_text(run.stdout+run.stderr,encoding='utf-8')
        caught=run.returncode==1 and '1 failed' in run.stdout and '1 error' not in run.stdout
        results.append(dict(plant=name,test=test,caught=caught,returncode=run.returncode))
        write(out/'plants.json',results)
        knowledge=dict(question='Do weather-pilot tests reject deliberate recipe, grouping and native-gate failures?',
              design='Isolated in-memory source mutations; no checkpoint imports or native calls.',
              exact_findings=dict(tests_passed=5,plants_completed=len(results),plants_caught=sum(r['caught'] for r in results),native_launched=False),
              controls='Every mutation changes source and must yield a failed test, not collection errors.',
              limitations='Known-answer instrumentation audit; no forecasting evidence.',evidence_class='software verification',
              next_decision='Freeze pilot and independently replay analytic preparations; HOLD native until grant.')
        write(out/'knowledge.json',knowledge)
        assert caught,(name,run.stdout,run.stderr)
    write(out/'manifest.json',{f.name:sha(f) for f in out.iterdir() if f.is_file()})
    print(knowledge['exact_findings'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    audit(parser.parse_args().out)
