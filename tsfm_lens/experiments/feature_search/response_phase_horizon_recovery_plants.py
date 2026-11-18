"""In-memory regression checks for the permitted recovery instrumentation changes."""
import importlib.util
import json
from pathlib import Path
import pytest


def run():
    """Every new bound/persistence/grant fixture must detect its specific failure."""
    root=Path(__file__).resolve().parents[2];path=root/'experiments/feature_search/response_phase_horizon_recovery.py'
    spec=importlib.util.spec_from_file_location('recovery_tests',root/'tests/test_response_phase_horizon_recovery.py');tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    source=path.read_text(encoding='utf-8')
    cases=[('omit_cross_term','mse_change_bound','2*np.linalg.norm(e)*np.linalg.norm(d)+np.sum(d*d)','np.sum(d*d)','test_mse_change_bound_is_tight_for_aligned_perturbation'),
           ('omit_background_error','response_bounds',"response_delta=delta[g['indices']]-delta[g['base']]","response_delta=delta[g['indices']]",'test_response_difference_subtracts_background_and_triangle_bounds_it'),
           ('reject_before_persist','Recorder','full,prefix,calls=self.native.pair(query)','full,prefix,calls=self.native.pair(query);check_pair(full,prefix,calls,self.native.native.pipeline.model_prediction_length)','test_failed_prefix_is_persisted_before_gate_rejection'),
           ('ignore_amendment_grant','require_recovery_grant',"if record.get('amendment_sha256')!=sha(out/'amendment.json'):","if False:",'test_grant_must_bind_amendment_not_only_old_protocol')]
    for test in sorted({c[-1] for c in cases}):getattr(tests,test)()
    results=[]
    for name,symbol,old,new,test in cases:
        assert source.count(old)==1
        space=dict(__name__='recovery_mutant',__file__=str(path));exec(compile(source.replace(old,new),str(path),'exec'),space)
        original=getattr(tests.recovery,symbol);setattr(tests.recovery,symbol,space[symbol])
        try:
            try:getattr(tests,test)()
            except (AssertionError,pytest.fail.Exception):results.append(dict(plant=name,test=test,detected=True))
            else:raise AssertionError(f'Plant survived {name}')
        finally:setattr(tests.recovery,symbol,original)
    print(json.dumps(results,indent=2))


if __name__=='__main__':run()
