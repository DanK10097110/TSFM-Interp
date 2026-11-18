"""In-memory fault injection; never mutate the frozen transfer implementation."""
import importlib.util
import json
import pytest
from pathlib import Path


def run():
    """Require each load-bearing fit/design/capture test to reject its confusable fault."""
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('transfer_tests',root/'tests/test_response_phase_transfer.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    path=root/'experiments/feature_search/response_phase_transfer.py';source=path.read_text(encoding='utf-8')
    cases=[
        ('scalar_ignores_finite_window','fit_training','scalar = float(np.sum(ideal * target) / np.sum(ideal**2))','scalar = float(np.trace(np.linalg.lstsq(basis(frequency), np.column_stack([(target[0]-target[2])/2,(target[1]-target[3])/2]),rcond=None)[0])/2)','test_scalar_uses_waveform_geometry_and_bounds'),
        ('scalar_not_bounded','fit_training','scalar_bounded=float(np.clip(scalar, 0., 1.))','scalar_bounded=scalar','test_scalar_uses_waveform_geometry_and_bounds'),
        ('complex_r_sign','fit_training','qr[1] * J','-qr[1] * J','test_complex_full_and_zero_with_even_decoy'),
        ('zero_is_identity','fit_training','zero=np.zeros((2, 2))','zero=np.eye(2)','test_complex_full_and_zero_with_even_decoy'),
        ('delay_wrong_sign','delayed_map','angle = -(frequency - center)','angle = (frequency - center)','test_delay_transport_sign_and_center'),
        ('late_window_resets_clock','window_decomposition','b = basis(frequency)[start:stop]','b = basis(frequency)[:stop-start]','test_window_absolute_origin_and_even_decoy'),
        ('family_not_adjusted','paired_interval','[.025/family, 1-.025/family]','[.025, .975]','test_source_bootstrap_and_family_four'),
        ('phases_as_sources','source_draws','(nboot, np.sum(types == k))','(nboot, 4*np.sum(types == k))','test_source_bootstrap_and_family_four'),
        ('phase_input_gate_disabled','input_gates','if worst > 1e-5 or closure > 2e-5:','if False:','test_fresh_design_and_phase_swap_gate'),
        ('grant_bypassed','worker','require_grant(out, grant, model); p = validate(out, model)','p = validate(out, model)','test_worker_requires_grant_before_model_load'),
        ('locked_policy_not_saved','capture_bank','arrays[policy][batch] = pred','arrays[policy][batch] = inputs[batch,:32]','test_capture_saved_arithmetic_and_identity_gate'),
    ]
    results=[]
    for test in sorted({case[-1] for case in cases}): getattr(tests,test)()
    for name,symbol,old,new,test in cases:
        assert source.count(old)==1,(name,source.count(old))
        namespace=dict(__name__='transfer_mutant',__file__=str(path));exec(compile(source.replace(old,new),str(path),'exec'),namespace)
        namespace['validate']=lambda *args,**kwargs: tests.transfer.validate(*args,**kwargs)
        namespace['normalization']=lambda *args,**kwargs: tests.transfer.normalization(*args,**kwargs)
        original=getattr(tests.transfer,symbol)
        setattr(tests.transfer,symbol,namespace[symbol])
        try:
            try: getattr(tests,test)()
            except (AssertionError,ValueError,pytest.fail.Exception): results.append(dict(plant=name,test=test,detected=True))
            else: raise AssertionError(f'Plant survived {name}')
        finally: setattr(tests.transfer,symbol,original)
    print(json.dumps(results,indent=2))


if __name__=='__main__': run()
