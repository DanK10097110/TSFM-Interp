"""In-memory direct-head and dynamic-horizon regressions; no native loading."""
import importlib.util
import json
from pathlib import Path
import pytest


def run():
    """Clean fixtures must pass; every planted capability/statistical fault must fail."""
    root=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('horizon_tests',root/'tests/test_response_phase_horizon.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    cases=[
        ('accept_autoregression','driver','check_pair','if head!=64 or calls!=[1,1]:','if head!=64:','test_direct_head_gate_rejects_autoregression_and_nonidentical_prefix'),
        ('tolerate_prefix_drift','driver','check_pair','if not np.array_equal(full[:,:32],prefix):','if not np.allclose(full[:,:32],prefix):','test_direct_head_gate_rejects_autoregression_and_nonidentical_prefix'),
        ('missing_slot_guard','driver','worker',"require_grant(out,grant,'bolt');p=validate(out,'bolt')","p=validate(out,'bolt')",'test_static_head_mismatch_and_absent_slot_fail_before_native'),
        ('force_input_frequency','analysis','curve','index=int(np.argmin(cvloss.mean(1)))','index=OUTPUT_PERIODS.index(input_period)','test_frequency_shift_and_phase_only_at_both_horizons'),
        ('truncate64_design','analysis','design','t=np.arange(horizon,dtype=float)','t=np.arange(min(horizon,32),dtype=float)','test_64_design_uses_all_steps_and_dc_trend_is_matched'),
        ('receiver_in_cv','analysis','curve','axes[np.arange(len(y))!=held].mean(0)','axes.mean(0)','test_source_cv_and_waveform_capacity_are_preserved'),
        ('family_unadjusted','analysis','summary','[.00625,.99375]','[.025,.975]','test_paired_source_intervals_keep_family_four'),
    ]
    for name in sorted({r[-1] for r in cases}):getattr(tests,name)()
    results=[]
    for name,module,symbol,old,new,test in cases:
        target=getattr(tests,module);path=Path(target.__file__);source=path.read_text(encoding='utf-8')
        assert old in source
        namespace=dict(__name__='horizon_mutant',__file__=str(path));exec(compile(source.replace(old,new),str(path),'exec'),namespace)
        if module=='driver':namespace['validate']=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Reached native preparation without slot guard'))
        original=getattr(target,symbol);setattr(target,symbol,namespace[symbol])
        try:
            try:getattr(tests,test)()
            except (AssertionError,ValueError,pytest.fail.Exception):results.append(dict(plant=name,test=test,detected=True))
            else:raise AssertionError(f'Plant survived {name}')
        finally:setattr(target,symbol,original)
    print(json.dumps(results,indent=2))


if __name__=='__main__':run()
