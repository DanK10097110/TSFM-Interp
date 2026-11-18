"""In-memory oblique holdout faults; frozen capture source remains untouched."""
import importlib.util
import json
from pathlib import Path
import pytest


def run():
    """Verify angular alias, decomposition, input membership and source inference guards."""
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('oblique_tests',root/'tests/test_response_phase_oblique.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    path=root/'experiments/feature_search/response_phase_oblique.py';source=path.read_text(encoding='utf-8')
    cases=[
        ('unapproved_refit','prediction','return amplitude*c@np.asarray(h,float).T@b.T','return amplitude*c@np.linalg.lstsq(np.eye(2),np.asarray(h,float).T,rcond=None)[0]@b.T','test_linear_phase_law_has_equal_pooled_error_without_any_fit'),
        ('phase_coordinate_swapped','prediction','[np.cos(angles),np.sin(angles)]','[np.sin(angles),np.cos(angles)]','test_cardinal_third_harmonic_alias_breaks_on_oblique_holdout'),
        ('even_factor_missing','score','even_nmse=float(np.sum(even**2)/(2*denominator))','even_nmse=float(np.sum(even**2)/denominator)','test_even_response_and_phase_energy_close'),
        ('mean_phase_ratios','score','nmse=float(np.sum(error**2)/denominator)','nmse=float(np.mean(np.sum(error**2,axis=1)/np.sum(ideal**2,axis=1)))','test_even_response_and_phase_energy_close'),
        ('oblique_offset_removed','bank_from_prior','for angle in ANGLES])','for angle in ANGLES-np.pi/8])','test_oblique_bank_reuses_exact_backgrounds_and_unseen_angles'),
        ('phase_rows_inflate_n','source_draws','size=(16384,np.sum(types==k))','size=(16384,8*np.sum(types==k))','test_source_inference_and_family_four_do_not_count_angles_as_n'),
        ('family_ignored','interval','[.00625,.99375]','[.025,.975]','test_source_inference_and_family_four_do_not_count_angles_as_n'),
        ('background_parity_ignored','assert_background_parity','if not np.array_equal(new,old):','if False:','test_changed_background_and_absent_slot_fail_closed'),
    ]
    for test in sorted({case[-1] for case in cases}):getattr(tests,test)()
    results=[]
    for name,symbol,old,new,test in cases:
        assert source.count(old)==1,(name,source.count(old))
        space=dict(__name__='oblique_mutant',__file__=str(path));exec(compile(source.replace(old,new),str(path),'exec'),space)
        original=getattr(tests.study,symbol);setattr(tests.study,symbol,space[symbol])
        try:
            try:getattr(tests,test)()
            except (AssertionError,ValueError,pytest.fail.Exception):results.append(dict(plant=name,test=test,detected=True))
            else:raise AssertionError(f'Plant survived: {name}')
        finally:setattr(tests.study,symbol,original)
    print(json.dumps(results,indent=2))


if __name__=='__main__':run()
