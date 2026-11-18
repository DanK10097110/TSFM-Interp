"""In-memory faults validating conditional phase-analysis tests without touching frozen code."""
import importlib.util
import json
from pathlib import Path


def run():
    """Each confusable algebra/grouping error must be rejected by a known-answer fixture."""
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('phase_tests',root/'tests/test_response_phase_study.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    path=root/'experiments/feature_search/response_phase_study.py';original=path.read_text(encoding='utf-8')
    cases=[
        ('rotation_sign','response_matrix',"r=float((h[1,0]-h[0,1])/2)","r=float((h[0,1]-h[1,0])/2)",'test_rotation_has_gain_phase_without_anisotropy_or_even_response'),
        ('rank_uses_smax_twice','split_matrix',"singular=np.linalg.svd(h,compute_uv=False)","singular=np.repeat(np.linalg.svd(h,compute_uv=False)[0],2)",'test_rank_one_and_reflection_do_not_conflate_ellipse_shape_and_phase_conjugation'),
        ('even_missing_base','response_matrix',"y[0]+y[2]-2*base,y[1]+y[3]-2*base","y[0]+y[2],y[1]+y[3]",'test_off_frequency_and_nonlinear_even_energy_close_exactly'),
        ('DC_control_ignored','response_matrix',"H_dc_adjusted=dc_fit[:2].tolist()","H_dc_adjusted=h.tolist()",'test_finite_window_dc_leakage_is_explicitly_detected'),
        ('receiver_in_LOSO','loso_mean',"matrices[np.arange(len(matrices))!=i].mean(0)","matrices.mean(0)",'test_loso_matrix_excludes_whole_receiver_source'),
        ('phases_count_as_source_bootstrap','bootstrap_indices',"size=(nboot,np.sum(source_types==kind))","size=(nboot,4*np.sum(source_types==kind))",'test_bootstrap_resamples_sources_with_type_balance_not_phases'),
        ('mean_phase_ratios','waveform_loss',"return float(np.sum((np.asarray(pred,float)-actual)**2)/np.sum(ideal**2))","return float(np.mean(np.sum((np.asarray(pred,float)-actual)**2,axis=1)/np.sum(ideal**2,axis=1)))",'test_phase_energy_pooled_score_is_not_mean_of_phase_ratios'),
    ]
    results=[]
    for name,symbol,old,new,test in cases:
        assert original.count(old)==1
        namespace=dict(__name__='phase_mutant',__file__=str(path));exec(compile(original.replace(old,new),str(path),'exec'),namespace)
        correct=getattr(tests,symbol);setattr(tests,symbol,namespace[symbol])
        try:
            try:getattr(tests,test)()
            except AssertionError:results.append(dict(plant=name,test=test,detected=True))
            else:raise AssertionError(f'Plant survived: {name}')
        finally:setattr(tests,symbol,correct)
    print(json.dumps(results,indent=2))


if __name__=='__main__':run()
