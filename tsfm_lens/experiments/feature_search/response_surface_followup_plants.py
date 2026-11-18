"""In-memory mutation checks for the normalizer metadata followup; no frozen edits."""
import importlib.util
import json
from pathlib import Path


def run():
    """Each load-bearing check must reject a deliberately confusable implementation."""
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('normalizer_tests',root/'tests/test_response_surface_normalization.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    source=root/'experiments/feature_search/response_surface_normalization.py'
    original=source.read_text(encoding='utf-8')
    cases=[
        ('DC_leak','project','+[np.ones(len(t))]', '+[]','test_fourier_projection_separates_dc_and_both_offgrid_tones'),
        ('omit_background_inverse','restore','+(sigma_q/sigma_b-1)[..., None]*background_coeff','+0*background_coeff','test_normalized_difference_inverse_requires_observed_background_term'),
        ('global_instead_of_patch_inverse','restore','sigma_q[..., None]*normalized[:, 0]','sigma_q[:, -1:, None]*normalized[:, 0]','test_endpoint_inverse_uses_each_native_prefix_scale'),
        ('global_instead_of_prefix_stats','native_metadata','mus.append(mean); sigmas.append(scale)','mus.append(x.mean(1)); sigmas.append(x.std(1, correction=0))','test_timesfm_stats_are_cumulative_prefix_not_global_or_perpatch'),
        ('missing_scale_restore','restore',"if route == 'normalized_delta':", "if route == 'normalized_delta': return prediction\n    if route == 'normalized_delta':",'test_omitted_scale_plant_recovers_with_fair_inverse_not_direct_linear_fit'),
        ('test_source_scaler_leak','ridge_grouped','scale = float(np.sqrt(np.mean(x*x)))','scale = float(np.sqrt(np.mean(block*block)))','test_grouped_scaling_and_metadata_block_are_not_learned_from_test_source'),
    ]
    results=[]
    for name,symbol,old,new,test in cases:
        assert original.count(old)==1
        namespace=dict(__name__='normalizer_mutant',__file__=str(source))
        exec(compile(original.replace(old,new),str(source),'exec'),namespace)
        correct=getattr(tests,symbol);setattr(tests,symbol,namespace[symbol])
        try:
            try:getattr(tests,test)()
            except AssertionError:results.append(dict(plant=name,test=test,detected=True))
            else:raise AssertionError(f'Plant survived: {name}')
        finally:setattr(tests,symbol,correct)
    print(json.dumps(results,indent=2))


if __name__=='__main__':run()
