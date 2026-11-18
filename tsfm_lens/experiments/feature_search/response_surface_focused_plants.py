"""In-memory planted faults for focused-v2 preparation, grant, capture and analysis checks."""
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile


def run():
    """Require each known implementation fault to fail its corresponding focused test."""
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('focused_tests',root/'tests/test_response_surface_focused.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    cases=[
        ('off_diagonal_pairs','response_surface_focused','focused_subset',"row['source_id']<count and row['phase_a']==row['phase_b']","row['source_id']<count",'test_focused_membership_radius_and_analytic_midpoint_are_explicit'),
        ('missing_budget_margin','response_surface_focused','budget',"buffered_seconds=r['seconds']*nodes/pilot_nodes*1.1","buffered_seconds=r['seconds']*nodes/pilot_nodes",'test_budget_includes_margin_and_rejects_double_size'),
        ('stale_protocol_grant','response_surface_focused','require_grant',"record.get('protocol_sha256')!=sha(out/'protocol.json')","False",'test_grant_required_and_protocol_model_bound'),
        ('missing_interaction_base','response_surface_focused_analysis','score_arrays',"quartets.append(yab.astype(float)-ya.astype(float)-yb.astype(float)+y0.astype(float))","quartets.append(yab.astype(float)-ya.astype(float)-yb.astype(float))",'test_standin_capture_both_policies_and_saved_arithmetic'),
        ('phase_cells_counted_as_sources','response_surface_focused_analysis','convergence',"n_sources=len(own)","n_sources=sum(r['dependent_cells_per_radius'] for r in own)",'test_source_convergence_does_not_count_phase_cells_as_sources'),
        ('changed_input_accepted','response_surface_focused','validate',"if sha(out/file)!=digest:","if False:",'test_changed_input_hash_is_rejected_before_capture'),
        ('radius_metadata_trusted','response_surface_focused','focused_subset',"if abs(r['actual_radius']-r['radius'])>1e-7:","if False:",'test_subset_rejects_wrong_actual_radius_before_freezing'),
    ]
    results=[]
    for name,module,symbol,old,new,test in cases:
        path=root/'experiments/feature_search'/f'{module}.py';text=path.read_text(encoding='utf-8')
        assert text.count(old)==1
        namespace=dict(__name__='focused_mutant',__file__=str(path));exec(compile(text.replace(old,new),str(path),'exec'),namespace)
        correct=getattr(tests,symbol);setattr(tests,symbol,namespace[symbol])
        try:
            with tempfile.TemporaryDirectory(prefix='response-surface-focused-plant-') as scratch:
                function=getattr(tests,test)
                try:function(Path(scratch)) if inspect.signature(function).parameters else function()
                except BaseException as error:
                    if type(error).__name__ not in ('AssertionError','Failed','ValueError'):raise
                    results.append(dict(plant=name,test=test,detected=True,error_type=type(error).__name__))
                else:raise AssertionError(f'Plant survived: {name}')
        finally:setattr(tests,symbol,correct)
    print(json.dumps(results,indent=2))


if __name__=='__main__':run()
