"""In-memory faults for geometry reconstruction and strict recovery gates."""
import importlib.util
import json
from pathlib import Path


def run():
    """Clean fixtures must pass before each fault is confirmed to fail an assertion."""
    root=Path(__file__).resolve().parents[2];path=root/'experiments/feature_search/response_phase_replay_preflight.py'
    spec=importlib.util.spec_from_file_location('replay_tests',root/'tests/test_response_phase_replay_preflight.py');tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    source=path.read_text(encoding='utf-8')
    cases=[
        ('sort_original_nodes','original_batch',"ids=[r['index'] for r in nodes if r['source_id']==source]","ids=sorted(r['index'] for r in nodes if r['source_id']==source)",'test_original_batch_preserves_source_list_order_and_row'),
        ('allclose_hides_diff','differences','bool(np.array_equal(a,b))','bool(np.allclose(a,b))','test_differences_keep_one_ulp_and_full_arrays'),
        ('ignore_original_failure','classify',"r['same_geometry_exact'] and r['current_prefix_exact']","r['current_prefix_exact']",'test_recovery_needs_four_exact_original_repeats_and_reproduced_geometry_effect'),
        ('unreproduced_mismatch_accepted','classify','eligible=bool(good and changed)','eligible=bool(good)','test_recovery_needs_four_exact_original_repeats_and_reproduced_geometry_effect'),
    ]
    for name in sorted({r[-1] for r in cases}):getattr(tests,name)()
    result=[]
    for name,symbol,old,new,test in cases:
        assert source.count(old)==1
        space=dict(__name__='replay_mutant',__file__=str(path));exec(compile(source.replace(old,new),str(path),'exec'),space)
        correct=getattr(tests,symbol);setattr(tests,symbol,space[symbol])
        try:
            try:getattr(tests,test)()
            except AssertionError:result.append(dict(plant=name,test=test,detected=True))
            else:raise AssertionError(f'Plant survived {name}')
        finally:setattr(tests,symbol,correct)
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
