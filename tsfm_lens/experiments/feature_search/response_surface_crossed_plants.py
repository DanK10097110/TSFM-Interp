"""Known-fault checks for actual-distance matching and crossed-phase membership."""
import importlib.util
import json
from pathlib import Path


def run():
    """Execute source mutants in memory; preserve all original and frozen files."""
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('crossed_tests',root/'tests/test_response_surface_crossed.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    source=root/'experiments/feature_search/response_surface_crossed.py';original=source.read_text(encoding='utf-8')
    cases=[('frequency_distance_ignores_amplitude','local_triplet',
            'return amplitude*rms((wave(frequency+step,phase)-wave(frequency-step,phase))/2)',
            'return rms((wave(frequency+step,phase)-wave(frequency-step,phase))/2)',
            'test_actual_radius_is_matched_across_frequency_amplitude_and_phase'),
           ('only_one_phase_per_source','make_design','for phase_id,phase in enumerate(PHASES):',
            'for phase_id,phase in enumerate(PHASES[:1]):',
            'test_all_phases_are_crossed_with_each_source_and_interaction_pair')]
    results=[]
    for name,symbol,old,new,test in cases:
        assert original.count(old)==1
        namespace=dict(__name__='crossed_mutant',__file__=str(source));exec(compile(original.replace(old,new),str(source),'exec'),namespace)
        correct=getattr(tests,symbol);setattr(tests,symbol,namespace[symbol])
        try:
            try:getattr(tests,test)()
            except (AssertionError,ValueError) as error:results.append(dict(plant=name,test=test,detected=True,error_type=type(error).__name__))
            else:raise AssertionError(f'Plant survived: {name}')
        finally:setattr(tests,symbol,correct)
    print(json.dumps(results,indent=2))


if __name__=='__main__':run()
