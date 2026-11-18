"""Known-answer checks for the fresh phase-crossed, actual-distance-matched design."""
import numpy as np
from experiments.feature_search.response_surface_crossed import local_triplet,background,make_design,PHASES,CENTERS,RADII


def test_actual_radius_is_matched_across_frequency_amplitude_and_phase():
    """A20-fold peak-amplitude change must not become a20-fold history-distance change."""
    _,x=background(4954000)
    for axis in ('frequency','amplitude'):
        for amplitude in (.1,2.):
            for phase in PHASES:
                for frequency in CENTERS:
                    for radius in RADII:
                        minus,zero,plus,midpoint,meta=local_triplet(x,frequency,phase,amplitude,radius,axis)
                        actual=np.sqrt(np.mean(((plus.astype(float)-minus)/2)**2))/x.astype(float).std()
                        assert abs(actual-radius)<1e-7
                        np.testing.assert_array_equal(midpoint,((plus.astype(float)+minus)/2).astype(np.float32))


def test_all_phases_are_crossed_with_each_source_and_interaction_pair():
    """Phases are conditions within sources, independent of their background phase draws."""
    inputs,nodes,paths,pairs,sources=make_design([4954000,4954001])
    assert len(paths)==2*4*2*3*2*3 and len(pairs)==2*4*4*3
    assert sources[0]['target_phase']!=sources[1]['target_phase']
    for source in sources:
        own=[r for r in paths if r['source']==source['source']]
        assert set(r['phase'] for r in own)==set(PHASES)
        assert all(source['target_phase']!=p for p in PHASES)
        ownpairs=[r for r in pairs if r['source']==source['source']]
        assert set((r['phase_a'],r['phase_b']) for r in ownpairs)=={(a,b) for a in range(4) for b in range(4)}
        for row in ownpairs:
            closure=inputs[row['ab']].astype(float)-inputs[row['a']]-inputs[row['b']]+inputs[row['base']]
            assert np.sqrt(np.mean(closure**2))/row['input_scale']<2e-7
            assert abs(row['input_a']-row['radius'])<1e-7 and abs(row['input_b']-row['radius'])<1e-7
