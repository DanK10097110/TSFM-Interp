"""Known-answer alias validation, complete bundle scopes and closed negatives."""
import json
import numpy as np
import pytest
from experiments.feature_search import phase_orbit as orbit
from experiments.feature_search import phase_orbit_alias as study


def oracle_case(pair=(40,20)):
    _,coarse,old_rows=orbit.make_orbit(pair)
    _,fresh,new_rows=study.requests(pair)
    return coarse,fresh,new_rows,old_rows


def test_requests_use_all_eight_offgrid_and_only_fixed_four_fine_bundles():
    x,y,rows=study.requests((32,16))
    fine=[r for r in rows if r['request_kind']=='fine_new'];off=[r for r in rows if r['request_kind']=='offgrid']
    assert x.shape==(896,128) and y.shape==(896,64)
    assert len(fine)==768 and {r['bundle'] for r in fine}=={0,3,4,7}
    assert len(off)==128 and {r['bundle'] for r in off}==set(range(8))
    assert all(sum(r['bundle']==b for r in off)==16 for b in range(8))
    assert len(set(study.ids(rows,'p32_16')))==896
    assert all(not(r['cell']//16%2==0 and r['cell']%16%2==0) for r in fine)


def test_oracle_full_waveform_passes_separate_four_and_eight_scopes():
    coarse,new,rows,old=oracle_case()
    result=study.assess_case(coarse,new,rows,old,(40,20))
    assert result['all_four_low_order_stable']
    assert result['all_four_beyond_order3_pass']
    assert result['offgrid_pass_count_all_eight']==8
    assert result['status']=='stable_and_adequate'
    assert set(result['fine_bundles'])=={'0','3','4','7'}
    assert set(result['offgrid_bundles'])=={str(i) for i in range(8)}


def test_offgrid_failures_outside_fine_subset_cannot_be_hidden():
    coarse,new,rows,old=oracle_case()
    damaged=new.copy()
    for i,row in enumerate(rows):
        if row['request_kind']=='offgrid' and row['bundle'] in (1,2):damaged[i]+=row['scale']
    result=study.assess_case(coarse,damaged,rows,old,(40,20))
    assert result['all_four_low_order_stable'] and result['all_four_beyond_order3_pass']
    assert result['offgrid_pass_count_all_eight']==6
    assert result['status']=='stable_low_order_partial_coverage'


def test_alias_and_omitted_fourth_mode_are_different_negative_statuses():
    coarse,new,rows,old=oracle_case()
    t=np.arange(64)
    def shifted(values,metadata,harmonic):
        return values+np.asarray([.5*r['scale']*np.sin(2*np.pi*harmonic*t/40+harmonic*r['phi']) for r in metadata])
    alias=study.assess_case(shifted(coarse,old,9),shifted(new,rows,9),rows,old,(40,20))
    assert not alias['all_four_low_order_stable']
    assert alias['status']=='unresolved_or_aliased'
    partial=study.assess_case(shifted(coarse,old,4),shifted(new,rows,4),rows,old,(40,20))
    assert partial['all_four_low_order_stable']
    assert not partial['all_four_beyond_order3_pass']
    assert partial['status']=='stable_low_order_partial_coverage'


def test_one_bad_fine_bundle_fails_all_four_guards():
    coarse,new,rows,old=oracle_case();t=np.arange(64)
    low_bad=new.copy();high_bad=new.copy()
    for i,row in enumerate(rows):
        if row['request_kind']=='fine_new' and row['bundle']==3:
            low_bad[i]+=.8*row['scale']*np.sin(2*np.pi*t/40+row['phi'])
            high_bad[i]+=.8*row['scale']*np.sin(8*np.pi*t/40+4*row['phi'])
    low=study.assess_case(coarse,low_bad,rows,old,(40,20))
    assert not low['all_four_low_order_stable']
    assert sum(r['stable_low_order'] for r in low['fine_bundles'].values())==3
    high=study.assess_case(coarse,high_bad,rows,old,(40,20))
    assert high['all_four_low_order_stable']
    assert not high['all_four_beyond_order3_pass']
    assert sum(r['beyond_order3_energy']<=.05 for r in high['fine_bundles'].values())==3


def test_both_models_must_pass_and_negative_pilot_stays_closed():
    good=dict(all_four_low_order_stable=True,adequate_full_response_coverage=True)
    bad=dict(all_four_low_order_stable=False,adequate_full_response_coverage=False)
    responses={m:{s:dict(good) for s in ('p32_16','p40_20')} for m in ('timesfm','bolt','chronos2')}
    candidates={f'{a}-{b}/{s}':dict(promotes_alias_validation=True) for a,b in [('timesfm','bolt'),('timesfm','chronos2'),('bolt','chronos2')] for s in ('p32_16','p40_20')}
    candidates['timesfm-bolt/p32_16']['promotes_alias_validation']=False
    responses['chronos2']['p40_20']=bad
    result=study.comparison_statuses(responses,candidates,True)
    assert len(result)==6
    assert result['timesfm-bolt/p32_16']['status']=='pilot_negative_closed'
    assert result['timesfm-chronos2/p40_20']['status']=='unresolved_or_aliased'
    assert result['bolt-chronos2/p40_20']['status']=='unresolved_or_aliased'
    assert result['timesfm-bolt/p40_20']['status']=='stable_and_adequate'
    assert all(not r['hidden_capture_authorized'] for r in result.values())
    failed=study.comparison_statuses(responses,candidates,False)
    assert failed['timesfm-bolt/p40_20']['status']=='new_native_control_failed'


def test_new_forecast_binding_refuses_offgrid_row_swaps(tmp_path):
    _,y,rows=study.requests((40,20));stem='p40_20'
    np.savez_compressed(tmp_path/f'{stem}_requests.npz',clean_future=y)
    (tmp_path/f'{stem}_requests.json').write_text(json.dumps(rows),encoding='utf-8')
    (tmp_path/'protocol.json').write_text('{}',encoding='utf-8')
    payload=dict(prediction=y,model=np.asarray('bolt'),case=np.asarray(stem),phase_row_ids=study.ids(rows,stem),
        request_corpus_sha256=np.asarray(study.native.digest(tmp_path/f'{stem}_requests.npz')),
        request_metadata_sha256=np.asarray(study.native.digest(tmp_path/f'{stem}_requests.json')),
        protocol_sha256=np.asarray(study.native.digest(tmp_path/'protocol.json')))
    path=tmp_path/'forecast.npz';np.savez_compressed(path,**payload)
    np.testing.assert_array_equal(study.load_new_prediction(path,'bolt',stem,tmp_path),y)
    payload['phase_row_ids']=payload['phase_row_ids'][::-1];np.savez_compressed(path,**payload)
    with pytest.raises(ValueError,match='orderedphase'):study.load_new_prediction(path,'bolt',stem,tmp_path)
