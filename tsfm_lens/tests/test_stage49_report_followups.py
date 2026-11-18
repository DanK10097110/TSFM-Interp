"""Scientific status aggregation with partial-coverage and signed-share decoys."""
import base64
import hashlib
import pytest
from experiments.feature_search import stage49_report as report


def fixture():
    def response(stable,adequate):return dict(all_four_low_order_stable=stable,
        adequate_full_response_coverage=adequate,status='unresolved_or_aliased' if not stable else 'stable_and_adequate' if adequate else 'stable_low_order_partial_coverage')
    responses=dict(timesfm=dict(p32_16=response(False,False),p40_20=response(True,True)),
        bolt=dict(p32_16=response(False,False),p40_20=response(True,False)),
        chronos2=dict(p32_16=response(True,True),p40_20=response(True,True)))
    phase=dict(responses=responses,new_native_controls_pass=True,comparisons={})
    for first,second in [('timesfm','bolt'),('timesfm','chronos2'),('bolt','chronos2')]:
        for pair in ('p32_16','p40_20'):
            key=f'{first}-{second}/{pair}';eligible=key not in ('timesfm-bolt/p32_16','timesfm-chronos2/p40_20')
            sides=[responses[m][pair] for m in (first,second)]
            status='pilot_negative_closed' if not eligible else 'unresolved_or_aliased' if not all(r['all_four_low_order_stable'] for r in sides) else 'stable_low_order_partial_coverage'
            phase['comparisons'][key]=dict(pilot_numeric_gate=eligible,status=status)
    recovery=dict(models={m:dict(fresh_gate_pass=False,datasets=dict(fresh={k:dict(mean=v) for k,v in [('native',.3),('output_readout',.02),('frozen_head_readout',.001),('output_minus_head_excess',.019)]})) for m in ('bolt','timesfm')})
    homogeneous=dict(calibration=dict(passed=True,maximum_common_scale_waveform_RMS=5e-6),mean_normalized_energies={},
        attribution=dict(components={k:dict(aggregate_signed_projection_share=v) for k,v in [('fixed_normalizer',1.4),('radial',-.4),('replay',0.)]},
                         vector_closure_max=1e-15,energy_closure_max=1e-15,native_normalized_energy=[.5]*16))
    return phase,recovery,homogeneous


def test_completed_partial_and_pilot_negatives_cannot_be_promoted():
    phase,recovery,homogeneous=fixture();summary=report.summarize_followups(phase,recovery,homogeneous)
    assert summary['phase']['status']=='completed_partial_closed'
    assert len(summary['phase']['comparisons'])==6
    assert not summary['phase']['any_eligible_complete_response']
    assert summary['phase']['comparisons']['timesfm-chronos2/p40_20']=='pilot_negative_closed'
    assert summary['proposed_intervention']['status']=='not_run'
    phase['comparisons']['timesfm-bolt/p40_20']['status']='stable_and_adequate'
    with pytest.raises(ValueError,match='coverage'):report.summarize_followups(phase,recovery,homogeneous)
    del phase['comparisons']['timesfm-bolt/p40_20']
    with pytest.raises(ValueError,match='six'):report.summarize_followups(phase,recovery,homogeneous)


def test_recovery_excess_gate_and_signed_projections_keep_their_meaning():
    phase,recovery,homogeneous=fixture()
    recovery['models']['bolt']['datasets']['fresh']['output_readout']['mean']=.005
    summary=report.summarize_followups(phase,recovery,homogeneous)
    assert not summary['recoverability']['bolt']['passed']  # Excess still exceeds .01.
    assert summary['homogeneous']['signed_projection_shares']['fixed_normalizer']==1.4
    assert summary['homogeneous']['signed_projection_shares']['radial']==-.4
    text=report.followup_html(dict(followups=summary))
    assert 'not additive energy fractions' in text and 'off the native unit-RMS manifold' in text
    assert 'not yet run' in text and 'new held-out confirmation' in text
    recovery['models']['bolt']['fresh_gate_pass']=True
    with pytest.raises(ValueError,match='point gate'):report.summarize_followups(phase,recovery,homogeneous)
    recovery['models']['bolt']['fresh_gate_pass']=False
    homogeneous['calibration']['passed']=False
    with pytest.raises(ValueError,match='calibration'):report.summarize_followups(phase,recovery,homogeneous)


def test_portable_original_assets_require_exact_bytes_and_safe_paths(tmp_path):
    content=b'original figure bytes';record=dict(base64=base64.b64encode(content).decode(),sha256=hashlib.sha256(content).hexdigest())
    report.write_assets(dict(portable_assets={'figure.png':record}),tmp_path)
    assert (tmp_path/'figure.png').read_bytes()==content
    with pytest.raises(ValueError,match='Unsafe'):report.write_assets(dict(portable_assets={'../outside':record}),tmp_path)
    corrupt=dict(record,sha256='wrong')
    with pytest.raises(ValueError,match='checksum'):report.write_assets(dict(portable_assets={'figure.png':corrupt}),tmp_path)
