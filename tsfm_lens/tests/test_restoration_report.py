"""Report status guards never promote resource failures or partial replications."""
from experiments.feature_search.restoration_report import discovery_status,replication_status,render


def complete_replication(qualified=False):
    return dict(status='complete',family_size=2,qualified=qualified,per_seed=dict(seed_ids=list(range(3700000,3700064))),
        claims=dict(native_gain=dict(value=.6),random_advantage=dict(value=.6)),gates=dict(passed=qualified),
        mean_losses=dict(native=1.,anchored=.1,random_mean=.3,component_reference=.01),scope='Frozen full cohort')


def test_discovery_resource_or_numerical_failure_overrides_selection(tmp_path):
    result=dict(numerical_checks=dict(passed=True),effective_earliest_cut=15,peak_resident_bytes=1000)
    assert discovery_status(result)['effective_cut']==15
    result['resource_limit_exceeded']=True
    assert discovery_status(result)==dict(valid=False,effective_cut=None)
    data=dict(discovery=result,discovery_state=dict(valid=True,effective_cut=15),discovery_audit={},replication=replication_status())
    render(data,tmp_path)
    text=(tmp_path/'index.html').read_text()
    assert 'no suffix qualifies' in text
    assert 'Discovery selects the state' not in text
    result['resource_limit_exceeded']=False
    result['numerical_checks']['passed']=False
    assert discovery_status(result)['effective_cut'] is None


def test_pending_and_partial_replication_never_expose_confirmatory_claims():
    assert replication_status()['claims']=={}
    partial=complete_replication(True)
    partial['status']='failed_incomplete'
    partial['reason']='Primary geometry failed; no partial inference'
    state=replication_status(partial)
    assert not state['complete'] and state['claims']=={}
    assert state['reason']==partial['reason']
    partial['status']='complete'
    partial['per_seed']['seed_ids']=partial['per_seed']['seed_ids'][:16]
    assert not replication_status(partial)['complete']


def test_complete_negative_replication_is_displayed_without_rescue():
    record=complete_replication(False)
    state=replication_status(record)
    assert state['complete'] and not state['qualified']
    assert state['claims']==record['claims']
    record['resource_limit_exceeded']=True
    assert not replication_status(record)['complete']


def test_replication_identity_rejects_float_or_missing_claim_decoys():
    record=complete_replication(True)
    record['per_seed']['seed_ids']=[float(seed) for seed in record['per_seed']['seed_ids']]
    assert not replication_status(record)['complete']
    record=complete_replication(True)
    record['claims'].pop('native_gain')
    assert not replication_status(record)['complete']


def test_unscorable_cut10_random_curve_has_gap_but_anchor_remains():
    from experiments.feature_search.restoration_report import discovery_curve
    audit=dict(cuts={str(c):dict(anchor_loss_mean=.04,mean_random_loss=.3,control_failed_seed_indices=[]) for c in [0,5,10,15,20]})
    audit['cuts']['10']['control_failed_seed_indices']=[10,12,14]
    curve=discovery_curve(audit)
    assert curve['anchor'][2]==.04
    assert curve['random'][2] is None
    assert curve['random'][3]==.3
