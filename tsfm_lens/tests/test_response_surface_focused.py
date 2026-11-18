"""Focused-v2 guards and an end-to-end deterministic stand-in; no foundation weights loaded."""
import contextlib
import numpy as np
import pytest
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.response_surface_crossed import make_design
from experiments.feature_search.response_surface_focused import focused_subset,budget,require_grant,capture_bank,validate
from experiments.feature_search.response_surface_focused_analysis import score_arrays,convergence,independent_audit


def fixture_design():
    """Four independent source types supply every phase and radius without native inference."""
    inputs,nodes,paths,pairs,sources=make_design(range(4954000,4954004))
    return focused_subset(inputs,nodes,paths,pairs,sources,count=4)


def test_focused_membership_radius_and_analytic_midpoint_are_explicit():
    """The subset removes amplitudes and off-diagonal phases, not independent-source crossing."""
    bank,nodes,paths,pairs,signals,sources=fixture_design()
    assert len(bank)==660 and len(paths)==192 and len(pairs)==48 and len(signals)==32
    assert [s['background_type'] for s in sources]==list(range(4))
    for s in sources:
        own=[r for r in paths if r['source']==s['source']]
        assert len(own)==48 and set(r['phase_id'] for r in own)==set(range(4))
    for row in paths:
        actual=np.sqrt(np.mean(((bank[row['plus']].astype(float)-bank[row['minus']])/2)**2))/row['input_scale']
        assert row['amplitude']==.1 and abs(actual-row['radius'])<1e-7
        assert row['chord_input_rounding']<2e-7
        if row['axis']=='amplitude':assert row['midpoint']==row['zero']
        else:assert row['midpoint']!=row['zero']
    assert all(r['phase_a']==r['phase_b'] for r in pairs)
    assert len(set(r['original_index'] for r in nodes))==len(nodes)


def test_budget_includes_margin_and_rejects_double_size():
    """Use measured complete runtime rather than discounting removed hooks speculatively."""
    pilot=dict(bolt=dict(seconds=35.83567283593584),timesfm=dict(seconds=169.5613895469578))
    estimates=budget(2640,pilot)
    np.testing.assert_allclose(estimates['timesfm']['buffered_seconds'],580.6677774108084)
    assert all(r['fits600'] for r in estimates.values())
    assert budget(5280,pilot)['timesfm']['buffered_seconds']>900


def test_grant_required_and_protocol_model_bound(tmp_path):
    """A missing, stale or wrong-model release must never authorize native loading."""
    write(tmp_path/'protocol.json',dict(stage='held'))
    with pytest.raises(PermissionError,match='HELD'):require_grant(tmp_path,None,'bolt')
    grant=tmp_path/'grant.json';write(grant,dict(protocol_sha256='stale',explicit_slot_release=True,slot=2,models=['bolt']))
    with pytest.raises(PermissionError,match='HELD'):require_grant(tmp_path,grant,'bolt')
    write(grant,dict(protocol_sha256=sha(tmp_path/'protocol.json'),explicit_slot_release=True,slot=2,models=['bolt']))
    with pytest.raises(PermissionError,match='not granted'):require_grant(tmp_path,grant,'timesfm')
    assert require_grant(tmp_path,grant,'bolt')['slot']==2


def fake_native():
    """A six-identity-block receiver with a real reversible normalizer for instrument testing."""
    import torch
    class Norm(torch.nn.Module):
        """Pure FP32 centering/scaling with supplied-stat support."""
        def forward(self,x,loc_scale=None):
            """Match the native normalizer's accepted intervention interface."""
            if loc_scale is None:
                mu=x.mean(-1,keepdim=True);sd=((x-mu)**2).mean(-1,keepdim=True).sqrt();loc_scale=(mu,sd)
            mu,sd=loc_scale
            return (x-mu)/sd,loc_scale
    class Fake:
        """Deterministic physical last32 readout with observable intermediate states."""
        def __init__(self):
            """Only tiny Identity modules, never a pretrained constructor."""
            self.name='bolt';self.model=torch.nn.Module();self.model.instance_norm=Norm()
            self.model.encoder=torch.nn.Module();self.model.encoder.block=torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])
            self.layer=self.model.encoder.block[-1]
        def predict(self,x,replacement=None):
            """Run the miniature stack and honor the same selected-layer replacement contract."""
            with torch.inference_mode():
                z,(mu,sd)=self.model.instance_norm(torch.as_tensor(np.asarray(x,np.float32).copy()))
                h=z.reshape(-1,4,32)
                for layer in self.model.encoder.block:
                    h=layer(h)
                    if layer is self.layer and replacement is not None:h=torch.as_tensor(np.asarray(replacement,np.float32))
                y=h[:,-1]*sd+mu
            return y.numpy(),h.numpy()
    return Fake()


def test_standin_capture_both_policies_and_saved_arithmetic(tmp_path):
    """Exercise hooks, normalizer locking, self-patching, replay and analysis without TSFM weights."""
    inputs,nodes,paths,pairs,signals,sources=fixture_design();np.save(tmp_path/'inputs.npy',inputs)
    write(tmp_path/'paths.json',paths);write(tmp_path/'interactions.json',pairs)
    p=dict(cuts={'bolt':['0','3']},seconds_per_model=60,worker_cap=3*2**30)
    with contextlib.redirect_stdout(None):record=capture_bank(tmp_path,fake_native(),p,inputs,nodes)
    assert record['batch_replay_exact'] and not any(record['gates'].values())
    assert set(record['cuts'])=={'0','3'} and len(record['hashes'])==6
    local=[];mixed=[];responses=[]
    for policy in ('native','locked'):
        for cut in ['forecast','0','3']:
            suffix='forecast' if cut=='forecast' else 'hidden_'+cut
            values=np.load(tmp_path/f'bolt_{policy}_{suffix}.npy')
            a,b,c,saved=score_arrays(values,inputs,paths,pairs,signals,'bolt',policy,cut)
            local.extend(a);mixed.extend(b);responses.extend(c)
            if saved:np.savez_compressed(tmp_path/f'bolt_{policy}_arithmetic.npz',**saved)
    audit=independent_audit(tmp_path,'bolt',local,mixed,responses)
    assert audit['passed'] and audit['checks']==5728
    forecast=[r for r in mixed if r['cut']=='forecast']
    assert max(r['interaction_rms'] for r in forecast)<1e-6
    assert all(not r['above_roundoff_proxy'] for r in forecast)


def test_source_convergence_does_not_count_phase_cells_as_sources():
    """Different numbers of dependent cells cannot inflate source n or alter equal-source summaries."""
    records=[]
    for source,count in [('a',20),('b',2)]:
        for radius in [.002,.005,.01]:
            for _ in range(count):records.append(dict(model='m',policy='native',cut='0',axis='frequency',source=source,
                radius=radius,chord_relative=radius*(1 if source=='a' else 3),distance_sensitivity=2.,above_roundoff_proxy=True))
    rows,summary=convergence(records)
    assert len(rows)==2 and summary[0]['n_sources']==2 and summary[0]['shrink_source_count']==2
    np.testing.assert_allclose(summary[0]['small_curvature']['mean'],.004)


def test_changed_input_hash_is_rejected_before_capture(tmp_path):
    """Changing an input byte must invalidate a prepared run."""
    bank=tmp_path/'inputs.npy';np.save(bank,np.ones((2,128),np.float32))
    write(tmp_path/'protocol.json',dict(source_hashes={},hashes={'inputs.npy':sha(bank)}))
    validate(tmp_path)
    np.save(bank,np.zeros((2,128),np.float32))
    with pytest.raises(ValueError,match='Bound input changed'):validate(tmp_path)


def test_subset_rejects_wrong_actual_radius_before_freezing():
    """Correct metadata cannot hide a collapsed perturbation in the actual input bank."""
    inputs,nodes,paths,pairs,sources=make_design(range(4954000,4954004))
    row=next(r for r in paths if r['amplitude']==.1 and r['axis']=='amplitude')
    inputs[row['plus']]=inputs[row['zero']];inputs[row['minus']]=inputs[row['zero']]
    with pytest.raises(ValueError,match='Actual RMS radius'):
        focused_subset(inputs,nodes,paths,pairs,sources,count=4)
