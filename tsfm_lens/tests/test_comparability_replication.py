"""Native-free replication plants: frozen probes, source dependence and launch gates."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.feature_search import comparability_replication as r
from experiments.feature_search import comparability_core as c


def test_fresh_generator_matches_old_five_conditions():
    """Identical old-seed replay catches changes to phase/noise/amplitude construction."""
    rows,x,y=r.generate([4942000])
    old=r.read(r.ROOT/'runs/selective_replication4942000/rows.json')
    ids=[next(i for i,a in enumerate(old) if a.get('seed')==4942000 and a['case']==row['case']) for row in rows]
    with np.load(r.ROOT/'runs/selective_replication4942000/inputs.npz') as f:
        np.testing.assert_array_equal(x.astype(np.float32),f['history'][ids])
        with pytest.raises(AssertionError):np.testing.assert_array_equal((x+0.01).astype(np.float32),f['history'][ids])
    with np.load(r.ROOT/'runs/selective_replication4942000/targets.npz') as f:np.testing.assert_allclose(y,f['future'][ids],atol=0,rtol=0)
    fresh,_,_=r.generate(range(4952000,4952064))
    assert len(fresh)==320 and len({a['source'] for a in fresh})==64
    assert all(a['split']=='test' for a in fresh)


def test_source_matched_permutation_and_row_shuffle_plant():
    """Condition labels and dependent bundles must survive every null permutation."""
    rows,_,_=r.generate(range(4952000,4952064));perm=r.source_permutation(rows,4952100)
    def check(p):
        for start in range(0,len(rows),5):
            assert len({rows[p[i]]['source'] for i in range(start,start+5)})==1
            assert all(rows[p[i]]['case']==rows[i]['case'] for i in range(start,start+5))
    check(perm)
    assert not np.array_equal(perm,np.arange(len(rows)))
    with pytest.raises(AssertionError):check(np.random.default_rng(2).permutation(len(rows)))


def test_slot_grant_required_and_bound(tmp_path):
    """Absent, wrong-slot and wrong-protocol grants cannot launch a worker."""
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(RuntimeError):r.require_grant(tmp_path)
    path=tmp_path/'slot_grant.json'
    for slot,digest in [(1,r.sha(tmp_path/'protocol.json')),(2,'wrong')]:
        path.write_text(json.dumps(dict(granted=True,slot=slot,protocol_sha256=digest)))
        with pytest.raises(RuntimeError):r.require_grant(tmp_path)
    path.write_text(json.dumps(dict(granted=True,slot=2,protocol_sha256=r.sha(tmp_path/'protocol.json'))))
    r.require_grant(tmp_path)


def test_exact_native_gates_reject_tiny_broken_replay():
    """A deliberately tiny nonzero error fails each exact gate independently."""
    clean=dict(repeat=0.,normalizer_identity=0.,first_self=0.,half_self=0.,input_closure=1e-7)
    r.check_gates(clean)
    for name in ('repeat','normalizer_identity','first_self','half_self'):
        with pytest.raises(ValueError):r.check_gates(dict(clean,**{name:1e-12}))
    with pytest.raises(ValueError):r.check_gates(dict(clean,input_closure=3e-6))


def test_frozen_kernel_train_block_ignores_fresh_features():
    """Fresh test features cannot alter preprocessing or a frozen coefficient fit."""
    rng=np.random.default_rng(77);x=rng.normal(size=(60,12));h=rng.normal(size=(60,8))
    rows=[dict(source=f's{i}',domain='synthetic') for i in range(60)];tr=np.arange(40)
    bank=c.kernel_bank(x,{'A_hidden':h@h.T},rows,tr)
    changed=x.copy();changed[40:]*=1000
    hc=h.copy();hc[40:]+=1e6
    changed_bank=c.kernel_bank(changed,{'A_hidden':hc@hc.T},rows,tr)
    for key in bank['input+A_hidden']:
        np.testing.assert_allclose(bank['input+A_hidden'][key][:40,:40],changed_bank['input+A_hidden'][key][:40,:40],atol=1e-10)
    leaked=c.kernel_bank(changed,{'A_hidden':hc@hc.T},rows,np.arange(60))
    with pytest.raises(AssertionError):np.testing.assert_allclose(bank['input+A_hidden']['pca8+pca8'][:40,:40],leaked['input+A_hidden']['pca8+pca8'][:40,:40],atol=1e-5)


def test_exploratory_addition_recovers_missing_signal():
    """The added hidden block must beat an observable-only decoy on held-out sources."""
    rng=np.random.default_rng(17);x=rng.normal(size=(100,4));h=rng.normal(size=(100,3));f=rng.normal(size=(100,2));y=(h[:,:1]+x[:,:1])
    rows=[dict(source=f's{i}',domain='synthetic') for i in range(100)];tr=np.arange(72);te=np.arange(72,100)
    grams={'A_hidden':h@h.T,'A_forecast':f@f.T}
    banks=[(t,v,c.kernel_bank(x,grams,rows,t)) for t,v in c.folds(rows,tr)];final=c.kernel_bank(x,grams,rows,tr)
    spec=r.choose_extension(banks,y,rows,'history_ridge+pca8')
    good=c.ridge(r.extension_kernel(final,spec),y,tr,te,c.weights(rows,tr),spec['alpha'])
    bad=c.ridge(final['input+A_forecast']['history_ridge+pca8'],y,tr,te,c.weights(rows,tr),spec['alpha'])
    assert np.mean((good-y[te])**2)<.01
    with pytest.raises(AssertionError):assert np.mean((bad-y[te])**2)<.01
    changed=y.copy();changed[te]=1e9
    assert r.choose_extension(banks,changed,rows,'history_ridge+pca8')==spec


def test_module_import_has_no_native_side_effect():
    """Preparation/import cannot silently acquire the native inference slot."""
    code='import sys; import experiments.feature_search.comparability_replication; assert "torch" not in sys.modules; assert "chronos" not in sys.modules'
    run=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert run.returncode==0,run.stderr


def test_knowledge_preserves_failed_instrumentation(tmp_path):
    """A later successful step must not erase an earlier instrumentation failure."""
    r.knowledge(tmp_path,dict(experiment='gate',status='failed',error='planted'))
    r.knowledge(tmp_path,dict(experiment='gate_v2',status='passed'))
    events=r.read(tmp_path/'knowledge.json')['events']
    assert [e['status'] for e in events]==['failed','passed']
    with pytest.raises(AssertionError):assert len(events)==1
