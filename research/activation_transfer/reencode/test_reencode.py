import numpy as np
import torch

from research.activation_transfer.reencode.study import (
    bootstrap, compact_ids, counts, encode, event_masks, pairs_for, predicted_anchor, project_positive,
)


def test_target_anchor_changes_identity_at_identical_displacement():
    class Toy:
        k=2

        def pre_activations(self,x):
            return x

        def sparsify(self,x):
            val,idx=torch.topk(x,self.k,dim=-1)
            return torch.zeros_like(x).scatter_(-1,idx,val)
    toy=Toy()
    # The second history already has feature 2 active. Broadcasting the first
    # history's clean support onto it would invent an entry and exit.
    h=np.array([[3.,2.,1.99,0.],[3.,.1,2.,0.]],dtype="f4")
    v=np.array([[[0.,0.,.1,0.]]],dtype="f4")
    for g in ("hard","shoulder"):
        base,end=encode(toy,h)[g],encode(toy,h[:,None]+v)[g]
        entry,leave=event_masks(base,end)
        assert compact_ids(entry,2).tolist()==[[[2,-1]],[[-1,-1]]]
        assert compact_ids(leave,2).tolist()==[[[1,-1]],[[-1,-1]]]
        test=counts((entry,leave),(entry[::-1],leave[::-1]))
        assert test[...,0].sum()==1 and test[...,2].sum()==0 and test[...,8].sum()==0
        np.testing.assert_array_equal(project_positive(base,end-base[:,None],2)>0,end>0)


def test_clean_bridge_and_sham_pairs():
    rng=np.random.default_rng(75)
    h=rng.normal(size=(256,8))
    change=rng.normal(size=(8,13))
    y=h@change+1.25
    test=rng.normal(size=(12,8))
    pred,_=predicted_anchor(h,y,test)
    np.testing.assert_allclose(pred,test@change+1.25,atol=1e-5)
    donor,pairs=pairs_for(80)
    np.testing.assert_array_equal(donor[donor],np.arange(80))
    assert np.all(donor!=np.arange(80))
    w=bootstrap(80,pairs)
    np.testing.assert_equal(w.sum(axis=1),80)
    np.testing.assert_array_equal(w[:,pairs[:,0]],w[:,pairs[:,1]])


def test_published_events_replay_with_exact_source_item_and_target_feature_ids():
    import json
    from research.activation_transfer.reencode.study import DATASETS, GATES, METHODS, OUT

    def sets(arr):
        return [set(row[row>=0].tolist()) for row in arr.reshape(-1,arr.shape[-1])]

    for target, source in (("bolt","timesfm"),("timesfm","bolt")):
        data=json.loads((OUT/f"{source}_to_{target}.json").read_text())
        for domain in DATASETS:
            with np.load(OUT/f"{domain}_{target}_sufficient.npz") as z:
                assert len(set(z["ids"]))==80
                if domain=="m4":
                    assert not set(z["ids"]).intersection(data["training_ids"])
                    assert not set(z["ids"]).intersection(data["validation_ids"])
                for gate in GATES:
                    t_ent=sets(z[gate+"_true_enter"])
                    t_leave=sets(z[gate+"_true_leave"])
                    for method in METHODS:
                        p_ent=sets(z[f"{gate}_{method}_enter"])
                        p_leave=sets(z[f"{gate}_{method}_leave"])
                        report=data["datasets"][domain]["gates"][gate]["methods"][method]
                        for kind,test,pred in (("enter",t_ent,p_ent),("leave",t_leave,p_leave)):
                            assert sum(len(a & b) for a,b in zip(test,pred))==report["counts"]["correct_"+kind]
                            assert sum(map(len,test))==report["counts"]["true_"+kind]
                            assert sum(map(len,pred))==report["counts"]["pred_"+kind]
                        ds=z[f"{gate}_{method}_energy"]
                        dot,t2,p2,_=ds.sum(axis=(0,1))
                        np.testing.assert_allclose(1-(p2+t2-2*dot)/t2,report["recovery"]["value"],atol=1e-9)


def test_weekly_forecast_summaries_and_native_reference_match_independent_boundary_run():
    import json
    from pathlib import Path
    from research.activation_transfer.reencode.study import GATES, METHODS, OUT

    boundary = Path(__file__).resolve().parents[1] / "boundary"
    for target in ("bolt","timesfm"):
        with np.load(OUT/f"weekly_{target}_forecast.npz") as data:
            report=json.loads((OUT/f"weekly_{target}_forecast.json").read_text())
            previous=json.loads((boundary/f"weekly_smooth_{target}_forecasts.json").read_text())
            np.testing.assert_array_equal(data["ids"],previous["ids"])
            for gate in GATES:
                truth=data[gate+"_native_add"]
                reference_key="native_decoded" if gate=="hard" else "native_shoulder"
                np.testing.assert_allclose(truth,previous["forecast_changes"][reference_key],atol=2e-5)
                for operation,ref in (("add",truth),("ablate",-truth)):
                    for method in METHODS:
                        pred=data[f"{gate}_{operation}_{method}"].astype(float)
                        y=ref.astype(float)
                        recovery=1-np.sum((pred-y)**2)/np.sum(y*y)
                        np.testing.assert_allclose(recovery,report["gates"][gate][operation][method]["recovery"]["value"],atol=1e-8)
