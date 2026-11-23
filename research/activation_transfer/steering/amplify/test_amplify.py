import json

import numpy as np

from research.activation_transfer.steering.amplify.analyze import effects_recovery, score_metrics
from research.activation_transfer.steering.amplify.data import OUT, excluded_items
from research.activation_transfer.steering.amplify.experiment import read, same_axis_read, score


def test_read_decouples_from_write_projection():
    rng=np.random.default_rng(4)
    y=rng.normal(size=(300,6))
    test=rng.normal(size=(40,6))
    d=np.array([[1.,0,0,0,0,0],[0,0,1.,0,0,0]])
    # In a sheared coordinate system the write axis alone cannot recover the original read.
    score_directions=np.array([[1,2,0,0,0,0],[0,0,1,-3,0,0]])
    subject=y@score_directions.T+.6
    truth=test@score_directions.T+.6
    naive,scaled,_=same_axis_read(y,subject,test,d)
    recovered,_=read(y,subject,test)
    assert np.mean((recovered-truth)**2)<1e-7
    assert np.mean((scaled-truth)**2)>1 and np.mean((naive-truth)**2)>1


def test_public_projection_units_and_statistics():
    d=np.array([[2.,0.,1.],[.5,1.,0.]])
    mid=np.array([[2.,3.,4.],[1.,0.,1.]])
    plus=np.stack([mid[j]+d[j]/2 for j in range(2)])
    minus=np.stack([mid[j]-d[j]/2 for j in range(2)])
    np.testing.assert_allclose(np.diag(score(plus,d,mid)-score(minus,d,mid)),1)
    rng=np.random.default_rng(3)
    t=rng.normal(size=(20,2))
    w=np.ones((3,20))
    stats=score_metrics(t,t,np.zeros(2),w)
    for v in stats.values():
        np.testing.assert_allclose(v,1)
    effect=rng.normal(size=(20,2,8))
    np.testing.assert_allclose(effects_recovery(effect,effect,w),1)
    np.testing.assert_allclose(effects_recovery(np.zeros_like(effect),effect,w,True),0)


def test_frozen_ids_are_new_and_have_expected_split():
    items=json.loads((OUT/"manifest.json").read_text())["items"]
    assert not set(items).intersection(excluded_items())
    assert {k:list(items.values()).count(k) for k in set(items.values())}==dict(clean=320,cue=40,daily=80,weekly=32)


def test_published_effects_and_score_summaries_match_series_arrays():
    manifest=json.loads((OUT/"manifest.json").read_text())["items"]
    for stem in ("timesfm_to_bolt","bolt_to_timesfm"):
        result=json.loads((OUT/(stem+"_results.json")).read_text())
        with np.load(OUT/(stem+".npz")) as z:
            assert list(z["domain"])==[manifest[i] for i in z["ids"]]
            for group in ("daily","weekly"):
                mask=z["domain"]==group
                row=result[group]
                for key in ("naive","scalar","read","shuffle_read"):
                    t=z["true_score"][mask]
                    s=z["score_"+key][mask]
                    r2=1-np.sum((t-s)**2,axis=0)/np.sum((t-z["train_score_mean"])**2,axis=0)
                    np.testing.assert_allclose(r2.mean(),row["score"][key]["r2"]["macro"]["value"])
                for ref,methods in (("source_add",["fixed"]),("target_add",["fixed"]),
                                    ("source_amplify",["oracle","read","naive","scalar"]),
                                    ("target_amplify",["oracle","read","naive","scalar"]),
                                    ("oracle",["oracle","read","naive","scalar","shuffle_read","shuffle_both"])):
                    t=z[ref][mask]
                    for m in methods:
                        p=z[m][mask]
                        by_cue=1-np.sum((p-t)**2,axis=(0,2))/np.sum(t*t,axis=(0,2))
                        np.testing.assert_allclose(by_cue.mean(),row[ref]["full"][m]["macro"]["value"])
