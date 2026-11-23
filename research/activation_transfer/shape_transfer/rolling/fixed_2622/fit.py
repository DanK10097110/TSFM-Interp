"""Freeze slopes and matched receiver directions using only the old 96 calibration histories."""
import json
import numpy as np
import torch

from research.activation_transfer.shape_transfer.feature_groups.data import dictionaries,load
from research.activation_transfer.shape_transfer.rolling.fixed_2622.data import OUT
from research.activation_transfer.shape_transfer.rolling.data import CACHE as DISCOVERY
from research.activation_transfer.shape_transfer.sparse_writes.run import weight_leads

PRIMARY=2622
DONOR=2362


def fit():
    dic=dictionaries("timesfm")
    cal=load("timesfm",dic,"cal")
    assert len(cal["ids"])==96 and cal["ref"] is None
    n,leads,width=cal["tj"].shape
    responses=(dic.decoder @ cal["tj"].reshape(n*leads,width).T).reshape(len(dic.decoder),n,leads)
    full=np.sqrt(np.square(responses).mean((1,2)))
    shape=np.sqrt(np.square(responses-responses.mean(2,keepdims=True)).mean((1,2)))
    occupancy=(cal["strengths"]>0).mean(0)
    descriptors=np.stack([occupancy,full,shape],axis=1)
    distance=np.square((descriptors-descriptors[PRIMARY])/descriptors.std(0)).sum(1)
    distance[PRIMARY]=np.inf
    candidates=[PRIMARY,*np.argsort(distance,kind="stable")[:3].tolist()]
    median=np.median(cal["strengths"][cal["strengths"]>0])
    fallback=[]
    for a in candidates:
        v=cal["strengths"][:,a]
        fallback.append(float(np.median(v[v>0])) if np.any(v>0) else float(median))
    slopes=np.empty((len(dic.atoms),len(candidates)))
    for j in range(len(dic.atoms)):
        c=cal["codes"][:,j,None]
        wants=weight_leads(-c*(cal["sj"]@dic.donor[j]),dic.weight)
        for h,atom in enumerate(candidates):
            predicts=weight_leads(c*responses[atom],dic.weight)
            slopes[j,h]=np.sum(wants*predicts)/max(np.sum(predicts**2),1e-20)
    with np.load(DISCOVERY/"timesfm_native.npz") as old, np.load(DISCOVERY/"timesfm_geometry.npz") as sg, np.load(DISCOVERY/"bolt_geometry.npz") as tg:
        rows=np.flatnonzero(np.char.startswith(old["ids"],"0/monash_traffic/"))
        assert len(rows)==96
        np.testing.assert_array_equal(old["ids"],sg["ids"])
        np.testing.assert_array_equal(old["ids"],tg["ids"])
        target_signature=tg["jac"][rows].astype(float)@dic.decoder[PRIMARY]
        source_jac=sg["jac"][rows].astype(float)
        traffic_codes=old["codes"][rows]
        traffic_slopes=[]
        for j in range(len(dic.atoms)):
            c=traffic_codes[:,j,None]
            w=weight_leads(-c*(source_jac@dic.donor[j]),dic.weight)
            p=weight_leads(c*target_signature,dic.weight)
            traffic_slopes.append(float(np.sum(w*p)/max(np.sum(p**2),1e-20)))
        traffic_ids=old["ids"][rows].tolist()
    record=dict(traffic_slopes=traffic_slopes,traffic_calibration_ids=traffic_ids,candidates=candidates,descriptors=descriptors[candidates].tolist(),distance2=distance[candidates[1:]].tolist(),
                slopes=slopes.tolist(),fallback_activation=fallback,donors=dic.atoms.tolist(),calibration_ids=cal["ids"].tolist(),
                descriptor_fields=["prevalence","full_response_rms","centered_response_rms"])
    (OUT/"frozen.json").write_text(json.dumps(record,indent=2)+"\n")
    print(candidates,"old calibration positive",occupancy[candidates]*96,"slopes for 2362",slopes[list(dic.atoms).index(DONOR)],flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    fit()
