"""Physically execute fixed decoder 2622 and calibration-matched control directions."""
import json
import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import change_forecasts,last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.feature_groups.data import dictionaries
from research.activation_transfer.shape_transfer.rolling.fixed_2622.data import CACHE,N,OUT,PHASES
from research.activation_transfer.shape_transfer.rolling.intervene import optimal
from research.activation_transfer.shape_transfer.sparse_stability.run import active_codes
from research.activation_transfer.shape_transfer.sparse_writes.run import weight_leads

ARMS=("cal_old","cal_now","traffic_old","traffic_now","anchor","gated","rescaled","current_offsupport",
      "fresh","old_winner","sham1","sham2","sham3")


def project(profile, goal, bound):
    return float(np.clip((profile@goal)/max(profile@profile,1e-24),-bound,bound))


def run():
    dic=dictionaries("timesfm")
    spec=json.loads((OUT/"frozen.json").read_text())
    native=dict(np.load(CACHE/"timesfm_native.npz"))
    raw=dict(np.load(CACHE/"data.npz"))
    s=dict(np.load(CACHE/"timesfm_geometry.npz"));t=dict(np.load(CACHE/"bolt_geometry.npz"))
    tokens,pos,ids=read_materials(CACHE,"bolt")
    for z in (native,raw,s,t):np.testing.assert_array_equal(ids,z["ids"])
    unique=[r["series"] for r in json.loads((OUT/"manifest.json").read_text())["series"]]
    np.testing.assert_array_equal(ids,[f"{p}/{l}/{q}" for p in PHASES for l in (0,32) for q in unique])
    assert set(unique).isdisjoint(spec["calibration_ids"]) and not set("0/"+u for u in unique).intersection(spec["traffic_calibration_ids"])
    support, strengths=active_codes(dic.sae,tokens)
    donor_codes=native["codes"]
    candidates=np.asarray(spec["candidates"])
    slopes=np.asarray(spec["slopes"])
    dom_slopes=np.asarray(spec["traffic_slopes"])
    norm=np.linalg.norm(dic.decoder,axis=1)
    source_profile=dic.donor @ s["jac"].astype(float).transpose(0,2,1)
    old=np.r_[0:N,2*N:3*N]; now=old+N
    rec=dict(ids=ids[now],split=raw["split"][now],atoms=dic.atoms,candidates=candidates,
             old_codes=donor_codes[old],now_codes=donor_codes[now],native=native["native"][now],
             original_active=strengths[old][:,candidates]>0,current_active=strengths[now][:,candidates]>0,
             original_target_code=strengths[old][:,candidates],current_target_code=strengths[now][:,candidates],
             anchor_budget=np.zeros((len(old),len(candidates)),dtype=float))
    refshape=donor_codes[old].shape
    for arm in ARMS:
        rec[arm+"_coef"]=np.zeros(refshape,dtype="f4")
        rec[arm+"_id"]=np.full(refshape,-1,dtype=int)
        rec[arm+"_effect"]=np.zeros((*refshape,8),dtype="f4")
    for r,(a,b) in enumerate(zip(old,now)):
        fixed_x=weight_leads(dic.decoder[candidates] @ t["jac"][a].astype(float).T,dic.weight)
        fixed_y=weight_leads(dic.decoder[candidates] @ t["jac"][b].astype(float).T,dic.weight)
        # The anchor budget allows an edit of a dormant coordinate without pretending that it is active.
        base=strengths[a,candidates]
        bound=np.minimum(np.where(base>0,base,spec["fallback_activation"]),.2*dic.radius/norm[candidates])
        rec["anchor_budget"][r]=bound
        sa,sb=support[a],support[b]
        rx=weight_leads(dic.decoder[sa]@t["jac"][a].astype(float).T,dic.weight)
        ry=weight_leads(dic.decoder[sb]@t["jac"][b].astype(float).T,dic.weight)
        for j in np.flatnonzero((donor_codes[a]>0)|(donor_codes[b]>0)):
            c0,c1=donor_codes[a,j],donor_codes[b,j]
            w0=weight_leads(-c0*source_profile[a,j],dic.weight)
            w1=weight_leads(-c1*source_profile[b,j],dic.weight)
            vals=[project(v,w0,limit) for v,limit in zip(fixed_x,bound)]
            anchor=vals[0];raw_bound=bound[0];fixed=candidates[0]
            proposals={
                "cal_old":(fixed,np.clip(c0*slopes[j,0],-raw_bound,raw_bound)),
                "cal_now":(fixed,np.clip(c1*slopes[j,0],-raw_bound,raw_bound)),
                "traffic_old":(fixed,np.clip(c0*dom_slopes[j],-raw_bound,raw_bound)),
                "traffic_now":(fixed,np.clip(c1*dom_slopes[j],-raw_bound,raw_bound)),
                "anchor":(fixed,anchor),
                "gated":(fixed,anchor if strengths[b,fixed]>0 else 0.),
                "rescaled":(fixed,np.clip(anchor*(c1/c0),-raw_bound,raw_bound) if c0>0 else 0.),
                "current_offsupport":(fixed,project(fixed_y[0],w1,raw_bound)),
                "fresh":optimal(sb,ry,strengths[b,sb],w1,None,norm,.2*dic.radius),
                "old_winner":optimal(sa,rx,strengths[a,sa],w0,None,norm,.2*dic.radius)}
            proposals.update({f"sham{k}":(candidates[k],vals[k]) for k in (1,2,3)})
            for key,(atom,coef) in proposals.items():
                if coef!=0.:
                    rec[key+"_id"][r,j]=atom;rec[key+"_coef"][r,j]=coef
    original_fixed=rec["old_winner_id"]==candidates[0]
    np.testing.assert_allclose(rec["anchor_coef"][original_fixed],rec["old_winner_coef"][original_fixed],atol=5e-6,rtol=5e-5)
    np.testing.assert_array_equal(rec["gated_coef"],np.where(rec["current_active"][:,0:1],rec["anchor_coef"],0.))
    moves={}
    for arm in ARMS:
        for r,j in np.argwhere(rec[arm+"_id"]>=0):
            key=(int(r),int(rec[arm+"_id"][r,j]),float(rec[arm+"_coef"][r,j]))
            moves.setdefault(key,[]).append((arm,r,j))
    keys=list(moves)
    model=adapter_for("bolt")
    selector=last_window_slice(model,pos)
    for k in range(0,len(keys),32):
        batch=keys[k:k+32]
        q=now[[v[0] for v in batch]]
        edits=(np.array([v[2] for v in batch])[:,None]*dic.decoder[[v[1] for v in batch]]).astype("f4")
        effect=change_forecasts(model,MODELS["bolt"][-1],raw["x"][q],tokens[q],edits,selector,32)-t["forecast"][q]
        for action,wave in zip(batch,effect):
            for arm,r,j in moves[action]:rec[arm+"_effect"][r,j]=wave
    rec["unique_writes"]=len(keys)
    np.savez_compressed(OUT/"effects.npz",**rec)
    print(len(keys),"actual receiver forecasts",flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    run()
