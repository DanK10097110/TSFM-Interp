"""Paired per-sensor estimates plus evidence on which donor identities generalize."""
import json
import numpy as np

from research.activation_transfer.shape_transfer.feature_groups.score import recover,transform,weights
from research.activation_transfer.shape_transfer.rolling.fixed_2622.data import OUT,PHASES
from research.activation_transfer.shape_transfer.rolling.fixed_2622.fit import DONOR,PRIMARY
from research.activation_transfer.shape_transfer.rolling.fixed_2622.intervene import ARMS
from research.activation_transfer.shape_transfer.sparse_stability.score import interval


def score_row(f,mask,draw,idx):
    n=int(mask.sum())
    values={"cases":n,"sensors":int(mask.any(1).sum())}
    for part in ("full","shape"):
        y=transform(f["native"][:,idx],part)
        scores={key:recover(mask,y,transform(f[key+"_effect"][:,idx],part),draw,True) for key in (*ARMS,"norm1","norm2","norm3")}
        scores["matched_mean"]=np.mean([scores[f"sham{k}"] for k in (1,2,3)],axis=0)
        scores["norm_matched_mean"]=np.mean([scores[f"norm{k}"] for k in (1,2,3)],axis=0)
        values[part]={k:interval(v) for k,v in scores.items()}
        for a,b in (("anchor","gated"),("anchor","fresh"),("anchor","old_winner"),("anchor","matched_mean"),("anchor","norm_matched_mean"),("norm_matched_mean","fresh"),
                    ("anchor","cal_old"),("anchor","traffic_old"),("current_offsupport","anchor"),
                    ("rescaled","anchor"),("traffic_now","traffic_old"),("cal_now","cal_old")):
            values[part][a+"_minus_"+b]=interval(scores[a]-scores[b])
    values["median_coefficients"]={k:float(np.median(f[k+"_coef"][:,idx][mask])) if n else None
                                   for k in ("cal_old","traffic_old","anchor","gated","rescaled","current_offsupport")}
    bound=np.broadcast_to(f["anchor_budget"][:,0:1],mask.shape)
    values["median_original_receiver_budget"]=float(np.median(bound[mask])) if n else None
    values["anchor_saturating_fraction"]=float(np.mean(np.isclose(np.abs(f["anchor_coef"][:,idx][mask]),bound[mask],atol=2e-6))) if n else None
    values["anchor_positive_fraction"]=float(np.mean(f["anchor_coef"][:,idx][mask]>0)) if n else None
    return values


def donor_summary(f,active,idx,draw,seed):
    n=len(idx)
    result={"eligible_ids":f["atoms"][idx].tolist(),"n_donors":n}
    if not n:return result
    rng=np.random.default_rng(seed)
    # One donor cannot identify between-donor uncertainty.
    multiplier=rng.multinomial(n,np.full(n,1/n),len(draw)-1) if n>=2 else None
    for part in ("full","shape"):
        per={}
        for arm in ("anchor","gated","fresh","traffic_old","cal_old","old_winner"):
            vals=[]
            for j in idx:
                mask=active[:,j:j+1]
                vals.append(recover(mask,transform(f["native"][:,j:j+1],part),
                                    transform(f[arm+"_effect"][:,j:j+1],part),draw,True))
            table=np.stack(vals,axis=1)
            per[arm] = {"point_by_donor":table[0].tolist(),"sensor_ci":interval(np.nanmean(table,axis=1))}
            if multiplier is not None:
                # Resample donors among the fixed, measurable set as well as resampling sensors.
                measure=np.isfinite(table[1:])
                count=(multiplier*measure).sum(1)
                two=np.divide(np.nansum(multiplier*table[1:],1),count,out=np.full(len(count),np.nan),where=count>0)
                per[arm]["donor_and_sensor_resampling"]=interval(np.r_[table[0].mean(),two])
            else:per[arm]["donor_and_sensor_resampling"]=None
        result[part]=per
    return result


def run():
    f=dict(np.load(OUT/"effects.npz"))
    with np.load(OUT/"norm_null.npz") as norms:
        for key in ("ids","atoms","candidates"):
            np.testing.assert_array_equal(norms[key],f[key])
        f.update({key:norms[key] for key in norms.files if key.startswith("norm")})
    out={}
    j=int(np.flatnonzero(f["atoms"]==DONOR)[0])
    for phase in PHASES:
        rows=f["split"]==phase
        z={k:(v if k in ("atoms","candidates","unique_writes") else v[rows]) for k,v in f.items()}
        draw=weights(len(z["ids"]))
        old,new=z["old_codes"]>0,z["now_codes"]>0
        persisted=old&new
        m=persisted[:,j:j+1]
        won=z["old_winner_id"][:,j:j+1]==PRIMARY
        before=z["original_active"][:,0:1];after=z["current_active"][:,0:1]
        masks={"persistent":m,"all_current":new[:,j:j+1],"original_winner":m&won,"other_original_winner":m&~won,
               "original_target_inactive":m&~before,"original_target_active":m&before,
               "current_target_inactive":m&~after,"current_target_active":m&after}
        record=dict(n=len(old),original_firing=int(old[:,j].sum()),current_firing=int(new[:,j].sum()),
                    donors=z["atoms"].tolist(),persistent_counts=persisted.sum(0).tolist(),
                    primary={name:score_row(z,mask,draw,np.array([j])) for name,mask in masks.items()})
        eligible=np.flatnonzero(persisted.sum(0)>=8)
        record["per_donor"]={str(z["atoms"][k]):score_row(z,persisted[:,k:k+1],draw,np.array([k])) for k in eligible}
        record["eligible_macro"]=donor_summary(z,persisted,eligible,draw,20261121342)
        record["without_primary"]=donor_summary(z,persisted,eligible[eligible!=j],draw,20261121342)
        out[phase]=record
    (OUT/"summary.json").write_text(json.dumps(out,allow_nan=False,separators=(",",":"))+"\n")


if __name__=="__main__":run()
