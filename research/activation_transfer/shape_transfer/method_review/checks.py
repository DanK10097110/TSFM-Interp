"""Retrospective read-only checks on stored nonlinear outputs, with no new forecaster interventions."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import ndtri
from scipy.stats import beta

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def self_check():
    path=ROOT/"self_transfer/results"
    scores=json.loads((path/"summary.json").read_text())
    with np.load("/tmp/tsfm_self_transfer_20261121/timer.npz") as t:timer_width=t["dec"].shape[0]
    out={}
    for model,width in (("bolt",512),("timesfm",1280),("timer",timer_width)):
        with np.load(path/f"{model}_to_{model}.npz") as f:
            domain=f["split"]=="extra_daily";active=f["active"][domain];cols=active.sum(0)>=8
            i,j=np.nonzero(active&cols);fractions=f["projection_norm_fraction"][domain][i]
            true=fractions[np.arange(len(i)),j]
            other=fractions.copy();other[np.arange(len(i)),j]=np.nan
            s=scores[model]["extra_daily"]
            out[model]=dict(width=width,rank=8,n=len(i),observed_visible_median=float(np.median(true)),
                isotropic_visible_median=float(np.sqrt(beta.ppf(.5,4,(width-8)/2))),
                other_selected_atom_visible_median=float(np.nanmedian(other)),
                correct_beats_other_fraction=float(np.nansum(true[:,None]>other)/(len(true)*(other.shape[1]-1))),
                separately_applied_sum_late_skill=s["scores"]["late_shape"]["separate_sum"]["value"],
                projected_late_skill=s["scores"]["late_shape"]["projection"]["value"],
                null_late_skill=s["scores"]["late_shape"]["null"]["value"])
    return out


def cubic_check():
    path=ROOT/"quantiles/results"
    x=ndtri(np.arange(1,10)/10)
    visible=np.array([0,2,4,6,8]);hidden=np.array([1,3,5,7])
    basis=np.eye(5)
    spline=CubicSpline(x[visible],basis,axis=0,bc_type="natural")(x[hidden]).T
    linear=np.array([np.interp(x[hidden],x[visible],col) for col in basis])
    out={}
    for donor in ("bolt","timesfm"):
        with np.load(path/(donor+"_normal.npz")) as f:
            data=dict(f)
        observed=data["source"][:,:,:8].astype(float);receiver=data["five"][:,:,:8].astype(float)
        med,width,asym,left,right=np.moveaxis(data["linear"].astype(float),-1,0)
        ratio=abs(ndtri(.3)/ndtri(.1))
        estimate=np.stack((med+asym-width,med+ratio*(asym-width)+left,med,
                           med+ratio*(asym+width)+right,med+asym+width),axis=-1)
        estimated_spline=estimate@spline
        actual_spline=observed[...,visible]@spline
        domains={}
        for domain in np.unique(data["split"]):
            idx=data["split"]==domain
            active=data["active"][idx]
            keep=active.sum(0)>=8
            mask=active[:,keep]
            s=resampling_counts(len(active),3000)
            p=int(keep.sum());fcounts=np.random.default_rng(870012).multinomial(p,np.full(p,1/p),size=len(s))
            def select(a):return a[idx][:,keep].reshape(*mask.shape,-1)
            ref=select(observed[...,hidden]-observed[...,visible]@linear)
            candidates=dict(physical=select(receiver[...,hidden]-receiver[...,visible]@linear),
                            matched_cubic=select(estimated_spline-estimate@linear),
                            privileged_cubic=select(actual_spline-observed[...,visible]@linear))
            raw={k:intervals(mask,v,ref,s,fcounts) for k,v in candidates.items()}
            row={k:dict(skill=float(v[0]),crossed95=np.nanquantile(v[2],[.025,.975]).tolist()) for k,v in raw.items()}
            a,b=raw["physical"],raw["matched_cubic"]
            row["physical_minus_matched_cubic"]=dict(skill=float(a[0]-b[0]),crossed95=np.nanquantile(a[2]-b[2],[.025,.975]).tolist())
            row["n_atoms"]=p
            row["rms"]=float(np.sqrt(np.mean(ref[mask]**2)))
            domains[domain]=row
        out[donor]=domains
    return out


def interaction_check():
    path=ROOT/"interactions/results"
    out={}
    for donor,target in (("bolt","timesfm"),("timesfm","bolt")):
        files=[path/f"{donor}_{n}.npz" for n in ("native",donor,target)]
        if not all(p.exists() for p in files):continue
        arrays=[]
        for p in files:
            with np.load(p) as f:arrays.append(dict(f))
        native,self_arm,cross=arrays
        for a in arrays[1:]:
            for k in ("ids","rows","pairs","atoms","indices"):np.testing.assert_array_equal(a[k],native[k])
        domains={}
        for lab in np.unique(native["split"]):
            case=(native["split"]==lab)[native["rows"]]
            pair=native["pairs"]
            count=np.bincount(pair[case],minlength=len(native["indices"]))
            freq=count>=8
            row=dict(n_pairs=int(freq.sum()),n_all_cofirings=int(case.sum()))
            if not freq.any():
                domains[lab]=row;continue
            mask=case&freq[pair]
            nc=int(mask.sum())
            row.update(n_eligible_cofirings=nc,self_clipped=int((self_arm["scale"][mask]<1).sum()),
                       cross_clipped=int((cross["scale"][mask]<1).sum()))
            single_a=native["a"][mask,:8].astype(float);single_b=native["b"][mask,:8].astype(float)
            center=lambda v:v-v.mean(-1,keepdims=True)
            a,b=center(single_a),center(single_b)
            rms=lambda q:float(np.sqrt(np.mean(q*q)))
            interaction=[];zero=[]
            for arm in arrays:
                interaction.append(center(arm["joint"][mask,:8].astype(float)-arm["a"][mask,:8]-arm["b"][mask,:8]))
                zero.append(center(arm["noise"][mask,:8].astype(float)))
            pool=pair[mask]
            def measure(ref,est):
                values=[]
                for j in np.flatnonzero(freq):
                    i=pool==j
                    if np.mean(ref[i]**2)>1e-10:values.append(1-np.sum((ref[i]-est[i])**2)/np.sum(ref[i]**2))
                return (float(np.mean(values)),len(values)) if values else (None,0)
            unadj=measure(interaction[0],interaction[2])
            adj=measure(interaction[0]+zero[0],interaction[2]+zero[2])
            source_adj=interaction[0]+zero[0];cross_adj=interaction[2]+zero[2]
            by_pair={}
            for j in np.flatnonzero(freq):
                k=pool==j
                if np.mean(source_adj[k]**2)>1e-10:
                    by_pair[int(j)]=1-np.sum((source_adj[k]-cross_adj[k])**2)/np.sum(source_adj[k]**2)
            edges=native["indices"]
            lout=[]
            for node in np.unique(edges[list(by_pair)]):
                values=[float(v) for j,v in by_pair.items() if node not in edges[j]]
                if values:lout.append(np.mean(values))
            row["leave_one_source_atom_out_cross_skill_range"]=[float(min(lout)),float(max(lout))] if lout else None
            row.update(source_raw_interaction_rms=rms(interaction[0]),source_zero_rms=rms(zero[0]),
                       target_raw_interaction_rms=rms(interaction[2]),target_zero_rms=rms(zero[2]),
                       target_raw_interaction_skill=unadj[0],target_zero_adjusted_interaction_skill=adj[0],
                       measurable_unadjusted=unadj[1],measurable_zero_adjusted=adj[1],
                       native_interaction_over_rms_of_singleton_sum=rms(interaction[0])/rms(a+b),
                       native_interaction_over_quadrature=rms(interaction[0])/np.sqrt(rms(a)**2+rms(b)**2))
            both=case&(self_arm["scale"]>=1)&(cross["scale"]>=1)
            retained=np.bincount(pair[both],minlength=len(freq))
            row.update(both_unclipped_pairs_frequent_after_filter=int((retained>=8).sum()),
                       both_unclipped_rms_rows_all_domain_eligible=int(np.sum(both&freq[pair])),
                       both_unclipped_rms_rows_stratum_eligible=int(np.sum(both&(retained>=8)[pair])))
            domains[lab]=row
        out[donor]=domains
    return out


if __name__=="__main__":
    hashes={name:sha256((ROOT/"quantiles/results"/(name+"_normal.npz")).read_bytes()).hexdigest() for name in ("bolt","timesfm")}
    (HERE/"checks.json").write_text(json.dumps(dict(final_quantile_archives_sha256=hashes,self_transfer=self_check(),five_anchor_cubic=cubic_check(),
                                                interactions=interaction_check()),indent=2,allow_nan=False)+"\n")
