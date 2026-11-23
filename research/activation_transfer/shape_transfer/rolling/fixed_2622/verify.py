"""Independently reconstruct fixed-direction anchor doses and replay current forecasts."""
import json
import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.feature_groups.data import dictionaries
from research.activation_transfer.shape_transfer.rolling.fixed_2622.data import CACHE,N,OUT,PHASES
from research.activation_transfer.shape_transfer.rolling.fixed_2622.fit import DONOR,PRIMARY
from research.activation_transfer.shape_transfer.rolling.verify import replay,weighted


def run():
    f=dict(np.load(OUT/"effects.npz"))
    with np.load(OUT/"norm_null.npz") as shams:
        np.testing.assert_array_equal(shams["ids"],f["ids"])
        for k in (1,2,3):
            f[f"norm{k}_coef"]=shams[f"norm{k}_coef"]
            f[f"norm{k}_effect"]=shams[f"norm{k}_effect"]
    params=json.loads((OUT/"frozen.json").read_text())
    raw=dict(np.load(CACHE/"data.npz"))
    target,source=[dict(np.load(CACHE/(m+"_geometry.npz"))) for m in ("bolt","timesfm")]
    dic=dictionaries("timesfm")
    h,p,ids=read_materials(CACHE,"bolt")
    chron=adapter_for("bolt");ts=last_window_slice(chron,p)
    donor_idx=int(np.flatnonzero(f["atoms"]==DONOR)[0])
    matched=params["candidates"]
    rng=np.random.default_rng(20261121348)
    checked=0;errors=[]
    for phase in PHASES:
        r=np.flatnonzero((f["split"]==phase)&(f["old_codes"][:,donor_idx]>0)&(f["now_codes"][:,donor_idx]>0))
        winner=f["old_winner_id"][r,donor_idx]
        masks=(winner==PRIMARY,winner!=PRIMARY,f["current_active"][r,0],~f["current_active"][r,0])
        cases={int(r[rng.choice(np.flatnonzero(mask))]) for mask in masks if mask.any()}
        for row in sorted(cases):
            a=row if phase=="tail" else row+N;b=a+N
            np.testing.assert_array_equal(ids[b],f["ids"][row])
            co=f["old_codes"][row,donor_idx];cn=f["now_codes"][row,donor_idx]
            signature=weighted(-co*(source["jac"][a].astype(float)@dic.donor[donor_idx]),dic.weight)
            for k,atom in enumerate(matched):
                anchor=f["original_target_code"][row,k]
                cap=min(.2*dic.radius/np.linalg.norm(dic.decoder[atom]),anchor if anchor>0 else params["fallback_activation"][k])
                basis=weighted(target["jac"][a].astype(float)@dic.decoder[atom],dic.weight)
                gain=np.dot(basis,signature)/np.dot(basis,basis)
                coef=min(cap,max(-cap,gain))
                amounts={("anchor" if k==0 else "sham"+str(k)):coef}
                if k>0:
                    radius=abs(f["anchor_coef"][row,donor_idx])*np.linalg.norm(dic.decoder[PRIMARY])
                    sgn=1 if np.dot(basis,signature)>=0. else -1
                    amounts["norm"+str(k)]=sgn*radius/np.linalg.norm(dic.decoder[atom])
                if k==0:
                    amounts["gated"]=coef if f["current_active"][row,0] else 0.
                    amounts["traffic_old"]=np.clip(co*params["traffic_slopes"][donor_idx],-cap,cap)
                    amounts["traffic_now"]=np.clip(cn*params["traffic_slopes"][donor_idx],-cap,cap)
                for name,val in amounts.items():
                    np.testing.assert_allclose(val,f[name+"_coef"][row,donor_idx],atol=5e-6,rtol=1e-5)
                    y=replay(chron,MODELS["bolt"][-1],ts,raw["x"][b:b+1],h[b:b+1],val*dic.decoder[atom])
                    errors.append(float(np.max(np.abs(y-f[name+"_effect"][row,donor_idx]))));checked+=1
    assert max(errors)<8e-5
    (OUT/"replay.json").write_text(json.dumps(dict(fixed_and_control_writes_replayed=checked,max_absolute_error=max(errors)),indent=2)+"\n")


if __name__=="__main__":
    torch.set_num_threads(2)
    run()
