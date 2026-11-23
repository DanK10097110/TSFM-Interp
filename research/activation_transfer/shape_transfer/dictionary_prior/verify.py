"""Reconstruct selected cases with scalar candidate losses and an independent SVD inverse."""
import json
import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.dictionary_prior.run import OUT,load_target
from tsfm_lens.extraction.hooks import token_patch


def verify(source,target):
    f=dict(np.load(OUT/f"{source}_to_{target}.npz"))
    t=load_target(target)
    np.testing.assert_array_equal(t["ids"],f["ids"])
    model=adapter_for(target)
    selector=last_window_slice(model,t["positions"])
    w=float(f["shape_weight"]);p=np.full((8,8),1/8);weights=p+w*(np.eye(8)-p)
    rng=np.random.default_rng(20261121407)
    replay_errors=[];selected=[]
    for domain in ("daily","extra_daily"):
        cases=np.argwhere((f["split"]==domain)[:,None] & f["active"])
        for row,col in cases[rng.choice(len(cases),3,replace=False)]:
            current=t["jac"][row];eligible=t["support"][row]
            goal=f["measured_source"][row,col,:8].astype(float)
            y=weights@goal
            candidates=[];ranks=[];costs=[];fracs=[];levels=[]
            for k,atom in enumerate(eligible):
                move=-t["doses"][row,k]*t["decoder"][atom]
                candidates.append(move)
                full_wave=current@move
                levels.append(abs(full_wave.mean()-goal.mean()))
                wave=weights@full_wave
                ranks.append(np.sum((wave-y)**2))
                dose=min(1.,max(0.,float(wave@y)/max(float(wave@wave),1e-24)))
                costs.append(float(np.sum((dose*wave-y)**2)));fracs.append(dose)
            order=np.argsort(ranks,kind="stable")
            np.testing.assert_array_equal(order,f["natural_rank"][row,col])
            np.testing.assert_array_equal(np.argsort(levels,kind="stable"),f["level_rank"][row,col])
            k=int(order[0]);h=int(np.argmin(costs))
            assert h==f["attenuated_rank"][row,col,0]
            np.testing.assert_allclose(fracs[h],f["attenuation_fraction"][row,col],rtol=1e-6,atol=1e-7)
            mat=weights@current
            u,sv,v=np.linalg.svd(mat,full_matrices=False)
            lam=.001*sv[0]**2
            reg=v.T@(np.diag(sv/(sv**2+lam))@u.T)
            moves={"dense":reg@y}
            priors=dict(natural=candidates[k],attenuated=fracs[h]*candidates[h])
            for idx,num in ((7,1),(17,2)):
                q=candidates[(k+idx)%32]
                priors[f"wrong{num}"]=q*np.linalg.norm(candidates[k])/np.linalg.norm(q)
            for name,prior in priors.items():
                moves[name]=prior
                moves[name+"_corrected"]=prior+reg@(y-mat@prior)
            if source==target:
                key=np.flatnonzero(eligible==f["atoms"][col])[0]
                moves["known_capped"]=candidates[key]
            for arm,move in moves.items():
                factor=min(1.,.2*t["radius"]/max(np.linalg.norm(move),1e-20))
                np.testing.assert_allclose(np.linalg.norm(move),f[arm+"_norm"][row,col],rtol=3e-5,atol=1e-5)
                delta=(factor*move).astype("f4")
                history=t["x"][row:row+1]
                clean=model.predict(history,16,[.5])["quantiles"][0,:,0]
                edited=torch.from_numpy(t["states"][row:row+1]+delta[None,None,:])
                with token_patch(model.module,MODELS[target][-1],selector,edited):
                    patch=model.predict(history,16,[.5])["quantiles"][0,:,0]
                error=float(np.max(np.abs(patch-clean-f[arm+"_effect"][row,col])))
                replay_errors.append(error)
            selected.append(dict(series=str(f["ids"][row]),donor=int(f["atoms"][col]),candidate=int(eligible[k])))
    if max(replay_errors)>1e-4:raise ValueError(f"{source} → {target} replay error {max(replay_errors)}")
    return dict(cases=selected,physical_writes=len(replay_errors),max_abs_error=max(replay_errors))


if __name__=="__main__":
    torch.set_num_threads(2)
    out={f"{s}_to_{t}":verify(s,t) for s in ("bolt","timesfm") for t in ("bolt","timesfm")}
    (OUT/"replay.json").write_text(json.dumps(out,indent=2)+"\n")
