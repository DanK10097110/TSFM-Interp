"""Direct-hook replay and independent scalar minimization for selected series at each lag."""
import json
import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.feature_groups.data import dictionaries
from research.activation_transfer.shape_transfer.rolling.data import CACHE,LAGS,OUT
from research.activation_transfer.shape_transfer.sparse_stability.run import active_codes
from tsfm_lens.extraction.hooks import token_patch


def replay(model,layer,selector,history,tokens,delta):
    standard = model.predict(history,8,[.5])["quantiles"][0,:8,0]
    patched = torch.as_tensor(tokens + delta.astype("f4")[None,None,:])
    with token_patch(model.module,layer,selector,patched):
        value=model.predict(history,8,[.5])["quantiles"][0,:8,0]
    return value-standard


def weighted(v,factor):
    return v.mean() + factor*(v-v.mean())


def audit(source):
    result=dict(np.load(OUT/(source+".npz")))
    data=dict(np.load(CACHE/"data.npz"))
    dic=dictionaries(source)
    target=dic.target
    sgeom,tgeom=[dict(np.load(CACHE/(m+"_geometry.npz"))) for m in (source,target)]
    s_tokens,s_pos,s_ids=read_materials(CACHE,source)
    t_tokens,t_pos,t_ids=read_materials(CACHE,target)
    for ids in (s_ids,t_ids,result["ids"],sgeom["ids"],tgeom["ids"]):
        np.testing.assert_array_equal(ids,data["ids"])
    sm,tm=[adapter_for(m) for m in (source,target)]
    ss,ts=last_window_slice(sm,s_pos),last_window_slice(tm,t_pos)
    support,z=active_codes(dic.sae,t_tokens)
    np.testing.assert_array_equal(support,result["support"])
    n=len(s_ids)//len(LAGS)
    rng=np.random.default_rng(20261121337)
    errors=[];donor_errors=[]; selected=0;stopped=0
    for k,lag in enumerate(LAGS):
        for domain in ("daily","monash_traffic"):
            starts=np.flatnonzero(result["split"][:n]==domain)
            rows=starts+k*n
            old=result["codes"][starts]>0;now=result["codes"][rows]>0
            groups = [old & now]
            if lag:
                groups.append(old & ~now)
            for pool in groups:
                ij=np.argwhere(pool)
                if not len(ij):continue
                r,j=ij[rng.integers(len(ij))]
                anchor,i=starts[r],rows[r]
                hist=data["x"][i:i+1]
                src= -result["codes"][i,j]*dic.donor[j]
                oy=replay(sm,MODELS[source][-1],ss,hist,s_tokens[i:i+1],src)
                donor_errors.append(float(np.max(abs(oy-result["native"][i,j]))))
                if not now[r,j]:stopped+=1
                for name in ("stale","bounded","refit","fresh","old_signature"):
                    old_id = result["fresh_id"][anchor,j]; old_c=result["fresh_coef"][anchor,j]
                    active=support[i]
                    if name == "stale":
                        atom,coef=old_id,old_c
                    elif name == "bounded":
                        atom,coef=(old_id,float(np.clip(old_c,-z[i,old_id],z[i,old_id]))) if old_id in active else (-1,0.)
                    elif not now[r,j]:
                        atom,coef=-1,0.
                    else:
                        donor_point = anchor if name=="old_signature" else i
                        goal=weighted(-result["codes"][i,j]*(sgeom["jac"][donor_point].astype(float)@dic.donor[j]),
                                      dic.weight)
                        best=float(goal@goal);atom,coef=-1,0.
                        for candidate in active:
                            if name=="refit" and candidate!=old_id:continue
                            wave=weighted(tgeom["jac"][i].astype(float)@dic.decoder[candidate],dic.weight)
                            val=np.clip((wave@goal)/max(wave@wave,1e-24),-z[i,candidate],z[i,candidate])
                            loss=np.sum((goal-val*wave)**2)
                            if loss<best:
                                best=loss;atom,coef=int(candidate),float(val)
                        if atom>=0:
                            coef*=min(1.,.2*dic.radius/max(np.linalg.norm(coef*dic.decoder[atom]),1e-20))
                    assert atom==result[name+"_id"][i,j]
                    np.testing.assert_allclose(coef,result[name+"_coef"][i,j],rtol=1e-5,atol=5e-6)
                    delta=coef*dic.decoder[atom] if atom>=0 else np.zeros(dic.decoder.shape[1])
                    y=replay(tm,MODELS[target][-1],ts,hist,t_tokens[i:i+1],delta)
                    errors.append(float(np.max(abs(y-result[name+"_effect"][i,j]))));selected+=1
    if max(donor_errors+errors)>8e-5:
        raise ValueError(f"{source} intervention replay mismatch")
    return dict(receiver_actions=selected,target_abs_max=max(errors),source_actions=len(donor_errors),
                source_abs_max=max(donor_errors),donor_stopped_samples=stopped)


if __name__=="__main__":
    torch.set_num_threads(2)
    result={m:audit(m) for m in ("bolt","timesfm")}
    (OUT/"replay.json").write_text(json.dumps(result,indent=2)+"\n")
