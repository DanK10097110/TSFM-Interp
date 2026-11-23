"""Execute residual-preserving atom removals and their mapped alternatives on real series."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import CACHE, HERE, adapter_for, MODELS
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice


def exact_removal_effects(effects, strength):
    """A contribution of exactly zero is a no-op even if re-batching changes floating arithmetic."""
    return np.where(np.asarray(strength)[:, :, None] == 0, 0, effects)


def run_arm(adapter, layer, selector, x, tokens, ordinary, active, strength, vectors, batch):
    """Batch only the source-firing atom--series pairs; preserve all untouched token residuals."""
    rows, atoms=np.nonzero(active)
    effects=np.zeros((*active.shape,ordinary.shape[1]),dtype="f4")
    for start in range(0,len(rows),batch):
        i=rows[start:start+batch]
        j=atoms[start:start+batch]
        delta=(-strength[i,j,None]*vectors[j]).astype("f4")
        altered=change_forecasts(adapter,layer,x[i],tokens[i],delta,selector,batch)
        effects[i,j]=altered-ordinary[i]
    return exact_removal_effects(effects,strength)


def patch(model,cache,out,batch):
    source=model
    target=next(m for m in MODELS if m!=source)
    with np.load(cache/"data.npz") as data:
        sel=data["split"]!="train"
        x,ids,split=data["x"][sel],data["ids"][sel],data["split"][sel]
    with np.load(cache/(source+"_vectors.npz")) as vec:
        rec=dict(vec)
    np.testing.assert_array_equal(rec["ids"][sel],ids)
    z=rec["source_codes"][sel]
    active=z>0
    if np.min(np.sum(active[split=="cal"],axis=0))==0:
        print("Some atoms inactive in calibration",flush=True)
    out.mkdir(exist_ok=True,parents=True)
    data_file=out/(source+"_effects.npz")
    arrays={}
    diagnostics={}
    for name in [source,target]:
        enc=adapter_for(name)
        layer=MODELS[name][-1]
        with np.load(cache/(name+".npz")) as local:
            tokens=local["tokens"][sel]
            selector=last_window_slice(enc,local["positions"])
        ordinary=change_forecasts(enc,layer,x,tokens,None,selector,batch)
        zeros=np.zeros((len(x),tokens.shape[-1]),dtype="f4")
        clean=change_forecasts(enc,layer,x,tokens,zeros,selector,batch)
        replay=float(np.max(np.abs(clean-ordinary)))
        if replay>5e-5:
            raise ValueError(f"Self-patch drift {replay}")
        diagnostics[name]=dict(layer=layer,clean_patch_error=replay)
        if name==source:
            arms={"source_native":(z,rec["source_decoder"]),"source_projected":(z,rec["source_projection"])}
        else:
            arms={k:(z,rec[k]) for k in ["target_dense","target_sham","target_omp1","target_omp4","target_omp16"]}
            arms["target_coactive"]=(rec["target_match_codes"][sel],rec["target_match_decoder"])
            for i,wrong in enumerate(rec["wrong_pairs"]):
                v=rec["target_dense"][wrong]
                v=v*np.linalg.norm(rec["target_dense"],axis=1,keepdims=True)/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-15)
                arms["target_wrong"+str(i)]=(z,v)
        for arm,(strength,directions) in arms.items():
            scratch=cache/(source+"_"+arm+".npz")
            # Older scratch files without a fingerprint are recomputed, preserving checkpoints
            # while preventing accidental reuse after a direction, amplitude or batching change.
            digest=sha256(tokens.tobytes())
            for value in (x,active,strength,directions,rec["source_atoms"]):
                digest.update(np.ascontiguousarray(value).tobytes())
            digest.update(str(batch).encode())
            digest.update(str(layer).encode())
            signature=digest.hexdigest()
            reusable=False
            if scratch.exists():
                with np.load(scratch) as stored:
                    reusable="fingerprint" in stored.files and str(stored["fingerprint"])==signature
            if reusable:
                with np.load(scratch) as stored:
                    np.testing.assert_array_equal(stored["ids"],ids)
                    np.testing.assert_array_equal(stored["feature_ids"],rec["source_atoms"])
                    arrays[arm]=stored["effect"]
            else:
                effect=run_arm(enc,layer,selector,x,tokens,clean,active,strength,directions,batch)
                # Different batched trajectories must agree for a true one-row intervention.
                i,j=np.argwhere(active)[0]
                single=change_forecasts(enc,layer,x[i:i+1],tokens[i:i+1],
                       (-strength[i,j]*directions[j])[None].astype("f4"),selector,1)
                native_single=change_forecasts(enc,layer,x[i:i+1],tokens[i:i+1],None,selector,1)
                np.testing.assert_allclose(effect[i,j],(single-native_single)[0],atol=2e-4,rtol=.002)
                np.savez_compressed(scratch,effect=effect,ids=ids,feature_ids=rec["source_atoms"],fingerprint=signature)
                arrays[arm]=effect
            print("patched source",source,"arm",arm,"active pairs",active.sum(),flush=True)
    match_codes=rec["target_match_codes"][sel]
    arrays["target_coactive"]=exact_removal_effects(arrays["target_coactive"],match_codes)
    np.savez_compressed(data_file,active=active,source_codes=z,target_match_codes=match_codes,
                        ids=ids,split=split,**arrays)
    weights=[{k:float(np.linalg.norm(rec[k][j])) for k in ("source_decoder","source_projection",
              "target_dense","target_sham","target_omp1","target_omp4","target_omp16")}
             for j in range(len(rec["source_atoms"]))]
    info=dict(source=source,target=target,sae_sha256={n:str(rec[side+"_sae_sha"]) for n,side in [(source,"source"),(target,"target")]},
              source_atoms=rec["source_atoms"].tolist(),target_atom_matches=rec["target_match"].tolist(),
              target_match_train_correlation=rec["target_match_corr"].tolist(),eligible=int(rec["target_eligible"]),
              target_omp_indices={str(k):rec[f"target_omp{k}_ids"].tolist() for k in (1,4,16)},
              control_pair_permutations=rec["wrong_pairs"].tolist(),norms=weights,diagnostics=diagnostics,
              activation_rate_train=(rec["source_codes"][~sel]>0).mean(0).tolist())
    (out/(source+"_metadata.json")).write_text(json.dumps(info,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=MODELS);p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--out",type=Path,default=HERE/"results");p.add_argument("--batch",type=int,default=24)
    a=p.parse_args()
    torch.set_num_threads(3)
    patch(a.source,a.cache,a.out,a.batch)
