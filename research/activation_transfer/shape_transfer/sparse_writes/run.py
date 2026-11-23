"""Select and physically edit one or four firing receiver SAE features by their forecast response."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.replication.data import CACHE, HERE as REPLICATION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.full_space.transfer import HERE as FULL, SCRATCH
from tsfm_lens.sae.train import load_sae_checkpoint

HERE = Path(__file__).resolve().parent
ARMS = ("fixed_removal","local_removal","one_signed","four_signed","permuted_four")
EXTRA = ("partial_removal","fixed_signed")


def weight_leads(wave, factor):
    mean = wave.mean(axis=-1,keepdims=True)
    return mean + factor*(wave-mean)


def removal_gains(profiles, strengths, goal):
    candidates = -strengths[:,None]*profiles
    return 2*(candidates@goal) - np.sum(candidates**2,axis=1)


def pursuit(profiles, strengths, goal, count, attenuation_only=False):
    """One-use signed changes within each firing feature's original amplitude."""
    residual = goal.copy()
    selected, coefs = [], []
    norm = np.einsum("ij,ij->i",profiles,profiles)
    for _ in range(count):
        inner = profiles @ residual
        change = np.clip(inner/np.maximum(norm,1e-24),-strengths,strengths)
        if attenuation_only:
            change = np.minimum(change,0.)
        gain = 2*change*inner - change**2*norm
        gain[selected] = -np.inf
        if not len(gain) or gain.max() <= 0:
            break
        index = int(np.argmax(gain))
        selected.append(index)
        coefs.append(change[index])
        residual -= change[index]*profiles[index]
    return np.asarray(selected,int),np.asarray(coefs)


def stable_partners(cal, codes, target_z, rows, decoder, jac, goal, weight, signed=False):
    gain = np.zeros((codes.shape[1],len(decoder)))
    for i in np.flatnonzero(cal):
        idx = rows[i]
        profiles = weight_leads(decoder[idx] @ jac[i].T,weight)
        for j in np.flatnonzero(codes[i]>0):
            bounds = target_z[i,idx]
            if signed:
                norm = np.sum(profiles**2,axis=1)
                projection = profiles @ goal[i,j]
                change = np.clip(projection/np.maximum(norm,1e-24),-bounds,bounds)
                gain[j,idx] += 2*projection*change - norm*change**2
            else:
                gain[j,idx] += removal_gains(profiles,bounds,goal[i,j])
    return np.where(gain.max(axis=1)>0,gain.argmax(axis=1),-1)


def run(source,cache,scratch,out,batch,extra=False):
    target = next(n for n in MODELS if n!=source)
    with np.load(REPLICATION/"results"/(source+"_seed0.npz")) as f:
        ids,split,codes,atoms = f["ids"],f["split"],f["source_codes"],f["atoms"]
    cal = split=="cal"
    ev = ~cal
    with np.load(scratch/(target+"_jac.npz")) as t,np.load(scratch/(source+"_jac.npz")) as s:
        np.testing.assert_array_equal(ids,t["ids"])
        np.testing.assert_array_equal(ids,s["ids"])
        target_jac,radius,base = t["jac"].astype(float),float(t["radius"]),t["forecast"]
        source_dirs = load_sae_checkpoint(str(dictionary_path(source))).W_dec.detach().numpy()[atoms]
        source_profiles = source_dirs.astype(float) @ s["jac"].astype(float).transpose(0,2,1)
    with np.load(FULL/"results"/(source+".npz")) as f:
        weight = float(f["shape_weight"])
        np.testing.assert_array_equal(ids[ev],f["ids"])
    with np.load(cache/"data.npz") as f:
        sel = f["split"]!="train"
        np.testing.assert_array_equal(ids,f["ids"][sel])
        x = f["x"][sel]
    tokens,positions,item_ids = read_materials(cache,target)
    np.testing.assert_array_equal(item_ids[sel],ids)
    tokens = tokens[sel]
    sae = load_sae_checkpoint(str(dictionary_path(target)))
    decoder = sae.W_dec.detach().numpy().astype(float)
    with torch.inference_mode():
        target_z = sae.encode(torch.from_numpy(tokens.mean(axis=1))).numpy()
    active_ids = [np.flatnonzero(z>0) for z in target_z]
    if min(map(len,active_ids))<4:
        raise ValueError("Fewer than four candidate atoms")
    goal = weight_leads(-codes[:,:,None]*source_profiles,weight)
    fixed = stable_partners(cal,codes,target_z,active_ids,decoder,target_jac,goal,weight,signed=extra)
    perm = np.random.default_rng(12871).permutation(len(decoder))
    adapter = adapter_for(target)
    layer = MODELS[target][-1]
    select = last_window_slice(adapter,positions)
    rec = dict(ids=ids[ev],atoms=atoms,split=split[ev],active=codes[ev]>0,fixed=fixed,
               candidate_count=np.array([len(v) for v in active_ids])[ev])
    rows,features = np.nonzero(codes[ev]>0)
    global_rows = np.flatnonzero(ev)[rows]
    for arm in EXTRA if extra else ARMS:
        delta = np.zeros((len(rows),decoder.shape[-1]),dtype="f4")
        ids_used = np.full((*codes[ev].shape,4),-1,dtype=int)
        assigned = np.zeros(ids_used.shape,dtype="f4")
        for row in np.unique(global_rows):
            valid = active_ids[row]
            coeffs = target_z[row,valid]
            candidates = perm[valid] if arm=="permuted_four" else valid
            weights = decoder[candidates]
            profiles = weight_leads(weights @ target_jac[row].T,weight)
            for p in np.flatnonzero(global_rows==row):
                j = features[p]
                if arm in ("fixed_removal","fixed_signed"):
                    which = np.flatnonzero(valid==fixed[j])
                    if arm=="fixed_signed" and len(which):
                        feature = profiles[which[0]]
                        amounts = np.clip(feature @ goal[row,j]/max(feature@feature,1e-24),-coeffs[which],coeffs[which])
                    else:
                        amounts = -coeffs[which]
                elif arm=="local_removal":
                    gains = removal_gains(profiles,coeffs,goal[row,j])
                    which = np.array([gains.argmax()],int) if gains.max()>0 else np.empty(0,int)
                    amounts = -coeffs[which]
                else:
                    count = 1 if arm in ("one_signed","partial_removal") else 4
                    which,amounts = pursuit(profiles,coeffs,goal[row,j],count,attenuation_only=arm=="partial_removal")
                if not len(which):
                    continue
                direction = amounts @ weights[which]
                literal = arm in ("fixed_removal","local_removal")
                cap = 1. if literal else min(1.,.2*radius/max(np.linalg.norm(direction),1e-20))
                delta[p] = cap*direction
                ids_used[rows[p],j,:len(which)] = candidates[which]
                assigned[rows[p],j,:len(which)] = cap*amounts
        effects = np.zeros((*codes[ev].shape,8),dtype="f4")
        for start in range(0,len(rows),batch):
            sl = slice(start,start+batch)
            r = global_rows[sl]
            pred = change_forecasts(adapter,layer,x[r],tokens[r],delta[sl],select,batch)
            effect = pred-base[r]
            effect[~np.any(delta[sl],axis=1)] = 0.
            effects[rows[sl],features[sl]] = effect
        rec[arm] = effects
        rec[arm+"_ids"] = ids_used
        rec[arm+"_coefficients"] = assigned
    out.mkdir(exist_ok=True,parents=True)
    suffix = "_identity.npz" if extra else ".npz"
    np.savez_compressed(out/(source+suffix),**rec)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=MODELS)
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--scratch",type=Path,default=SCRATCH)
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--extra",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(2)
    run(args.source,args.cache,args.scratch,args.out,args.batch,args.extra)
