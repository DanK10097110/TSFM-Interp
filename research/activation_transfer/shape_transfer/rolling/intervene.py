"""Execute transported receiver edits at each new forecast origin."""
import argparse
import json
import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import change_forecasts,last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.feature_groups.data import dictionaries
from research.activation_transfer.shape_transfer.rolling.data import CACHE, LAGS, OUT
from research.activation_transfer.shape_transfer.sparse_stability.run import active_codes
from research.activation_transfer.shape_transfer.sparse_writes.run import weight_leads

ARMS = ("stale", "bounded", "refit", "fresh", "old_signature")


def optimal(valid, profiles, bounds, goal, atom, norms, cap):
    """Choose a single current feature or constrain the fit to the old ID."""
    eligible = np.arange(len(valid)) if atom is None else np.flatnonzero(valid == atom)
    if not len(eligible):
        return -1, 0.
    vectors = profiles[eligible]
    scores = vectors @ goal
    energy = np.square(vectors).sum(1)
    change = np.clip(scores / np.maximum(energy,1e-24), -bounds[eligible], bounds[eligible])
    gain = 2*change*scores-change**2*energy
    ix = int(gain.argmax())
    if gain[ix] <= 0.:
        return -1, 0.
    best = int(valid[eligible[ix]])
    value = float(change[ix])
    return best, value * min(1.,cap / max(abs(value)*norms[best],1e-20))


def carried(atom, coef, present, bound):
    """Preserve the blind old write; enforce present support only in the bounded policy."""
    exact = (int(atom), float(coef))
    constrained = (int(atom), float(np.clip(coef,-bound,bound))) if present else (-1,0.)
    return exact, constrained


def run(source):
    dic = dictionaries(source)
    with np.load(CACHE/"data.npz") as f:
        ids,x,domain = f["ids"],f["x"],f["split"]
    h,pos,keys = read_materials(CACHE,dic.target)
    np.testing.assert_array_equal(ids,keys)
    with np.load(CACHE/(source+"_native.npz")) as s:
        np.testing.assert_array_equal(s["ids"],ids)
        np.testing.assert_array_equal(s["atoms"],dic.atoms)
        codes, native = s["codes"],s["native"]
    with np.load(CACHE/(source+"_geometry.npz")) as s,np.load(CACHE/(dic.target+"_geometry.npz")) as t:
        for record in (s,t):
            np.testing.assert_array_equal(ids,record["ids"])
        sj,tj,base = s["jac"].astype(float),t["jac"].astype(float),t["forecast"]
    per = len(ids)//len(LAGS)
    manifest=json.loads((OUT/"manifest.json").read_text())
    expected = [f"{lag}/{v['series']}" for lag in LAGS for v in manifest["series"]]
    np.testing.assert_array_equal(ids,expected)
    support,z = active_codes(dic.sae,h)
    decoder_norm = np.linalg.norm(dic.decoder,axis=1)
    src_unit = dic.donor @ sj.transpose(0,2,1)
    anchor_active = codes[:per]>0
    selected={k:np.full(codes.shape,-1,dtype=np.int32) for k in ARMS}
    amounts={k:np.zeros(codes.shape,dtype=np.float32) for k in ARMS}
    radius = .2 * dic.radius
    for i,active in enumerate(support):
        anchor = i % per
        prof = weight_leads(dic.decoder[active] @ tj[i].T,dic.weight)
        bounds = z[i,active]
        for j in np.flatnonzero(anchor_active[anchor]):
            if i == anchor:
                goal = weight_leads(-codes[i,j]*src_unit[i,j],dic.weight)
                action = optimal(active,prof,bounds,goal,None,decoder_norm,radius)
                for name in ARMS:
                    selected[name][i,j],amounts[name][i,j]=action
                continue
            oldid = selected["fresh"][anchor,j]
            if oldid < 0:
                raise ValueError("Original active donor had no recipient action")
            oldcoef = amounts["fresh"][anchor,j]
            exact, constrained = carried(oldid,oldcoef,z[i,oldid]>0,z[i,oldid])
            selected["stale"][i,j],amounts["stale"][i,j]=exact
            selected["bounded"][i,j],amounts["bounded"][i,j]=constrained
            if codes[i,j]<=0:
                continue
            for arm in ("refit","fresh","old_signature"):
                unit = src_unit[anchor if arm=="old_signature" else i,j]
                goal = weight_leads(-codes[i,j]*unit,dic.weight)
                action = optimal(active,prof,bounds,goal,oldid if arm=="refit" else None,decoder_norm,radius)
                selected[arm][i,j],amounts[arm][i,j]=action
    model = adapter_for(dic.target)
    selector = last_window_slice(model,pos)
    layer = MODELS[dic.target][-1]
    effects={key:np.zeros((*codes.shape,8),dtype="f4") for key in ARMS}
    total=0
    # Enumerate once per origin: identical target writes get the identical production result.
    for lag_idx,lag in enumerate(LAGS):
        lo,hi=lag_idx*per,(lag_idx+1)*per
        snapshot = CACHE / (source + "_lag" + str(lag) + "_effects.npz")
        if snapshot.exists():
            with np.load(snapshot) as saved:
                np.testing.assert_array_equal(saved["ids"],ids[lo:hi])
                for arm in ARMS:
                    np.testing.assert_array_equal(saved[arm+"_id"],selected[arm][lo:hi])
                    np.testing.assert_array_equal(saved[arm+"_coef"],amounts[arm][lo:hi])
                    effects[arm][lo:hi] = saved[arm+"_effect"]
                total += int(saved["count"])
            continue
        unique={}
        for key in ARMS:
            for rr,j in np.argwhere(selected[key][lo:hi]>=0):
                i=int(rr+lo)
                atom=int(selected[key][i,j])
                coef=float(amounts[key][i,j])
                if abs(coef)>0:
                    unique.setdefault((i,atom,coef),[]).append((key,i,j))
        writes=list(unique)
        for start in range(0,len(writes),32):
            chosen=writes[start:start+32]
            rows=np.array([w[0] for w in chosen])
            atoms=np.array([w[1] for w in chosen])
            coefs=np.array([w[2] for w in chosen])
            delta=(coefs[:,None]*dic.decoder[atoms]).astype("f4")
            y=change_forecasts(model,layer,x[rows],h[rows],delta,selector,32)-base[rows]
            for value,w in zip(y,chosen):
                for name,i,j in unique[w]:
                    effects[name][i,j]=value
        shard = {"ids":ids[lo:hi],"count":len(writes)}
        for arm in ARMS:
            shard[arm+"_id"] = selected[arm][lo:hi]
            shard[arm+"_coef"] = amounts[arm][lo:hi]
            shard[arm+"_effect"] = effects[arm][lo:hi]
        temporary = snapshot.with_suffix(".partial.npz")
        np.savez_compressed(temporary,**shard)
        temporary.replace(snapshot)
        total+=len(writes)
        print(source,lag,'unique physically executed receiver writes',len(writes),flush=True)
    record=dict(ids=ids,split=domain,atoms=dic.atoms,codes=codes,support=support,native=native,
                source_linear=(-codes[:,:,None]*src_unit).astype("f4"),source_unit=src_unit.astype("f4"),
                unique_target_writes=total)
    for name in ARMS:
        record[name+"_id"]=selected[name]
        record[name+"_coef"]=amounts[name]
        record[name+"_effect"]=effects[name]
        np.testing.assert_array_equal(effects[name][:per],effects["fresh"][:per])
    np.savez_compressed(OUT/(source+".npz"),**record)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=MODELS)
    torch.set_num_threads(2)
    run(p.parse_args().source)
