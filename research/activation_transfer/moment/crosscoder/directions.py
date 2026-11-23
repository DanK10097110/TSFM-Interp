"""Compare paired decoder edits on identical source-selected coordinates and contexts."""
import argparse
import json

import numpy as np
import torch

from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.mechanism import map_inputs, OUT, SPLIT
from research.activation_transfer.moment.dictionary_transfer.match import hidden
from research.activation_transfer.moment.dictionary_transfer.functional.audit import addition
from research.activation_transfer.moment.token_paths.geometry import CACHE, histories, load

OUT = OUT / "directions"
B = 4000
SEED = 22112411


def ordinary_map():
    meta = json.loads((OUT.parent / "mechanism.json").read_text())
    best = meta["fits"]["map_all_exactgate"]["relative"]
    x,y = (hidden(s)[:SPLIT].reshape(-1,d) for s,d in (("small",512),("large",1024)))
    query = np.vstack((np.eye(512), np.zeros(512)))
    for lam, p in map_inputs(x,y,query):
        if lam==best: return p[:-1]-p[-1]
    raise ValueError("Unknown penalty")


def read_views():
    x,ids,domains = histories(False)
    states, jac = {}, {}
    for size in cd.SIZES:
        with np.load(CACHE/size/"eval.npz") as f:
            np.testing.assert_array_equal(ids, f["ids"])
            np.testing.assert_array_equal(domains, f["split"])
            states[size] = f["middle"][:,0].astype("f4")
            jac[size] = f["middle_jac"][:,:,0].astype(float)
    return x, ids, domains, states, jac


def compare(waves):
    src, tar, dense = (waves[k] for k in ("small","large","dense"))
    def normalize(a):
        centered=a-a.mean(-1,keepdims=True)
        norms=np.linalg.norm(centered,axis=-1,keepdims=True)
        if np.any(norms < 1e-13): raise ValueError("Zero centered response")
        return centered / norms
    a,b,d = (normalize(v) for v in (src,tar,dense))
    match = (a*b).sum(-1)
    mapped = (a*d).sum(-1)
    off = ((a*b.sum(1,keepdims=True)).sum(-1)-match)/(b.shape[1]-1)
    return dict(paired=match, dense=mapped, within_history=off)


def describe(scores):
    rand=np.random.default_rng(SEED+7)
    sums=rand.multinomial(len(scores), np.full(len(scores),1/len(scores)), B)
    b=sums@scores/len(scores)
    return [float(scores.mean()), *np.quantile(b,[.025,.975]).tolist()]


def jacobian(method,seed,w):
    x, ids, domains, states, jac = read_views()
    model,means=cd.load_artifact(method,seed)
    with torch.inference_mode():
        codes=cd.encode_views(model,means,states["small"]).numpy()
    assert np.all((codes>0).sum(1)==cd.TOPK)
    # In each history all qualifying coordinates are retained. Sorting IDs fixes the sampler convention.
    atom=np.where(codes>0)[1].reshape(len(x),cd.TOPK)
    dose=np.take_along_axis(codes,atom,axis=1)
    with np.load(cd.CACHE/f"{method}_{seed}.npz") as f:
        ds,dl=f["decoder_small"].astype(float), f["decoder_large"].astype(float)
    dense=ds@w
    dense*=np.linalg.norm(dl,axis=1,keepdims=True)/np.linalg.norm(dense,axis=1,keepdims=True)
    waves={}
    for key, basis, site in (("small",ds,"small"),("large",dl,"large"),("dense",dense,"large")):
        waves[key]=-dose[:,:,None]*(basis[atom]@jac[site].transpose(0,2,1))
    scores=compare(waves)
    out={}
    for name in np.unique(domains):
        ix=np.flatnonzero(domains==name)
        mse={}
        for key,wave in waves.items():
            sig=(wave[ix]-wave[ix].mean(-1,keepdims=True))/x[ix].std(1)[:,None,None]
            r=np.sqrt(np.mean(sig**2,-1))
            mse[key]=np.quantile(r,[.1,.5,.9]).tolist()
        delta=(scores["paired"]-scores["within_history"])[ix]
        atoms=atom[ix].ravel(); counts=np.bincount(atoms,minlength=cd.WIDTH)
        eligible=counts>=8
        totals=np.bincount(atoms,weights=delta.ravel(),minlength=cd.WIDTH)
        out[str(name)]=dict(**{k:describe(v[ix].mean(1)) for k,v in scores.items()},
            paired_minus_within=describe(delta.mean(1)),
            paired_minus_dense=describe((scores["paired"]-scores["dense"])[ix].mean(1)),
            eligible_atoms=int(eligible.sum()), atom_equal_paired_minus_within=float((totals[eligible]/counts[eligible]).mean()),
            centered_linear_rms_history_sd_quantiles=mse)
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/f"{method}_{seed}.npz",ids=ids,domains=domains,atom=atom,dose=dose,**scores)
    (OUT/f"{method}_{seed}.json").write_text(json.dumps(out,indent=2)+"\n")
    return ds,dl,dense,waves


def sample(domains):
    g=np.random.default_rng(SEED+9)
    rows=np.concatenate([np.sort(g.choice(np.flatnonzero(domains==domain),32,replace=False)) for domain in np.unique(domains)])
    pick=g.integers(0,cd.TOPK,len(rows))
    other=(pick+g.integers(1,cd.TOPK,len(rows)))%cd.TOPK
    return rows,pick,other


def finite(method,seed,parameters,adapters):
    ds,dl,dense,jw=parameters
    x,ids,dom,_,jac=read_views()
    rows,pick,other=sample(dom)
    with np.load(OUT/f"{method}_{seed}.npz") as f:
        np.testing.assert_array_equal(f["ids"],ids)
        atom,alt=f["atom"][rows,pick],f["atom"][rows,other]
        amp=f["dose"][rows,pick,None]
    assert np.all(atom!=alt)
    large_decoy=dl[alt]*np.linalg.norm(dl[atom],axis=1,keepdims=True)/np.linalg.norm(dl[alt],axis=1,keepdims=True)
    moves={"small":-amp*ds[atom],"large":-amp*dl[atom],"dense":-amp*dense[atom],"decoy":-amp*large_decoy}
    for k in ("dense","decoy"):
        np.testing.assert_allclose(np.linalg.norm(moves[k],axis=1),np.linalg.norm(moves["large"],axis=1),rtol=1e-10)
    waves,linear={},{}
    for size,a in adapters.items():
        for key in (("small",) if size=="small" else ("large","dense","decoy")):
            values=[]
            for i in range(0,len(rows),16):
                ix=slice(i,i+16)
                batch=x[rows[ix]]
                clean=a.predict(batch,16,[.5])["point"]
                with addition(a,moves[key][ix]): edited=a.predict(batch,16,[.5])["point"]
                values.append(edited-clean)
            waves[key]=np.concatenate(values)
            linear[key]=np.einsum("nhd,nd->nh",jac[size][rows],moves[key])
            if key in jw: np.testing.assert_allclose(linear[key],jw[key][rows,pick],rtol=3e-5,atol=3e-6)
    scores={}
    def centered(z): return z-z.mean(-1,keepdims=True)
    def cos(a,b):
        return (a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))
    for domain in np.unique(dom):
        ix=np.flatnonzero(dom[rows]==domain)
        ref=centered(waves["small"][ix].astype(float))/x[rows[ix]].std(1)[:,None]
        out={}
        allcos={}
        for arm,wave in waves.items():
            z=centered(wave[ix].astype(float))/x[rows[ix]].std(1)[:,None]
            finite_to_linear=cos(centered(wave[ix]),centered(linear[arm][ix]))
            allcos[arm]=cos(ref,z)
            out[arm]=dict(signed_cosine=describe(allcos[arm]),finite_vs_linear_median=float(np.median(finite_to_linear)),
                shape_rms_in_history_sd=float(np.sqrt(np.mean(z**2))),source_wave_skill=float(1-((z-ref)**2).sum()/(ref**2).sum()))
        for arm in ("dense","decoy"):
            out["large_minus_"+arm]=describe(allcos["large"]-allcos[arm])
        scores[str(domain)]=out
    np.savez_compressed(OUT/f"finite_{method}_{seed}.npz",ids=ids[rows],domains=dom[rows],row=rows,atom=atom,decoy_atom=alt,
        dose=amp[:,0],history_sd=x[rows].std(1),**waves,**{"jac_"+k:v for k,v in linear.items()})
    (OUT/f"finite_{method}_{seed}.json").write_text(json.dumps(scores,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method",choices=cd.METHODS)
    parser.add_argument("--finite",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2)
    w=ordinary_map()
    adapters={size:load(size) for size in cd.SIZES} if args.finite else {}
    for seed in cd.SEEDS:
        result=jacobian(args.method,seed,w)
        if args.finite: finite(args.method,seed,result,adapters)
