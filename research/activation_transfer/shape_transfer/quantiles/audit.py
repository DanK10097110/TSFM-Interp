"""Independent raw-quantile, codec, physical intervention, and whole-cohort metric replay."""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor,cho_solve,eigh
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.replication.data import CACHE as OLD, HERE as REP
from research.activation_transfer.shape_transfer.fresh_cohort.data import CACHE as NEW
from research.activation_transfer.shape_transfer.full_space.audit import direct_scores,hook
from tsfm_lens.sae.train import load_sae_checkpoint

HERE=Path(__file__).resolve().parent
CACHE=Path("/tmp/tsfm_quantiles_20261121")
C=np.array([[0.,1,0],[-.5,0,.5],[.5,-1,.5]])
LEVELS=[.1,.5,.9]
P=np.ones((8,8))/8
ARMS=("source","median_only","joint_local","wrong_width","joint_fixed")


def get_data():
    with np.load(OLD/"data.npz") as a,np.load(NEW/"data.npz") as b:
        old=a["split"]!="train"
        return np.r_[a["ids"][old],b["ids"]],np.concatenate((a["x"][old],b["x"])),np.r_[
            a["split"][old],np.char.add("extra_",b["split"])]


def weights(cal):
    # Apply an explicit 3×3 codec matrix to each eight-lead Jacobian.
    ch=np.einsum("cq,nhqd->nchd",C,cal,optimize=True)
    avg=ch.mean(2,keepdims=True)
    w=np.sqrt(np.mean(avg**2,axis=(0,2,3))/np.mean((ch-avg)**2,axis=(0,2,3)))
    rms=np.sqrt(np.mean((avg+w[None,:,None,None]*(ch-avg))**2,axis=(0,2,3)))
    return w,rms[0]/rms


def weighted(j,w,g):
    # j has a leading case dimension, followed by lead, raw channel, hidden.
    vals=[]
    for channel in range(3):
        v=np.einsum("q,nhqd->nhd",C[channel],j,optimize=True)
        vals.append(g[channel]*((P+w[channel]*(np.eye(8)-P)) @ v))
    return np.concatenate(vals,axis=1)


def inverse(j):
    u,s,v=np.linalg.svd(j,full_matrices=False)
    return (v.T*(s/(s**2+.001*s[0]**2)))@u.T


def fixed_inverse(j):
    # Independent primal ridge, computed once for the target, rather than solving in calibration-row space.
    n,h,d=j.shape
    design=j.reshape(n*h,d)
    gram=design.T@design
    lam=.001*eigh(gram,subset_by_index=[d-1,d-1],eigvals_only=True)[0]
    return cho_solve(cho_factor(gram+lam*np.eye(d)),j.sum(axis=0).T)


def forward(enc,name,x,v=None,capture=None):
    if v is None and capture is None:
        return enc.predict(x,8,LEVELS)["quantiles"][0].astype(float)
    width=512 if name=="bolt" else 1280
    delta=torch.zeros((1,width)) if v is None else torch.from_numpy(v[None].astype("f4"))
    with hook(enc,name,delta,capture):
        return enc.predict(x,8,LEVELS)["quantiles"][0].astype(float)


def audit_metrics(archive,report):
    errors=[];cross_err=[];out={}
    inverse_codec=np.linalg.inv(C)
    for domain in np.unique(archive["split"]):
        chosen=archive["split"]==domain
        eligible=archive["active"][chosen].sum(0)>=8
        mask=archive["active"][chosen][:,eligible]
        dr=report["domains"][str(domain)]
        expect=archive["atoms"][eligible].tolist()
        assert expect==dr["features"]
        selected=lambda x:x[chosen][:,eligible]
        truth=selected(archive["source"]).astype(float)@C.T
        np.testing.assert_allclose(truth@inverse_codec.T,selected(archive["source"]),atol=5e-15)
        outputs={k:selected(archive[k]).astype(float)@C.T for k in ARMS[1:]}
        outputs["donor_linear"]=selected(archive["source_linear"]).astype(float)
        outputs["repeat_observed_donor_median"]=np.repeat(selected(archive["source"])[...,1,None].astype(float),3,axis=-1)@C.T
        energy=np.mean(truth[mask]**2,axis=(0,1))
        metrics_sample={}
        for h,component in enumerate(("median","halfwidth","asymmetry")):
            for cent in (False,True):
                term=component+("_shape" if cent else "_full")
                ref=truth[...,h]
                if cent:ref=ref-ref.mean(-1,keepdims=True)
                for k,p in outputs.items():
                    y=p[...,h]
                    if cent:y=y-y.mean(-1,keepdims=True)
                    calc=direct_scores(ref,y,mask,bootstrap=k=="joint_local")
                    entry=dr["scores"][term][k]
                    assert calc["n_scored"]==entry["measurable"]
                    errors.append(abs(calc["skill"]-entry["skill"]["value"]))
                    if k=="joint_local":
                        errors.extend(abs(np.array(calc["ci95"])-entry["skill"]["ci95"]))
                # A translated triplet preserves both interval coordinates exactly.
                same=np.repeat(ref[...,None],3,axis=-1)@C.T
                np.testing.assert_array_equal(same[...,1:],0.)
                metrics_sample[term]=dict(median_only=dr["scores"][term]["median_only"]["skill"]["value"],
                                          joint_local=dr["scores"][term]["joint_local"]["skill"]["value"])
        crossing={}
        for key,prefix in (("source","donor"),("joint_local","target_joint"),("median_only","target_median")):
            base=archive[("donor" if key=="source" else "target")+"_baseline"][chosen]
            a,b=np.nonzero(mask)
            q0=base[a]
            q1=q0+selected(archive[key])[a,b]
            row={}
            for label,vals in (("clean",q0),("edited",q1)):
                adjacent=vals[...,:2]>vals[...,1:]
                row[label]=dict(any_rate=float(np.any(adjacent,axis=-1).mean()),
                    lower_above_median=int(adjacent[...,0].sum()),median_above_upper=int(adjacent[...,1].sum()),
                    lower_above_upper=int((vals[...,0]>vals[...,2]).sum()),denominator=int(np.prod(vals.shape[:2])))
                cross_err.append(abs(row[label]["any_rate"]-dr[prefix+"_"+label+"_crossing_rate"]))
            crossing[key]=row
        out[str(domain)]=dict(active=int(mask.sum()),n_atoms=eligible.sum().item(),rms_all_firings=np.sqrt(energy).tolist(),
                              selected_scores=metrics_sample,crossings=crossing)
    if max(errors)>5e-9 or max(cross_err)>1e-12:raise ValueError("Summary/scoring or crossing mismatch")
    return dict(max_score_or_CI_error=max(errors),max_crossing_rate_error=max(cross_err),domains=out)


def run(source,enc):
    target=next(k for k in MODELS if k!=source)
    ids,x,labels=get_data();cal=labels=="cal";ev=~cal
    bound={}
    js={}
    for name in (source,target):
        with np.load(CACHE/(name+".npz")) as f:
            np.testing.assert_array_equal(ids,f["ids"]);np.testing.assert_array_equal(labels,f["split"])
            js[name]=f["jac"].astype(float);bound[name]=float(f["radius"])
    with np.load(HERE/"results"/(source+".npz")) as f: a=dict(f)
    np.testing.assert_array_equal(a["ids"],ids[ev]);np.testing.assert_array_equal(a["split"],labels[ev])
    report=json.loads((HERE/"results/summary.json").read_text())[source]
    summary=audit_metrics(a,report)
    w,g=weights(js[source][cal])
    np.testing.assert_allclose(w,a["balance"],rtol=2e-10);np.testing.assert_allclose(g,a["gains"],rtol=2e-10)
    library=fixed_inverse(weighted(js[target][cal],w,g))
    liberr=np.linalg.norm(library-a["library"])/np.linalg.norm(library)
    assert liberr<3e-9
    sae=load_sae_checkpoint(str(dictionary_path(source)))
    codes_ref=[]
    with np.load(REP/"results"/(source+"_seed0.npz")) as old:
        codes_ref.extend(old["source_codes"][old["split"]!="cal"])
    from research.activation_transfer.shape_transfer.fresh_cohort.data import OUT
    with np.load(OUT/(source+"_native.npz")) as new: codes_ref.extend(new["codes"])
    codes_ref=np.array(codes_ref)
    active=a["active"]
    rng=np.random.default_rng(20261121509)
    results=[]
    for domain in ("daily","machines","extra_daily","extra_monash_traffic"):
        local=a["split"]==domain
        eligible=active[local].sum(0)>=8
        candidates=np.argwhere(active & local[:,None] & eligible)
        if not len(candidates):continue
        ri,ai=candidates[rng.integers(len(candidates))]
        i=int(np.flatnonzero(ev)[ri])
        atom=int(a["atoms"][ai]);obs=x[i:i+1]
        dec=sae.W_dec[atom].detach().numpy().astype(float)
        q0={};cur={}
        for n in (source,target):
            cap=[]
            q0[n]=forward(enc[n],n,obs,None,cap)
            # Independent one-at-a-time requested native levels verify stacking and lead order.
            singles=np.stack([enc[n].predict(obs,8,[q])["quantiles"][0,:,0] for q in LEVELS],axis=-1)
            np.testing.assert_allclose(q0[n],singles,atol=1e-5,rtol=1e-5)
            cur[n]=cap[0][0].mean(0)
        z=float(sae.encode(torch.from_numpy(cur[source][None])).detach().numpy()[0,atom])
        np.testing.assert_allclose(z,codes_ref[ri,ai],atol=2e-4,rtol=8e-5);assert z>0
        sj,tj=js[source][i],js[target][i]
        demand=weighted(sj[None],w,g)[0]@dec
        one=weighted(tj[None],w,g)[0]
        right=weighted(tj[:,::-1,:][None],w,g)[0]  # swap actual raw tail channels before computing the codec
        med=(P+w[0]*(np.eye(8)-P))
        dirs=dict(source=dec,joint_local=inverse(one)@demand,
                  joint_fixed=library@demand,wrong_width=inverse(right)@demand,
                  median_only=inverse(med@tj[:,1,:])@(med@sj[:,1,:]@dec))
        measured={}
        for key,direction in dirs.items():
            n=source if key=="source" else target
            v=-z*direction
            frac=np.linalg.norm(v)/bound[n]
            if key!="source":
                np.testing.assert_allclose(frac,a[key+"_raw_fraction"][ri,ai],atol=3e-5,rtol=1e-4)
                if frac>.2:v=v*(.2/frac)
                assert (frac>.2)==bool(a[key+"_clipped"][ri,ai])
            raw=forward(enc[n],n,obs,v)-q0[n]
            expected=a[key][ri,ai]
            err=np.max(np.abs(raw-expected))
            np.testing.assert_allclose(raw,expected,atol=2e-5,rtol=2e-4)
            measured[key]=dict(max_raw_error=float(err),
                               max_component_error=float(np.max(np.abs(raw@C.T-expected@C.T))),
                               raw_norm_fraction=float(frac))
        derivatives={};larger={}
        for n in (source,target):
            u=rng.normal(size=js[n].shape[-1])
            u*=.002*bound[n]/np.linalg.norm(u)
            fd=(forward(enc[n],n,obs,u)-forward(enc[n],n,obs,-u))/2 @ C.T
            pred=(js[n][i]@u)@C.T
            rel=np.linalg.norm(pred-fd,axis=0)/np.maximum(np.linalg.norm(fd,axis=0),1e-12)
            if np.max(rel)>.12:raise ValueError(f"Independent {n} {domain} projected FD {rel}")
            derivatives[n]=rel.tolist()
            fd_large=(forward(enc[n],n,obs,5*u)-forward(enc[n],n,obs,-5*u))/10 @ C.T
            larger[n]=(np.linalg.norm(pred-fd_large,axis=0)/np.maximum(np.linalg.norm(fd_large,axis=0),1e-12)).tolist()
        results.append(dict(domain=domain,item=str(ids[i]),atom=atom,source_code=z,patches=measured,
                            directional_fd_relative_median_width_asymmetry=derivatives,
                            directional_fd_relative_at_one_percent_radius=larger))
    return dict(primal_fixed_inverse_relative=float(liberr),population_scores=summary,replayed=results)


def main():
    enc={k:adapter_for(k) for k in MODELS}
    results={n:run(n,enc) for n in MODELS}
    results["semantics"]=dict(levels=LEVELS,codec=C.tolist(),inverse=np.linalg.inv(C).tolist(),
          crossings="Active case × forecast lead. Either q10 > q50 or q50 > q90 (strict). Each item occurs once per firing atom.",
          interpretation="Half-width and asymmetry are signed changes of the native output quantiles. Asymmetry measures median displacement relative to the interval midpoint, with positive value when midpoint exceeds median. They do not assess distribution calibration, distributional skewness, or forecast coverage.")
    (HERE/"results/audit.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    torch.set_num_threads(1)
    main()
