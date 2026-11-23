"""Keep the edited patch identical while testing whether older traffic rhythms affect direction transfer."""
import argparse
import json
import numpy as np
import torch

from research.activation_transfer.moment.crosscoder.cross_family_directions import data as d
from research.activation_transfer.moment.crosscoder.cross_family_directions.directions import agreements,native_directions
from research.activation_transfer.moment.crosscoder.patch_probe import crossed
from research.activation_transfer.shape_transfer.physical_ramp.study import true_captures
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from tsfm_lens.extraction.hooks import token_patch

OUT=d.OUT/"traffic_phase"


def variants():
    _,_,x,ids,domains=d.histories()
    rows=np.flatnonzero(domains=="monash_traffic");x=x[rows]
    generator=np.random.default_rng(22112501)
    whole=x.copy();split=x.copy()
    for i,z in enumerate(x):
        blocks=z[16:-32].reshape(18,24)[generator.permutation(18)]
        whole[i,16:-32]=blocks.ravel()
        shifts=generator.integers(1,24,len(blocks))
        split[i,16:-32]=np.concatenate([np.roll(b,s) for b,s in zip(blocks,shifts)])
    views=dict(clean=x,phase_preserved=whole,phase_scrambled=split)
    for transformed in views.values():
        np.testing.assert_array_equal(x[:,-32:],transformed[:,-32:])
        np.testing.assert_array_equal(x[:,:16],transformed[:,:16])
        np.testing.assert_array_equal(np.sort(x,axis=1),np.sort(transformed,axis=1))
    return views,rows,ids[rows]


def corr(a,b):
    a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
    return (a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))


def capture(model):
    views,rows,ids=variants();a=d.adapter_for(model);site=d.MODELS[model][-1]
    pos=d.groups(a.token_time_spans())[-1];sel=d.last_window_slice(a,pos)
    OUT.mkdir(parents=True,exist_ok=True)
    for label,x in views.items():
        if label=="clean":continue
        h,base=true_captures(a,site,x,pos,16)
        jac=[]
        for lo in range(0,len(x),8):
            state=torch.from_numpy(h[lo:lo+8].copy())
            move=torch.zeros_like(state[:,0],requires_grad=True)
            with torch.enable_grad(),token_patch(a.module,site,sel,state+move[:,None,:]):
                y=a.predict_differentiable(x[lo:lo+8],16)
                grads=[torch.autograd.grad(y[:,lead].sum(),move,retain_graph=lead!=15)[0] for lead in range(16)]
            np.testing.assert_allclose(y.detach().numpy(),base[lo:lo+8],atol=2e-4,rtol=2e-4)
            jac.append(torch.stack(grads,dim=1).detach().numpy())
        jac=np.concatenate(jac)
        # Symmetric finite perturbation of a random write double-checks the altered-context Jacobian.
        pick=np.arange(0,len(x),16)
        v=np.random.default_rng(22112502).normal(size=(len(pick),jac.shape[2])).astype("f4")
        v*=.0005*np.median(np.linalg.norm(h.mean(1),axis=1))/np.linalg.norm(v,axis=1,keepdims=True)
        hi,lo=[forecast16(a,site,x[pick],h[pick],sign*v,sel,8) for sign in (1,-1)]
        true=(hi-lo)/2;est=np.einsum("nhd,nd->nh",jac[pick],v)
        rel=np.linalg.norm(est-true)/np.linalg.norm(true)
        assert rel<.1,rel
        np.savez_compressed(d.CACHE/f"{model}_{label}_phase_regular.npz",ids=ids,jac=jac)
        np.savez_compressed(OUT/f"{model}_{label}.npz",ids=ids,base=base,source_sha=d.fingerprint(x),
                          audit_relative=rel,actual=true,linear=est)


def run():
    views,rows,ids=variants();meta={"lag24_mean":{}, "forecast_change_rms":{}, "results":{}}
    for k,values in views.items():
        v=values[:,16:-32].astype(float);meta["lag24_mean"][k]=float(corr(v[:,24:],v[:,:-24]).mean())
    jac={}
    for model in d.VIEWS:
        jac[model]={"clean":np.load(d.CACHE/(model+"_jac.npy"))[rows].astype(float)}
        with np.load(d.OUT/f"{model}_alignment.npz") as f: clean=f["ordinary"][rows]
        for lab in list(views)[1:]:
            with np.load(d.CACHE/f"{model}_{lab}_phase_regular.npz") as f:
                np.testing.assert_array_equal(f["ids"],ids);jac[model][lab]=f["jac"].astype(float)
            with np.load(OUT/f"{model}_{lab}.npz") as f:
                assert str(f["source_sha"])==d.fingerprint(views[lab])
                delta=(f["base"]-clean)/views["clean"].std(1,keepdims=True)
                meta["forecast_change_rms"][model+"_"+lab]=float(np.sqrt(np.mean(delta**2)))
    for method in d.METHODS:
        for seed in d.SEEDS:
            name=f"{method}_{seed}"
            with np.load(d.OUT/f"{name}_dict.npz") as f:
                atom=f["atom"][rows];active=f["dose"][rows]>0;assert np.all(active)
                ds,dt=native_directions(f)
            results={};energy=None
            for lab in views:
                s=ds[atom]@jac["timesfm"][lab].transpose(0,2,1)
                t=dt[atom]@jac["bolt"][lab].transpose(0,2,1)
                if lab=="clean":energy=np.linalg.norm(s-s.mean(-1,keepdims=True),axis=-1)
                same,other=agreements(s,t,active)
                results[lab]=same-other
            with np.load(d.OUT/f"{name}_directions.npz") as f:
                np.testing.assert_allclose(results["clean"],f["forecast16_adv"][rows],rtol=1e-7,atol=1e-7)
            results["phase_preserved_minus_scrambled"]=results["phase_preserved"]-results["phase_scrambled"]
            results["clean_minus_phase_preserved"]=results["clean"]-results["phase_preserved"]
            report=crossed(results,atom,None,False)
            use=energy>np.median(energy)
            delta=results["phase_preserved_minus_scrambled"]
            report["high_source_response_difference"]=float(delta[use].mean())
            meta["results"][name]=report
            np.savez_compressed(OUT/f"{name}.npz",ids=ids,atom=atom,**results)
    (OUT/"summary.json").write_text(json.dumps(meta,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=(*d.VIEWS,"score"))
    args=p.parse_args();torch.set_num_threads(2)
    if args.stage=="score":run()
    else:capture(args.stage)
