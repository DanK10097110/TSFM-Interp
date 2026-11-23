"""Apply the archived eight-lead lift policies and measure actual 32-lead forecasts."""
import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.forecast_requests.run import ALIGN, DATA_ROOT, DEPTHS, HERE, MODELS, N_ALIGN, TEST, adapter_for, data, learn_directions
from tsfm_lens.steering import SteeringModel
from tsfm_lens.steering.model import InterventionSite, _get

OUT = Path(__file__).resolve().parent / "results"
AMP = .05
TOL = 3e-4


class Monitor:
    """Verify invocation count, incoming hidden geometry and batch size on every call."""
    def __init__(self, model, sites):
        self.model = model
        self.sites = sites
        self.calls = Counter()
        self.traces = {}

    def install(self, stack):
        for site in self.sites:
            def observe(module, args, kwargs, site=site):
                state = _get(dict(args=args, kwargs=kwargs),site.path)
                expected_shape = (self.batch,*site.shape)
                if tuple(state.shape) != expected_shape:
                    raise RuntimeError(f"{site.module}: got {tuple(state.shape)}, expected {expected_shape}")
                self.traces[site.module].append(state[:,-1].detach().numpy().copy() if self.retain else None)
            handle = self.model.module.get_submodule(site.module).register_forward_pre_hook(observe,with_kwargs=True)
            stack.callback(handle.remove)

    def call(self, x, h, policy=None, retain=False):
        self.batch = len(x)
        self.traces = {s.module:[] for s in self.sites}
        self.retain = retain
        kind = "clean" if policy is None else "edited"
        if policy is None:
            y = self.model.forecast(x,h)
        else:
            site, axes, coeff = policy
            y = self.model.evaluate(x,h,site,axes,coeff)
        for layer,observed in self.traces.items():
            if len(observed)!=1:
                raise RuntimeError(f"{layer}: {len(observed)} live invocations at requested horizon {h}")
            self.calls[(h,kind,layer)] += 1
        return y[...,0]


def select(ids):
    return np.concatenate([np.flatnonzero(np.char.startswith(ids,d+"/"))[:64] for d in ("daily","weekly")])


def directions(target, i, fits, archived, sham_archive):
    ts = fits[target]["s"+str(i)]
    rng = np.random.default_rng(202611207)
    perms = [rng.permutation(N_ALIGN) for _ in range(4)]
    axes = [fits[target]["d"+str(i)][0]]
    labels = ["native"]
    for source in MODELS:
        if source==target:
            continue
        a = fits[source]
        mapped,shuffled = learn_directions(a["s"+str(i)][ALIGN],ts[ALIGN],a["d"+str(i)],perms)
        key = f"{i}_{source}"
        axes.append(mapped[0]*archived["applied_dose"][key][0])
        labels.append(source)
        for j,scale in enumerate(sham_archive[key+"_scales"][0]):
            axes.append(shuffled[0,j]*scale)
            labels.append(f"{source}_shuffle{j}")
    return labels,np.stack(axes)


def edited(monitor,x,norm,site,axes,batch=32):
    forecast = []
    for start in range(0,len(x)*len(axes),batch):
        flat = np.arange(start,min(start+batch,len(x)*len(axes)))
        row,col = flat//len(axes),flat%len(axes)
        coeff = np.zeros((len(flat),len(axes)))
        coeff[np.arange(len(flat)),col]=1/np.maximum(norm[row],1e-6)
        forecast.append(monitor.call(x[row],32,(site,axes,coeff)))
    return np.concatenate(forecast).reshape(len(x),len(axes),32)


def run(target,cache,old,out):
    ids,all_x=data(DATA_ROOT)
    with ExitStack() as stack:
        fits={k:stack.enter_context(np.load(cache/(k+"_fit.npz"))) for k in MODELS}
        sham=stack.enter_context(np.load(old/("dose_shams_"+target+".npz")))
        original=stack.enter_context(np.load(old/(target+".npz")))
        applied=json.loads((old/(target+".json")).read_text())
        for archive in fits.values():
            np.testing.assert_array_equal(archive["ids"],ids)
        np.testing.assert_array_equal(original["ids"],ids[TEST])
        np.testing.assert_array_equal(sham["ids"],ids[TEST])
        idx=select(ids[TEST])
        if len(idx)!=128:
            raise ValueError("expected 64 daily and 64 weekly original held-out histories")
        selected=ids[TEST][idx]
        x=all_x[TEST][idx]
        meta=json.loads(str(fits[target]["metadata"]))
        sites=[InterventionSite(**s) for s in meta["site"]]
        model=SteeringModel.from_adapter(adapter_for(target),quantiles=[.5])
        monitor=Monitor(model,sites)
        monitor.install(stack)
        sample=np.array([0,1,64,65])
        p8=monitor.call(x[sample],8,retain=True)
        h8={k:v[0] for k,v in monitor.traces.items()}
        p32=monitor.call(x[sample],32,retain=True)
        hidden_diffs={k:float(np.abs(v[0]-h8[k]).max()) for k,v in monitor.traces.items()}
        for i,s in enumerate(sites):
            np.testing.assert_allclose(h8[s.module],fits[target]["s"+str(i)][TEST][idx][sample],atol=TOL,rtol=1e-4)
            np.testing.assert_allclose(monitor.traces[s.module][0],h8[s.module],atol=TOL,rtol=1e-4)
        np.testing.assert_allclose(p32[:,:8],p8,atol=TOL,rtol=0)
        baseline=np.concatenate([monitor.call(x[j:j+32],32) for j in range(0,len(x),32)])
        info={"target":target,"model":MODELS[target],"sites":[dict(layer=s.module,shape=s.shape,mode=s.mode) for s in sites],
              "early_clean_forecast_max_abs":float(np.abs(p32[:,:8]-p8).max()),"hidden_8_to_32_max_abs":hidden_diffs,
              "selected_index_sha256":sha256(idx.tobytes()).hexdigest(),"sites_checks":{}}
        saved={"ids":selected,"context_sha256":sha256(x.tobytes()).hexdigest()}
        for i,(depth,site) in enumerate(zip(DEPTHS,sites)):
            layers,axes=directions(target,i,fits,applied,sham)
            norms=np.linalg.norm(fits[target]["s"+str(i)][TEST][idx],axis=1)
            effect=edited(monitor,x,norms,site,axes)-baseline[:,None,:]
            refs=[original[f"{i}_native"][idx,0]]
            for source in MODELS:
                if source==target:
                    continue
                key=f"{i}_{source}"
                refs.append(original[key][idx,0,1])
                refs.extend(sham[key][idx,0,j] for j in range(4))
            ref=np.stack(refs,axis=1)
            diff=effect[:,:,:8]-ref
            max_abs=np.max(np.abs(diff),axis=(0,2))
            np.testing.assert_allclose(effect[:,:,:8],ref,atol=TOL,rtol=0)
            score=lambda v:1-((v-AMP)**2).mean(-1)/AMP**2
            delta=score(effect[:,:,:8])-score(ref.astype(float))
            # Independently exercise a real patched 8-lead forward at each site.
            coeff=np.zeros((len(sample),len(axes)))
            coeff[:,0]=1/np.maximum(norms[sample],1e-6)
            native8=monitor.call(x[sample],8,(site,axes,coeff))-p8
            np.testing.assert_allclose(native8,effect[sample,0,:8],atol=TOL,rtol=0)
            info["sites_checks"][str(depth)]={"early_max_abs":dict(zip(layers,max_abs.tolist())),
              "early_max_item_skill_abs":dict(zip(layers,np.abs(delta).max(0).tolist())),
              "early_max_mean_skill_abs":dict(zip(layers,np.abs(delta.mean(0)).tolist())),
              "native_8_to_32_max_abs":float(np.abs(native8-effect[sample,0,:8]).max())}
            saved[str(i)+"_arms"]=layers
            saved[str(i)+"_effect"]=effect.astype("f4")
            print(target,depth,"prefix max error",max_abs.max(),flush=True)
        info["monitored_calls"]={f"{h}_{kind}_{layer}":n for (h,kind,layer),n in monitor.calls.items()}
        info["multiplicity"]=1
        out.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(out/(target+".npz"),**saved)
        (out/(target+"_replay.json")).write_text(json.dumps(info,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("target",choices=MODELS)
    p.add_argument("--cache",type=Path,default=Path("/tmp/tsfm_forecast_requests_20261120"))
    p.add_argument("--original",type=Path,default=HERE/"results")
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    torch.set_num_threads(2)
    run(a.target,a.cache,a.original,a.out)
