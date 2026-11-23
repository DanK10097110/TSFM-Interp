"""Fit reusable source steering requests and replay them across three actual checkpoints."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from research.activation_transfer.steering.study import DATA_ROOT, ROOT, learn_directions, load_histories, patch_effect, sites_and_states
from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.models import build_adapter
from tsfm_lens.steering import SteeringModel, solve_direction

MODELS = {
    "bolt": ("chronos_bolt", "amazon/chronos-bolt-small", "772f3d25d38aec6d914c8949dab4462e2d46f5d8"),
    "timesfm": ("timesfm", "google/timesfm-2.5-200m-pytorch", "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"),
    "timer": ("generic_hf", "thuml/timer-base-84m", "70077a71acce1b4c00d98332fcaabc694255d8e5"),
}
CUES = ["level","tilt","sine"]
DEPTHS = [.5,.75]
N_ALIGN, N_FIT, N_CAL, N_EVAL = 640,64,32,128
ALIGN = slice(0,N_ALIGN)
FIT = slice(N_ALIGN,N_ALIGN+N_FIT)
CAL = slice(FIT.stop,FIT.stop+N_CAL)
TEST = slice(CAL.stop,None)
RANK=24


def desired():
    """Each output request has equal energy on the eight-step horizon."""
    t = np.arange(8)
    shapes = np.stack([np.ones(8),np.linspace(-1,1,8),np.sin(2*np.pi*t/8)])
    return .05 * shapes / np.sqrt(np.mean(shapes**2,axis=1,keepdims=True))


def adapter_for(name):
    kind,repo,revision=MODELS[name]
    path=Path.home()/".cache/huggingface/hub"/("models--"+repo.replace("/","--"))/"snapshots"/revision
    extra={"auto_class":"AutoModelForCausalLM"} if name=="timer" else {}
    cfg=ModelConfig(name=name,adapter=kind,checkpoint=str(path),kwargs=extra,batch_size=128)
    adapter=build_adapter(cfg,DataConfig(context_len=192,horizon=8),torch.device("cpu"),torch.float32)
    adapter.ensure_loaded()
    return adapter


def data(root):
    seen=set()
    did,daily=load_histories("daily",CAL.stop+N_EVAL,202611205,root,context=192,dedupe=seen)
    wid,weekly=load_histories("weekly",N_EVAL,202611206,root,context=192,dedupe=seen)
    return np.r_[np.char.add("daily/",did), np.char.add("weekly/",wid)],np.vstack([daily,weekly])


def fit_policy(model,site,states,inputs,batch):
    """Fit a common additive edit from finite differences across the source calibration rows."""
    base=states[ALIGN].astype(float)
    _,_,basis=np.linalg.svd(base-base.mean(0),full_matrices=False)
    basis=basis[:RANK]
    radius=float(np.median(np.linalg.norm(base,axis=1)))
    step=.01*radius
    effects=patch_effect(model,site,inputs[FIT],np.linalg.norm(states[FIT],axis=1),basis*step,batch)
    jac=(effects/step).transpose(0,2,1).reshape(-1,RANK)
    penalty=.01*np.linalg.svd(jac,compute_uv=False)[0]**2
    if penalty <= 0:
        raise ValueError("No measurable response")
    solutions=[solve_direction(jac,np.tile(wave,N_FIT),ridge=penalty,max_norm=.20*radius) for wave in desired()]
    return np.stack([sol.delta @ basis for sol in solutions]),dict(
        radius=radius,step=step,ridge=float(penalty),coef_norm=[float(np.linalg.norm(sol.delta)) for sol in solutions],
        clipped=[sol.clipped for sol in solutions])


def collect(cache,root,batch):
    ids,x=data(root)
    for name in MODELS:
        path=cache/(name+"_fit.npz")
        if path.exists():
            with np.load(path) as archive:
                np.testing.assert_array_equal(archive["ids"],ids)
            continue
        adapter=adapter_for(name)
        model=SteeringModel.from_adapter(adapter,quantiles=[.5])
        all_layers=adapter.all_layer_names()
        chosen=[all_layers[int(f*len(all_layers))] for f in DEPTHS]
        sites,values=sites_and_states(model,chosen,x,batch)
        record={"checkpoint":MODELS[name],"site":[],"fit":[]}
        arrays={}
        for i,layer in enumerate(chosen):
            site,states=sites[layer],values[layer]
            controls,meta=fit_policy(model,site,states,x,batch)
            arrays["s"+str(i)]=states
            arrays["d"+str(i)]=controls
            record["site"].append(asdict(site))
            record["fit"].append(meta)
            print("fit",name,site.module,flush=True)
        np.savez_compressed(path,ids=ids,metadata=json.dumps(record),**arrays)


def live_effect(model, site, x, norms, directions, batch):
    """Positive edit minus ordinary forecast, including nonlinear and asymmetric responses."""
    ordinary = np.concatenate([model.forecast(x[i:i+batch],8)[...,0] for i in range(0,len(x),batch)])
    predictions = []
    k = len(directions)
    for i in range(0,len(x)*k,batch):
        flat = np.arange(i,min(i+batch,len(x)*k))
        row, arm = flat//k, flat%k
        coef = np.zeros((len(flat),k))
        coef[np.arange(len(row)),arm] = 1 / np.maximum(norms[row],1e-6)
        predictions.append(model.evaluate(x[row],8,site,directions,coef)[...,0])
    return np.concatenate(predictions).reshape(len(x),k,8) - ordinary[:,None,:]


def scale_factors(effects,requests):
    """One nonnegative scalar per waveform estimated on paired calibration items."""
    return np.clip(np.einsum("nth,th->t",effects,requests)/
        np.maximum(np.sum(effects**2,axis=(0,2)),1e-15),0,8)


def apply(target,cache,out,root,batch):
    from tsfm_lens.steering.model import InterventionSite
    ids,x=data(root)
    fits={name:np.load(cache/(name+"_fit.npz")) for name in MODELS}
    adapter=adapter_for(target)
    model=SteeringModel.from_adapter(adapter,quantiles=[.5])
    meta=json.loads(str(fits[target]["metadata"]))
    all_meta={name:json.loads(str(z["metadata"])) for name,z in fits.items()}
    for archive in fits.values():
        np.testing.assert_array_equal(archive["ids"],ids)
    rng=np.random.default_rng(202611207)
    perms=[rng.permutation(N_ALIGN) for _ in range(4)]
    effects={}
    factors={}
    for i,depth in enumerate(DEPTHS):
        site=InterventionSite(**meta["site"][i])
        target_states=fits[target]["s"+str(i)]
        norms=np.linalg.norm(target_states,axis=1)
        native=fits[target]["d"+str(i)]
        native_effect=live_effect(model,site,x[TEST],norms[TEST],native,batch)
        np.testing.assert_allclose(native_effect[:1],live_effect(model,site,x[TEST][:1],norms[TEST][:1],native,1),rtol=2e-3,atol=3e-4)
        effects[f"{i}_native"]=native_effect
        for source,archive in fits.items():
            if source==target:
                continue
            key=f"{i}_{source}"
            dirs,shams=learn_directions(archive["s"+str(i)][ALIGN],target_states[ALIGN],archive["d"+str(i)],perms)
            cal=live_effect(model,site,x[CAL],norms[CAL],dirs,batch)
            scalars=scale_factors(cal,desired())
            factors[key]=scalars.tolist()
            # Per cue: zero-shot, dose-calibrated, then each norm-matched pairing sham.
            actions=np.concatenate([dirs[:,None],(scalars[:,None]*dirs)[:,None],shams],axis=1)
            output=live_effect(model,site,x[TEST],norms[TEST],actions.reshape(-1,actions.shape[-1]),batch)
            effects[key]=output.reshape(len(x[TEST]),len(CUES),actions.shape[1],8)
            print("patched",source,target,depth,"calibrated doses",scalars,flush=True)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/(target+".npz"),**{k:v.astype("f4") for k,v in effects.items()},
                        ids=ids[TEST],desired=desired())
    info=dict(target=target,source_fits=all_meta,applied_dose=factors,
              calibration_ids=ids[CAL].tolist(),clean_fit_ids=ids[ALIGN].tolist(),source_fit_ids=ids[FIT].tolist(),
              test_ids=ids[TEST].tolist(),depths=DEPTHS,requests=CUES)
    (out/(target+".json")).write_text(json.dumps(info,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["fit",*MODELS])
    parser.add_argument("--cache",type=Path,default=Path("/tmp/tsfm_forecast_requests_20261120"))
    parser.add_argument("--out",type=Path,default=HERE/"results")
    parser.add_argument("--data-root",type=Path,default=DATA_ROOT)
    parser.add_argument("--batch-size",type=int,default=32)
    a=parser.parse_args()
    a.cache.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if a.stage=="fit":
        collect(a.cache,a.data_root,a.batch_size)
    else:
        apply(a.stage,a.cache,a.out,a.data_root,a.batch_size)
