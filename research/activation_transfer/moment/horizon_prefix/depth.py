"""Open access to the second masked forecast token independently over three depths."""
import argparse
from contextlib import contextmanager,ExitStack
from itertools import product
import json
import numpy as np
import torch
from research.activation_transfer.moment.head_test.capture import adapter
from research.activation_transfer.moment.horizon_identity.factorial import horizon_factors
from research.activation_transfer.moment.horizon_prefix.run import OUT,WEATHER
from research.activation_transfer.moment.input_paths.run import shapley
from research.activation_transfer.moment.forecast_quality.score import boot_weights


@contextmanager
def band_access(model,bits,attention=None):
    """Reconstruct the clean shared position bias independently at each block boundary."""
    blocks=model.module.encoder.block
    cache={}
    with ExitStack() as handles:
        for i,block in enumerate(blocks):
            module=block.layer[0].SelfAttention
            allowed=bits[i*3//len(blocks)]
            def prepare(mod,args,kwargs,idx=i,take=allowed):
                if idx==0:
                    t=args[0]
                    if t.shape[1]!=64:raise ValueError(t.shape)
                    # T5's native positional bias for an encoder-only 64-token forward.
                    b=mod.compute_bias(64,64,device=t.device,cache_position=torch.arange(64,device=t.device))
                    mask=kwargs.get("mask")
                    if mask is not None:b=b+mask
                    cache["original"]=b
                b=cache["original"]
                if not take:
                    b=b.clone()
                    b[:,:,:,-1]=torch.finfo(b.dtype).min
                new={**kwargs,"position_bias":b}
                if attention is not None:new["output_attentions"]=True
                return args,new
            pre=module.register_forward_pre_hook(prepare,with_kwargs=True)
            handles.callback(pre.remove)
            if attention is not None:
                def capture(mod,args,out,idx=i):
                    prob=out[-1].detach().cpu()
                    if prob.ndim!=4:raise ValueError(prob.shape)
                    attention[idx].append(torch.stack([prob[:,:,2:62,-1].mean((1,2)),prob[:,:,62,-1].mean(1),
                                                       prob[:,:,63,-1].mean(1)],dim=1).numpy())
                h=module.register_forward_hook(capture);handles.callback(h.remove)
        yield


def run(position,batch):
    with np.load(WEATHER/"panel.npz") as f:x=f["context"].reshape(-1,480);ids=f["ids"]
    with np.load(OUT/"noaa_large_factorial.npz") as f:
        lower=f["both8" if position=="h8" else "one_mask"][:,:8]
        upper=f["position8" if position=="h8" else "native16"][:,:8]
    model=adapter("large",0)
    result={}
    from research.activation_transfer.moment.token_paths.geometry import CACHE
    folder=CACHE/"horizon_depth"/position;folder.mkdir(exist_ok=True,parents=True)
    for bits in product((0,1),repeat=3):
        label="cell_"+"".join(map(str,bits));dest=folder/(label+".npy")
        if not dest.exists():
            ys=[]
            with horizon_factors(model,position=="h8",False),band_access(model,bits):
                for k in range(0,len(x),batch):ys.append(model.predict(x[k:k+batch],16,[.5])["point"][:,:8])
            temp=dest.with_suffix(".tmp.npy");np.save(temp,np.concatenate(ys));temp.replace(dest)
        result[label]=np.load(dest)
        if bits==(0,0,0):np.testing.assert_allclose(result[label],lower,atol=4e-5,rtol=1e-5)
        if bits==(1,1,1):np.testing.assert_allclose(result[label],upper,atol=4e-5,rtol=1e-5)
        print(position,bits,flush=True)
    result["true"]=result["cell_111"]-result["cell_000"]
    waves=shapley(result)
    stats={}
    weights=boot_weights("noaa-horizon-band",len(ids))
    ref=result["true"].reshape(len(ids),-1)
    energy=(ref**2).mean(1)
    for k,v in zip(("early","middle","late"),waves):
        v=v.reshape(len(ids),-1);dot=(ref*v).mean(1)
        share=(weights@dot)/(weights@energy)
        amplitude=np.sqrt(np.mean(v.reshape(len(ids),4,8)**2,axis=2)).mean(1)
        stats[k]=dict(share=[float(dot.sum()/energy.sum()),*np.quantile(share,[.025,.975]).tolist()],
                      mean_RMS_C=[float(amplitude.mean()),*np.quantile(weights@amplitude,[.025,.975]).tolist()])
    att=[[] for _ in model.module.encoder.block]
    ix=np.arange(0,len(x),7)
    with horizon_factors(model,position=="h8",False),band_access(model,(1,1,1),att):
        y=model.predict(x[ix],16,[.5])["point"][:,:8]
        np.testing.assert_allclose(y,upper[ix],atol=3e-5,rtol=3e-5)
    result["attention_mass"]=np.stack([np.concatenate(v) for v in att]);result["attention_rows"]=ix
    stats["ordinary_attention_sample"]=len(ix)
    for i,k in enumerate(("early","middle","late")):
        mass=np.array_split(result["attention_mass"],3)[i].mean(axis=(0,1))
        stats[k]["attention_to_extra_observed_first_future_second_future"]=mass.tolist()
    stats["lower_upper_replay_max_C"]=[float(np.max(abs(result["cell_000"]-lower))),float(np.max(abs(result["cell_111"]-upper)))]
    np.savez_compressed(OUT/f"noaa_large_layers_{position}.npz",ids=ids,**result)
    (OUT/f"noaa_large_layers_{position}.json").write_text(json.dumps(stats,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("position",choices=("h8","h16"))
    p.add_argument("--batch",type=int,default=16);a=p.parse_args();torch.set_num_threads(2);run(a.position,a.batch)
