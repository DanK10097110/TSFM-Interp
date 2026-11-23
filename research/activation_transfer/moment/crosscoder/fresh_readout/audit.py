"""Independent full-call replay with native rolled mask and direct flat readout slicing."""
import json
from contextlib import contextmanager
from hashlib import sha256, file_digest

import numpy as np
import torch

from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.directions import ordinary_map
from research.activation_transfer.moment.crosscoder.evaluate import write_json
from research.activation_transfer.moment.crosscoder.fresh_readout.panel import OUT
from research.activation_transfer.moment.token_paths.geometry import ROOT, load


@contextmanager
def native_hook(block,changes,states):
    def visit(module,args,out):
        states.append(out[0][:,61].detach().numpy().copy())
        if changes is None:return out
        # A replacement tensor with one slice edited, registered directly on the native block.
        edited=out[0].clone()
        edited[:,61,:]+=torch.tensor(changes,dtype=edited.dtype)
        return (edited,)+out[1:]
    handle=block.register_forward_hook(visit)
    try:yield
    finally:handle.remove()


@torch.inference_mode()
def forward(a,x,move=None):
    n=len(x);past,mask=a.prepare(x)
    unrolled=past[:,0].detach().numpy()
    np.testing.assert_array_equal(unrolled[:,-480:],x)
    np.testing.assert_array_equal(torch.roll(past,-16,dims=2)[:,0,480:496].numpy(),x[:,-16:])
    actual_mask=mask.roll(-16,dims=1).numpy()
    assert np.all(actual_mask[:,480:496]) and not np.any(actual_mask[:,496:])
    states=[];block=a.module.encoder.block[len(a.module.encoder.block)//2-1]
    with native_hook(block,move,states):
        real=a.module.short_forecast(x_enc=past,input_mask=mask,forecast_horizon=16)
    # Separate native observed reconstruction and actual forecast fields.
    flat=real.reconstruction[:,0].numpy();future=real.forecast[:,0].numpy()
    np.testing.assert_array_equal(future,flat[:,-16:])
    return np.stack([flat[:,480:488],flat[:,488:496],future[:,:8],future[:,8:]],1),states[0]


def main():
    torch.set_num_threads(2)
    with np.load(OUT/"panel.npz") as f:x=f["x"].reshape(-1,480)
    rows=np.array([4,20,115,201,286]);g=np.random.default_rng(22112433)
    pick=g.integers(0,32,len(x));other=(pick+g.integers(1,32,len(x)))%32
    adapters={s:load(s) for s in cd.SIZES}
    clean={};hidden={};report={}
    m=ordinary_map()
    report["sha256"]={}
    for s in cd.SIZES:
        for path in (ROOT/s/"model.safetensors",ROOT/s/"config.json"):
            with path.open("rb") as stream:report["sha256"][s+"/"+path.name]=file_digest(stream,"sha256").hexdigest()

    with np.load(OUT/"ordinary.npz") as base:
        for s,a in adapters.items():
            raw=base[s][:,:2].astype(float);observed=x[:,-16:].reshape(-1,2,8).astype(float)
            err=raw-observed
            report[s+"_clean_visible_rmse_C"]=np.sqrt((err**2).mean(axis=(0,2))).tolist()
            clean[s],hidden[s]=forward(a,x[rows])
            np.testing.assert_allclose(clean[s],base[s][rows],atol=.0001,rtol=0)
            # Same independent official path and same batch with a literal zero perturbation.
            replay,_=forward(a,x[rows],np.zeros_like(hidden[s]))
            report[s+"_zero_max_C"]=float(np.max(abs(replay-clean[s])))
    for method in cd.METHODS:
        for seed in cd.SEEDS:
            stem=f"{method}_{seed}"
            with (cd.CACHE/(stem+".pt")).open("rb") as stream:
                report["sha256"][stem+".pt"]=file_digest(stream,"sha256").hexdigest()
            with np.load(OUT/(stem+".npz")) as f,np.load(cd.CACHE/(stem+".npz")) as d:
                assert str(f["input_sha256"])==sha256(x.tobytes()).hexdigest()
                src,tgt=d["decoder_small"],d["decoder_large"]
                model,means=cd.load_artifact(method,seed)
                with torch.inference_mode():
                    code=cd.encode_views(model,means,hidden["small"]).numpy()
                ss=np.stack([np.flatnonzero(z>0) for z in code])
                ids=ss[np.arange(len(rows)),pick[rows]];alt=ss[np.arange(len(rows)),other[rows]]
                np.testing.assert_array_equal(ids,f["source_atom"][rows]);np.testing.assert_array_equal(alt,f["other_atom"][rows])
                amp=code[np.arange(len(rows)),ids]
                np.testing.assert_allclose(amp,f["dose"][rows],rtol=1e-5,atol=1e-5)
                native_norm=np.linalg.norm(tgt[ids],axis=1)
                row={"small":src[ids],"paired":tgt[ids],"decoy":tgt[alt],"dense":src[ids].astype(float)@m}
                errors={}
                for arm,vec in row.items():
                    if arm in ("dense","decoy"):
                        vec=vec*(native_norm/np.linalg.norm(vec,axis=1))[:,None]
                    s="small" if arm=="small" else "large"
                    changed,_=forward(adapters[s],x[rows],-amp[:,None]*vec)
                    error=float(np.max(abs(changed-clean[s]-f[arm][rows])))
                    assert error<.0002,error
                    errors[arm]=error
                report[stem]=errors
    report["rows"]=rows.tolist()
    write_json(OUT/"audit.json",report);print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
