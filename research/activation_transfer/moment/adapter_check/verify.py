"""Check the registered MOMENT adapter against independently loaded official pretrained weights."""
import argparse
import gc
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from momentfm import MOMENTPipeline
from safetensors import safe_open

from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.models import build_adapter
from tsfm_lens.models.conformance import check_adapter_conformance

HERE = Path(__file__).resolve().parent


def official(model, x, horizon):
    """Construct native input/mask independently, preserving raw units and future-step convention."""
    history = x[:, -512:]
    left = 512 - history.shape[1]
    full = np.zeros((len(x), 512), dtype="f4")
    valid = np.zeros((len(x), 512), dtype=np.int64)
    full[:, left:] = history
    valid[:, left:] = np.isfinite(history)
    with torch.no_grad():
        out = model.short_forecast(x_enc=torch.tensor(full[:, None]), input_mask=torch.tensor(valid),
                                   forecast_horizon=horizon)
    return out.forecast[:, 0].numpy()


def cases(length):
    """Three histories with different offsets and scales, including a flat series."""
    u = np.arange(length)
    return np.stack((28 + .013*u + 2.7*np.sin(u/6.4), -1.8+.17*np.cos(u/3.4)-.0008*u,
                     np.full(length, -24.)), axis=0).astype("f4")


def add_at(delta, region):
    """Apply an independent post-block additive write without the project's replacement helper."""
    def change(_model, _args, out):
        value = out[0] if isinstance(out, tuple) else out
        new = value.clone()
        new[:, region] = value[:, region] + delta[:, None, :]
        return (new, *out[1:]) if isinstance(out, tuple) else new
    return change


def sensitivity(adapter, x):
    """Compare one physical-direction derivative to actual symmetric forecasts at two token sites."""
    outs = {}
    names = adapter.all_layer_names()
    middle = names[len(names)//2-1]
    final = names[-1]
    observed, future = adapter.patch_layout(480, 16)
    recent = slice(observed.stop-4, observed.stop)
    rng = np.random.default_rng(410827)
    for name, layer, slots in (("middle_context",middle,recent), ("terminal_context",final,recent),
                                ("terminal_masked",final,future)):
        block = adapter.find_module(layer)
        buffer = []
        hook = block.register_forward_hook(lambda _,__,out: buffer.append(out[0].detach()) or None)
        baseline = adapter.predict(x,16,[.5])["point"]
        hook.remove()
        state = buffer[0]
        assert state.shape[:2] == (len(x),64)
        v = torch.tensor(rng.normal(size=(len(x),state.shape[-1])),dtype=torch.float32)
        v /= v.norm(dim=1,keepdim=True)
        radius = .008*float(state[:,slots].norm(dim=-1).mean())
        alpha = torch.zeros(len(x),requires_grad=True)
        handle = block.register_forward_hook(add_at(alpha[:,None]*v,slots))
        try:
            graph = adapter.predict_differentiable(x,16)
        finally:
            handle.remove()
        np.testing.assert_allclose(graph.detach().numpy(),baseline,rtol=0,atol=0)
        j = torch.stack([torch.autograd.grad(graph[:,i].sum(),alpha,retain_graph=i<15)[0] for i in range(16)],dim=1)
        effects=[]
        for sign in (1,-1):
            handle = block.register_forward_hook(add_at(sign*radius*v,slots))
            try:
                effects.append(adapter.predict(x,16,[.5])["point"])
            finally:
                handle.remove()
        fd=(effects[0].astype(float)-effects[1].astype(float))/2
        want=radius*j.numpy().astype(float)
        e=float(np.linalg.norm(fd-want)/max(np.linalg.norm(fd),1e-14))
        mx=float(np.max(np.abs(fd)))
        if name=="terminal_context":
            assert mx==0 and not j.count_nonzero()
        else:
            assert mx>1e-6 and e<.07,(name,mx,e)
        outs[name]={"site":layer,"slice":[slots.start,slots.stop],"physical_norm":radius,
                    "fd_max_abs_in_forecast_units":mx,"analytic_relative_error":e}
    return outs


def check(model_name, path):
    """Run direct native, rolling, normalization, attention-mask, token and intervention checks."""
    cfg=ModelConfig(name=model_name,adapter="moment",checkpoint=str(path))
    a=build_adapter(cfg,DataConfig(context_len=480,horizon=16),torch.device("cpu"),torch.float32)
    a.ensure_loaded()
    native=MOMENTPipeline.from_pretrained(str(path),strict=True,model_kwargs={"task_name":"reconstruction"}).eval()
    live_state = a.module.state_dict()
    with safe_open(path/"model.safetensors", framework="pt", device="cpu") as release:
        published=set(release.keys())
        assert published==set(live_state),set(live_state)^published
        for key in sorted(published):
            tensor=release.get_tensor(key)
            assert tensor.dtype==live_state[key].dtype and torch.equal(tensor,live_state[key]),key
        critical={k:hashlib.sha256(release.get_tensor(k).contiguous().numpy().tobytes()).hexdigest()
                  for k in ("patch_embedding.mask_embedding","head.linear.weight","head.linear.bias")}
    cases_run=[]
    for length,h in [(480,1),(480,7),(480,8),(480,9),(480,16),(480,64),(480,65),(477,17),
                     (523,9),(7,8),(64,505),(64,512)]:
        x=cases(length)
        if length==477:
            x[0,32]=np.nan
        ref=official(native,x,h)
        raw=a.predict(x,h,[.1,.5,.9])
        err=float(np.max(np.abs(raw["point"]-ref)))
        np.testing.assert_allclose(raw["point"],ref,rtol=1e-5,atol=1e-5)
        np.testing.assert_array_equal(raw["quantiles"],np.repeat(raw["point"][:,:,None],3,axis=2))
        assert raw["point"].shape==(3,h) and np.isfinite(raw["point"]).all()
        cases_run.append({"length":length,"horizon":h,"max_error":err})
    x=cases(480)[:2]
    r=a.predict(x,16,[.5])["point"]
    shift=11.;factor=2.3
    translated=a.predict(x+shift,16,[.5])["point"]; scaled=a.predict(factor*x,16,[.5])["point"]
    shift_err=float(np.max(np.abs(translated-(r+shift))))
    scale_err=float(np.max(np.abs(scaled-factor*r)))
    assert shift_err<2e-4 and scale_err<4e-4
    x2=cases(523)[:1]
    pred=a.predict(x2,513,[.5])["point"]
    history=x2.copy();pieces=[]
    for take in (256,256,1):
        block=official(native,history,take)
        history=np.concatenate((history,block),axis=1)[:,-512:]
        pieces.append(block)
    ext=float(np.max(np.abs(pred-np.concatenate(pieces,axis=1))))
    assert ext<3e-5
    masked=cases(477)[:2];masked[1,100]=np.nan
    captured={}
    def get_input(_module, _args, kwargs):
        captured["attn"]=kwargs["attention_mask"].detach().numpy()
        captured["emb"]=kwargs["inputs_embeds"].detach().numpy()
    h=a.module.encoder.register_forward_pre_hook(get_input,with_kwargs=True)
    try:
        a.predict(masked,17,[.5])
    finally:
        h.remove()
    mask=np.pad(np.isfinite(masked),((0,0),(35,0)))
    shifted=np.roll(mask,-24,axis=-1);shifted[:,-24:]=True
    np.testing.assert_array_equal(captured["attn"],shifted.reshape(2,64,8).all(-1))
    np.testing.assert_array_equal(captured["emb"][0,-3:],captured["emb"][1,-3:])
    obs,fut=a.patch_layout(477,17)
    first=np.flatnonzero(captured["attn"][0])[:1].item()
    assert first==obs.start and fut==slice(61,64) and obs.stop==61
    expected_embed={}
    def native_inputs(_module,_args,kwargs):
        expected_embed["emb"]=kwargs["inputs_embeds"].detach().numpy()
    watch=native.encoder.register_forward_pre_hook(native_inputs,with_kwargs=True)
    try:
        official(native,masked,17)
    finally:
        watch.remove()
    emb_max=float(np.max(np.abs(captured["emb"]-expected_embed["emb"])))
    assert emb_max<1e-6
    del native
    gc.collect()
    intervention=sensitivity(a,x)
    mechanics=check_adapter_conformance(a,window=32,n_series=3,horizon=16)
    with (path/"model.safetensors").open("rb") as f:digest=hashlib.file_digest(f,"sha256").hexdigest()
    null_check = {}
    if model_name == "small":
        torch.manual_seed(20261122)
        random_cfg = ModelConfig(name="random-small", adapter="moment", checkpoint=str(path), random_init=True)
        fresh = build_adapter(random_cfg, DataConfig(context_len=480, horizon=16), torch.device("cpu"), torch.float32)
        fresh_point = fresh.predict(x, 16, [.5])["point"]
        shapes = {k: tuple(v.shape) for k,v in fresh.module.state_dict().items()}
        assert shapes == {k: tuple(v.shape) for k,v in live_state.items()}
        max_change = float(np.max(np.abs(fresh_point-r)))
        assert fresh_point.shape == r.shape and np.isfinite(fresh_point).all() and max_change > 1e-3
        null_check = {"random_init_check": {"state_shapes_equal": True, "max_forecast_difference": max_change}}
    return {**null_check,"checkpoint_sha256":digest,"layer_count":len(a.all_layer_names()),"hidden_width":a.module.config.d_model,
            "parameter_count":sum(p.numel() for p in a.module.parameters()),"head":type(a.module.head).__name__,
            "strict_state_keys_exact":len(published),"published_critical_tensor_sha256":critical,"native_cases":cases_run,"shift_max":shift_err,"positive_scale_max":scale_err,"rolling_513_max":ext,
            "mask_probe":{"length":477,"horizon":17,"first_visible":int(first),"first_future":int(fut.start),
                          "incomplete_history_keys":int((captured["attn"][0]>captured["attn"][1]).sum()),
                          "identical_future_embeddings":True,"max_all_embeddings_against_native":emb_max},"intervention":intervention,"conformance":mechanics}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--small",type=Path,required=True);p.add_argument("--large",type=Path,required=True)
    p.add_argument("--output",type=Path,default=HERE/"verification.json")
    args=p.parse_args();torch.set_num_threads(1)
    out={"versions":{k:version(k) for k in ("momentfm","torch","transformers","huggingface-hub","safetensors","numpy")}}
    for name,path in (("small",args.small),("large",args.large)):
        out[name]=check(name,path.resolve())
        print(name,"passed",flush=True)
    args.output.write_text(json.dumps(out,indent=2)+"\n")
