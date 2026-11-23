"""Execute the authors' original short_forecast using official released tensors in their original library versions."""
import argparse
from argparse import Namespace
import gc
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import T5Config,T5EncoderModel


def histories(length):
    t=np.arange(length)
    ramp=np.linspace(-1.6,2.1,length)
    return np.stack((37+2.8*ramp+.6*np.cos(t/9), 83-6*ramp+.8*np.sin(t/2.7),
                     .71+.02*np.sin(t/4.3), np.full(length, -152.)),axis=0).astype("f4")


def native_inputs(x):
    pad=512-x.shape[1]
    value=np.pad(x,((0,0),(pad,0)))
    mask=np.pad(np.isfinite(x),((0,0),(pad,0)))
    return dict(x_enc=torch.tensor(value[:,None]),input_mask=torch.tensor(mask.astype("int64")))


def layer_outputs(model,attrs,x,h):
    """Record all model stages shared by the two implementations without changing computations."""
    states={}
    hooks=[]
    for label,node in attrs.items():
        def record(_m,_args,out,tag=label):
            if isinstance(out,tuple):out=out[0]
            states[tag]=out.detach().float().numpy()
        hooks.append(node.register_forward_hook(record))
    try:
        with torch.no_grad(): y=model.short_forecast(**native_inputs(x),forecast_horizon=h).forecast[:,0]
    finally:
        for handle in hooks:handle.remove()
    return y.numpy(),states


def stream(model):
    ix=(len(model.encoder.block)-1)//2
    return {"patches":model.patch_embedding,"first":model.encoder.block[0],"middle":model.encoder.block[ix],
            "last":model.encoder.block[-1],"head":model.head}


def run(folder,name):
    from moment.models.moment import MOMENT
    from momentfm import MOMENTPipeline

    cfg=json.loads((folder/name/"config.json").read_text())
    canonical=T5Config.from_pretrained(str(folder/("flan_"+name)),local_files_only=True)
    actual=T5Config.from_dict(cfg["t5_config"])
    fields=("d_model","d_ff","d_kv","num_layers","num_heads","feed_forward_proj","dense_act_fn",
            "is_gated_act","layer_norm_epsilon","relative_attention_max_distance","relative_attention_num_buckets")
    differences={k:[getattr(canonical,k),getattr(actual,k)] for k in fields if getattr(canonical,k)!=getattr(actual,k)}
    assert not differences, differences
    research_cfg={**cfg,"task_name":"short-horizon-forecasting","finetuning_mode":"zero-shot","d_model":canonical.d_model}
    def bootstrap(backbone):
        assert backbone==cfg["transformer_backbone"]
        return T5EncoderModel(canonical)
    # Replace only the initial text-backbone download. Every parameter and persistent buffer
    # of this object is overwritten with the checkpoint under strict=True before inference.
    with patch.object(T5EncoderModel,"from_pretrained",side_effect=bootstrap):
        original=MOMENT(Namespace(**research_cfg)).eval()
    state=load_file(str(folder/name/"model.safetensors"))
    original.load_state_dict(state,strict=True)
    del state
    public=MOMENTPipeline.from_pretrained(str(folder/name),model_kwargs={"task_name":"reconstruction"},strict=True).eval()
    rows=[];outs={}
    for L,H in ((480,8),(477,17),(512,64)):
        x=histories(L)
        if L==477:x[1,181]=np.nan
        a,sa=layer_outputs(original,stream(original),x,H)
        b,sb=layer_outputs(public,stream(public),x,H)
        rows.append(dict(length=L,horizon=H,max_forecast_error=float(np.max(np.abs(a-b))),
                         block_max_error={k:float(np.max(np.abs(sa[k]-sb[k]))) for k in sa}))
        assert rows[-1]["max_forecast_error"] < 1e-5
        assert max(rows[-1]["block_max_error"].values()) < 1e-5
        outs[f"{name}_{L}_{H}"]=a
    return {"checkpoint":name,"canonical_flan_config_differences":differences,
            "research_num_state_keys":len(original.state_dict()),"head":type(original.head).__name__,"cases":rows},outs


if __name__=="__main__":
    import importlib.metadata
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--research",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--reference",type=Path)
    args=p.parse_args();sys.path.insert(0,str(args.research));torch.set_num_threads(1)
    commit=subprocess.check_output(["git","-C",str(args.research),"rev-parse","HEAD"],text=True).strip()
    assert commit=="3ab637e413f35f2c317573c0ace280d825c558de"
    summary=dict(research_revision=commit,versions={k:importlib.metadata.version(k) for k in ("torch","transformers","huggingface-hub","numpy","scipy")},models={})
    values={}
    for n in ("small","base","large"):
        summary["models"][n],v=run(args.root,n);values.update(v);gc.collect()
        print(n,summary["models"][n]["cases"][0],flush=True)
    if args.reference:
        with np.load(args.reference) as f:
            summary["max_forecast_difference_from_other_environment"]={k:float(np.max(np.abs(values[k]-f[k]))) for k in values}
        assert max(summary["max_forecast_difference_from_other_environment"].values()) < 1e-4
    np.savez_compressed(args.output.with_suffix(".npz"),**values)
    args.output.write_text(json.dumps(summary,indent=2)+"\n")
