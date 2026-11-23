"""Inspect normal workflow artifacts and reproduce a profile-matched SAE ablation on the official model."""
import argparse
import html
import json
from pathlib import Path
import re

import numpy as np
import torch
import yaml
from momentfm import MOMENTPipeline


def report(run, checkpoint, steering_run):
    key, metadata = next(iter(json.loads((run/"sae/meta.json").read_text()).items()))
    name, site = key.split("/")
    path = run/"sae"/name
    saved = torch.load(next(path.glob("*.pt")), map_location="cpu",weights_only=True)
    w=saved["state_dict"]
    archive=json.loads(next(path.glob("*_ablation.json")).read_text())
    assert archive["ablation_null"]=="profile_matched" and archive["reach"]["reachable"]
    assert not archive["withheld"] and any(c["scorable"] for c in archive["candidates"])
    candidate=next(c for c in archive["candidates"] if c["scorable"] and c["forecasts"])
    feature=candidate["feature"]
    rows=candidate["forecasts"]
    assert len(rows)>=2
    x=torch.tensor([r["context"] for r in rows],dtype=torch.float32)
    model=MOMENTPipeline.from_pretrained(str(checkpoint),model_kwargs={"task_name":"reconstruction"},strict=True).eval()
    padded=torch.nn.functional.pad(x,(32,0))[:,None]
    mask=torch.nn.functional.pad(torch.ones_like(x,dtype=torch.int64),(32,0))
    H=16
    def native():
        return model.short_forecast(x_enc=padded,input_mask=mask.clone(),forecast_horizon=H).forecast[:,0].numpy().astype(float)
    clean=native(); module=model.get_submodule(site)
    def hook(omit, controlled):
        def replace(_module,_args,out):
            hidden=out[0]; assert hidden.shape[1]==64
            observed=hidden[:,2:62]
            pre=torch.relu((observed-w["b_dec"])@w["W_enc"]+w["b_enc"])
            values,indices=pre.topk(saved["k"],dim=-1)
            code=torch.zeros_like(pre).scatter_(-1,indices,values)
            controlled.append(float(code[...,feature].max()))
            if omit:code[...,feature]=0
            edited=hidden.clone();edited[:,2:62]=code@w["W_dec"]+w["b_dec"]
            return (edited,*out[1:])
        return replace
    amps=[];effects={}
    for omit,keyname in ((False,"with_feature"),(True,"without_feature")):
        handle=module.register_forward_hook(hook(omit,amps))
        try:effects[keyname]=native()
        finally:handle.remove()
    assert min(amps)>0
    sd=x.numpy().std(1)
    reference={k:np.asarray([r[k] for r in rows]) for k in ("unpatched","with_feature","without_feature")}
    effects["unpatched"]=clean
    errors={k:float(np.max(np.abs(y-reference[k])/sd[:,None])) for k,y in effects.items()}
    assert max(errors.values())<4e-5,errors
    delta=np.max(np.abs(effects["with_feature"]-effects["without_feature"])/sd[:,None])
    assert delta>1e-5,delta
    recipe = yaml.safe_load((run/"config_resolved.yaml").read_text())
    steering_recipe = yaml.safe_load((steering_run/"config_resolved.yaml").read_text())
    for field in ("models", "data", "steering"):
        assert recipe[field] == steering_recipe[field], field
    assert recipe["run"]["seed"] == steering_recipe["run"]["seed"]
    artifact=(steering_run/"steering/report.html").read_text()
    blocks=[]
    for blob in re.findall(r'<pre>(.*?)</pre>',artifact,flags=re.S):
        try:blocks.append(json.loads(html.unescape(blob)))
        except ValueError:pass
    stats=next(z["statistics"] for z in blocks if isinstance(z,dict) and "statistics" in z)
    assert stats["completed"]==stats["total"] and stats["replay_matches"]==stats["completed"],stats
    fields=("n_train_rows","n_benchmark_rows","dict_size","k","reconstruction_fidelity","reconstruction_fidelity_heldout","dead_feature_rate","dead_feature_rate_heldout")
    select={k:metadata[k] for k in fields}
    select.update(admission=metadata["admission"],heldout_token_preservation=metadata["forecast_preservation_token_heldout"],
                  feature_ablation_count=len(metadata["feature_ablation"]["features"]),
                  feature_steering_count=len(metadata["feature_steering"]["features"]),
                  ground_truth_matched=metadata["ground_truth_alignment"]["n_features_matched"])
    minimal=dict(name=name,site=site,steering_recipe_and_data_match=True,SAE=select,ablation=dict(null=archive["ablation_null"],
                candidate_count=len(archive["candidates"]),scorable_candidates=sum(c["scorable"] for c in archive["candidates"]),n_null=archive["n_null_directions"],
                selected_feature=feature,replay_ids=[r["series_id"] for r in rows],replay_error_in_history_sd=errors,
                max_ablation_effect_in_history_sd=float(delta),peak_active_code=max(amps),reach=archive["reach"]),
                steering=stats)
    return minimal


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=Path("/tmp/tsfm_moment_native"))
    p.add_argument("--runs",type=Path);p.add_argument("--size",choices=("small","large"),required=True)
    p.add_argument("--output",type=Path);p.add_argument("--run",type=Path);p.add_argument("--steering-run",type=Path)
    args=p.parse_args();runs=args.runs or args.root/"cli"
    torch.set_num_threads(1)
    run = args.run or runs/("core_"+args.size)
    with torch.inference_mode():result=report(run,args.root/args.size,args.steering_run or run)
    output=args.output or Path(__file__).with_name("workflow_"+args.size+".json")
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(args.size,result["ablation"],result["steering"],flush=True)
