"""Compare edge and zero padding on the original short-forecast model with an identical mask."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from momentfm import MOMENTPipeline


def compare(root):
    report={}
    for size in ("small","base","large"):
        model=MOMENTPipeline.from_pretrained(str(root/size),model_kwargs={"task_name":"reconstruction"},strict=True).eval()
        rows=[]
        for length,horizon in ((7,6),(23,6),(48,8),(97,18),(477,17),(480,16)):
            a=np.arange(length)
            x=np.stack([86+2.5*np.sin(a/11)+a/12, -2+.004*a+.3*np.cos(a/3)]).astype("f4")
            mask=torch.as_tensor(np.pad(np.ones_like(x,dtype=int),((0,0),(512-length,0))))
            def run(mode):
                values=torch.tensor(np.pad(x,((0,0),(512-length,0)),mode=mode))
                return model.short_forecast(x_enc=values[:,None],input_mask=mask.clone(),forecast_horizon=horizon).forecast[:,0].numpy()
            err=float(np.max(np.abs(run("edge")-run("constant"))))
            rows.append(dict(L=length,H=horizon,max_raw_forecast_difference=err))
            assert err<1e-5,(size,length,horizon,err)
        report[size]=rows
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=Path("/tmp/tsfm_moment_native"))
    args=p.parse_args();torch.set_num_threads(1)
    with torch.inference_mode():result=compare(args.root)
    Path(__file__).with_name("padding.json").write_text(json.dumps(result,indent=2)+"\n")
