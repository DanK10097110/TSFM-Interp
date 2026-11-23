"""Positive controls for detecting a leaking mask and a transposed within-patch forecast order."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch

from audit import HERE, QUALITY, STUDY, model_from_original, public_input


def run(root):
    sys.path.insert(0,str(root/"research_src"))
    model=model_from_original(root)
    with np.load("/tmp/tsfm_moment_nov22_heads/train.npz") as f:
        x=f["context"][:4];holes=f["holes"][:4]
    damaged=x.copy().reshape(4,60,8)
    damaged[np.arange(4)[:,None],holes] += 7777.
    damaged=damaged.reshape(4,480)
    original=model.pretraining
    def leaking(*,x_enc,input_mask,mask):
        return original(x_enc=x_enc,input_mask=input_mask,mask=torch.ones_like(mask))
    with patch.object(model,"pretraining",side_effect=leaking):
        h0, y0, mu0=reconstruction_without_reference(model,x,holes)
        h1, y1, mu1=reconstruction_without_reference(model,damaged,holes)
    leak=dict(hidden_max=float(np.max(np.abs(h0-h1))),raw_max=float(np.max(np.abs(y0-y1))),mu_max=float(np.max(np.abs(mu0-mu1))))
    assert all(v>1e-2 for v in leak.values())
    with np.load(QUALITY/"panel.npz") as f:
        row=np.flatnonzero((f['domain']=='m4_daily')&(f['split']=='holdout'))[0]
        context=f['context'][row]
    with np.load(STUDY/'small_0_heads.npz') as f:
        w,b=f['interior_hidden_weight'],f['interior_hidden_bias']
    with np.load(STUDY/'small_0_pred.npz') as f:ref=f['interior_hidden_16'][row]
    values,valid=public_input(context)
    def forecast(order):
        model.head.linear.weight.copy_(torch.tensor(w[:,order].T.copy()))
        model.head.linear.bias.copy_(torch.tensor(b[order].copy()))
        return model.short_forecast(x_enc=values,input_mask=valid.clone(),forecast_horizon=16).forecast[:,0].numpy()
    good=forecast(np.arange(8));wrong=forecast(np.arange(7,-1,-1));sd=context.std(1)[:,None]
    d0=float(np.max(np.abs(good-ref)/sd));d1=float(np.max(np.abs(wrong-ref)/sd))
    assert d0<3e-4 and d1>3e-4
    return dict(leaking_mask=leak,direct_forecast_max_error_in_history_sd=d0,reversed_output_rows_error_in_history_sd=d1)


def reconstruction_without_reference(model,x,holes):
    # The clean audit rejects an exposed mask at its independent visible-only normalization
    # assertion. Record the bad model's changes here so its magnitude also survives on disk.
    value,valid=public_input(x);idx=32+holes[:,:,None]*8+np.arange(8);keep=torch.ones_like(valid)
    keep[np.arange(len(x))[:,None,None],idx]=0;states=[]
    handle=model.head.linear.register_forward_pre_hook(lambda m,args:states.append(args[0].numpy().copy()))
    try:y=model.pretraining(x_enc=value,input_mask=valid,mask=keep).reconstruction.numpy()
    finally:handle.remove()
    return states[0],y,model.normalizer.mean.numpy()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/tmp/tsfm_moment_native'))
    a=p.parse_args();torch.set_num_threads(1)
    with torch.inference_mode():r=run(a.root)
    (HERE/'canary.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
