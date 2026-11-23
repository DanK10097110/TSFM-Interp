"""Validate total long-rollout gradients from a real write confined to the first native call."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.models import build_adapter


def check(size, root):
    a=build_adapter(ModelConfig(name=size,adapter="moment",checkpoint=str(root/size)),
                    DataConfig(context_len=480,horizon=513),torch.device("cpu"),torch.float32)
    x=np.cos(np.arange(480)/15)[None].astype("f4")
    a.ensure_loaded()
    width=a.module.config.d_model
    v=torch.tensor(np.random.default_rng(1301).normal(size=(1,width)).astype("f4"))
    v/=v.norm()
    def run(d,graph):
        n=0
        def hook(m,args,out):
            nonlocal n
            n+=1
            if n!=1:return None
            h=out[0].clone();h[:,32:]+=d*v[:,None]
            return (h,*out[1:])
        handle=a.module.encoder.block[-1].register_forward_hook(hook)
        try:
            y=a.predict_differentiable(x,513)[0] if graph else torch.tensor(a.predict(x,513,[.5])["point"][0])
        finally:handle.remove()
        assert n==3
        return torch.stack([y[:256].mean(),y[256:512].mean(),y[-1]])
    def gradient():
        dose=torch.tensor(0.,requires_grad=True)
        y=run(dose,True)
        g=torch.stack([torch.autograd.grad(o,dose,retain_graph=True)[0] for o in y])
        return y.detach().numpy(),g.numpy()
    value,correct=gradient()
    physical=((run(2,False)-run(-2,False))/4).numpy()
    with patch.object(a,"_feedback_forecast",side_effect=a._short):
        original,uncorrected=gradient()
    np.testing.assert_array_equal(value,original)
    err=np.abs(correct-physical)/np.maximum(abs(physical),1e-8)
    assert np.max(err)<.04,(size,correct,physical)
    return dict(milestones=["leads 1–256 mean","leads 257–512 mean","lead 513"],
                derivative_with_native_detachment=uncorrected.tolist(),corrected_gradient=correct.tolist(),
                physical_central_difference=physical.tolist(),relative_error=err.tolist(),
                zero_dose_values_bitwise_equal=True,finite_difference_half_width=2.)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/tmp/tsfm_moment_native'))
    args=p.parse_args();torch.set_num_threads(1)
    out={s:check(s,args.root) for s in ('small','large')}
    Path(__file__).with_name('rolling_gradient.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
