"""Exercise three native forecasting adapters in the same Lens ModelHub and Python runtime."""
from importlib import metadata
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.forecast_quality.post_release.data import OUT
from research.activation_transfer.moment.forecast_quality.predict import checkpoint
from tsfm_lens.config import ModelConfig, DataConfig
from tsfm_lens.models import ModelHub


def main():
    torch.set_num_threads(2)
    order = ("moment_large", "bolt", "timesfm", "moment_large")
    configs = []
    for name in dict.fromkeys(order):
        kind, _, path = checkpoint(name)
        configs.append(ModelConfig(name=name, adapter=kind, checkpoint=str(path), batch_size=16))
    hub = ModelHub(configs, DataConfig(context_len=480,horizon=16),torch.device("cpu"),torch.float32)
    with np.load(OUT/"panel.npz") as f:
        x=f["context"].reshape(-1,480)[:16]
    checks=[]
    for i,name in enumerate(order):
        a=hub.get(name)
        a.ensure_loaded()
        row=dict(model=name,pass_number=i+1)
        with np.load(OUT/(name+".npz")) as f, torch.inference_mode():
            for horizon in (8,16):
                v=a.predict(x,horizon,[.5])["quantiles"][...,0]
                reference=f["raw_"+str(horizon)].reshape(-1,horizon)[:len(x)]
                error=float(np.max(np.abs(v-reference)))
                assert error<.0001,(name,horizon,error)
                row["h"+str(horizon)+"_max_C"]=error
        hub.release(name)
        assert not a._loaded
        checks.append(row)
    result=dict(histories=len(x),units="degrees Celsius",dtype="float32",device="cpu",checks=checks,
                package_versions={k:metadata.version(k) for k in ("momentfm","chronos-forecasting","timesfm","transformers")})
    Path(__file__).with_name("mixed_hub.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
