"""Replay the pinned Hub ID forms via an isolated local Hub cache of official files."""
import os
from pathlib import Path
import json
import tempfile

from safetensors import safe_open
import torch
import numpy as np
from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.models import build_adapter


def audit(root):
    revisions={"small":"411e288267f82cce86296dbe4d6c8bc533cc162f","base":"5e44b0ea26376a176360f87831124e018f876d96","large":"ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc"}
    from huggingface_hub import constants
    hub=Path(constants.HF_HUB_CACHE)
    if not hub.is_relative_to(Path(tempfile.gettempdir())):
        raise ValueError("Use an isolated HF_HUB_CACHE within the temporary directory")
    rows={}
    torch.set_num_threads(1)
    x=np.cos(np.arange(480)/15)[None].astype("f4")
    for name, revision in revisions.items():
        repo="AutonLab/MOMENT-1-"+name
        snap=hub/("models--"+repo.replace("/","--"))/"snapshots"/revision
        snap.mkdir(exist_ok=True,parents=True)
        for fname in ("model.safetensors","config.json"):
            local=snap/fname
            if not local.exists():local.symlink_to((root/name/fname).resolve())
        a=build_adapter(ModelConfig(name=name,adapter="moment",checkpoint=repo,kwargs={"revision":revision}),
                        DataConfig(context_len=480,horizon=16),torch.device("cpu"),torch.float32)
        raw=a.predict(x,16,[.5])["point"]
        with safe_open(str(root/name/"model.safetensors"),framework="pt",device="cpu") as f:
            for key,v in a.module.state_dict().items():assert torch.equal(v,f.get_tensor(key)),key
        assert np.isfinite(raw).all() and raw.shape==(1,16)
        rows[name]=dict(model_id=repo,revision=revision,state_tensors=len(a.module.state_dict()),license="MIT",offline_id_load_passed=True)
        del a
    return rows


if __name__=="__main__":
    if os.environ.get("HF_HUB_OFFLINE")!="1":raise ValueError("This replay expects HF_HUB_OFFLINE=1")
    out=audit(Path(os.environ.get("TSFM_MOMENT_NATIVE","/tmp/tsfm_moment_native")))
    Path(__file__).with_name("hub_examples.json").write_text(json.dumps(out,indent=2)+"\n")
