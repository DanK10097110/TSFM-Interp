"""Compare pinned model state, reconstruction-head initialization geometry and serialization formats."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch


def digest(path):
    with path.open("rb") as reader:
        return hashlib.file_digest(reader,"sha256").hexdigest()


def inspect(root,name):
    directory=root/name
    filename=directory/"model.safetensors"
    with safe_open(filename,framework="np") as f:
        klist=f.keys()
        W=f.get_tensor("head.linear.weight").astype(float)
        bias=f.get_tensor("head.linear.bias").astype(float)
        vi=f.get_tensor("patch_embedding.value_embedding.weight").astype(float)
        mask=f.get_tensor("patch_embedding.mask_embedding").astype(float)
        norms=[f.get_tensor(k) for k in klist if k.startswith("encoder.") and k.endswith("layer_norm.weight")]
        gram=W@W.T
        rows=np.linalg.norm(W,axis=1)
        cs=gram/(rows[:,None]*rows[None,:])
        report=dict(state_keys=len(klist),archive_sha256=digest(filename),
                    head_bias_norm=float(np.linalg.norm(bias)),head_row_norm_q0_q50_q100=np.quantile(rows,[0,.5,1]).tolist(),
                    head_row_gram_max_departure_from_141_squared_identity=float(np.max(np.abs(gram-np.eye(len(gram))*1.41**2))),
                    head_adjacent_row_dot_median=float(np.median(np.diag(gram,1))),head_adjacent_row_cosine_median=float(np.median(np.diag(cs,1))),
                    value_column_gram_max_departure_from_141_squared_identity=float(np.max(np.abs(vi.T@vi-np.eye(vi.shape[1])*1.41**2))),
                    mask_embedding_norm=float(np.linalg.norm(mask)),encoder_norm_layers=len(norms),
                    encoder_norm_layers_identically_one=sum(int(np.all(z==1)) for z in norms))
        alternate=directory/"pytorch_model.bin"
        if alternate.is_file():
            raw=torch.load(alternate,map_location="cpu",weights_only=True)
            absent=set(klist)-set(raw);extra=set(raw)-set(klist)
            same={k:np.array_equal(f.get_tensor(k),raw[k].numpy()) for k in klist if k in raw}
            report["alternative_official_pytorch_archive"]=dict(sha256=digest(alternate),total=len(raw),
                missing=sorted(absent),extra=sorted(extra),equal=sum(same.values()),unequal=[k for k in same if not same[k]])
            assert not absent and not extra and all(same.values())
            del raw
        if name=="small":
            with safe_open(root/"flan_small"/"model.safetensors",framework="np") as init:
                details={}
                for k in (k for k in klist if k.startswith("encoder.")):
                    target="shared.weight" if k=="encoder.embed_tokens.weight" else k
                    x=f.get_tensor(k);y=init.get_tensor(target)
                    details[k]={"exact":bool(np.array_equal(x,y)),
                                "relative_l2":float(np.linalg.norm(x-y)/np.linalg.norm(y))}
            relative=[z["relative_l2"] for z in details.values()]
            report["canonical_flan_small"]=dict(compared=len(details),equal=sum(int(z["exact"]) for z in details.values()),
                relative_l2_min_median_max=np.quantile(relative,[0,.5,1]).tolist(),by_key=details)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,default=Path(__file__).with_name("release.json"))
    a=p.parse_args();torch.set_num_threads(1)
    report={name:inspect(a.root,name) for name in ("small","base","large")}
    a.output.write_text(json.dumps(report,indent=2)+"\n")
