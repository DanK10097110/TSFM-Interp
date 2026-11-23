"""Check the same-model ridge control in a deliberately rotated coordinate system."""
import argparse
from pathlib import Path
import json

import numpy as np
import torch

from research.activation_transfer.atoms.materials import CACHE, HERE, MODELS, adapter_for
from research.activation_transfer.atoms.intervene import run_arm
from research.activation_transfer.atoms.vectors import bridge
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice


def check(model, cache, folder):
    with np.load(cache/"data.npz") as f:
        train=f["split"]=="train"
        daily=np.flatnonzero(f["split"]=="daily")[:32]
        xs=f["x"][daily]
        ids=f["ids"][daily]
    with np.load(cache/(model+".npz")) as f:
        tokens=f["tokens"]
        x=tokens[train].mean(axis=1)
        local=tokens[daily]
        positions=f["positions"]
    with np.load(cache/(model+"_vectors.npz")) as f:
        decoder=f["source_decoder"]
        z=f["source_codes"][daily]
        projected=f["source_projection"]
        atoms=f["source_atoms"]
    rng=np.random.default_rng(2026112096)
    perm=rng.permutation(x.shape[1])
    signs=rng.choice([-1.,1.],size=x.shape[1])
    def forward(rows):
        return rows[:,perm]*signs
    def inverse(rows):
        return (rows*signs)[:,np.argsort(perm)]
    transformed,_,_=bridge(x,forward(x)+7,decoder,2026112018)
    estimated=inverse(transformed)
    np.testing.assert_allclose(estimated,projected,rtol=1e-6,atol=1e-7)
    true=inverse(forward(decoder))
    np.testing.assert_array_equal(true,decoder)
    adapter=adapter_for(model)
    site=MODELS[model][-1]
    selector=last_window_slice(adapter,positions)
    base=change_forecasts(adapter,site,xs,local,None,selector,24)
    full=run_arm(adapter,site,selector,xs,local,base,z>0,z,true,24)
    shrunk=run_arm(adapter,site,selector,xs,local,base,z>0,z,estimated,24)
    with np.load(folder/(model+"_effects.npz")) as f:
        indices=np.flatnonzero(f["split"]=="daily")[:len(daily)]
        np.testing.assert_array_equal(ids,f["ids"][indices])
        full_drift=float(np.max(np.abs(full-f["source_native"][indices])))
        shrunk_drift=float(np.max(np.abs(shrunk-f["source_projected"][indices])))
    if max(full_drift,shrunk_drift)>1e-5:
        raise ValueError("Known-frame replay disagreed with archived physical interventions")
    return dict(known_transform="signed permutation plus intercept 7",series=ids.tolist(),atoms=atoms.tolist(),
                active_pairs=int(np.sum(z>0)),direction_max_absolute_change=float(np.max(np.abs(estimated-projected))),
                oracle_forecast_max_absolute_change=full_drift,ridge_forecast_max_absolute_change=shrunk_drift)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--folder",type=Path,default=HERE/"results")
    a=p.parse_args()
    torch.set_num_threads(3)
    record={model:check(model,a.cache,a.folder) for model in MODELS}
    (a.folder/"gauge_check.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
