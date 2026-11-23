"""Measure how the unsupervised clean ridge penalty changes a source atom's causal transfer."""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigh

from research.activation_transfer.atoms.intervene import run_arm
from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE, HERE, ORIGINAL
from research.activation_transfer.replication.experiment import clean_forecast, read_materials
from research.activation_transfer.atoms.materials import dictionary_path
from tsfm_lens.sae.train import load_sae_checkpoint

PENALTIES = (1e-6, 1e-4, .01, .1)


def spectral_map(x, y, queries):
    """Map query displacements at four ridge strengths using one decomposition."""
    x, y = x - x.mean(axis=0, dtype=float), y-y.mean(axis=0, dtype=float)
    eig, vec = eigh(x@x.T)
    eig = np.maximum(eig, 0)
    projected_query = queries @ x.T @ vec
    projected_target = vec.T @ y
    return np.stack([(projected_query/(eig+p*eig[-1]))@projected_target for p in PENALTIES])


def cross_validate(x, y):
    losses = []
    folds = np.arange(len(x))%4
    for i in range(4):
        train = folds!=i
        pred = spectral_map(x[train],y[train],x[~train]-x[train].mean(axis=0,dtype=float))
        residual = pred - (y[~train]-y[train].mean(axis=0,dtype=float))
        losses.append(np.mean(residual**2,axis=(1,2)))
    return np.stack(losses)


def run(source, cache, out, batch):
    target = next(m for m in MODELS if m!=source)
    with np.load(cache/"data.npz") as f:
        x, ids, split = f["x"], f["ids"], f["split"]
    s, _, sids = read_materials(cache, source)
    t, pos, tids = read_materials(cache, target)
    np.testing.assert_array_equal(ids, sids)
    np.testing.assert_array_equal(ids, tids)
    tr, test = split=="train", np.isin(split, ["daily","machines"])
    xs, yt = s[tr].mean(axis=1), t[tr].mean(axis=1)
    with np.load(ORIGINAL/(source+"_vectors.npz")) as prior:
        atoms = prior["source_atoms"]
        directions = spectral_map(xs, yt, prior["source_decoder"])
        np.testing.assert_allclose(directions[2], prior["target_dense"], rtol=1e-4, atol=1e-5)
    cv = cross_validate(xs, yt)
    sae = load_sae_checkpoint(str(dictionary_path(source)))
    with torch.inference_mode():
        z = sae.encode(torch.from_numpy(s[test].mean(axis=1))).numpy()[:,atoms]
    adapter = adapter_for(target)
    layer = MODELS[target][-1]
    select = last_window_slice(adapter, pos)
    base = clean_forecast(adapter, layer, x[test], t[test], select, batch)
    effects = [run_arm(adapter,layer,select,x[test],t[test],base,z>0,z,dirs,batch) for dirs in directions]
    np.savez_compressed(out/(source+"_penalty.npz"), ids=ids[test], split=split[test], atoms=atoms,
                        cv=cv, penalties=PENALTIES, picked=int(np.argmin(cv.mean(axis=0))),
                        map_norm=np.linalg.norm(directions,axis=-1), effects=np.stack(effects))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=MODELS)
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--batch",type=int,default=32)
    a=p.parse_args()
    torch.set_num_threads(3)
    run(a.source,a.cache,a.out,a.batch)
