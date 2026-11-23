"""Verify every coefficient budget and independently replay live sparse additions."""
import json

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.sparse_writes.run import ARMS, EXTRA, HERE
from tsfm_lens.extraction.hooks import _primary, _rebuild
from tsfm_lens.sae.train import load_sae_checkpoint


def audit():
    report = {}
    with np.load(CACHE/"data.npz") as f:
        ev = np.isin(f["split"],["daily","machines"])
        inputs,ids,split = f["x"][ev],f["ids"][ev],f["split"][ev]
    for source in MODELS:
        target = next(m for m in MODELS if m!=source)
        tokens,pos,token_ids = read_materials(CACHE,target)
        np.testing.assert_array_equal(token_ids[ev],ids)
        enc = adapter_for(target)
        block = enc.module.get_submodule(MODELS[target][-1])
        select = last_window_slice(enc,pos)
        sae = load_sae_checkpoint(str(dictionary_path(target)))
        with torch.inference_mode():
            z = sae.encode(torch.from_numpy(tokens[ev].mean(axis=1))).numpy()
        decoder = sae.W_dec.detach().numpy()
        inverse_perm = np.argsort(np.random.default_rng(12871).permutation(len(decoder)))
        measured = []
        samples,valid = [],{}
        with np.load(HERE/"results"/(source+".npz")) as a,np.load(HERE/"results"/(source+"_identity.npz")) as b:
            for arm in (*ARMS,*EXTRA):
                f = a if arm in ARMS else b
                np.testing.assert_array_equal(ids,f["ids"])
                active = f["active"]
                selected,alpha = f[arm+"_ids"],f[arm+"_coefficients"]
                rows,feat,slot = np.nonzero(selected>=0)
                dec_ids = selected[rows,feat,slot]
                candidate_ids = inverse_perm[dec_ids] if arm=="permuted_four" else dec_ids
                amounts,original = alpha[rows,feat,slot],z[rows,candidate_ids]
                if np.any(original<=0) or not active[rows,feat].all() or np.max(np.abs(amounts)-original,initial=0)>5e-5:
                    raise ValueError("An edit exceeded the available feature budget")
                if arm in ("fixed_removal","local_removal"):
                    np.testing.assert_allclose(amounts,-original,atol=1e-6,rtol=1e-6)
                if arm=="partial_removal" and np.any(amounts>0):
                    raise ValueError("A partial removal increased a feature")
                np.testing.assert_array_equal(alpha[selected<0],0.)
                valid[arm]=len(amounts)
                rng=np.random.default_rng(1999)
                for domain in ("daily","machines"):
                    candidates=np.argwhere(active & (split[:,None]==domain) & np.any(selected>=0,axis=2))
                    if not len(candidates):
                        candidates=np.argwhere(active & (split[:,None]==domain))
                    for row,j in candidates[rng.choice(len(candidates),size=min(3,len(candidates)),replace=False)]:
                        choice=selected[row,j]>=0
                        delta=alpha[row,j,choice].astype(float) @ decoder[selected[row,j,choice]].astype(float)
                        samples.append((row,delta))
                        measured.append(f[arm][row,j])
        replay=[]
        for start in range(0,len(samples),8):
            rows,dirs=zip(*samples[start:start+8])
            obs=inputs[list(rows)]
            ordinary=enc.predict(obs,8,[.5])["quantiles"][...,0]
            calls=[]
            def add(module,args,output):
                h=_primary(output)
                new=h.clone()
                new[:,select(h.shape[1])] += torch.as_tensor(np.array(dirs),dtype=h.dtype)[:,None]
                calls.append(True)
                return _rebuild(output,new)
            with block.register_forward_hook(add):
                pred=enc.predict(obs,8,[.5])["quantiles"][...,0]
            if len(calls)!=1:
                raise ValueError("Replay hook must fire once")
            replay.extend(pred-ordinary)
        replay=np.array(replay)
        np.testing.assert_allclose(replay,measured,atol=5e-5,rtol=.002)
        report[source]=dict(coefficient_entries_checked=valid,live_forecasts=len(samples),
                            max_abs_difference=float(np.max(np.abs(replay-measured))))
    (HERE/"results"/"audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    torch.set_num_threads(2)
    audit()
