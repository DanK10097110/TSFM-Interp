"""Singleton real-model checks using a separate inverse and raw Celsius additions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,dictionary_path
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.forward_only.audit import add_native
from research.activation_transfer.shape_transfer.support_confirmation.audit import ordinary,reconstruct
from research.activation_transfer.shape_transfer.support_confirmation.data import adapter_for_length
from research.activation_transfer.shape_transfer.support_confirmation.noaa.data import CACHE,OUT
from research.activation_transfer.shape_transfer.support_confirmation.noaa.run import SEEDS,SUPPORT
from tsfm_lens.sae.train import load_sae_checkpoint


def check(target,cache,out):
    f=dict(np.load(out/(target+'.npz')));src=str(f['source_model'])
    with np.load(cache/'data.npz') as d:
        x=d['x'];sd=x.astype(float).std(1)
    np.testing.assert_equal(f['sd'],sd)
    j=np.load(cache/(target+'_early.npz'))['jac'].astype(float)/sd[:,None,None]
    sj=np.load(cache/(src+'_early.npz'))['jac'].astype(float)/sd[:,None,None]
    target_model=adapter_for_length(target,480);donor_model=adapter_for_length(src,480)
    tm=MODELS[target][-1];sm=MODELS[src][-1]
    ts=last_window_slice(target_model,f['target_pos']);ss=last_window_slice(donor_model,f['source_pos'])
    dct=load_sae_checkpoint(str(dictionary_path(target)));src_dict=load_sae_checkpoint(str(dictionary_path(src)))
    dw=dct.W_dec.detach().numpy().astype(float);sw=src_dict.W_dec.detach().numpy().astype(float)
    th,_,_=read_materials(cache,target);sh,_,_=read_materials(cache,src)
    report=[]
    for ev in np.unique(np.linspace(0,len(f['rows'])-1,4,dtype=int)):
        r=f['rows'][ev];atom=f['atoms'][f['cols'][ev]]
        with torch.inference_mode():
            code=src_dict.encode(torch.from_numpy(sh[r].mean(0,keepdims=True)))[0,atom].item()
            native=dct.encode(torch.from_numpy(th[r].mean(0,keepdims=True)))[0].topk(32).indices.numpy()
        np.testing.assert_array_equal(np.sort(native),np.sort(f['native_ids'][r]))
        donor_vector=(-code*sw[atom]).astype('f4')
        dense=reconstruct(j[r],sj[r]@donor_vector,float(f['weight']),float(f['cap']))
        np.testing.assert_allclose(dense,f['dense_vector'][ev],atol=3e-5,rtol=.001)
        target_goal=j[r]@dense
        np.testing.assert_allclose(target_goal,f['current_goal'][ev],atol=5e-6,rtol=.001)
        budget=np.linalg.norm(dense.astype(float))
        base_s=ordinary(donor_model,x[r]);base_t=ordinary(target_model,x[r])
        with add_native(donor_model,sm,ss,donor_vector):donor=ordinary(donor_model,x[r])-base_s
        with add_native(target_model,tm,ts,dense):full=ordinary(target_model,x[r])-base_t
        gaps={}
        for name,y in [('source',donor),('dense',full)]:
            np.testing.assert_allclose(y,f[name+'_celsius'][ev],atol=.0004,rtol=.001)
            gaps[name]=float(np.max(abs(y/sd[r]-f[name][ev])))
        seed=SEEDS[ev%len(SEEDS)];other=f[f'assigned_{seed}'][r]
        assert f['station'][other]!=f['station'][r] and f['origin'][other]==f['origin'][r]
        for support in SUPPORT:
            axes=np.linalg.svd(dw[np.sort(f[support+'_ids'][r])],full_matrices=False)[2]
            for name,read in [('current',r),(str(seed),other)]:
                v=reconstruct(j[read],target_goal,float(f['weight']),budget,axes)
                with add_native(target_model,tm,ts,v):y=ordinary(target_model,x[r])-base_t
                key=support+'_'+name;expected=f[key+'_celsius'][ev]
                np.testing.assert_allclose(y,expected,atol=.0006,rtol=.003)
                np.testing.assert_allclose(y/sd[r],f[key][ev],atol=.0001,rtol=.003)
                gaps[key]=float(np.max(abs(y/sd[r]-f[key][ev])))
        report.append(dict(item=str(f['ids'][r]),atom=int(atom),sd_errors=gaps))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();torch.set_num_threads(1);r={m:check(m,a.cache,a.out) for m in MODELS}
    (a.out/'replay.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
