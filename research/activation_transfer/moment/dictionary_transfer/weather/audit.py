"""Independently rebuild selected directions and rerun the native checkpoint on singleton inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.match import OUT as MATCH,dictionary
from research.activation_transfer.moment.dictionary_transfer.weather.study import OUT,SEEDS,histories
from research.activation_transfer.moment.token_paths.geometry import layers,load
from tsfm_lens.extraction.hooks import ActivationCatcher,token_patch


def replay(seed,model,out):
    with np.load(out/(str(seed)+'.npz')) as f:a=dict(f)
    with np.load(MATCH/f'large_{seed}_to_large_matches.npz') as f:match=dict(f)
    x,ids,*_=histories();np.testing.assert_equal(a['ids'],ids)
    sd=dictionary('large',seed);td=dictionary('large',int(match['recipient_seed']))
    tvec=td.W_dec.detach().numpy();svec=sd.W_dec.detach().numpy()
    site=layers(model)['middle'];samples=[]
    on=a['target_z'][a['rows'],a['cols']]>0
    ev=np.unique(np.r_[np.linspace(0,len(on)-1,min(3,len(on)),dtype=int),
                       np.flatnonzero(on)[:1],np.flatnonzero(~on)[:1]])
    for i in ev:
        r,c=int(a['rows'][i]),int(a['cols'][i]);row=x[r:r+1];atom=int(a['atoms'][c]);target=int(match['corr_id'][c])
        with ActivationCatcher(model.module,[site]) as collect:
            base=model.predict(row,16,[.5])['point'][0].astype(float)
            hidden=collect.collect()[site]
        selector=model.token_slice(hidden.shape[1])
        slot=selector.stop-1
        np.testing.assert_equal(a['pos'],slot)
        ordinary=hidden[0,slot:slot+1]
        with torch.no_grad():
            sa=sd.encode(ordinary)[0,atom].item();ta=td.encode(ordinary)[0,target].item()
        np.testing.assert_allclose(sa,a['z'][r,c],atol=3e-4,rtol=3e-5)
        np.testing.assert_allclose(ta,a['target_z'][r,c],atol=3e-4,rtol=3e-5)
        original=-sa*svec[atom].astype(float);dose=np.linalg.norm(original)
        cut=td.W_dec.detach().numpy()[a['group_ids'][c]].astype(float)
        combo=a['group_coeff'][c]@cut
        vecs={'donor':original,'corr_own':-ta*tvec[target].astype(float),'group16':-sa*combo}
        for key,which in [('corr_forced','corr_id'),('random','random_id'),('geom_forced','geom_id')]:
            axis=tvec[int(match[which][c])].astype(float)
            vecs[key]=-dose*axis/np.linalg.norm(axis)
        gaps={}
        for key,v in vecs.items():
            overwrite=(ordinary+torch.from_numpy(v.astype('f4')))[None]
            with token_patch(model.module,site,lambda n:slice(model.token_slice(n).stop-1,model.token_slice(n).stop),overwrite):
                patched=model.predict(row,16,[.5])['point'][0].astype(float)
            archived=(a[key]-a['noop'])[i]
            gap=patched-base-archived;gaps[key]=float(np.abs(gap).max())
            np.testing.assert_allclose(patched-base,archived,atol=.00025,rtol=.002)
        samples.append(dict(id=str(ids[r]),atom=atom,target=target,source_coefficient=sa,target_coefficient=ta,
                            gap_celsius=gaps,ordinary_archive_max_celsius=float(np.abs(base-a['base'][r]).max())))
    return samples


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();torch.set_num_threads(1)
    model=load('large');r={str(seed):replay(seed,model,a.out) for seed in SEEDS}
    (a.out/'replay.json').write_text(json.dumps(r,indent=2)+'\n')
