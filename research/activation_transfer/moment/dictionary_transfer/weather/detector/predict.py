"""Receiver-only test path. This module never loads a donor SAE, weather donor coefficients or donor responses."""
import argparse
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.match import dictionary,encode
from research.activation_transfer.moment.dictionary_transfer.weather.detector.fit import OUT,infer_doses,TARGET
from research.activation_transfer.moment.dictionary_transfer.weather.prefix import obtain,cached_middle
from research.activation_transfer.moment.dictionary_transfer.weather.study import CACHE,histories,live
from research.activation_transfer.moment.token_paths.geometry import load


def run(cache,out,arm,batch):
    f=dict(np.load(out/'fit.npz'))
    x,ids,station,origin=histories();clean=dict(np.load(cache/'clean.npz'))
    np.testing.assert_array_equal(ids,clean['ids'])
    sae=dictionary('large',TARGET);codes=encode(sae,clean['middle']).astype(float)
    if arm=='inferred':
        raw,dose=infer_doses(codes,f);directions=f['group']
    else:
        dose=codes[:,f['match']];raw=dose;directions=sae.W_dec.detach().numpy()[f['match']]
    rows,cols=np.nonzero(dose>0);delta=(-dose[rows,cols,None]*directions[cols]).astype('f4')
    model=load('large')
    stamp=sha256(np.ascontiguousarray(clean['middle']).tobytes()+str(clean['checkpoint_sha256']).encode()+str(clean['context_sha256']).encode()).hexdigest()
    full=obtain(model,x,str(clean['site']),int(clean['pos']),cache,stamp,batch)
    effects=np.zeros((*dose.shape,16));shifts=np.zeros_like(effects);diffs=[]
    scratch=cache/('detector_'+arm);scratch.mkdir(exist_ok=True)
    checksum=sha256()
    for obj in (ids,f['atoms'],dose,directions,rows,cols,[batch],stamp):checksum.update(np.ascontiguousarray(obj).tobytes())
    sig=checksum.hexdigest()
    for i in range(0,len(rows),batch):
        ix=slice(i,i+batch);r=rows[ix];c=cols[ix];path=scratch/(str(i)+'.npz')
        if path.exists():
            with np.load(path) as prev:
                assert str(prev['signature'])==sig
                effects[r,c]=prev['native'];shifts[r,c]=prev['shadow'];diffs.append(float(prev['floor']))
            continue
        with cached_middle(model,full[r],str(clean['site'])):
            ordinary,original,_,sd=live(model,x[r],np.zeros_like(delta[ix]))
            changed,updated,_,_=live(model,x[r],delta[ix])
        native=changed.astype(float)-ordinary
        shadow=((updated.astype(float)-original.astype(float))@clean['head'].astype(float).T).reshape(-1,16)*sd[:,None]
        floor=np.max(np.abs(ordinary.astype(float)-clean['base'][r]))
        np.testing.assert_allclose(native,shadow,atol=.00012,rtol=.001)
        effects[r,c]=native;shifts[r,c]=shadow;diffs.append(float(floor))
        tmp=path.with_suffix('.partial.npz');np.savez_compressed(tmp,native=native,shadow=shadow,floor=floor,signature=sig);tmp.replace(path)
        if i%(8*batch)==0:print(arm,i,len(rows),flush=True)
    np.savez_compressed(out/(arm+'.npz'),ids=ids,station=station,origin=origin,dose=dose,raw=raw,atoms=f['atoms'],
                        native=effects,head=shifts,input_sd=clean['input_sd'],max_batch_ordinary_gap=float(max(diffs,default=0.)))
    print(arm,len(rows),'of',dose.size,'nonzero receiving writes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=('inferred','natural'));p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--batch',type=int,default=24);a=p.parse_args();torch.set_num_threads(1);run(a.cache,a.out,a.arm,a.batch)
