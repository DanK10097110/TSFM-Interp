"""Apply two frozen model-to-model ordinary ridge operators to specific measured changes in Celsius."""
import argparse
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch

from research.activation_transfer.atoms.materials import CACHE as ORIGINAL,MODELS,adapter_for
from research.activation_transfer.atoms.vectors import bridge
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.support_confirmation.noaa.data import CACHE as WEATHER,PANEL,SIGNATURE
from tsfm_lens.extraction.hooks import ActivationCatcher

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
CACHE=Path('/tmp/tsfm_physical_ramp_20261122')
ARM=('mapped','wrong_step','other_station')
OTHER={'bolt':'timesfm','timesfm':'bolt'}


def hashed(x):return sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def history():
    with np.load(PANEL) as f:raw=f['context'];stations=f['ids']
    assert raw.shape==(46,4,480) and hashed(raw)==SIGNATURE
    with np.load(WEATHER/'data.npz') as f:
        x=f['x'];ids=f['ids'];station=f['station'];origin=f['origin']
        np.testing.assert_equal(x,raw.reshape(-1,480).astype('f4'))
    np.testing.assert_equal(station,np.repeat(np.arange(len(stations)),4));np.testing.assert_equal(origin,np.tile(range(4),len(stations)))
    return x,ids,station,origin


def edits(x):
    ramp=x.copy();step=x.copy()
    ramp[:,-8:]+=np.linspace(-2.,2.,8,dtype=np.float32)
    step[:,-8:]+=2.
    return ramp,step


def training(source):
    target=OTHER[source]
    hs,ps,ids=read_materials(ORIGINAL,source);ht,pt,ti=read_materials(ORIGINAL,target)
    with np.load(ORIGINAL/'data.npz') as f:
        keep=f['split']=='train';np.testing.assert_equal(f['ids'],ids)
    np.testing.assert_equal(ti,ids);assert keep.sum()==1024 and len(set(ids[keep]))==1024
    return hs[keep].mean(1),ht[keep].mean(1),ids[keep],ps,pt


def freeze(source,cache,out):
    x,y,ids,sp,tpos=training(source)
    with np.load(ORIGINAL/(source+'_vectors.npz')) as old:
        dec=old['source_decoder'];directions=old['target_dense'];atoms=old['source_atoms']
        assert np.all(np.isin(ids,old['ids']))
    matrix,_,_=bridge(x,y,np.eye(x.shape[1]),2026112018)
    err=np.max(abs(dec@matrix-directions))
    np.testing.assert_allclose(dec@matrix,directions,rtol=1e-8,atol=1e-8)
    cache.mkdir(exist_ok=True,parents=True);out.mkdir(exist_ok=True,parents=True)
    np.save(cache/(source+'_linear.npy'),matrix)
    np.savez_compressed(out/(source+'_map.npz'),ids=ids,source=source,target=OTHER[source],source_pos=sp,target_pos=tpos,
                        train_source_sha=hashed(x),train_target_sha=hashed(y),linear_sha=hashed(matrix),linear=matrix,atoms=atoms,
                        mapped_check=dec@matrix,archived_mapped=directions,checksum_max_abs=err)
    print(source,'frozen',matrix.shape,'archived atom error',err,flush=True)


def true_captures(a,site,x,pos,batch):
    windows=[];forecasts=[]
    for j in range(0,len(x),batch):
        with torch.inference_mode(),ActivationCatcher(a.module,[site]) as cap:
            forecasts.append(a.predict(x[j:j+batch],16,[.5])['quantiles'][...,0])
            raw=cap.collect()[site];w=a.postprocess_tokens(site,raw)[:,pos].float().numpy().copy()
            windows.append(w)
    return np.concatenate(windows),np.concatenate(forecasts)


def capture(name,out,batch):
    x,ids,station,origin=history()
    prior,pos,ordinary_ids=read_materials(WEATHER,name);np.testing.assert_equal(ordinary_ids,ids)
    with np.load(out/(name+'_map.npz')) as f:np.testing.assert_equal(f['source_pos'],pos)
    site=MODELS[name][-1];a=adapter_for(name)
    with np.load(WEATHER/(name+'_early.npz')) as f:archive=f['forecast'];np.testing.assert_equal(f['ids'],ids)
    clean,pub=true_captures(a,site,x,pos,batch)
    np.testing.assert_allclose(clean,prior,atol=5e-5,rtol=1e-5)
    np.testing.assert_allclose(pub,archive,atol=.0001,rtol=1e-5)
    select=last_window_slice(a,pos)
    base=forecast16(a,site,x,clean,np.zeros_like(clean[:,0]),select,batch)
    np.testing.assert_allclose(pub,base,atol=.0001,rtol=1e-5)
    rec=dict(ids=ids,station=station,origin=origin,pos=pos,clean_token=clean,base=base,sd=x.astype(float).std(1),
             clean_cache_max=float(abs(clean-prior).max()),ordinary_cache_max=float(abs(pub-archive).max()),zero_max=float(abs(pub-base).max()))
    for kind,changed in zip(('ramp','step'),edits(x)):
        h,y=true_captures(a,site,changed,pos,batch)
        token_delta=h.astype(float)-clean.astype(float);vec=token_delta.mean(1)
        rec[kind+'_token_delta']=token_delta.astype('f4');rec[kind+'_mean']=vec
        rec[kind+'_input_effect']=y.astype(float)-pub
        if kind=='ramp':rec['ramp_isolated']=forecast16(a,site,x,clean,vec.astype('f4'),select,batch).astype(float)-base
    tmp=out/(name+'_native.partial.npz');np.savez_compressed(tmp,**rec);tmp.replace(out/(name+'_native.npz'))
    print(name,'captured raw and isolated',flush=True)


def wrong_rows(stations,origins):
    n=len(np.unique(stations));order=np.random.default_rng(22112391).permutation(n)
    wrong=np.empty(n,int);wrong[order]=np.roll(order,1)
    lookup={(s,o):i for i,(s,o) in enumerate(zip(stations,origins))}
    idx=np.array([lookup[(wrong[s],o)] for s,o in zip(stations,origins)])
    assert np.all(stations[idx]!=stations);np.testing.assert_equal(origins[idx],origins);return idx


def match_norm(v,budget):
    budget=np.asarray(budget,dtype=float)
    n=np.linalg.norm(v,axis=1)
    if np.any((n==0)&(budget>0)):raise ValueError('Control is zero with a nonzero budget')
    scale=np.divide(budget,n,where=n>0,out=np.zeros_like(n));return v*scale[:,None]


def project(source,cache,out,batch):
    target=OTHER[source];m=np.load(cache/(source+'_linear.npy'));fit=dict(np.load(out/(source+'_map.npz')))
    assert str(fit['linear_sha'])==hashed(m)
    src=dict(np.load(out/(source+'_native.npz')));rec=dict(np.load(out/(target+'_native.npz')))
    x,ids,st,origin=history()
    for s in (src,rec):np.testing.assert_equal(ids,s['ids'])
    np.testing.assert_equal(rec['pos'],fit['target_pos'])
    mapped=src['ramp_mean']@m;budget=np.linalg.norm(mapped,axis=1);wrong=wrong_rows(st,origin)
    vectors=dict(mapped=mapped,wrong_step=match_norm(src['step_mean']@m,budget),other_station=match_norm(mapped[wrong],budget))
    a=adapter_for(target);site=MODELS[target][-1];select=last_window_slice(a,rec['pos'])
    result=dict(ids=ids,station=st,origin=origin,source=source,target=target,sd=rec['sd'],wrong=wrong,budget=budget,
                truth=rec['ramp_isolated'],target_hidden=rec['ramp_mean'],donor_native=src['ramp_isolated'],
                target_whole_input=rec['ramp_input_effect'],map_sha=fit['linear_sha'],source_map_atom_error=fit['checksum_max_abs'])
    for key,v in vectors.items():
        np.testing.assert_allclose(np.linalg.norm(v,axis=1),budget,atol=1e-8)
        result[key+'_vec']=v
        result[key]=forecast16(a,site,x,rec['clean_token'],v.astype('f4'),select,batch).astype(float)-rec['base']
    tmp=out/(source+'_transfer.partial.npz');np.savez_compressed(tmp,**result);tmp.replace(out/(source+'_transfer.npz'))
    print(source,'to',target,'all physical writes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=OTHER);p.add_argument('stage',choices=('freeze','capture','project'));p.add_argument('--batch',type=int,default=24)
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--cache',type=Path,default=CACHE);arg=p.parse_args();torch.set_num_threads(1)
    if arg.stage=='freeze':freeze(arg.model,arg.cache,arg.out)
    elif arg.stage=='capture':capture(arg.model,arg.out,arg.batch)
    else:project(arg.model,arg.cache,arg.out,arg.batch)
