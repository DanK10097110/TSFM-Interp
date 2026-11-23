"""Single atoms and frozen signed decoder groups across two SAEs of the same native backbone."""
import argparse
import fcntl
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.boundary.capture import DEFAULT_SCRATCH
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.moment.dictionary_transfer.groups import pursuit
from research.activation_transfer.moment.dictionary_transfer.match import encode
from research.activation_transfer.moment.dictionary_transfer.weather.study import file_hash
from research.activation_transfer.replication.data import CACHE as TRAIN,HERE as REP
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND,HERE as BLIND_ROOT
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.support_confirmation.noaa.data import CACHE as NOAA,OUT as NOAA_OUT
from tsfm_lens.sae.train import load_sae_checkpoint

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
CACHE=Path('/tmp/tsfm_same_encoder_groups_20261122')
ARM=('own','forced','rank1','rank4','rank16','rotated16')


def orthogonal(d,seed):
    q,r=np.linalg.qr(np.random.default_rng(seed).normal(size=(d,d)))
    q=q*np.where(np.diag(r)<0,-1,1)
    return q


def scale_like(v,ref):
    have=np.linalg.norm(v,axis=-1,keepdims=True)
    if np.any(have==0):raise ValueError('Cannot assign a nonzero norm to a zero group')
    return v*np.linalg.norm(ref,axis=-1,keepdims=True)/have


def models(name):
    return (load_sae_checkpoint(str(dictionary_path(name))),
            load_sae_checkpoint(str(DEFAULT_SCRATCH/f'seed1_{name}.pt')))


def fit(name,out):
    out.mkdir(exist_ok=True,parents=True)
    with np.load(TRAIN/'data.npz') as f:ids=f['ids'];sel=f['split']=='train'
    assert sel.sum()==1024 and len(set(ids[sel]))==1024
    h,_,hid=read_materials(TRAIN,name);np.testing.assert_array_equal(hid,ids)
    with np.load(REP/'results'/(name+'_seed0.npz')) as f:atoms=f['atoms']
    sa,ta=models(name);x=h[sel].mean(1)
    a=encode(sa,x)[:,atoms].astype(float);b=encode(ta,x).astype(float)
    count=(b>0).sum(0);pool=np.flatnonzero(count>=8)
    a-=a.mean(0);b=b[:,pool];b-=b.mean(0)
    den=np.linalg.norm(a,axis=0)[:,None]*np.linalg.norm(b,axis=0)
    corr=np.divide(a.T@b,den,where=den>0,out=np.full(den.shape,-np.inf))
    ix=corr.argmax(1);matches=pool[ix]
    source=sa.W_dec.detach().numpy()[atoms].astype(float);all_dec=ta.W_dec.detach().numpy().astype(float)
    axes=all_dec[pool]
    np.testing.assert_allclose(np.linalg.norm(axes,axis=1),1.,atol=2e-4)
    np.testing.assert_allclose(np.linalg.norm(source,axis=1),1.,atol=2e-4)
    seed=26112301 if name=='bolt' else 26112302
    rotated_axes=axes@orthogonal(axes.shape[1],seed)
    np.testing.assert_allclose(axes[:24]@axes[:24].T,rotated_axes[:24]@rotated_axes[:24].T,atol=1e-10)
    rec=dict(atoms=atoms,train_ids=ids[sel],match_ids=matches,match_corr=corr[np.arange(len(ix)),ix],
             cos_at_match=np.einsum('ij,ij->i',source,all_dec[matches]),pool=pool,counts=count,
             donor=source,match=all_dec[matches],rotation_seed=seed,
             donor_sha=file_hash(dictionary_path(name)),recipient_sha=file_hash(DEFAULT_SCRATCH/f'seed1_{name}.pt'))
    for key,basis in [('rank',axes),('rotated',rotated_axes)]:
        sets=[pursuit(v,basis,pool) for v in source]
        for r in ((1,4,16) if key=='rank' else (16,)):
            chosen,coef,vec=map(np.stack,zip(*(s[r] for s in sets)));lab=key+str(r)
            rec[lab+'_ids']=chosen;rec[lab+'_coeff']=coef;rec[lab+'_unscaled']=vec
            if key=='rotated':vec=scale_like(vec,rec['rank16'])
            rec[lab]=vec;rec[lab+'_r2']=1-np.sum((source-vec)**2,1)/np.sum(source*source,1)
    np.savez_compressed(out/(name+'_fit.npz'),**rec)
    print('fit',name,len(pool),flush=True)


def population(name,pop):
    weather=pop=='noaa';loc=NOAA if weather else BLIND
    with np.load(loc/'data.npz') as f:
        keep=np.ones(len(f['ids']),bool) if weather else f['split']==pop
        x=f['x'][keep];ids=f['ids'][keep];station=f['station'][keep] if weather else np.arange(len(x))
    h,pos,tid=read_materials(loc,name);np.testing.assert_equal(tid[keep],ids);h=h[keep]
    with np.load(loc/(name+'_early.npz')) as f:base=f['forecast'][keep];np.testing.assert_equal(f['ids'][keep],ids)
    if weather:
        other='timesfm' if name=='bolt' else 'bolt'
        fname=NOAA_OUT/(other+'.npz')
        with np.load(fname) as v:
            np.testing.assert_equal(v['ids'],ids);assert str(v['source_model'])==name
            atoms=v['atoms'];z=v['codes'];r,c=np.nonzero(z>0)
            np.testing.assert_equal(v['rows'],r);np.testing.assert_equal(v['cols'],c);donor=v['source_celsius']
    else:
        fname=BLIND_ROOT/'results'/(name+'_fresh.npz')
        with np.load(fname) as v:
            np.testing.assert_equal(v['ids'][keep],ids);atoms=v['atoms'];z=v['codes'][keep];r,c=np.nonzero(z>0);donor=v['source'][keep][r,c]
    return x,h,pos,ids,station,base,atoms,z,r,c,donor,fname


def physical(name,pop,cache,out,batch):
    fitdata=dict(np.load(out/(name+'_fit.npz')))
    x,h,pos,ids,station,base,atoms,z,r,c,donor,archive=population(name,pop)
    assert not set(ids)&set(fitdata['train_ids'])
    np.testing.assert_equal(atoms,fitdata['atoms'])
    sa,ta=models(name);hs=h.mean(1)
    code=encode(sa,hs)[:,atoms];other=encode(ta,hs)[:,fitdata['match_ids']]
    np.testing.assert_allclose(z,code,atol=2e-5,rtol=1e-5)
    np.testing.assert_equal(z>0,code>0)
    amp=z[r,c,None];ref=-amp*fitdata['donor'][c]
    def axis(arm):
        if arm=='own':return -other[r,c,None]*fitdata['match'][c]
        if arm=='forced':return scale_like(-fitdata['match'][c],ref)
        return -amp*fitdata[arm][c]
    a=adapter_for(name);sel=last_window_slice(a,pos);site=MODELS[name][-1];sd=x.astype(float).std(1)
    rec=dict(ids=ids,station=station,sd=sd,z=z,target_z=other,rows=r,cols=c,donor=donor,atoms=atoms,pos=pos,site=site,base=base,
             archive_sha=file_hash(archive))
    folder=cache/name/pop;folder.mkdir(exist_ok=True,parents=True)
    # Group output checks are made on the actual raw H16 point at a common finite additive dose.
    zi=np.unique(np.linspace(0,len(r)-1,min(len(r),12)).astype(int))
    zero=forecast16(a,site,x[r[zi]],h[r[zi]],np.zeros_like(ref[zi],dtype='f4'),sel,batch)
    rerun=forecast16(a,site,x[r[zi]],h[r[zi]],ref[zi].astype('f4'),sel,batch)-zero
    rec['donor_replay_max']=np.max(abs(rerun-donor[zi]));rec['zero_max']=np.max(abs(zero-base[r[zi]]))
    np.testing.assert_allclose(rerun,donor[zi],atol=.00028,rtol=.003)
    for arm in ARM:
        dest=folder/(arm+'.npz');delta=axis(arm).astype('f4')
        checksum=sha256()
        for v in (ids,r,c,delta,h,base,x,[batch],fitdata['donor_sha']):checksum.update(np.ascontiguousarray(v).tobytes())
        sign=checksum.hexdigest()
        if dest.exists():
            with np.load(dest) as f:
                assert str(f['signature'])==sign
                rec[arm]=f['effect'];rec[arm+'_norm']=f['norm']
            continue
        live_rows=np.flatnonzero(np.linalg.norm(delta,axis=1)>0)
        value=np.zeros((len(r),16),np.float32)
        for i in range(0,len(live_rows),batch):
            ix=live_rows[i:i+batch];idx=r[ix]
            part=folder/f'{arm}_{i}.npz'
            if part.exists():
                with np.load(part) as f:assert str(f['signature'])==sign;value[ix]=f['value']
                continue
            y=forecast16(a,site,x[idx],h[idx],delta[ix],sel,batch)-base[idx]
            value[ix]=y
            t=part.with_suffix('.partial.npz');np.savez_compressed(t,value=y,signature=sign);t.replace(part)
        rec[arm]=value;rec[arm+'_norm']=np.linalg.norm(delta.astype(float),axis=1)
        tmp=dest.with_suffix('.partial.npz');np.savez_compressed(tmp,effect=value,norm=rec[arm+'_norm'],signature=sign);tmp.replace(dest)
        print(name,pop,arm,len(live_rows),flush=True)
    tmp=out/(name+'_'+pop+'.partial.npz');np.savez_compressed(tmp,**rec);tmp.replace(out/(name+'_'+pop+'.npz'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=('bolt','timesfm'));p.add_argument('stage',choices=('fit','daily','monash_traffic','noaa'))
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--batch',type=int,default=24)
    a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='fit':fit(a.model,a.out)
    else:
        loc=a.cache/a.model/a.stage;loc.mkdir(parents=True,exist_ok=True)
        with (loc/'stage.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            physical(a.model,a.stage,a.cache,a.out,a.batch)
