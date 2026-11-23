"""Finite H16 forecasts for two frozen MOMENT-large dictionaries on raw station histories."""
import argparse
import fcntl
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.groups import pursuit
from research.activation_transfer.moment.dictionary_transfer.match import OUT as MATCH, dictionary, encode
from research.activation_transfer.moment.dictionary_transfer.weather.prefix import cached_middle,obtain
from research.activation_transfer.moment.head_test.capture import scaling
from research.activation_transfer.moment.horizon_identity.train import SEEDS
from research.activation_transfer.moment.token_paths.geometry import ROOT, CACHE as MATERIAL, layers, load
from research.activation_transfer.shape_transfer.support_confirmation.noaa.data import ORIGINS, PANEL, SIGNATURE
from tsfm_lens.extraction.hooks import ActivationCatcher

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
CACHE=Path('/tmp/tsfm_moment_dictionary_weather_20261122')
ARMS=('donor','corr_own','corr_forced','random','geom_forced','group16')


def file_hash(p):
    checksum=sha256()
    with p.open('rb') as f:
        for v in iter(lambda:f.read(1024*1024),b''):checksum.update(v)
    return checksum.hexdigest()


def histories():
    with np.load(PANEL) as f:
        raw=f['context'];ids=f['ids']
    assert sha256(raw.tobytes()).hexdigest()==SIGNATURE
    x=raw.reshape(-1,480).astype('f4')
    station=np.repeat(np.arange(len(ids)),4);origin=np.tile(np.arange(4),len(ids))
    return x,np.array([f'{ids[s]}/{ORIGINS[o]}' for s,o in zip(station,origin)]),station,origin


def last_observed(adapter,length,horizon=16):
    observed,future=adapter.patch_layout(length,horizon)
    spans=adapter.token_time_spans()
    assert len(spans)==observed.stop-observed.start and observed.stop==future.start
    assert spans[-1,1]==length and np.all(spans[-1]==[length-8,length])
    return observed.stop-1,future


def live(a,x,delta=None):
    """One physical additive hook; capture the exact published head input for precision checks."""
    pos,future=last_observed(a,x.shape[1]);site=layers(a)['middle']
    seen={}
    def hook(module,args,out):
        hidden=out[0] if isinstance(out,tuple) else out
        assert future.stop==hidden.shape[1]
        seen['middle']=hidden[:,pos].detach().numpy().copy()
        if delta is None:return
        changed=hidden.clone()
        changed[:,pos]+=torch.as_tensor(delta,dtype=hidden.dtype,device=hidden.device)
        return (changed,*out[1:]) if isinstance(out,tuple) else changed
    h=a.module.get_submodule(site).register_forward_hook(hook)
    try:
        with torch.inference_mode(),ActivationCatcher(a.module,['encoder.final_layer_norm']) as cap:
            y=a.predict(x,16,[.5])['point']
            final=cap.collect()['encoder.final_layer_norm'][:,future].numpy()
            _,sd=scaling(a.module.normalizer)
        return y,final,seen['middle'],sd
    finally:h.remove()


def clean(cache,batch):
    x,ids,station,origin=histories();a=load('large')
    pos,future=last_observed(a,x.shape[1])
    _,present=a.prepare(x)
    shifted=present.roll(-16,dims=1).reshape(len(x),-1,8).all(2).numpy()
    actual=np.flatnonzero(shifted[0,:future.start])
    np.testing.assert_equal(actual[-1],pos);assert not shifted[:,future].any()
    pieces=[live(a,x[i:i+batch]) for i in range(0,len(x),batch)]
    y,final,middle,scale=[np.concatenate(v) for v in zip(*pieces)]
    # Against a separately frozen ordinary full-forward archive where locally available.
    archived=PANEL.parent/'moment_large.npz'
    if archived.exists():
        with np.load(archived) as f:np.testing.assert_allclose(y,f['raw_16'].reshape(y.shape),atol=5e-5,rtol=1e-5)
    cache.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(cache/'clean.npz',ids=ids,station=station,origin=origin,base=y,final=final,middle=middle,
                        scale=scale,input_sd=x.astype(float).std(1),pos=pos,future=np.array([future.start,future.stop]),
                        site=layers(a)['middle'],head=a.module.head.linear.weight.detach().numpy(),
                        checkpoint_sha256=file_hash(ROOT/'large/model.safetensors'),context_sha256=SIGNATURE)
    print('captured ordinary',y.shape,pos,future,flush=True)


def vectors(seed,clean,out):
    mat=MATCH/f'large_{seed}_to_large_matches.npz';rec=dict(np.load(mat))
    other=int(rec['recipient_seed']);sae=dictionary('large',seed);alt=dictionary('large',other)
    source_codes=encode(sae,clean['middle'])[:,rec['atoms']]
    other_codes=encode(alt,clean['middle']);rows,cols=np.nonzero(source_codes>0)
    meta=dict(atoms=rec['atoms'],rows=rows,cols=cols,z=source_codes,target_z=other_codes[:,rec['corr_id']],
              matched_ids=rec['corr_id'],random_ids=rec['random_id'],geom_ids=rec['geom_id'],corr=rec['corr_best'],
              cos_at_corr=rec['cos_at_corr'],match_sha256=file_hash(mat),recipient_seed=other,
              source_sae_sha256=file_hash(MATERIAL/'large'/f'sae_{seed}.pt'),
              recipient_sae_sha256=file_hash(MATERIAL/'large'/f'sae_{other}.pt'))
    dw=alt.W_dec.detach().numpy().astype(float);source=sae.W_dec.detach().numpy()[rec['atoms']].astype(float)
    np.testing.assert_allclose(source,rec['source_decoder'],atol=1e-6)
    fname=MATCH/f'large_{seed}_to_large_groups.npz'
    if fname.exists():
        with np.load(fname) as f:g=f['sixteen_vector'];gids=f['sixteen_ids'];gc=f['sixteen_coeff']
    else:
        eligible=np.flatnonzero(rec['eligible_target'])
        fits=[pursuit(v,dw[eligible],eligible)[16] for v in source]
        gids,gc,g=map(np.stack,zip(*fits))
    np.testing.assert_allclose(g,np.einsum('nk,nkd->nd',gc,dw[gids]),atol=1e-6)
    meta.update(group_ids=gids,group_coeff=gc,group_r2=1-np.sum((g-source)**2,axis=1)/np.sum(source**2,axis=1))
    amp=source_codes[rows,cols,None];donor=-amp*source[cols];dose=np.linalg.norm(donor,axis=1)[:,None]
    def matched_norm(ids):
        axes=dw[ids[cols]];return -dose*axes/np.linalg.norm(axes,axis=1,keepdims=True)
    moves=dict(donor=donor,corr_own=-meta['target_z'][rows,cols,None]*dw[rec['corr_id'][cols]],
               corr_forced=matched_norm(rec['corr_id']),random=matched_norm(rec['random_id']),
               geom_forced=matched_norm(rec['geom_id']),group16=-amp*g[cols])
    for k in ARMS:meta[k+'_norm']=np.linalg.norm(moves[k],axis=1)
    return meta,{k:v.astype('f4') for k,v in moves.items()}


def run(seed,cache,out,batch,arm):
    hist,ids,*_=histories();clean_data=dict(np.load(cache/'clean.npz'))
    np.testing.assert_array_equal(clean_data['ids'],ids)
    meta,moves=vectors(seed,clean_data,out);r=meta['rows'];c=meta['cols'];m=clean_data
    loc=cache/str(seed);loc.mkdir(exist_ok=True,parents=True);out.mkdir(exist_ok=True,parents=True)
    print(seed,len(r),'positive history/atom pairs',flush=True)
    if arm=='merge':
        for key in ('noop',*ARMS):
            with np.load(loc/(key+'.npz')) as f:
                np.testing.assert_array_equal(f['rows'],r);np.testing.assert_array_equal(f['cols'],c)
                meta[key]=f['native'];meta[key+'_head']=f['head']
        dest=out/(str(seed)+'.npz')
        np.savez_compressed(dest,**meta,**{k:m[k] for k in ('ids','station','origin','input_sd','scale','pos','future','base','site','checkpoint_sha256','context_sha256')})
        return
    key=arm;d=np.zeros_like(moves['donor']) if key=='noop' else moves[key]
    # Resume only against the same public checkpoint, firing events, vectors, masks and batch grouping.
    stamp=sha256()
    for value in (m['checkpoint_sha256'],ids,hist,r,c,d,m['base'],m['middle'],m['final'],m['head'],m['scale'],[batch]):
        stamp.update(np.ascontiguousarray(value).tobytes())
    signature=stamp.hexdigest()
    partial=loc/(key+'_batches');partial.mkdir(exist_ok=True)
    a=load('large');native=[];head=[]
    sig=sha256(np.ascontiguousarray(m['middle']).tobytes()+str(m['checkpoint_sha256']).encode()+str(m['context_sha256']).encode()).hexdigest()
    prefix=obtain(a,hist,str(m['site']),int(m['pos']),cache,sig,batch)
    np.testing.assert_allclose(prefix[:,int(m['pos'])],m['middle'],atol=3e-5,rtol=1e-6)
    # Independently run a true full native intervention, then validate the prefix shortcut at the same batch shape.
    check=r[:min(4,len(r))];edit=d[:len(check)]
    full,full_final,_,_=live(a,hist[check],edit)
    with cached_middle(a,prefix[check],str(m['site'])):
        rec,rec_final,_,_=live(a,hist[check],edit)
    np.testing.assert_allclose(full,rec,atol=.00015,rtol=1e-5)
    np.testing.assert_allclose(full_final,rec_final,atol=.0004,rtol=3e-5)
    for start in range(0,len(r),batch):
        sel=slice(start,start+batch);rows=r[sel]
        piece=partial/(str(start)+'.npz')
        if piece.exists():
            with np.load(piece) as prev:
                assert str(prev['signature'])==signature
                native.append(prev['native']);head.append(prev['head'])
            continue
        with cached_middle(a,prefix[rows],str(m['site'])):
            y,fin,hidden,sd=live(a,hist[rows],d[sel])
        np.testing.assert_allclose(hidden,m['middle'][rows],atol=5e-4,rtol=3e-5)
        np.testing.assert_allclose(sd,m['scale'][rows],rtol=2e-6)
        native.append(y.astype(float)-m['base'][rows])
        hd=(fin.astype(float)-m['final'][rows])@m['head'].astype(float).T
        head.append(hd.reshape(-1,16)*sd[:,None])
        tmp=piece.with_suffix('.partial.npz')
        np.savez_compressed(tmp,native=native[-1],head=head[-1],signature=signature);tmp.replace(piece)
        if start%(batch*8)==0:print(seed,key,start,len(r),flush=True)
    native=np.concatenate(native);head=np.concatenate(head)
    np.testing.assert_allclose(native,head,atol=.00015,rtol=.001)
    path=loc/(key+'.tmp.npz');np.savez_compressed(path,native=native,head=head,rows=r,cols=c);path.replace(loc/(key+'.npz'))
    print(seed,key,len(r),float(np.max(abs(native-head))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('clean','noop',*ARMS,'merge'));p.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--batch',type=int,default=24)
    arg=p.parse_args();torch.set_num_threads(1)
    if arg.stage=='clean':clean(arg.cache,arg.batch)
    else:
        lockdir=arg.cache/str(arg.seed);lockdir.mkdir(exist_ok=True,parents=True)
        with (lockdir/(arg.stage+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            run(arg.seed,arg.cache,arg.out,arg.batch,arg.stage)
