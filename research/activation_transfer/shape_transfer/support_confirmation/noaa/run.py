"""Apply frozen support/derivative policies to observed NOAA station histories."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import HERE as REPLICATION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.continuation.probe import HERE as CONTINUATION
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape,solve_adaptive
from research.activation_transfer.shape_transfer.support_confirmation.data import adapter_for_length
from research.activation_transfer.shape_transfer.support_confirmation.derivative import fit_goal
from research.activation_transfer.shape_transfer.support_confirmation.noaa.data import CACHE,OUT
from research.activation_transfer.shape_transfer.support_confirmation.run import archive_inputs,limit_norm
from research.activation_transfer.shape_transfer.support_confirmation.supports import SEEDS,active_ids,basis,count_overlap,fixed_directions
from tsfm_lens.sae.train import load_sae_checkpoint

SUPPORT=('native','frequent','zero')


def station_derangement(station,origin,seed):
    sites=np.unique(station)
    order=np.random.default_rng(seed).permutation(sites)
    donor=dict(zip(order,np.roll(order,1)))
    lookup={(int(s),int(o)):i for i,(s,o) in enumerate(zip(station,origin))}
    if len(lookup)!=len(station):raise ValueError('Duplicate station and origin')
    given=np.array([lookup[donor[s],o] for s,o in zip(station,origin)])
    np.testing.assert_array_equal(np.sort(given),np.arange(len(station)))
    assert np.all(station[given]!=station)
    np.testing.assert_array_equal(origin[given],origin)
    return given


def physical_policy(support,target,axes,x,h,rows,jt,goal,limits,weight,base,sd,pos,reassign,batch,enc,checkpoints=None):
    checksum=sha256()
    if checkpoints is not None:
        checkpoints.mkdir(exist_ok=True,parents=True)
        for value in (x,rows,axes,goal,limits,base,sd,[weight]):checksum.update(np.ascontiguousarray(value).tobytes())
    result={};site=MODELS[target][-1];sel=last_window_slice(enc,pos)
    for label,assigned in reassign.items():
        fx=[];norm=[];key=support+'_'+label
        piece=checkpoints/(target+'_'+key+'.npz') if checkpoints is not None else None
        signature=checksum.copy();signature.update(assigned.tobytes())
        stamp=signature.hexdigest()
        if piece is not None and piece.exists():
            with np.load(piece) as saved:
                assert str(saved['signature'])==stamp
                result[key+'_celsius']=saved['celsius'];result[key+'_norm']=saved['norm']
            result[key]=result[key+'_celsius']/sd[rows,None]
            continue
        for i in range(0,len(rows),batch):
            sl=slice(i,i+batch);r=rows[sl]
            delta=fit_goal(jt[assigned[r]],axes[r],goal[sl],limits[sl],weight,'balanced')
            y=forecast16(enc,site,x[r],h[r],delta,sel,batch)-base[sl]
            fx.append(y);norm.append(np.linalg.norm(delta.astype(float),axis=1))
        result[key+'_celsius']=np.concatenate(fx);result[key]=result[key+'_celsius']/sd[rows,None]
        result[key+'_norm']=np.concatenate(norm)
        if piece is not None:
            tmp=piece.with_suffix('.partial.npz')
            np.savez_compressed(tmp,signature=stamp,celsius=result[key+'_celsius'],norm=result[key+'_norm']);tmp.replace(piece)
        print(target,key,len(rows),flush=True)
    return result


def from_reference(target,stage,cache,out,batch):
    initial=dict(np.load(cache/(target+'_reference.npz')))
    if stage=='merge':
        for name in SUPPORT:
            with np.load(cache/'arms'/(target+'_'+name+'.npz')) as arm:
                np.testing.assert_array_equal(arm['ids'],initial['ids'])
                np.testing.assert_array_equal(arm['rows'],initial['rows'])
                initial.update({key:arm[key] for key in arm if key not in ('rows','ids')})
        finish=out/(target+'.partial.npz');np.savez_compressed(finish,**initial);finish.replace(out/(target+'.npz'));return
    with np.load(cache/'data.npz') as d:x,sd=d['x'],d['sd']
    with np.load(cache/(target+'_early.npz')) as d:jt=d['jac'].astype(float)/sd[:,None,None]
    h,pos,ids=read_materials(cache,target);np.testing.assert_array_equal(ids,initial['ids'])
    sae=load_sae_checkpoint(str(dictionary_path(target)));dec=sae.W_dec.detach().numpy()
    axes=basis(dec[initial[stage+'_ids']]);enc=adapter_for_length(target,480)
    reassign={'current':np.arange(len(ids)),**{str(seed):initial[f'assigned_{seed}'] for seed in SEEDS}}
    effects=physical_policy(stage,target,axes,x,h,initial['rows'],jt,initial['current_goal'],initial['budget'],
                            float(initial['weight']),initial['target_ordinary_celsius'],sd,pos,reassign,batch,enc,cache/'policies')
    segment=cache/'arms';segment.mkdir(exist_ok=True)
    tmp=segment/(target+'_'+stage+'.partial.npz')
    np.savez_compressed(tmp,ids=ids,rows=initial['rows'],**effects);tmp.replace(segment/(target+'_'+stage+'.npz'))


def run(target,cache,out,batch,stage='all'):
    if stage in (*SUPPORT,'merge'):
        from_reference(target,stage,cache,out,batch);return
    source=next(k for k in MODELS if k!=target)
    with np.load(cache/'data.npz') as f:
        x,ids,station,origin,stations,sd=[f[k] for k in ('x','ids','station','origin','source_stations','sd')]
    hs,ps,js,base_s,_,fs=archive_inputs(source,cache,ids)
    ht,pt,jt,base_t,radius,ft=archive_inputs(target,cache,ids)
    js/=sd[:,None,None];jt/=sd[:,None,None]
    with np.load(REPLICATION/'results'/(source+'_seed0.npz')) as f:atoms=f['atoms']
    with np.load(CONTINUATION/'results'/(source+'.npz')) as f:weight=float(f['weight'])
    source_sae=load_sae_checkpoint(str(dictionary_path(source)))
    with torch.inference_mode():z=source_sae.encode(torch.from_numpy(hs.mean(1))).numpy()[:,atoms]
    rows,cols=np.nonzero(z>0)
    decoder=source_sae.W_dec.detach().numpy()
    source_write=(-z[rows,cols,None]*decoder[atoms[cols]].astype(float)).astype('f4')
    source_early=np.einsum('nhd,nd->nh',js[rows],source_write.astype(float))
    proposed=solve_adaptive(boost_shape(jt[rows],weight),boost_shape(source_early[:,:,None],weight))[:,:,0]
    dense,requested,scale=limit_norm(proposed,.2*radius)
    limits=np.linalg.norm(dense.astype(float),axis=1)
    assert (limits>0).all()
    goal=np.einsum('nhd,nd->nh',jt[rows],dense.astype(float))
    src=adapter_for_length(source,480);srcsel=last_window_slice(src,ps)
    start=forecast16(src,MODELS[source][-1],x[rows],hs[rows],np.zeros_like(source_write),srcsel,batch)
    np.testing.assert_allclose(start,base_s[rows],atol=.00025,rtol=.001)
    source_c=forecast16(src,MODELS[source][-1],x[rows],hs[rows],source_write,srcsel,batch)-start
    del src
    enc=adapter_for_length(target,480);sel=last_window_slice(enc,pt);site=MODELS[target][-1]
    base=forecast16(enc,site,x[rows],ht[rows],np.zeros_like(dense),sel,batch)
    np.testing.assert_allclose(base,base_t[rows],atol=.00025,rtol=.001)
    dense_c=forecast16(enc,site,x[rows],ht[rows],dense,sel,batch)-base
    sae=load_sae_checkpoint(str(dictionary_path(target)));raw=sae.W_dec.detach().numpy()
    active=active_ids(sae,ht.mean(1));ranking,counts,_,fit_ids=fixed_directions(target,sae)
    chosen={'native':active,'frequent':np.broadcast_to(ranking[:32],active.shape),
            'zero':np.stack([ranking[~np.isin(ranking,vec)][:32] for vec in active])}
    np.testing.assert_equal(count_overlap(active,chosen['zero']),0)
    rec=dict(ids=ids,station=station,origin=origin,source_stations=stations,sd=sd,source_model=source,target_model=target,
             atoms=atoms,rows=rows,cols=cols,codes=z,source_celsius=source_c,dense_celsius=dense_c,
             source=source_c/sd[rows,None],dense=dense_c/sd[rows,None],
             source_ordinary_celsius=start,target_ordinary_celsius=base,source_early_derivative=source_early,
             current_goal=goal,dense_vector=dense,budget=limits,cap=.2*radius,weight=weight,
             requested=requested,scale=scale,source_pos=ps,target_pos=pt,fd_source=fs,fd_target=ft,
             freq_fit_ids=fit_ids,freq_training_counts=counts)
    reassign={'current':np.arange(len(ids))}
    for seed in SEEDS:
        given=station_derangement(station,origin,seed)
        rec[f'assigned_{seed}']=given
        rec[f'shared_{seed}']=count_overlap(active,active[given])
        reassign[str(seed)]=given
    for name in SUPPORT:rec[name+'_ids']=chosen[name]
    tmp=cache/(target+'_reference.partial.npz')
    np.savez_compressed(tmp,**rec);tmp.replace(cache/(target+'_reference.npz'))
    if stage=='reference':return
    for name,ids_for_arm in chosen.items():
        axes=basis(raw[ids_for_arm])
        rec.update(physical_policy(name,target,axes,x,ht,rows,jt,goal,limits,weight,base,sd,pt,reassign,batch,enc))
    np.savez_compressed(out/(target+'.npz'),**rec)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('target',choices=MODELS);p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--stage',choices=['all','reference',*SUPPORT,'merge'],default='all');p.add_argument('--out',type=Path,default=OUT);p.add_argument('--batch',type=int,default=24)
    a=p.parse_args();torch.set_num_threads(1);run(a.target,a.cache,a.out,a.batch,a.stage)
