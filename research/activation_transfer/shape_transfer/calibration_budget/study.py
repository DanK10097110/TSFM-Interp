"""Fit compact recipient libraries, physically measure pilot axes and new source probes, run live writes."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.replication.data import CACHE as PRIOR_CACHE
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.fresh_cohort.data import CACHE as FRESH_CACHE, OUT as FRESH_RESULTS
from research.activation_transfer.shape_transfer.forward_only.probe import central, source_steps, HERE as FORWARD
from research.activation_transfer.shape_transfer.forward_only.transfer import fit_library, shape_weight
from research.activation_transfer.shape_transfer.full_space.transfer import apply, boost_shape
from tsfm_lens.sae.train import load_sae_checkpoint

HERE=Path(__file__).resolve().parent
SEEDS=(2026112101,2026112102,2026112103)
DESIGN=((4,8),(4,24),(12,8),(16,8),(16,24),(48,8))
FRESH_ARMS=('4x24','12x8','16x24','48x8','pilot','wrong')
LEAD_ORDER=np.array([3,6,1,7,0,2,5,4])


def draws(n):
    return np.stack([np.random.default_rng(seed).permutation(n) for seed in SEEDS])


def select_pilot(jac,basis,weight):
    weighted=boost_shape(jac,weight).mean(axis=0)
    _,s,v=np.linalg.svd(weighted,full_matrices=False)
    axes=v[:8]@basis
    np.testing.assert_allclose(axes@axes.T,np.eye(8),atol=1e-9)
    return axes,s


def materials(name,cache):
    with np.load(cache/'data.npz') as f:
        x,labels,ids=f['x'],f['split'],f['ids']
    h,pos,token_ids=read_materials(cache,name)
    np.testing.assert_array_equal(ids,token_ids)
    return x,h,pos,ids,labels


def probe_and_fit(target,prior,fresh,out,batch):
    other=next(m for m in MODELS if m!=target)
    with np.load(FORWARD/'results'/(target+'_probes.npz')) as f:
        pc_jac,basis,radius,atoms=f['pc_jac'].astype(float),f['pc_basis'],float(f['radius']),f['atoms']
        training_ids,cal_ids=f['train_ids'],f['cal_ids']
    with np.load(FORWARD/'results'/(other+'_probes.npz')) as f:
        weight=shape_weight(f['random_jac'].astype(float))
    cal_x,cal_h,pos,ids,splits=materials(target,prior)
    np.testing.assert_array_equal(ids[splits=='cal'],cal_ids)
    np.testing.assert_array_equal(ids[splits=='train'],training_ids)
    cal_x,cal_h=cal_x[splits=='cal'],cal_h[splits=='cal']
    model=adapter_for(target);layer=MODELS[target][-1];selector=last_window_slice(model,pos)
    perms=draws(len(cal_x));out.mkdir(parents=True,exist_ok=True)
    record=dict(permutations=perms,cal_ids=cal_ids,seeds=np.array(SEEDS),weight=weight,radius=radius,basis=basis)
    for j,perm in enumerate(perms):
        for n,k in DESIGN:
            lib,_=fit_library(pc_jac[perm[:n],:,:k],basis[:k],weight)
            record[f'{n}x{k}_{j}']=lib
        pilot,fit=perm[:4],perm[4:40]
        assert set(pilot).isdisjoint(fit) and len(fit)==36
        axes,s=select_pilot(pc_jac[pilot],basis,weight)
        mask=np.ones((len(fit),len(axes)),bool)
        jac,_=central(model,layer,selector,cal_x[fit],cal_h[fit],axes,mask,np.full(mask.shape,.002*radius),batch)
        lib,_=fit_library(jac.transpose(0,2,1).astype(float),axes,weight)
        record[f'pilot_{j}']=lib
        record[f'pilot_axes_{j}']=axes
        record[f'pilot_singular_{j}']=s
        record[f'pilot_measured_{j}']=jac
        print(target,'library draw',j,flush=True)
    lib,_=fit_library(pc_jac,basis,weight)
    with np.load(FORWARD/'results'/(other+'_effects.npz')) as f:
        np.testing.assert_allclose(lib,f['pc_library'],rtol=1e-12,atol=1e-12)
    record['baseline']=lib
    np.savez_compressed(out/(target+'_libraries.npz'),**record)
    x,h,new_pos,fresh_ids,labels=materials(target,fresh)
    np.testing.assert_array_equal(pos,new_pos)
    assert set(fresh_ids).isdisjoint(set(ids))
    with np.load(FRESH_RESULTS/(target+'_native.npz')) as f:
        np.testing.assert_array_equal(atoms,f['atoms']);np.testing.assert_array_equal(fresh_ids,f['ids'])
        codes=f['codes']
    sae=load_sae_checkpoint(str(dictionary_path(target)))
    with torch.inference_mode():
        recomputed=sae.encode(torch.from_numpy(h.mean(axis=1))).numpy()[:,atoms]
    np.testing.assert_allclose(codes,recomputed,atol=2e-4,rtol=1e-4)
    decoder=sae.W_dec.detach().numpy()[atoms].astype(float)
    norm=np.linalg.norm(decoder,axis=1)
    steps=source_steps(codes,norm,radius)
    source_der,half=central(model,layer,selector,x,h,decoder/norm[:,None],codes>0,steps,batch)
    np.savez_compressed(out/(target+'_fresh_probes.npz'),ids=fresh_ids,split=labels,atoms=atoms,codes=codes,
        test_decoder_der=source_der*norm[None,:,None],steps=steps,half_difference=half,decoder_norm=norm)
    print(target,'fresh probes',int((codes>0).sum()),flush=True)


def dataset(source,target,cohort,prior,fresh,out,model,batch):
    probe=(FORWARD/'results'/(source+'_probes.npz') if cohort=='original' else out/(source+'_fresh_probes.npz'))
    with np.load(probe) as f:
        names,labels,atoms,codes,der=f['ids'],f['split'],f['atoms'],f['codes'],f['test_decoder_der']
    cache=prior if cohort=='original' else fresh
    x,h,pos,ids,splits=materials(target,cache)
    keep=np.isin(splits,('daily','machines')) if cohort=='original' else np.ones(len(ids),bool)
    np.testing.assert_array_equal(names,ids[keep]);np.testing.assert_array_equal(labels,splits[keep])
    x,h=x[keep],h[keep]
    layer=MODELS[target][-1];selector=last_window_slice(model,pos)
    base=change_forecasts(model,layer,x,h,None,selector,batch)
    state_args=(model,layer,selector,x,h,base,codes)
    requested=der.transpose(0,2,1)
    arms=[f'{n}x{k}' for n,k in DESIGN]+['pilot','wrong'] if cohort=='original' else list(FRESH_ARMS)
    with np.load(out/(target+'_libraries.npz')) as library:
        radius=float(library['radius'])
        archive=dict(ids=names,split=labels,atoms=atoms,codes=codes,base=base,source_est=(-codes[:,:,None]*der).astype('f4'))
        if cohort=='original':
            with np.load(FORWARD/'results'/(source+'_effects.npz')) as f:
                np.testing.assert_array_equal(names,f['ids']);np.testing.assert_allclose(base,f['base'],atol=3e-5)
                archive['baseline'],archive['baseline_clip']=f['pc_current'],f['pc_current_clipped']
        else:
            archive['baseline'],archive['baseline_clip']=apply(*state_args,library['baseline']@requested,radius,batch)
        for arm in arms:
            start=time.perf_counter(); effects=[];clips=[]
            for j in range(len(SEEDS)):
                lib=library[f"{'16x24' if arm=='wrong' else arm}_{j}"]
                ask=requested[:,LEAD_ORDER] if arm=='wrong' else requested
                result,clipped=apply(*state_args,lib@ask,radius,batch)
                effects.append(result);clips.append(clipped)
            archive[arm],archive[arm+'_clip']=np.stack(effects),np.stack(clips)
            print(source,cohort,arm,'three seeds',round(time.perf_counter()-start,1),flush=True)
            # Preserve finished physical writes across lengthy forward-only runs.
            path=out/(source+'_'+cohort+'.npz');tmp=out/(source+'_'+cohort+'.partial.npz')
            np.savez_compressed(tmp,**archive);tmp.replace(path)


def patches(source,prior,fresh,out,batch):
    target=next(m for m in MODELS if m!=source)
    model=adapter_for(target)
    for cohort in ('original','fresh'):
        dataset(source,target,cohort,prior,fresh,out,model,batch)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','apply'));p.add_argument('model',choices=MODELS)
    p.add_argument('--prior',type=Path,default=PRIOR_CACHE);p.add_argument('--fresh',type=Path,default=FRESH_CACHE)
    p.add_argument('--out',type=Path,default=HERE/'results');p.add_argument('--batch',type=int,default=16);p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(a.threads)
    (probe_and_fit if a.stage=='prepare' else patches)(a.model,a.prior,a.fresh,a.out,a.batch)
