"""Test the role of the ordered interior using the identical edit values and fixed endpoints."""
import argparse
from pathlib import Path
import numpy as np
import torch
from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.physical_ramp.study import CACHE, hashed, match_norm, true_captures
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.run import OUT as NEW, panel

OUT=Path(__file__).resolve().parent/'results'
ORDER=[0,6,5,4,3,2,1,7]

def reverse_interior(x):
    b=x.copy()
    perturbation=np.linspace(-2.,2.,8,dtype='f4')[ORDER]
    b[:,-8:]+=perturbation
    return b

def capture(name,out,new,batch):
    x,ids,_,_,_,_=panel()
    with np.load(new/(name+'_native.npz')) as f:
        np.testing.assert_equal(f['ids'],ids)
        assert str(f['raw_input_sha'])==hashed(x)
        clean=f['window'];base=f['base'];pos=f['pos']
    a=adapter_for(name);site=MODELS[name][-1]
    h,changed=true_captures(a,site,reverse_interior(x),pos,batch)
    vec=(h.astype(float)-clean.astype(float)).mean(1)
    out.mkdir(parents=True,exist_ok=True)
    result=dict(ids=ids,delta=vec,whole=changed.astype(float)-base)
    if name=='bolt':
        result['truth']=forecast16(a,site,x,clean,vec.astype('f4'),last_window_slice(a,pos),batch).astype(float)-base
    tmp=out/(name+'_perm.partial.npz');np.savez_compressed(tmp,**result);tmp.replace(out/(name+'_perm.npz'))
    print(name,'captured actual order reversal',flush=True)

def project(out,new,cache,batch):
    x,ids,st,dt,stations,dates=panel()
    with np.load(new/'transfer.npz') as first:
        np.testing.assert_equal(ids,first['ids'])
        truth_a=first['truth'];map_a=first['mapped_vec'];response_aa=first['mapped'];whole_a=first['whole']
        m=np.load(cache/'timesfm_linear.npy');assert hashed(m)==str(first['map_sha'])
    with np.load(new/'timesfm_native.npz') as f:
        np.testing.assert_equal(f['ids'],ids);np.testing.assert_allclose(f['ramp']@m,map_a,atol=1e-11)
    with np.load(out/'timesfm_perm.npz') as f:
        np.testing.assert_equal(ids,f['ids']);map_b=f['delta']@m
    with np.load(out/'bolt_perm.npz') as f:
        np.testing.assert_equal(ids,f['ids']);truth_b=f['truth'];whole_b=f['whole']
    with np.load(new/'bolt_native.npz') as f:
        np.testing.assert_equal(ids,f['ids']);clean=f['window'];base=f['base'];pos=f['pos']
    norm_a=np.linalg.norm(map_a,axis=1);norm_b=np.linalg.norm(map_b,axis=1)
    # Two capital letters are requested target reference, then donor edit.
    vectors=dict(AB=match_norm(map_b,norm_a),BA=match_norm(map_a,norm_b),BB=map_b)
    a=adapter_for('bolt');site=MODELS['bolt'][-1];sel=last_window_slice(a,pos)
    store=dict(ids=ids,station=st,origin=dt,stations=stations,dates=dates,raw_input_sha=hashed(x),map_sha=hashed(m),
               truth_A=truth_a,truth_B=truth_b,whole_A=whole_a,whole_B=whole_b,
               AA=response_aa,AA_vec=map_a,norm_A=norm_a,norm_B=norm_b)
    for code,vec in vectors.items():
        np.testing.assert_allclose(np.linalg.norm(vec,axis=1),store['norm_'+code[0]],atol=1e-10)
        store[code+'_vec']=vec
        store[code]=forecast16(a,site,x,clean,vec.astype('f4'),sel,batch).astype(float)-base
    np.savez_compressed(out/'order.npz',**store)
    print('2 x 2 genuine H16 writes ready',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('bolt','timesfm','project'));p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--new',type=Path,default=NEW);p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--batch',type=int,default=24)
    args=p.parse_args();torch.set_num_threads(1)
    if args.stage=='project':project(args.out,args.new,args.cache,args.batch)
    else:capture(args.stage,args.out,args.new,args.batch)
