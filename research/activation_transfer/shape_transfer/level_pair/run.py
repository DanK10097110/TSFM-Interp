"""Read archived SAE coordinates on controlled input edits and test decoder terms in live forecasts."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.replication.data import HERE as REP
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.level_pair.inputs import CACHE,CUES,HERE
from tsfm_lens.extraction.extract import capture_raw_tokens
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.sae.train import load_sae_checkpoint

NAMED={'bolt':5432,'timesfm':742}
ARMS=('intact','named','all','tokens','control0','control1','control2')


def capture(model,layer,pos,x,batch):
    return np.concatenate([capture_raw_tokens(model,x[i:i+batch],[layer])[layer][:,pos].numpy()
                           for i in range(0,len(x),batch)])


def replacement_forecast(model,layer,selector,x,replacement,batch):
    outputs=[]
    for i in range(0,len(x),batch):
        block=torch.from_numpy(replacement[i:i+batch])
        with token_patch(model.module,layer,selector,block):
            outputs.append(model.predict(x[i:i+batch],8,[.5])['quantiles'][:,:,0])
    return np.concatenate(outputs)


def decode_states(sae,h,atom):
    with torch.inference_mode():
        hidden=torch.from_numpy(h.mean(axis=1))
        pre=sae.pre_activations(hidden)
        code=sae.sparsify(pre)
        return code[:,atom].numpy(),pre[:,atom].numpy(),sae.decode(code).numpy()


def term_corrections(z0,z,decoded0,decoded,unit,named,controls):
    scalar=z0-z
    correction=scalar[:,None]*unit[named]
    norm=np.linalg.norm(unit[named])
    moves={'named':correction,'all':decoded0-decoded}
    for j,k in enumerate(controls):
        moves['control'+str(j)]=scalar[:,None]*unit[k]*(norm/np.linalg.norm(unit[k]))
    return moves


def merge_chunks(name, out, chunks):
    parts=[]
    for cue in CUES:
        with np.load(chunks/(name+'_'+cue+'.npz')) as f:parts.append(dict(f))
    final={}
    for key in parts[0]:
        if key.startswith('cube_'):
            final[key[5:]]=np.stack([part[key] for part in parts],axis=1)
        else:
            value=parts[0][key]
            for part in parts[1:]:np.testing.assert_array_equal(part[key],value)
            final[key]=value
    path=out/(name+'_effects.npz');temp=out/(name+'_effects.partial.npz')
    np.savez_compressed(temp,**final);temp.replace(path)


def run(name,cache,out,batch,only=None,chunks=None):
    with np.load(out/'inputs.npz') as f:
        x,edits,ids,labels=f['inputs'],f['edits'],f['ids'],f['split']
    model=adapter_for(name);layer=MODELS[name][-1]
    saved,pos,cache_ids=read_materials(cache,name);np.testing.assert_array_equal(ids,cache_ids)
    selector=last_window_slice(model,pos)
    clean=capture(model,layer,pos,x,batch)
    np.testing.assert_allclose(clean,saved,atol=6e-5,rtol=2e-5)
    base=change_forecasts(model,layer,x,clean,None,selector,batch)
    same=replacement_forecast(model,layer,selector,x,clean,batch)
    np.testing.assert_allclose(base,same,atol=5e-5,rtol=.001)
    with np.load(cache/(name+'_early.npz')) as g:
        np.testing.assert_array_equal(ids,g['ids']);np.testing.assert_allclose(base,g['forecast'][:,:8],atol=1e-4,rtol=.002)
    atom=NAMED[name];sae=load_sae_checkpoint(str(dictionary_path(name)));decoder=sae.W_dec.detach().numpy()
    with np.load(REP/'results'/(name+'_seed0.npz')) as a:
        controls=np.array([int(i) for i in a['atoms'] if i!=atom][:3])
    assert len(set(controls))==3 and atom not in controls
    z0,p0,d0=decode_states(sae,clean,atom)
    shape=(len(ids),len(CUES),2)
    record=dict(ids=ids,split=labels,clean_forecast=base,clean_z=z0,clean_pre=p0,atoms=np.r_[atom,controls],
                positions=pos,cues=np.array(CUES),arms=np.array(ARMS),site=layer,
                capture_max_abs=np.max(abs(clean-saved)),clean_self_patch_max_abs=np.max(abs(base-same)))
    for arm in ARMS:record[arm]=np.empty((*shape,8),dtype='f4')
    for label in ('edited_z','edited_pre','named_lock_z','named_lock_pre','all_lock_z','all_lock_pre'):
        record[label]=np.empty(shape,dtype='f4')
    for key in ARMS[1:]:record[key+'_norm']=np.empty(shape,dtype='f4')
    for i,cue in enumerate(CUES):
        if only is not None and cue!=only:continue
        for j in range(2):
            obs=edits[:,i,j];h=capture(model,layer,pos,obs,batch)
            ordinary=change_forecasts(model,layer,obs,h,None,selector,batch)
            record['intact'][:,i,j]=ordinary
            z,pre,d=decode_states(sae,h,atom)
            record['edited_z'][:,i,j]=z;record['edited_pre'][:,i,j]=pre
            moves=term_corrections(z0,z,d0,d,decoder,atom,controls)
            for arm in ARMS[1:]:
                if arm=='tokens':
                    replacement=clean
                    length=np.linalg.norm((h-clean).reshape(len(h),-1),axis=1)
                else:
                    dr=moves[arm].astype('f4');replacement=h+dr[:,None]
                    length=np.linalg.norm(dr.astype(float),axis=1)
                record[arm+'_norm'][:,i,j]=length
                if arm in ('named','all'):
                    post,postpre,_=decode_states(sae,replacement,atom)
                    record[arm+'_lock_z'][:,i,j]=post;record[arm+'_lock_pre'][:,i,j]=postpre
                active=length>0
                response=ordinary.copy()
                if active.any():response[active]=replacement_forecast(model,layer,selector,obs[active],replacement[active],batch)
                record[arm][:,i,j]=response
            for k in range(3):
                np.testing.assert_allclose(record['control'+str(k)+'_norm'][:,i,j],record['named_norm'][:,i,j],atol=5e-5,rtol=1e-5)
        if chunks is not None:
            chunks.mkdir(parents=True,exist_ok=True)
            part={}
            for key,val in record.items():
                value=np.asarray(val)
                if value.ndim>=3 and value.shape[:3]==shape:part['cube_'+key]=value[:,i]
                else:part[key]=value
            dest=chunks/(name+'_'+cue+'.npz');tmp=chunks/(name+'_'+cue+'.partial.npz')
            np.savez_compressed(tmp,**part);tmp.replace(dest)
        print(name,cue,'complete',flush=True)
    if only is None:np.savez_compressed(out/(name+'_effects.npz'),**record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',choices=MODELS);p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--out',type=Path,default=HERE/'results');p.add_argument('--batch',type=int,default=16)
    p.add_argument('--cue',choices=(*CUES,'merge'));p.add_argument('--chunks',type=Path);a=p.parse_args()
    if a.cue is not None and a.chunks is None:p.error('--cue requires --chunks')
    torch.set_num_threads(1)
    if a.cue=='merge':merge_chunks(a.model,a.out,a.chunks)
    else:run(a.model,a.cache,a.out,a.batch,a.cue,a.chunks)
