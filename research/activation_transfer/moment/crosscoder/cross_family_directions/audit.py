"""Independent code, physical token alignment, matched/distractor readouts and direct live additions."""
import json
import numpy as np
import torch
from tsfm_lens.extraction.hooks import ActivationCatcher
from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.forward_only.audit import add_native
from research.activation_transfer.moment.crosscoder.cross_family_directions.data import OUT,CACHE,TEST,VIEWS,METHODS,SEEDS,histories

def corr(x,y):
    return float(np.corrcoef(x,y)[0,1])

def run():
    _,_,x,ids,dom=histories()
    ix=np.array([0,63,127,128,191,255]);views={}
    for name in VIEWS:
        a=adapter_for(name);place=MODELS[name][-1]
        with np.load(OUT/(name+'_alignment.npz')) as f:positions=f['token_ids'][-1]
        with torch.inference_mode(),ActivationCatcher(a.module,[place]) as cap:
            pred=a.predict(x[ix],16,[.5])['quantiles'][...,0]
            h=a.postprocess_tokens(place,cap.collect()[place])[:,positions].numpy()
        pooled=np.load(CACHE/(name+'_pooled.npy'),mmap_mode='r')
        np.testing.assert_allclose(h.mean(1),pooled[1024+ix,-1],atol=8e-5,rtol=2e-5)
        jac=np.load(CACHE/(name+'_jac.npy'),mmap_mode='r')[ix]
        views[name]=(a,place,last_window_slice(a,positions),pred,jac)
    with np.load(OUT/'ridge.npz') as f:
        prob=[f[name+'_probe'] for name in VIEWS];dense=f['map']
    audits={}
    for m in METHODS:
        for seed in SEEDS:
            z=dict(np.load(OUT/f'{m}_{seed}_dict.npz'));meas=dict(np.load(OUT/f'{m}_{seed}_directions.npz'))
            f=dict(np.load(OUT/f'{m}_{seed}_precision.npz'));idx=4*ix+1
            np.testing.assert_array_equal(ix,f['rows'][idx])
            sf,alt,amplitude=f['feature'][idx],f['other_feature'][idx],f['dose'][idx]
            u,v=[z[k+'_decoder'].astype(float)*gain for k,gain in zip(('source','target'),z['scale'])]
            state=np.load(CACHE/'timesfm_pooled.npy',mmap_mode='r')[1024+ix,-1]
            scores=((state-z['source_mean'])/z['scale'][0]-z['source_bias'])@z['source_encoder']+z['encoder_bias']
            np.testing.assert_allclose(scores[np.arange(len(ix)),sf],amplitude,atol=2e-4,rtol=5e-5)
            np.testing.assert_array_equal(np.sort(np.argsort(scores,axis=1)[:,-32:],axis=1),meas['atom'][ix])
            vnorm=np.linalg.norm(v[sf],axis=1,keepdims=True);maprow=u[sf]@dense
            writes=dict(source=u[sf],paired=v[sf],other=v[alt]*vnorm/np.linalg.norm(v[alt],axis=1,keepdims=True),
                        dense=maprow*vnorm/np.linalg.norm(maprow,axis=1,keepdims=True))
            replay={}
            for key,vec in writes.items():
                model,place,sel,base,jac=views['timesfm' if key=='source' else 'bolt']
                with add_native(model,place,sel,(-vec*amplitude[:,None]).astype('f4')[:,None,:]):
                    y=model.predict(x[ix],16,[.5])['quantiles'][...,0].astype(float)-base
                err=float(abs(y-f[key][idx]).max());assert err<.00015
                replay[key]=err
            for q,history in enumerate(ix):
                ids_here=meas['atom'][history];j=np.flatnonzero(ids_here==sf[q]).item()
                for key,src,target in [('patch32',u[sf[q]]@prob[0],v[ids_here]@prob[1]),
                        ('forecast16',u[sf[q]]@views['timesfm'][4][q].T,v[ids_here]@views['bolt'][4][q].T)]:
                    cs=np.array([corr(src,b) for b in target])
                    np.testing.assert_allclose(meas[key+'_paired'][history,j],cs[j],atol=1e-7)
                    np.testing.assert_allclose(meas[key+'_other'][history,j],np.delete(cs,j).mean(),atol=1e-7)
            audits[m+'_'+str(seed)]=dict(cases=ids[ix].tolist(),features=sf.tolist(),native_max_by_arm=replay)
    (OUT/'independent_replay.json').write_text(json.dumps(audits,indent=2)+'\n')

if __name__=='__main__':
    torch.set_num_threads(2);run()
