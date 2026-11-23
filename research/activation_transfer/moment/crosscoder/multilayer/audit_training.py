"""Independently replay sources and four-view validation from the released dictionary arrays."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from research.activation_transfer.moment.crosscoder.multilayer.train import OUT,METHODS,NAMES,retrieve,cd

def reconstruct(f, values):
    pre=torch.from_numpy(f['latent_bias']).clone()
    for i,raw in enumerate(values):
        name=NAMES[i]
        t=torch.from_numpy(raw.copy())-torch.from_numpy(f['mean_'+name])
        normalized=t/float(f['scale'][i])-torch.from_numpy(f['bias_'+name])
        pre=pre+normalized@torch.from_numpy(f['encoder_'+name])
    vals,indices=torch.topk(pre,32,dim=1)
    sparse=torch.zeros_like(pre)
    sparse.scatter_(1,indices,vals.clamp_min(0))
    return sparse

def run(out=OUT):
    data=retrieve();report={}
    for m in METHODS:
        for s in cd.SEEDS:
            stem=f'{m}_{s}';f=dict(np.load(out/(stem+'.npz')))
            meta=json.loads((out/(stem+'.json')).read_text())
            for k in ('m4_ids','blind_ids','blind_domains','station_ids','dates'):
                np.testing.assert_array_equal(f[k],data[k])
            np.testing.assert_array_equal(f['train_ix'],np.arange(768))
            np.testing.assert_array_equal(f['validation_ix'],np.arange(768,1024))
            np.testing.assert_array_equal(f['view_order'],NAMES)
            np.testing.assert_array_equal(np.sort(f['history_permutation']),np.arange(768))
            assert np.all(f['history_permutation']==np.arange(768)) if m=='aligned' else np.all(f['history_permutation']!=np.arange(768))
            assert int(f['source_only_step'].sum())==500
            errors={}
            for key in ('m4','blind','noaa'):
                x=data[key]
                actual=torch.cat([reconstruct(f,[x[i:i+256]]) for i in range(0,len(x),256)]).numpy()
                wanted=f[key+'_source']
                np.testing.assert_array_equal(actual>0,wanted>0)
                np.testing.assert_allclose(actual,wanted,rtol=1e-6,atol=2e-5)
                errors[key]=float(np.max(abs(actual-wanted)))
            order=('small','large','small_final','large_final')
            weights=[torch.from_numpy(f['decoder_'+n]) for n in order]
            squared=sum((w/d).double().square().sum(1) for w,d in zip(weights,f['scale']))
            uniterr=float((squared-1).abs().max());assert uniterr<5e-6
            fidelity={}
            for key,raw in [('small_middle_only',data['val'][:1]),('all_four',data['val'])]:
                z=reconstruct(f,raw)
                fidelity[key]={}
                for i,name in enumerate(NAMES):
                    truth=data['val'][i].astype(float)-f['mean_'+name]
                    prediction=(z@weights[i]).numpy()+f['bias_'+name]*f['scale'][i]
                    q=float(1-np.mean((prediction-truth)**2)/np.mean(truth**2))
                    np.testing.assert_allclose(q,meta['validation_endpoint_r2'][key][name],atol=1e-5)
                    fidelity[key][name]=q
            report[stem]=dict(all_source_codes_max_error=errors,joint_decoder_unit_norm_squared_max_error=uniterr,
                validation_recomputed=fidelity)
    (out/'training_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);opts=p.parse_args()
    torch.set_num_threads(2);run(opts.out)
