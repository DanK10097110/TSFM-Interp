"""Fit the same paired-view objectives on the fifteen aligned cross-family intervals."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tsfm_lens.sae.crosscoder import CrosscoderSAE
from research.activation_transfer.moment.crosscoder import dictionaries as precedent
from research.activation_transfer.moment.crosscoder.cross_family_directions.data import CACHE,OUT,VIEWS,METHODS,SEEDS,TRAIN_COUNT,PATCHES,fingerprint

def arrays():
    return [np.load(CACHE/(name+'_pooled.npy'),mmap_mode='r') for name in VIEWS]

def selected_codes(model,means,endpoint):
    with torch.inference_mode():
        code=precedent.encode_views(model,means,endpoint.copy()).numpy()
    cols=np.sort(np.argpartition(-code,precedent.TOPK-1,axis=1)[:,:precedent.TOPK],axis=1)
    doses=np.take_along_axis(code,cols,axis=1)
    return cols,doses

def fidelity(model,means,arrays):
    result={}
    with torch.inference_mode():
        for section in ('all','endpoint'):
            vals=[a[TRAIN_COUNT:1024] if section=='all' else a[TRAIN_COUNT:1024,-1] for a in arrays]
            vals=[q.reshape(-1,q.shape[-1]) for q in vals]
            denom=np.zeros(2);sse=np.zeros((2,2))
            for k in range(0,len(vals[0]),256):
                current=[torch.from_numpy(v[k:k+256].copy()) for v in vals]
                truth=[v-m for v,m in zip(current,means)]
                denom+=[t.double().square().sum().item() for t in truth]
                for j,code in enumerate([precedent.encode_views(model,means,current[0]),precedent.encode_views(model,means,*current)]):
                    sse[j]+=[(r-t).double().square().sum().item() for r,t in zip(model.decode(code),truth)]
            result[section]={k:dict(zip(VIEWS,(1-sse[i]/denom).tolist())) for i,k in enumerate(['source_only','both'])}
    return result

def train(method,seed,out=OUT):
    a=arrays()
    means,scales=precedent.normalizers(a)
    m=CrosscoderSAE([x.shape[-1] for x in a],precedent.WIDTH,precedent.TOPK,source_scale=scales,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    shuffled=precedent.match_indices(method);sampler=np.random.default_rng(seed+100)
    masked=np.zeros(precedent.STEPS,dtype=bool)
    if method!='joint':
        masked[np.random.default_rng(seed+200).permutation(precedent.STEPS)[:precedent.STEPS//2]]=True
    order=np.empty(0,int);acc=np.zeros(2);log=[]
    for step in range(precedent.STEPS):
        if len(order)<precedent.BATCH:
            order=np.r_[order,sampler.permutation(TRAIN_COUNT*PATCHES)]
        rows,pos=np.divmod(order[:precedent.BATCH],PATCHES)
        order=order[precedent.BATCH:]
        inputs=[torch.from_numpy(a[0][rows,pos]),torch.from_numpy(a[1][shuffled[rows],pos])]
        c=precedent.encode_views(m,means,inputs[0],None if masked[step] else inputs[1])
        losses=[((fit-(raw-mean))/scale).square().mean() for fit,raw,mean,scale in zip(m.decode(c),inputs,means,scales)]
        opt.zero_grad(set_to_none=True);(losses[0]+losses[1]).backward();opt.step();m.normalize_decoder_()
        acc+=[float(l.detach()) for l in losses]
        if (step+1)%250==0:
            assert np.isfinite(acc).all()
            print(method,seed,step+1,acc/250,flush=True);log.append((acc/250).tolist());acc[:]=0
    m.eval();cols,dose=selected_codes(m,means,a[0][1024:,-1])
    active=(dose>0).sum(1)
    assert np.min(active)>=2
    out.mkdir(parents=True,exist_ok=True)
    filename=out/f'{method}_{seed}_dict.npz'
    # All trainable encoder/decoder tensors and both fixed view normalizations are retained losslessly.
    tmp=filename.with_suffix('.building')
    with tmp.open('wb') as f:
        np.savez_compressed(f,source_decoder=m.W_dec[0].detach().numpy(),target_decoder=m.W_dec[1].detach().numpy(),
                            source_encoder=m.W_enc[0].detach().numpy(),source_bias=m.b_dec[0].detach().numpy(),
                            encoder_bias=m.b_enc.detach().numpy(),source_mean=means[0],target_mean=means[1],scale=scales,
                            target_encoder=m.W_enc[1].detach().numpy(),target_bias=m.b_dec[1].detach().numpy(),
                            atom=cols,dose=dose,permutation=shuffled)
    tmp.replace(filename)
    report=dict(method=method,seed=seed,training_histories=768,physical_intervals=15,training_rows=11520,
        trained_updates=1000,masked_steps=int(masked.sum()),normalized_mse_every_250=log,validation=fidelity(m,means,a),
        positive_slots_min=int(active.min()),positive_slots_max=int(active.max()),decoder_unit_max_error=float(
        (torch.linalg.vector_norm(torch.cat(list(m.W_dec),dim=1),dim=1)-1).abs().max().detach()))
    (out/f'{method}_{seed}_training.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=METHODS);p.add_argument('seed',type=int,choices=SEEDS);p.add_argument('--out',type=Path,default=OUT);args=p.parse_args()
    torch.set_num_threads(2);train(args.method,args.seed,args.out)
