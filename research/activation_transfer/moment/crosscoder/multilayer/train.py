"""Four-view masked crosscoders at observed middle and final token states in two MOMENT checkpoints."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch

from tsfm_lens.sae.crosscoder import CrosscoderSAE
from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.mechanism import SCRATCH
from research.activation_transfer.moment.dictionary_transfer.weather.new_dates import run as weather
from research.activation_transfer.moment.token_paths.geometry import CACHE as GEOMETRY, histories

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
NAMES=('small_middle','large_middle','small_final','large_final')
METHODS=('aligned','shuffled_depth')

def sha(x):
    return sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def retrieve():
    ids,mid=cd.paired_m4()
    x,actual,_=histories(True)
    np.testing.assert_array_equal(actual,ids)
    sources=[arr[:cd.TRAIN] for arr in mid];val_middle=[];val_final=[];final=[]
    offsets={}
    for size,view in zip(cd.SIZES,mid):
        fm=np.load(SCRATCH/'depth_probe'/f'{size}_final.npy',mmap_mode='r')
        assert fm.shape==view[:cd.TRAIN].shape
        with np.load(GEOMETRY/size/'train.npz') as old:
            np.testing.assert_array_equal(ids,old['ids']);np.testing.assert_array_equal(old['positions'],[61,62,63])
            np.testing.assert_array_equal(view[:,-1],old['middle'][:,0])
            terminal=old['readout'][:,0]
        gap=float(np.max(abs(terminal[:cd.TRAIN]-fm[:,-1])))
        np.testing.assert_allclose(terminal[:cd.TRAIN],fm[:,-1],atol=2e-5,rtol=2e-5)
        final.append(fm);val_middle.append(view[cd.TRAIN:,-1]);val_final.append(terminal[cd.TRAIN:])
        offsets[size]=gap
    with np.load(GEOMETRY/'small'/'eval.npz') as f:
        blind=f['middle'][:,0];blind_ids=f['ids'];blind_domains=f['split']
    _,real_blind,domains=histories(False)
    np.testing.assert_array_equal(blind_ids,real_blind);np.testing.assert_array_equal(domains,blind_domains)
    assert not set(ids)&set(blind_ids)
    stations,dates,observed=weather.panel()
    with np.load(weather.WORK/'physical_ramp_confirmation'/'small_base.npz') as f:
        assert sha(observed)==str(f['sig']);noaa=f['arr']
    assert noaa.shape==(385,512) and blind.shape==(256,512)
    return dict(train=[*sources,*final],val=[*val_middle,*val_final],m4=mid[0][:,-1],noaa=noaa,blind=blind,
                m4_ids=ids,blind_ids=blind_ids,blind_domains=blind_domains,station_ids=stations,dates=np.asarray(dates),
                final_max_delta=offsets,train_x_sha=sha(x[:cd.TRAIN]),noaa_x_sha=sha(observed))


def plan(method,seed):
    if method not in METHODS:raise ValueError(method)
    perm=cd.match_indices('shuffled' if method=='shuffled_depth' else 'masked')
    singles=np.zeros(cd.STEPS,dtype=bool)
    singles[np.random.default_rng(seed+200).permutation(cd.STEPS)[:cd.STEPS//2]]=True
    return perm,singles


def minibatch(arrays,rows,pos,perm):
    # Same donor row for both middle views. Same (possibly deranged) donor for both final views.
    return [torch.from_numpy(view[rows if index<2 else perm[rows],pos]) for index,view in enumerate(arrays)]


def code(model,means,values,source_only):
    n=1 if source_only else len(NAMES)
    inputs=[values[i]-means[i] for i in range(n)]
    return model.sparsify(model.pre_activations(inputs))


@torch.inference_mode()
def source_codes(model,means,raw):
    return torch.cat([code(model,means,[torch.from_numpy(raw[i:i+cd.BATCH].copy())],True)
                      for i in range(0,len(raw),cd.BATCH)]).numpy()


@torch.inference_mode()
def validation(model,means,states):
    true=[torch.from_numpy(z.copy()) for z in states]
    scales=[(x-mu).double().square().sum().item() for x,mu in zip(true,means)]
    report={};source=None
    for name,hide in [('small_middle_only',True),('all_four',False)]:
        codes=code(model,means,true,hide)
        loss=[(r-(x-mu)).double().square().sum().item() for r,x,mu in zip(model.decode(codes),true,means)]
        report[name]={k:float(1-v/d) for k,v,d in zip(NAMES,loss,scales)}
        if hide:source=codes
        else:
            intersect=((source>0)&(codes>0)).sum(1)
            union=((source>0)|(codes>0)).sum(1)
            report['source_full_support_jaccard']=float((intersect/union).mean())
    return report


def fit(method,seed,out):
    data=retrieve();arrays=data['train']
    means,scales=cd.normalizers(arrays)
    np.testing.assert_equal([v.shape[:2] for v in arrays],[(768,60)]*4)
    dim=[v.shape[-1] for v in arrays]
    model=CrosscoderSAE(dim,cd.WIDTH,cd.TOPK,source_scale=scales,generator=torch.Generator().manual_seed(seed))
    optim=torch.optim.Adam(model.parameters(),lr=.001)
    perm,singles=plan(method,seed);rng=np.random.default_rng(seed+100)
    order=np.array([],int);running=np.zeros(4);losses=[]
    for step in range(cd.STEPS):
        if len(order)<cd.BATCH:order=np.r_[order,rng.permutation(cd.TRAIN*cd.POSITIONS)]
        rows,pos=np.divmod(order[:cd.BATCH],cd.POSITIONS);order=order[cd.BATCH:]
        x=minibatch(arrays,rows,pos,perm)
        c=code(model,means,x,singles[step])
        parts=[((r-(v-mu))/s).square().mean() for r,v,mu,s in zip(model.decode(c),x,means,scales)]
        optloss=sum(parts);assert torch.isfinite(optloss)
        optim.zero_grad(set_to_none=True);optloss.backward();optim.step();model.normalize_decoder_()
        running += [float(q.detach()) for q in parts]
        if (step+1)%250==0:
            losses.append((running/250).tolist());running[:]=0
            print(method,seed,step+1,losses[-1],flush=True)
    model.eval()
    val=validation(model,means,data['val'])
    out.mkdir(parents=True,exist_ok=True)
    saved={k:data[k] for k in ('m4_ids','blind_ids','blind_domains','station_ids','dates')}
    saved.update(view_order=np.asarray(NAMES),dims=dim,width=cd.WIDTH,topk=cd.TOPK,train_ix=np.arange(768),
                 validation_ix=np.arange(768,1024),history_permutation=perm,source_only_step=singles,
                 m4_source=source_codes(model,means,data['m4']),noaa_source=source_codes(model,means,data['noaa']),
                 blind_source=source_codes(model,means,data['blind']),scale=scales,
                 latent_bias=model.b_enc.detach().numpy(),seed=seed,method=method)
    for i,name in enumerate(NAMES):
        saved['mean_'+name]=means[i].numpy()
        saved['encoder_'+name]=model.W_enc[i].detach().numpy()
        saved['bias_'+name]=model.b_dec[i].detach().numpy()
        label=('small','large','small_final','large_final')[i]
        saved['decoder_'+label]=(model.W_dec[i].detach()*scales[i]).numpy()
    for key in ('m4_source','noaa_source','blind_source'):
        positive=(saved[key]>0).sum(1)
        assert positive.min()>0 and positive.max()==32 and np.isfinite(saved[key]).all()
    path=out/f'{method}_{seed}.npz';tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,**saved)
    tmp.replace(path)
    param_count=sum(p.numel() for p in model.parameters())
    report=dict(method=method,seed=seed,view_order=NAMES,params=param_count,width=cd.WIDTH,topk=cd.TOPK,
                updates=cd.STEPS,small_middle_only_updates=int(singles.sum()),batch=cd.BATCH,
                train_histories=cd.TRAIN,patches_per_history=cd.POSITIONS,train_rows=cd.TRAIN*cd.POSITIONS,
                nonidentity_final_history_pairs=int(np.sum(perm!=np.arange(cd.TRAIN))),
                rms_scales=scales,validation_endpoint_r2=val,normalized_view_mse_every_250=losses,
                selected_input_checks=dict(final_middle_cleanup_at_endpoint=data['final_max_delta'],
                    train_history_sha=data['train_x_sha'],noaa_history_sha=data['noaa_x_sha'],
                    ids_sha=sha(data['m4_ids']),training_ids_sha=sha(data['m4_ids'][:768])),
                firing_scopes={key:dict(rows=len(saved[key]),positive_min=int((saved[key]>0).sum(1).min()),
                    positive_max=int((saved[key]>0).sum(1).max()),distinct=int(((saved[key]>0).sum(0)>0).sum())) for key in ('m4_source','blind_source','noaa_source')})
    (out/f'{method}_{seed}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=METHODS);p.add_argument('seed',type=int,choices=cd.SEEDS)
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();torch.set_num_threads(args.threads);fit(args.method,args.seed,args.out)
