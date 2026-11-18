"""Fixed stronger-pattern control when conservative extraction abstains.

Same16 TEST real sources, amplitude8 at period19.2, unchanged extraction rule.
This is an exploratory feasibility control, not additional independent data.
"""
from pathlib import Path
import argparse,time,resource
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha,normalization
from experiments.feature_search.selective_tree import decompose,continuation,fuse


def worker(out,model):
    """Native queries use past histories only; actual targets are scored separately."""
    from experiments.feature_search.real_background_poc import Native
    from experiments.feature_search.selective_study import blocks,capture,patched
    from experiments.feature_search.composition_restoration import random_edits
    start=time.monotonic();p=read(out/'protocol.json');rows=read(out/'rows.json');x=np.load(out/'inputs.npz')['history'];native=Native(model,p['checkpoints'][model]['path']);all_sites=blocks(native);sites={k:all_sites[k] for k in [list(all_sites)[0],list(all_sites)[-1]]}
    prefix=model+'_strong';write(out/f'{prefix}_protocol.json',dict(amplitude=8,period=19.2,phase=.6,depths=[1,3],layers=list(sites),weightings=['equal','learned'],blend=1,source_sha256=sha(__file__),scope='Declared after moderate-injection extraction abstention; same TEST sources, fixed stronger amplitude, no tuning'))
    result=[];metadata=[];saved={};tone=8*np.sin(2*np.pi*np.arange(-128,32)/19.2+.6)
    for i,row in enumerate(rows):
        if row['case']!='real' or row['split']!='test':continue
        query=x[i:i+1]+tone[:128];levels,detail=decompose(query[0]);pred,h=capture(native,query,sites)
        metadata.append(dict(row=i,source=row['source'],domain=row['domain'],accepted=len(detail),details=detail))
        saved[f'{i}_full']=np.stack([h[k][0] for k in sites])
        result.append(dict(row=i,method='native',depth=0,cut='none',weight='none',blend=0.,forecast=pred[0].tolist()))
        for cut,module in sites.items():
            if not np.array_equal(patched(native,query,module,h[cut]),pred):raise ValueError('Strong self patch')
        with normalization(native,query,'background'):zero,hzero=capture(native,np.zeros_like(query),sites)
        saved[f'{i}_zero']=np.stack([hzero[k][0] for k in sites]);freezero=native.predict(np.zeros_like(query))[0][0]
        for requested in [1,3]:
            depth=min(requested,len(detail))
            if depth==0:continue
            leaves=np.asarray(levels[depth],np.float32);free=native.predict(leaves)[0]
            with normalization(native,np.repeat(query,len(leaves),axis=0),'background'):locked,hs=capture(native,leaves,sites)
            saved[f'{i}_depth{depth}']=np.stack([hs[k] for k in sites])
            for method,value in [('output_free',free.sum(0)-depth*freezero),('output_locked',locked.sum(0)-depth*zero[0]),('hybrid',continuation(detail,depth)+free[-1])]:result.append(dict(row=i,method=method,depth=requested,actual_depth=depth,cut='none',weight='none',blend=1.,forecast=value.tolist()))
            for cut,module in sites.items():
                variants=[];info=[]
                for weight in ['equal','learned']:
                    w=np.ones(depth+1)
                    if weight=='learned':a,b=p['learned_weights'][model][str(depth)][cut]['weights'];w[:-1]=a;w[-1]=b
                    proposed=fuse(hs[cut],hzero[cut][0],w);edit=proposed-h[cut][0]
                    random,invalid=random_edits(h[cut],edit[None],np.random.default_rng(4947000+i*10+int(cut)),count=1)
                    for method,value,valid in [('fusion',proposed,True),('random',h[cut][0]+random[0,0],not bool(invalid[0]))]:
                        variants.append(value);info.append(dict(row=i,method=method,depth=requested,actual_depth=depth,cut=cut,weight=weight,blend=1.,valid=valid,closure_nmse=float(np.mean((value-h[cut][0])**2)/max(float(np.mean((h[cut][0]-hzero[cut][0])**2)),1e-12))))
                forecast=patched(native,np.repeat(query,len(variants),axis=0),module,variants)
                result.extend(dict(meta,forecast=v.tolist()) for meta,v in zip(info,forecast))
        if time.monotonic()-start>180 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>3*2**30:raise RuntimeError('Strong control cap')
    write(out/f'{prefix}_forecasts.json',result);write(out/f'{prefix}_metadata.json',metadata);np.savez_compressed(out/f'{prefix}_states.npz',**saved);write(out/f'{prefix}_execution.json',dict(seconds=time.monotonic()-start,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,sha256=sha(out/f'{prefix}_forecasts.json')));print(model,'strong control',len(metadata),flush=True)


def analyze(out):
    """Compare consistent method changes on original and strongly injected contexts."""
    from experiments.feature_search.selective_analysis import resolve,key
    metadata=read(out/'rows.json');plans=read(out/'plans.json');x=np.load(out/'inputs.npz')['history'];y=np.load(out/'targets.npz')['future'];expected=8*np.sin(2*np.pi*np.arange(32)/19.2+.6);results=[]
    for model in ['bolt','timesfm']:
        f=read(out/f'{model}_forecasts.json');base_lookup={(r['row'],key(r)):r for r in f}
        strong=read(out/f'{model}_strong_forecasts.json');info={r['row']:r for r in read(out/f'{model}_strong_metadata.json')};lookup={(r['row'],key(r)):r for r in strong};cuts=read(out/f'{model}_strong_protocol.json')['layers']
        configs=[('native',0,'none','none',0.)]+[(m,d,'none','none',1.) for m in ['output_free','output_locked','hybrid'] for d in [1,3]]+[(m,d,c,w,1.) for m in ['fusion','random'] for d in [1,3] for c in cuts for w in ['equal','learned']]
        for i,meta in info.items():
            for config in configs:
                actual=lookup.get((i,config),lookup[i,('native',0,'none','none',0.)]);pred=np.asarray(actual['forecast']);base=np.asarray(resolve(config,i,base_lookup,plans)['forecast']);delta=pred-base
                results.append(dict(model=model,source=metadata[i]['source'],domain=metadata[i]['domain'],row=i,config=list(config),accepted=meta['accepted'],valid=actual.get('valid',True),response_nmse=float(np.mean((delta-expected)**2)/np.mean(expected**2)),loss_original_units=float(np.mean((pred-y[i]-expected)**2)/np.var(x[i].astype(float))),clean_loss=float(np.mean((base-y[i])**2)/np.var(x[i].astype(float))),closure_nmse=actual.get('closure_nmse')))
    write(out/'strong_signal_scores.json',results);print('strong scores',len(results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['worker','analyze']);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',choices=['bolt','timesfm']);a=p.parse_args()
    if a.command=='worker':worker(a.out,a.model)
    else:analyze(a.out)
