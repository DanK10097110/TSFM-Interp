"""Pre-inference balanced-mixture extension and dominance controls.

The native study remains source-frozen. This adds a specified family to the new
replication before inference, and reuses its saved native states for controls.
"""
from pathlib import Path
import argparse,time,resource
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.selective_tree import decompose


def prepare(out,cache):
    """Append equal-amplitude mixtures to the frozen fresh-source replication."""
    from experiments.feature_search.selective_study import prepare as base_prepare
    base_prepare(out,'replication',cache);protocol=read(out/'protocol.json');rows=read(out/'rows.json');plans=read(out/'plans.json')
    f=np.load(out/'inputs.npz');histories=list(f['history']);nodes=list(f['nodes']);targets=list(np.load(out/'targets.npz')['future'])
    for j,seed in enumerate(protocol['seeds']):
        rng=np.random.default_rng(seed);t=np.arange(-128,32);phi,psi=rng.uniform(-np.pi,np.pi,2);noise=rng.normal(0,.05,len(t))
        value=.5*np.sin(2*np.pi*5.35*t/128+phi)+.5*np.sin(2*np.pi*17.35*t/128+psi)+noise
        levels,details=decompose(value[:128]);ids=[]
        for level in levels:
            group=[]
            for part in level:group.append(len(nodes));nodes.append(part)
            ids.append(group)
        rows.append(dict(source=f'syn:{seed}',domain='synthetic',case='balanced_offgrid',split='fit' if j<len(protocol['seeds'])//2 else 'test',seed=seed,index=len(rows)))
        histories.append(value[:128]);targets.append(value[128:]);plans.append(dict(levels=ids,details=details))
    for base in list(rows):
        if base['domain']=='synthetic':continue
        t=np.arange(-128,32);tone=2*np.sin(2*np.pi*t/19.2+.6);i=base['index']
        history=np.asarray(histories[i],float)+tone[:128];future=np.asarray(targets[i],float)+tone[128:]
        levels,details=decompose(history);ids=[]
        for level in levels:
            group=[]
            for part in level:group.append(len(nodes));nodes.append(part)
            ids.append(group)
        rows.append(dict(base,index=len(rows),case='real_injected',base_index=i,injected_period=19.2,injected_amplitude=2.,injected_phase=.6))
        histories.append(history);targets.append(future);plans.append(dict(levels=ids,details=details))
    np.savez_compressed(out/'inputs.npz',history=np.asarray(histories,np.float32),nodes=np.asarray(nodes,np.float32));np.savez_compressed(out/'targets.npz',future=targets)
    write(out/'rows.json',rows);write(out/'plans.json',plans);protocol['rows']=len(rows)
    protocol['real_injection']='Paired +2 past-SD period19.2 oscillation on32 real histories and futures; same source split, never independent rows. FIT policies optimize mean across original/injected contexts.'
    protocol['extension']='Balanced off-grid mixture amplitude.5+.5, same32 seeds; tests whether composition closure only reflects a dominant component. Dominant-only state control at all cuts.'
    protocol['sources'][str(Path(__file__).resolve())]=sha(__file__)
    protocol['hashes']={name:sha(out/name) for name in protocol['hashes']};write(out/'protocol.json',protocol)
    print(dict(rows=len(rows),nodes=len(nodes),extra_case='balanced_offgrid'),flush=True)


def dominant(out,model):
    """Test whether the strongest branch alone accounts for apparent hidden closure."""
    from experiments.feature_search.real_background_poc import Native
    from experiments.feature_search.selective_study import blocks,patched
    started=time.monotonic();p=read(out/'protocol.json');native=Native(model,p['checkpoints'][model]['path']);sites=blocks(native)
    x=np.load(out/'inputs.npz')['history'];states=np.load(out/f'{model}_states.npz');plans=read(out/'plans.json');rows=[]
    for i in range(len(x)):
        if len(plans[i]['levels'])<2:continue
        full=states[f'{i}_full'].astype(float);zero=states[f'{i}_zero'].astype(float);first=states[f'{i}_depth1'][:,0].astype(float)
        for j,(cut,module) in enumerate(sites.items()):
            pred=patched(native,x[i:i+1],module,first[j:j+1])[0]
            denominator=max(float(np.mean((full[j]-zero[j])**2)),1e-12)
            rows.append(dict(row=i,cut=cut,forecast=pred.tolist(),closure_nmse=float(np.mean((first[j]-full[j])**2)/denominator)))
        if time.monotonic()-started>300:raise RuntimeError('Dominance control cap')
    write(out/f'{model}_dominant.json',rows);write(out/f'{model}_dominant_execution.json',dict(seconds=time.monotonic()-started,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,sha256=sha(out/f'{model}_dominant.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','dominant']);p.add_argument('--out',type=Path,required=True);p.add_argument('--cache',type=Path,default=Path.home()/'.cache/huggingface');p.add_argument('--model',choices=['bolt','timesfm']);a=p.parse_args()
    if a.command=='prepare':prepare(a.out,a.cache)
    else:dominant(a.out,a.model)
