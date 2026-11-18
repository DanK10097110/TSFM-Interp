"""Bounded frozen-dictionary native preservation; no feature removals or fitting.

Twenty baseline receivers and fifteen prespecified paired-response receivers
leave room for exact gates and independent replays inside399 singleton cells.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.feature_search.dictionary_response_poc import read, write, sha

METHODS=('lambda0','lambda1','pca')


def design(rows):
    """Reuse existing20 receivers; first3/domain get both response directions."""
    from experiments.feature_search.dictionary_native import receiver_indices
    base=receiver_indices(rows)
    paired=receiver_indices(rows,per_domain=3)
    cells=6*len(base)+18*len(paired)+9
    if len(base)!=20 or len(paired)!=15 or cells>400:
        raise ValueError('Unexpected receiver design')
    return dict(base=base,paired=paired,cells=cells,replay_receiver=base[0])


def distortion(pred,reference,scale):
    """FP64 baseline error in physical and history-variance units."""
    import numpy as np
    error=float(np.mean((np.asarray(pred,float)-np.asarray(reference,float))**2))
    return dict(mse=error,history_variance_nmse=error/scale**2)


def response_score(native_pair,rebuilt_pair,scale):
    """Actual forecast central difference; flag small denominators, never hide them."""
    import numpy as np
    actual=np.asarray(native_pair[0],float)-np.asarray(native_pair[1],float)
    rebuilt=np.asarray(rebuilt_pair[0],float)-np.asarray(rebuilt_pair[1],float)
    energy=float(np.mean(actual**2))
    error=float(np.mean((rebuilt-actual)**2))
    floor=1e-12*scale**2
    return dict(native_response_energy=energy,error_energy=error,denominator_floor=floor,
                small_response=energy<=floor,nmse=None if energy<=floor else error/energy,
                floored_nmse=error/max(energy,floor),error_history_variance=error/scale**2)


def raw_predict(native,query,site=None,replacement=None):
    """Independent Bolt pipeline path, with no hook unless replacement requested."""
    import numpy as np
    import torch
    from tsfm_lens.extraction.hooks import _primary,_rebuild
    fired=[]
    def hook(module,args,output):
        hidden=_primary(output)
        value=torch.as_tensor(replacement,dtype=hidden.dtype,device=hidden.device)
        if value.shape!=hidden.shape:
            raise ValueError('Replay shape mismatch')
        fired.append(1)
        return _rebuild(output,value)
    handle=site.register_forward_hook(hook) if replacement is not None else None
    try:
        with torch.inference_mode():
            _,forecast=native.pipeline.predict_quantiles(torch.as_tensor(np.ascontiguousarray(query),dtype=torch.float32),prediction_length=32,quantile_levels=[.5])
        if handle is not None and len(fired)!=1:
            raise ValueError('Independent hook execution mismatch')
        return forecast.numpy()
    finally:
        if handle is not None:
            handle.remove()


def summarize(records):
    """One receiver/source; no tokens, signs or methods treated as replicates."""
    import numpy as np
    result={}
    for method in METHODS:
        base=[r['baseline'][method]['history_variance_nmse'] for r in records]
        item=dict(baseline_n_sources=len(base),baseline_mean=float(np.mean(base)),responses={})
        for direction in ('sinusoid','noise'):
            own=[r['response'][direction][method] for r in records if direction in r['response']]
            valid=[r['nmse'] for r in own if r['nmse'] is not None]
            item['responses'][direction]=dict(n_sources=len(own),valid_sources=len(valid),small_response_sources=sum(r['small_response'] for r in own),
                mean_source_nmse=float(np.mean(valid)) if valid else None,
                mean_source_floored_nmse=float(np.mean([r['floored_nmse'] for r in own])),
                ratio_of_energies=float(sum(r['error_energy'] for r in own)/sum(r['native_response_energy'] for r in own)) if sum(r['native_response_energy'] for r in own)>0 else None)
        result[method]=item
    return result


def prepare(out):
    """Freeze exact checkpoints, dictionaries, receivers, budgets and analysis."""
    from experiments.feature_search import dictionary_core,dictionary_native,selective_study,coherence_study,real_background_poc
    source=Path('runs/selective_replication4942000').resolve()
    trained=Path('runs/dictionary_response4965000').resolve()
    old=Path('runs/dictionary_pilot4951000/recovery_exit_monitor').resolve()
    artifacts={m:trained/m/'dictionary.npz' for m in METHODS[:2]}
    artifacts['pca']=old/'fit_bolt_cut0_pca_k16_s4951000/dictionary.npz'
    assert read(trained/'comparison.json')['gate']['passed']
    selection=design(read(source/'rows.json'))
    files=[source/n for n in ('rows.json','inputs.npz','bolt_states.npz','bolt_worker.json','protocol.json')]
    files+=list(artifacts.values())+[trained/'protocol.json',trained/'comparison.json',old/'native_bolt_cut0_pca_k16_s4951000/results.json']
    files += [Path(__file__),Path('tests/test_dictionary_response_native.py')]+[Path(m.__file__) for m in (dictionary_core,dictionary_native,selective_study,coherence_study,real_background_poc)]
    cp=read(source/'protocol.json')['checkpoints']['bolt']
    files += [Path(cp['path'])/name for name in cp['weights']]
    files += [Path('tsfm_lens/sae/models.py'),Path('tsfm_lens/extraction/hooks.py'),Path('experiments/feature_search/dictionary_response_poc.py')]
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(model='bolt',cut='0',horizon=32,k=16,width=512,source=str(source),training=str(trained),old_root=str(old),
        artifacts={m:str(p) for m,p in artifacts.items()},design=selection,
        selection='same20 first existing TEST receivers,4/domain; paired subset first3/domain,15 total; outcomes never select receivers',
        cell_accounting='20*(base capture+self patch+3 reconstructions+true nohook restoration)=120;15*(4 locked captures+12 reconstructions+2 plus-sign identity replays)=270;9 independent pipeline replays=399',
        identity='every baseline exact saved state, original saved forecast, self patch, locked-anchor identity and final true nohook restoration; plus sign of each direction exact native self patch; minus branches lack extra self replay to respect cell cap',
        pair_definition='sin(2*pi*5.35*t/128+.3), noise rng(4944000+row), each RMS normalized to background SD, +/- .1, actual Bolt background-locked normalizer and inverse; reconstruct ACTUAL fresh singleton states',
        reinsertion='same perturbed input under same background lock; replace whole first-block output with frozen decoded state',
        denominators='baseline history variance; response actual native plus-minus energy; floor1e-12*history_variance; below-floor raw NMSE=null, retain explicitly floored NMSE and physical errors',
        aggregation='equal source mean; baseline20; response15 separately; domains retained; no CI, p-value, or pair-as-independent claims',
        independent_replay='first baseline receiver only: all3 dictionaries on base, sinusoid+, noise-; second pipeline/hook implementation, exact forecast equality; independent raw-array arithmetic all cells',
        seconds=180,rss_bytes=3*2**30,threads=1,max_native_cells=400,files={str(p.resolve()):sha(p) for p in files},
        limitations=['descriptive native forecast preservation, not semantics or future accuracy','single learned dictionary seed; no training or grid',
                    'paired subset15 sources,3/domain; baseline20 sources,4/domain; all sources previously inspected',
                    'PCA uses input-dependent topk support in a full frozen orthogonal basis, not fixed-rank projection',
                    'residual first-block output, not FFN neurons; no causal feature-removal controls'],
        closure='close branch regardless of outcome; no optimization or automatic additional native run')
    write(out/'protocol.json',protocol)
    write(out/'knowledge.json',dict(question='Does response-aware hidden reconstruction preserve actual native baseline and weak forecast responses?',status='frozen_ready',exact_results={'protocol':'protocol.json','cells':399},controls=[protocol['identity']],limitations=protocol['limitations'],next_step=protocol['closure']))


def worker(out):
    """Run only the frozen native plan; partial raw results survive any failure."""
    import numpy as np
    import torch
    from experiments.feature_search import dictionary_core as core
    from experiments.feature_search.dictionary_native import load_native,exact_gate
    from experiments.feature_search.selective_study import blocks,capture,patched
    from experiments.feature_search.coherence_study import normalization
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=read(out/'protocol.json')
    for path,digest in p['files'].items():
        if sha(path)!=digest:
            raise ValueError('Frozen input changed: '+path)
    source=Path(p['source']);rows=read(source/'rows.json')
    inputs=np.load(source/'inputs.npz')['history']
    artifacts={m:dict(np.load(path)) for m,path in p['artifacts'].items()}
    old={r['row']:r for r in read(Path(p['old_root'])/'native_bolt_cut0_pca_k16_s4951000/results.json')}
    native=load_native(source,'bolt');site=blocks(native)['0']
    calls=[0]
    original=native.predict
    def counted(*args,**kwargs):
        calls[0]+=1
        if calls[0]>p['max_native_cells']:
            raise ValueError('Native cell cap')
        return original(*args,**kwargs)
    native.predict=counted
    def independent(query,replacement=None):
        calls[0]+=1
        if calls[0]>p['max_native_cells']:
            raise ValueError('Native cell cap')
        return raw_predict(native,query,site,replacement)
    def reconstruction(h,method):
        a=artifacts[method]
        z=core.encode((h-a['center'])/float(a['scale']),a)
        return (core.reconstruct(z,a)*float(a['scale'])+a['center']).astype(np.float32)
    states={};forecasts={};records=[];replays=[]
    original_norm=native.model.instance_norm.forward
    cut=read(source/'bolt_worker.json')['cuts'].index('0')
    initial_hooks={id(m):len(m._forward_hooks) for m in native.model.modules()}
    with np.load(source/'bolt_states.npz') as cached:
        for i in p['design']['base']:
            query=inputs[i:i+1];sd=float(query.std())
            if sd<=1e-8:
                raise ValueError('Degenerate receiver scale')
            base,h=capture(native,query,{'0':site});h=h['0']
            exact_gate(h,cached[f'{i}_full'][cut:cut+1],'saved baseline state')
            exact_gate(base,np.asarray(old[i]['baseline'],np.float32)[None],'original baseline forecast')
            with normalization(native,query,'background'):
                exact_gate(base,patched(native,query,site,h),'locked baseline self identity')
            states[f'{i}_base']=h;forecasts[f'{i}_base_native']=base
            record=dict(row=i,source=rows[i]['source'],domain=rows[i]['domain'],history_sd=sd,baseline={},response={},exact_gates={})
            for method in METHODS:
                rec=reconstruction(h,method)
                pred=patched(native,query,site,rec)
                states[f'{i}_base_{method}']=rec;forecasts[f'{i}_base_{method}']=pred
                record['baseline'][method]=distortion(pred,base,sd)
                if i==p['design']['replay_receiver']:
                    check=independent(query,rec)
                    exact_gate(pred,check,'independent baseline replay')
                    forecasts[f'{i}_base_{method}_independent']=check
                    replays.append(dict(row=i,state='base',method=method,exact=True))
            if i in p['design']['paired']:
                for direction in ('sinusoid','noise'):
                    raw=np.sin(2*np.pi*5.35*np.arange(-128,0)/128+.3) if direction=='sinusoid' else np.random.default_rng(4944000+i).normal(size=128)
                    raw=raw/np.sqrt(np.mean(raw**2))*sd
                    natural=[];rebuilt={m:[] for m in METHODS}
                    for sign in (1,-1):
                        q=query+sign*.1*raw;key=f'{i}_{direction}_{sign}'
                        with normalization(native,query,'background'):
                            pred,hs=capture(native,q,{'0':site});hs=hs['0']
                            natural.append(pred);forecasts[key+'_native']=pred;states[key]=hs;states[key+'_input']=q
                            if sign==1:
                                exact_gate(pred,patched(native,q,site,hs),'locked plus self identity')
                            for method in METHODS:
                                rec=reconstruction(hs,method)
                                value=patched(native,q,site,rec)
                                rebuilt[method].append(value)
                                forecasts[key+'_'+method]=value;states[key+'_'+method]=rec
                                if i==p['design']['replay_receiver'] and ((direction=='sinusoid' and sign==1) or (direction=='noise' and sign==-1)):
                                    check=independent(q,rec)
                                    exact_gate(value,check,'independent paired replay')
                                    forecasts[key+'_'+method+'_independent']=check
                                    replays.append(dict(row=i,state=f'{direction}_{sign}',method=method,exact=True))
                    record['response'][direction]={m:response_score(natural,rebuilt[m],sd) for m in METHODS}
            if native.model.instance_norm.forward!=original_norm:
                raise ValueError('Normalizer restoration failed')
            if {id(m):len(m._forward_hooks) for m in native.model.modules()}!=initial_hooks:
                raise ValueError('Hook restoration failed')
            exact_gate(base,independent(query),'true nohook final restoration')
            record['exact_gates']=dict(saved_baseline=True,old_baseline=True,locked_self=True,plus_pair_self=i in p['design']['paired'],normalizer_restored=True,hooks_restored=True,true_nohook_restored=True)
            records.append(record)
            write(out/'partial.json',records)
            print(f'receivers={len(records)}/20 cells={calls[0]}/399',flush=True)
    if calls[0]!=p['design']['cells']:
        raise ValueError('Native call accounting mismatch')
    np.savez_compressed(out/'states.npz',**states)
    np.savez_compressed(out/'forecasts.npz',**forecasts)
    write(out/'results.json',records);write(out/'independent_replay.json',replays)
    write(out/'summary.json',summarize(records))
    write(out/'knowledge.json',dict(question='Does response-aware reconstruction preserve native forecasts and locked weak responses?',status='complete_branch_closed',slot1='released',
        exact_results=dict(summary='summary.json',per_source='results.json',raw_forecasts='forecasts.npz',actual_states='states.npz',native_cells=calls[0],independent_exact_replays=len(replays),identity_gate_receivers=20),
        controls=[p['identity'],p['denominators'],'all dictionary/input/checkpoint hashes checked; no fit; same receiver/input for methods'],
        limitations=p['limitations'],next_step='Branch closed regardless of tradeoff; no further fitting or native launch.'))


def launch(out):
    """External wall/RSS monitor includes model loading and parent memory."""
    p=read(out/'protocol.json');started=time.monotonic();peak=0;reason=None
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',DICTIONARY_NATIVE_RESPONSE_GRANTED='1')
    with (out/'worker.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-m','experiments.feature_search.dictionary_response_native','worker','--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            rss=0
            for pid in (os.getpid(),child.pid):
                try:rss+=int(next((s.split()[1] for s in Path(f'/proc/{pid}/status').read_text().splitlines() if s.startswith('VmRSS:')),'0'))*1024
                except FileNotFoundError:pass
            peak=max(peak,rss)
            if peak>p['rss_bytes'] or time.monotonic()-started>p['seconds']:
                reason='resource cap';os.killpg(child.pid,9);child.wait();break
            time.sleep(.05)
    execution=dict(returncode=child.returncode,reason=reason,seconds=time.monotonic()-started,peak_rss=peak,threads=1)
    write(out/'execution.json',execution)
    if child.returncode:
        write(out/'knowledge.json',dict(question='Bounded native preservation',status='failed_branch_closed',slot1='released',exact_results=execution,controls=['external180s/3GiB monitor'],limitations=['incomplete; no preservation verdict'],next_step='Retain partial failures; branch closed, no automatic retry.'))
    print(execution,flush=True)


def main():
    """Preparation and explicit slot grant are distinct from native execution."""
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','worker'])
    parser.add_argument('--out',type=Path,default=Path('runs/dictionary_response_native4965000'));parser.add_argument('--slot-granted',action='store_true');a=parser.parse_args()
    if a.command=='prepare':prepare(a.out)
    elif a.command=='run':
        if not a.slot_granted:parser.error('Native slot grant required')
        launch(a.out)
    else:
        if os.environ.get('DICTIONARY_NATIVE_RESPONSE_GRANTED')!='1':parser.error('Use monitored launch')
        worker(a.out)


if __name__=='__main__':main()
