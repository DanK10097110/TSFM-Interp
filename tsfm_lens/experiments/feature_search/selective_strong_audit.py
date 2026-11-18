"""Replay strong-pattern controls and require validity in both paired contexts."""
from pathlib import Path
import argparse
import numpy as np
from experiments.feature_search.coherence_study import read, write, sha
from experiments.feature_search.selective_strong_signal import analyze
from experiments.feature_search.selective_analysis import key, resolve
from experiments.feature_search.selective_tree import decompose


def run(out):
    """Audit saved forecasts and geometry; keep original/injected null flags explicit."""
    analyze(out)
    scores=read(out/'strong_signal_scores.json');p=read(out/'protocol.json')
    x=np.load(out/'inputs.npz')['history'];y=np.load(out/'targets.npz')['future']
    plans=read(out/'plans.json');tone=8*np.sin(2*np.pi*np.arange(-128,32)/19.2+.6)
    checked=0;closures=0;maximum=0.
    for model in ['bolt','timesfm']:
        prefix=model+'_strong';protocol=read(out/f'{prefix}_protocol.json')
        assert protocol['source_sha256']==sha(Path(__file__).with_name('selective_strong_signal.py'))
        assert read(out/f'{prefix}_execution.json')['sha256']==sha(out/f'{prefix}_forecasts.json')
        lookup={(r['row'],key(r)):r for r in read(out/f'{prefix}_forecasts.json')}
        original={(r['row'],key(r)):r for r in read(out/f'{model}_forecasts.json')}
        states=np.load(out/f'{prefix}_states.npz');cuts=protocol['layers']
        for meta in read(out/f'{prefix}_metadata.json'):
            i=meta['row'];query=x[i]+tone[:128];levels,details=decompose(query)
            assert len(details)==meta['accepted']
            for leaves in levels:
                err=float(np.max(np.abs(np.asarray(leaves,np.float32).sum(0)-query))/max(float(np.std(query)),1e-12))
                maximum=max(maximum,err);assert err<2e-6
        for r in [a for a in scores if a['model']==model]:
            i=r['row'];config=tuple(r['config']);actual=lookup.get((i,config),lookup[i,('native',0,'none','none',0.)])
            base=resolve(config,i,original,plans);pred=np.asarray(actual['forecast']);baseline=np.asarray(base['forecast'])
            r['valid_original']=base.get('valid',True);r['valid_injected']=actual.get('valid',True)
            r['valid']=bool(r['valid_original'] and r['valid_injected'])
            values=[np.mean((pred-baseline-tone[128:])**2)/np.mean(tone[128:]**2),np.mean((pred-y[i]-tone[128:])**2)/np.var(x[i].astype(float)),np.mean((baseline-y[i])**2)/np.var(x[i].astype(float))]
            np.testing.assert_allclose(values,[r['response_nmse'],r['loss_original_units'],r['clean_loss']],rtol=1e-12,atol=1e-12);checked+=1
            if actual['method']=='fusion':
                d=actual['actual_depth'];c=cuts.index(actual['cut']);h=states[f'{i}_full'][c];z=states[f'{i}_zero'][c];leaves=states[f'{i}_depth{d}'][c]
                w=np.ones(d+1)
                if actual['weight']=='learned':
                    a,b=p['learned_weights'][model][str(d)][actual['cut']]['weights'];w[:-1]=a;w[-1]=b
                composed=z.astype(float)+sum(weight*(leaf.astype(float)-z.astype(float)) for weight,leaf in zip(w,leaves))
                closure=np.mean((composed-h)**2)/max(float(np.mean((h-z)**2)),1e-12)
                np.testing.assert_allclose(closure,actual['closure_nmse'],rtol=1e-8,atol=1e-10);closures+=1
    write(out/'strong_signal_scores.json',scores)
    result=dict(passed=True,forecast_scores=checked,hidden_closures=closures,input_closure_max=maximum,random_invalid_pairs=sum(not r['valid'] for r in scores if r['config'][0]=='random'),source_sha256=sha(__file__))
    write(out/'strong_signal_audit.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
