"""Independent arithmetic audit of saved coherence and training results.

Uses raw forecasts and future differences, not the analysis module's scoring
helpers. A false saved result must fail even when inputs and weights are valid.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from experiments.feature_search.coherence_study import read,write,sha


def audit(out):
    """Replay every scalar and the twelve corrected directional tests."""
    rows=read(out/'rows.json'); measurements=read(out/'measurements.json');analysis=read(out/'analysis.json');future=np.load(out/'targets.npz')['future']
    predictions={model:{k:v.astype(float) for k,v in np.load(out/f'{model}.npz').items()} for model in ['bolt','timesfm']}
    c=np.load(out/'classical.npz');classic=dict(zip(c['ids'],c['predictions']))
    def key(r):return (r['source'],r.get('distractor_frequency'),r.get('amplitude'),r['length'],r['target_frequency'] if r['kind']=='interference' else None)
    bases={key(r):r for r in rows if r['sign']==0};checked=0;responses=0
    for item in measurements:
        r=rows[item['index']];base=bases[key(r)];model=item['model'];method=item['method']
        if model=='classical':pred=classic[r['root']];bp=classic[base['root']]
        elif method in ['background','shared6']:
            idx=r['locked'][method];pred=predictions[model]['locked'][idx];bp=predictions[model]['locked_base'][idx]
        else:
            f=predictions[model]['forecasts']
            def reconstruct(row):
                if method=='native':return f[row['root']]
                ids=row['methods']['fft'][:2] if method=='drop_high' else row['methods'][method]
                return np.sum(f[ids],axis=0)-(len(ids)-1)*f[row['zero']]
            pred=reconstruct(r);bp=reconstruct(base)
        loss=float(np.dot(pred-future[r['index']],pred-future[r['index']])/32)
        np.testing.assert_allclose(loss,item['mse'],rtol=1e-12,atol=1e-12);checked+=1
        if r['sign']:
            expected=future[r['index']]-future[base['index']]
            delta=pred-bp;err=delta-expected
            nmse=float(np.dot(err,err)/np.dot(expected,expected));gain=float(np.dot(delta,expected)/np.dot(expected,expected))
            np.testing.assert_allclose([nmse,gain],[item['response_nmse'],item['gain']],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(item['gain_error'],(gain-1)**2,rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(item['orthogonal_error'],nmse-(gain-1)**2,rtol=1e-10,atol=1e-10);responses+=1
    odd_checked=0
    if (out/'oddness_rows.json').exists():
        index={(r['source'],r['target_frequency'],r.get('distractor_frequency'),r.get('amplitude'),r['length'],r['sign']):r for r in rows}
        for item in read(out/'oddness_rows.json'):
            common=(item['source'],item['target_frequency'],item.get('distractor_frequency'),item.get('amplitude'),item['length'])
            positive=index[common+(1,)];negative=index[common+(-1,)];base=bases[key(positive)];f=predictions[item['model']]['forecasts'];method=item['method']
            def compose(r):
                if method=='native':return f[r['root']]
                ids=r['methods']['fft'][:2] if method=='drop_high' else r['methods'][method]
                return f[ids].sum(0)-(len(ids)-1)*f[r['zero']]
            difference=compose(positive)+compose(negative)-2*compose(base)
            expected=future[positive['index']]-future[base['index']]
            np.testing.assert_allclose(np.dot(difference,difference)/np.dot(expected,expected),item['oddness_nmse'],rtol=1e-10,atol=1e-10);odd_checked+=1
    primary=[]
    for claim in analysis['primary']:
        pieces=claim['claim'].split(':');model=pieces[0];metric='response_nmse';groups={}
        for r in measurements:
            if r['model']!=model:continue
            if len(pieces)==2:
                if r['domain']!='synthetic' or r['target_frequency']!=5 or r['distractor_frequency']!=17 or r['length']!=128 or not r['sign']:continue
                if pieces[1]=='amplitude_harm':
                    if r['method']!='native':continue
                    arm=1 if r['amplitude']==8 else -1
                else:
                    if r['amplitude']!=8 or r['method'] not in ['native','fft']:continue
                    arm=1 if r['method']=='native' else -1
            else:
                if r['domain']!=pieces[1] or r['split']!='test' or r['target_frequency']!=5 or r['method'] not in ['native','fft']:continue
                if pieces[2]=='forecast_harm':
                    if r['sign']!=0:continue
                    metric='mse';arm=1 if r['method']=='fft' else -1
                else:
                    if r['sign']==0:continue
                    arm=1 if r['method']=='native' else -1
            groups.setdefault(r['source'],{1:[],-1:[]})[arm].append(r[metric])
        values=np.asarray([np.mean(v[1])-np.mean(v[-1]) for v in groups.values()]);nonzero=values[abs(values)>1e-12]
        p=binomtest(sum(nonzero>0),len(nonzero),alternative='greater').pvalue if len(nonzero) else 1.
        np.testing.assert_allclose([values.mean(),p],[claim['mean'],claim['p']],rtol=1e-12,atol=1e-12);assert len(values)==claim['n']
        primary.append(float(p))
    order=np.argsort(primary);adjusted=np.empty(len(primary));adjusted[order]=np.minimum(1,np.maximum.accumulate(np.asarray(primary)[order]*np.arange(len(primary),0,-1)))
    np.testing.assert_allclose(adjusted,[r['holm_p'] for r in analysis['primary']])
    selector=read(out/'selector_rows.json');lookup={(r['source'],r['model']):r['mse'] for r in measurements if r['domain']!='synthetic' and r['sign']==0 and r['method']=='native'}
    for row in selector:
        np.testing.assert_allclose(row['loss'],lookup[row['source'],['bolt','timesfm'][row['choice']]],rtol=1e-12)
    prior=read(out.parent/'fourier_poc4929000/rows.json');pilot=read(out.parent/'coherence_pilot4931000/rows.json')
    sources={r['source'] for r in rows if r['kind']=='real'}
    assert not sources & {r['source'] for r in prior}
    earlier=read(out.parent/'real_background_poc492306/metadata.json')
    assert not sources & {r['dataset']+':'+r['id'] for r in earlier}
    if 'replication' in out.name:assert not sources & {r['source'] for r in pilot}
    for domain in {r['domain'] for r in rows if r['kind']=='real'}:
        fit={r['source'] for r in rows if r['domain']==domain and r['split']=='fit'};test={r['source'] for r in rows if r['domain']==domain and r['split']=='test'};assert not fit&test
    result=dict(losses=checked,responses=responses,oddness=odd_checked,primary_tests=len(primary),selector_rows=len(selector),source_disjoint=True,passed=True,source_sha256=sha(__file__))
    write(out/'independent_audit.json',result);return result


def training_audit(out):
    """Replay all training test losses, preserving corpus-level sample size."""
    results=read(out/'results.json');f=np.load(out/'predictions.npz');labels=read(out/'test_labels.json')
    count=0
    for row in results['results']:
        seed=row['seed'];key=f"{seed}_{row['architecture']}_{row['arm']}_{row['steps']}"
        e=np.mean((f[key]-f[f'{seed}_future'])**2,axis=1)/np.var(f[f'{seed}_history'],axis=1)
        for group,value in row['error'].items():
            np.testing.assert_allclose(np.mean(e[[i for i,r in enumerate(labels) if r['group']==group]]),value,rtol=1e-7);count+=1
    result=dict(checked=count,corpora=len({r['seed'] for r in results['results']}),passed=True);write(out/'independent_audit.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--training',action='store_true');a=p.parse_args();print(training_audit(a.out) if a.training else audit(a.out))
