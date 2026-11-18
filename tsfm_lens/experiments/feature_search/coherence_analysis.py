"""Source-paired coherence measurements and fixed, source-disjoint selectors.

Native inference is separate. This module never trains on evaluation futures.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.stats import binomtest
from experiments.feature_search.coherence_study import read, write, sha, load_bank, DOMAINS


def estimate(values):
    """Conditional source bootstrap and directional-majority test, distinct estimands."""
    x=np.asarray(values,dtype=float); nonzero=x[abs(x)>1e-12]
    if not len(x): raise ValueError('Empty contrast')
    rng=np.random.default_rng(4932999)
    draws=x[rng.integers(len(x),size=(4999,len(x)))].mean(1)
    return dict(n=len(x),mean=float(x.mean()),ci=np.quantile(draws,[.025,.975]).tolist(),positive=int(sum(x>1e-12)),
                p=float(binomtest(int(sum(nonzero>0)),len(nonzero),.5,alternative='greater').pvalue) if len(nonzero) else 1.)


def holm(records):
    """Retain the whole declared family, monotonically adjusted probabilities."""
    previous=0.
    for rank,i in enumerate(sorted(range(len(records)),key=lambda i:records[i]['p'])):
        previous=max(previous,min(1.,records[i]['p']*(len(records)-rank)))
        records[i]['holm_p']=previous
        records[i]['passes']=previous<=.05 and records[i]['mean']>=.1
    return records


def groupkey(row):
    """A real background is shared by both injected frequencies."""
    return (row['source'],row['target_frequency'] if row['kind']=='interference' else 0,
            row.get('distractor_frequency',0),row.get('amplitude',0),row['length'])


def predictions(row, f):
    """Subtract duplicated zero-context predictions when composing components."""
    raw=f['forecasts']; zero=raw[row['zero']].astype(float)
    result={'native':raw[row['root']].astype(float)}
    for name,ids in row['methods'].items(): result[name]=raw[ids].astype(float).sum(0)-3*zero
    if 'fft' in row['methods']: result['drop_high']=raw[row['methods']['fft'][:2]].astype(float).sum(0)-zero
    return result


def response_score(query, base, expected):
    """Counterfactual change must use a baseline under the same intervention."""
    delta=query-base; energy=np.mean(expected**2)
    gain=float(np.mean(delta*expected)/energy)
    return dict(response_nmse=float(np.mean((delta-expected)**2)/energy),gain=gain,
                gain_error=(gain-1)**2,orthogonal_error=float(np.mean((delta-gain*expected)**2)/energy))


def measure(out):
    """Recompute every loss and paired response from bound forecasts."""
    protocol=read(out/'protocol.json'); execution=read(out/'execution.json')
    if set(execution)!={'bolt','timesfm'} or not all(v['valid'] for v in execution.values()): raise ValueError('Invalid native run')
    for name,digest in protocol['hashes'].items():
        if sha(out/name)!=digest: raise ValueError('Input binding changed')
    rows=read(out/'rows.json'); y=np.load(out/'targets.npz')['future']; bases={groupkey(r):r for r in rows if r['sign']==0}; records=[]
    for model in ['bolt','timesfm']:
        if sha(out/f'{model}.npz')!=read(out/f'{model}_worker.json')['sha256']: raise ValueError('Forecast binding changed')
        f={k:v for k,v in np.load(out/f'{model}.npz').items()}
        for r in rows:
            base=bases[groupkey(r)]; pred=predictions(r,f); bp=predictions(base,f)
            for method,p in pred.items():
                meta={k:v for k,v in r.items() if k not in ['methods','locked','hindcast']}
                record=dict(meta,model=model,method=method,mse=float(np.mean((p-y[r['index']])**2)))
                if r['sign']:
                    expected=r['sign']*.5*np.sin(2*np.pi*r['target_frequency']*np.arange(32)/128+r['target_phase'])
                    record.update(response_score(p,bp[method],expected))
                if method=='fft':
                    e=pred['native']-y[r['index']]; d=p-pred['native']
                    record.update(error_cross=float(2*np.mean(e*d)),change_energy=float(np.mean(d*d)))
                    if not np.isclose(record['mse']-np.mean(e*e),record['error_cross']+record['change_energy'],atol=1e-10): raise ValueError('Error identity')
                records.append(record)
            for policy,i in r.get('locked',{}).items():
                p=f['locked'][i]; base_locked=f['locked_base'][i]
                record={k:v for k,v in r.items() if k not in ['methods','locked','hindcast']}
                records.append(dict(record,model=model,method=policy,mse=float(np.mean((p-y[r['index']])**2)),**response_score(p,base_locked,expected)))
    c=np.load(out/'classical.npz'); classical={int(i):p for i,p in zip(c['ids'],c['predictions'])}
    binding=read(out/'classical.json')
    if sha(out/'classical.npz')!=binding['sha256'] or sha(out/'bank.npz')!=binding['bank_sha256']: raise ValueError('Classical binding changed')
    for r in rows:
        if r['kind']!='interference': continue
        p=classical[r['root']];base=bases[groupkey(r)]
        record={k:v for k,v in r.items() if k not in ['methods','locked','hindcast']}
        record.update(model='classical',method='native',mse=float(np.mean((p-y[r['index']])**2)))
        if r['sign']:
            expected=r['sign']*.5*np.sin(2*np.pi*r['target_frequency']*np.arange(32)/128+r['target_phase'])
            record.update(response_score(p,classical[base['root']],expected))
        records.append(record)
    return records


def source_values(records,metric,**filters):
    """Average paired signs within source before any resampling."""
    groups={}
    for r in records:
        if metric in r and all(r.get(k)==v for k,v in filters.items()): groups.setdefault(r['source'],[]).append(r[metric])
    return {s:float(np.mean(v)) for s,v in groups.items()}


def contrast(records,metric,left,right,**filters):
    """Exact source membership intersection; no window-level inference."""
    a=source_values(records,metric,**dict(filters,**left));b=source_values(records,metric,**dict(filters,**right))
    if set(a)!=set(b): raise ValueError('Unmatched contrast')
    return estimate([a[s]-b[s] for s in sorted(a)])


def primary(records):
    """The twelve prospectively declared comparisons, including failed claims."""
    result=[]
    for model in ['bolt','timesfm']:
        filt=dict(model=model,domain='synthetic',target_frequency=5.,distractor_frequency=17.,length=128)
        result.append(dict(claim=model+':amplitude_harm',**contrast(records,'response_nmse',dict(method='native',amplitude=8.),dict(method='native',amplitude=.5),**filt)))
        result.append(dict(claim=model+':far_fft_gain',**contrast(records,'response_nmse',dict(method='native'),dict(method='fft'),amplitude=8.,**filt)))
        for domain in [DOMAINS[0],DOMAINS[2]]:
            filt=dict(model=model,domain=domain,split='test',target_frequency=5.,length=128)
            result.append(dict(claim=model+':'+domain+':response_gain',**contrast(records,'response_nmse',dict(method='native'),dict(method='fft'),**filt)))
            result.append(dict(claim=model+':'+domain+':forecast_harm',**contrast(records,'mse',dict(method='fft'),dict(method='native'),sign=0,**filt)))
    return holm(result)


def features(x,domain):
    """Past-only spectral, autocorrelation and shape descriptors."""
    z=x-x.mean(); power=abs(np.fft.rfft(z))**2; power/=max(power.sum(),1e-12)
    return [float(np.mean(z[k:]*z[:-k])/(np.mean(z*z)+1e-12)) for k in [1,2,4,8,16,24,32,64]]+ [float(a.sum()) for a in np.array_split(power,8)]+[float(abs(z).max()),float(x[-1]),float(np.polyfit(np.arange(len(x)),x,1)[0]),float(np.std(np.diff(x)))]+[float(domain==d) for d in DOMAINS]


def choose_models(x,diagnostic,loss,fit,domains,hindcast):
    """Freeze choices using FIT labels only; TEST labels cannot affect a choice."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.kernel_ridge import KernelRidge
    choices={};label=loss[:,1]-loss[:,0]
    choices['always_bolt']=np.zeros(len(x),dtype=int)
    choices['always_timesfm']=np.ones(len(x),dtype=int)
    choices['train_global']=np.full(len(x),int(np.argmin(loss[fit].mean(0))))
    choices['train_domain']=np.asarray([np.argmin(loss[fit & (domains==d)].mean(0)) for d in domains])
    choices['hindcast']=np.argmin(hindcast,axis=1)
    for inputs,values in [('input',x),('diagnostic',np.column_stack([x,diagnostic]))]:
        scaler=StandardScaler().fit(values[fit]);z=scaler.transform(values)
        for name,model in [('ridge',Ridge(alpha=10)),('rbf',KernelRidge(alpha=1,kernel='rbf',gamma=1/z.shape[1]))]:
            model.fit(z[fit],label[fit]);choices[inputs+'_'+name]=(model.predict(z)<0).astype(int)
    return choices


def selectors(out,records):
    """Score all fixed selectors once, reporting test source losses and regret."""
    rows=[r for r in read(out/'rows.json') if r['kind']=='real' and r['sign']==0];bank=load_bank(out/'bank.npz')
    loss=np.zeros((len(rows),2));hind=np.zeros_like(loss);diag=[];domains=np.asarray([r['domain'] for r in rows]);fit=np.asarray([r['split']=='fit' for r in rows])
    for j,model in enumerate(['bolt','timesfm']):
        f=np.load(out/f'{model}.npz')['forecasts']
        for i,r in enumerate(rows):
            loss[i,j]=next(v['mse'] for v in records if v['index']==r['index'] and v['model']==model and v['method']=='native')
            h=r['hindcast'];hind[i,j]=np.mean((f[h['bank_id']]*h['scale']+h['center']-h['target'])**2)
    for r in rows:
        diag.append([np.mean([v[metric] for v in records if v['source']==r['source'] and v['model']==model and v['method']=='native' and v['target_frequency']==freq and metric in v]) for model in ['bolt','timesfm'] for freq in [5.,128/24] for metric in ['response_nmse','gain']])
    x=np.asarray([features(bank[r['root']],r['domain']) for r in rows]);diag=np.asarray(diag)
    choices=choose_models(x,diag,loss,fit,domains,hind)
    changed=loss.copy();changed[~fit]=changed[~fit,::-1]*100+91
    alternate=choose_models(x,diag,changed,fit,domains,hind)
    if any(not np.array_equal(choices[k],alternate[k]) for k in choices): raise ValueError('Test target leakage')
    result=[]; saved=[]
    for name,choice in choices.items():
        selected=loss[np.arange(len(rows)),choice]
        for domain in DOMAINS:
            ix=(~fit)&(domains==domain); baseline=loss[np.arange(len(rows)),choices['train_domain']]
            result.append(dict(method=name,domain=domain,n=int(ix.sum()),loss=float(selected[ix].mean()),regret=float((selected-loss.min(1))[ix].mean()),timesfm_choices=int(choice[ix].sum()),vs_train_domain=estimate((baseline-selected)[ix])))
        saved.extend(dict(source=r['source'],split=r['split'],domain=r['domain'],method=name,choice=int(choice[i]),loss=float(selected[i]),bolt_loss=float(loss[i,0]),timesfm_loss=float(loss[i,1])) for i,r in enumerate(rows))
    write(out/'selector_rows.json',saved)
    return result


def summarize(records):
    """Keep every explored cell visible; no cherry-picked model/method averages."""
    keys=['model','domain','split','target_frequency','distractor_frequency','amplitude','length','method','sign']
    groups={}
    for r in records:
        key=tuple(r.get(k) for k in keys);groups.setdefault(key,[]).append(r)
    result=[]
    for key,values in groups.items():
        meta=dict(zip(keys,key));meta['n_sources']=len({v['source'] for v in values})
        for metric in ['mse','response_nmse','gain','gain_error','orthogonal_error','error_cross','change_energy']:
            if metric in values[0]:meta[metric]=float(np.mean([v[metric] for v in values]))
        result.append(meta)
    return result


def controls(records):
    """Exploratory paired mechanisms with scope explicitly retained."""
    result=[]
    def paired(label,left,right,metric='response_nmse',**filters):
        a=source_values(records,metric,**dict(filters,**left));b=source_values(records,metric,**dict(filters,**right))
        ids=sorted(set(a)&set(b))
        if ids:result.append(dict(contrast=label,**filters,**estimate([a[i]-b[i] for i in ids])))
    for model in ['bolt','timesfm']:
        for distractor in [6.,17.]:
            filt=dict(model=model,domain='synthetic',target_frequency=5.,distractor_frequency=distractor,amplitude=8.)
            for policy in ['background','shared6']:
                paired('native minus '+policy,dict(method='native'),dict(method=policy),length=128,**filt)
            paired('128 minus 256 history',dict(method='native',length=128),dict(method='native',length=256),**filt)
        paired('near native minus shifted split',dict(method='native'),dict(method='shift'),model=model,domain='synthetic',target_frequency=5.,distractor_frequency=6.,amplitude=8.,length=128)
        for domain in DOMAINS:
            filt=dict(model=model,domain=domain,split='test',length=128)
            paired('native minus bridge response',dict(method='native'),dict(method='bridge'),target_frequency=128/24,**filt)
            paired('FFT minus drop-high clean loss',dict(method='fft'),dict(method='drop_high'),metric='mse',sign=0,**filt)
            paired('FFT minus bridge clean loss',dict(method='fft'),dict(method='bridge'),metric='mse',sign=0,**filt)
    for distractor in [6.,17.,6.35,17.35]:
        target=5. if distractor in [6.,17.] else 5.35
        paired('TimesFM minus Bolt native response',dict(model='timesfm'),dict(model='bolt'),method='native',domain='synthetic',target_frequency=target,distractor_frequency=distractor,amplitude=8.,length=128)
    return result


def selector_contrasts(out):
    """Incremental diagnostic value over matching input-only estimators."""
    rows=read(out/'selector_rows.json');result=[]
    for domain in DOMAINS:
        for method in ['ridge','rbf']:
            left={r['source']:r['loss'] for r in rows if r['domain']==domain and r['split']=='test' and r['method']=='input_'+method}
            right={r['source']:r['loss'] for r in rows if r['domain']==domain and r['split']=='test' and r['method']=='diagnostic_'+method}
            assert set(left)==set(right)
            result.append(dict(domain=domain,estimator=method,**estimate([left[s]-right[s] for s in sorted(left)])))
    return result


def oddness(out):
    """Check sign symmetry separately from response correctness, with no new inference."""
    rows=read(out/'rows.json');bases={groupkey(r):r for r in rows if r['sign']==0}
    negatives={(groupkey(r),r['target_frequency']):r for r in rows if r['sign']==-1};records=[]
    for model in ['bolt','timesfm']:
        f={k:v for k,v in np.load(out/f'{model}.npz').items()}
        for r in rows:
            if r['sign']!=1:continue
            pos=predictions(r,f);neg=predictions(negatives[groupkey(r),r['target_frequency']],f);base=predictions(bases[groupkey(r)],f)
            expected=.5*np.sin(2*np.pi*r['target_frequency']*np.arange(32)/128+r['target_phase'])
            for method in pos:
                defect=pos[method]+neg[method]-2*base[method]
                records.append(dict(source=r['source'],domain=r['domain'],split=r['split'],model=model,method=method,target_frequency=r['target_frequency'],distractor_frequency=r.get('distractor_frequency'),amplitude=r.get('amplitude'),length=r['length'],sign=1,oddness_nmse=float(np.mean(defect**2)/np.mean(expected**2))))
    write(out/'oddness_rows.json',records)
    result=[]
    for model in ['bolt','timesfm']:
        for domain in ['synthetic']+list(DOMAINS):
            for method in ['native','fft']:
                filt=dict(model=model,domain=domain,method=method,split='test',target_frequency=5.,length=128)
                if domain=='synthetic':filt.update(distractor_frequency=17.,amplitude=8.)
                vals=source_values(records,'oddness_nmse',**filt)
                result.append(dict(**filt,n=len(vals),oddness_nmse=float(np.mean(list(vals.values())))))
    return result


def main(out):
    """Write reproducible analysis and the full long-form measurement table."""
    records=measure(out);write(out/'measurements.json',records)
    result=dict(primary=primary(records),controls=controls(records),cells=summarize(records),selectors=selectors(out,records),analysis_sha256=sha(__file__))
    result['selector_contrasts']=selector_contrasts(out);result['oddness']=oddness(out)
    write(out/'analysis.json',result)
    print(result['primary'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
