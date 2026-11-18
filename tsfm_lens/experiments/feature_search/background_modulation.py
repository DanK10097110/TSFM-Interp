"""Does off-tone response distortion resemble a changed background forecast?

Nested per-row projections are explanatory ceilings. The permuted-template
control measures how much an arbitrary extra direction can explain; it is not
a conditional-independence significance test or a deployable readout.
"""
from pathlib import Path
import argparse
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha


def fit(design, response):
    """Orthogonal least-squares projection, recording identifiability diagnostics."""
    coef,_,rank,s=np.linalg.lstsq(design,response,rcond=1e-10)
    pred=design@coef
    return pred,dict(rank=int(rank),columns=design.shape[1],
                     condition=float(s[0]/s[-1]) if s[-1]>0 else None)


def partition(response, signal, background, frequency, seed):
    """Partition known-signal error without mistaking a fitted template for cause."""
    response,signal,background=[np.asarray(x,float) for x in (response,signal,background)]
    if not (response.shape==signal.shape==background.shape): raise ValueError('Shape mismatch')
    energy=float(np.mean(signal**2))
    if energy<1e-12: raise ValueError('Degenerate known signal')
    t=np.arange(len(signal)); wave=np.column_stack([np.sin(2*np.pi*frequency*t/128),np.cos(2*np.pi*frequency*t/128)])
    low=np.column_stack([wave,np.ones(len(t)),(t-t.mean())/len(t)])
    whole=np.column_stack([low,background])
    w,gw=fit(wave,response);l,gl=fit(low,response);b,gb=fit(whole,response)
    norm=lambda v:float(np.mean(v**2)/energy)
    increments=[norm(response-w),norm(response-l),norm(response-b)]
    if increments[0]+1e-10<increments[1] or increments[1]+1e-10<increments[2]:
        raise ValueError('Nested projection increased error')
    rng=np.random.default_rng(seed);null=[]
    for _ in range(19):
        perm,_=fit(np.column_stack([low,rng.permutation(background)]),response)
        null.append(norm(response-l)-norm(response-perm))
    extra=increments[1]-increments[2]
    return dict(off_tone_error=increments[0],after_level_trend=increments[1],after_background=increments[2],
                level_trend_explained=increments[0]-increments[1],background_explained=extra,
                background_over_permuted_median=extra-float(np.median(null)),
                background_exceeds_permuted95=bool(extra>np.quantile(null,.95)),
                permuted_gains=null,gates=dict(wave=gw,low=gl,whole=gb)),(w,l,b)


def run(source,out):
    """Save raw explanatory fits and source-equal summaries in a fresh directory."""
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(source=str(source),code_sha256=sha(__file__),
        source_protocol_sha256=sha(source/'protocol.json'),scope='Post-pilot explanatory follow-up; fitted response ceilings, not causal attribution',
        inputs='Center frequencies only; all amplitudes; no native inference',
        controls='Nested projection energy identity;19 seeded time permutations of background template; native versus locked policies',
        limits='8 sources, reused pilot,32-step horizon; low-frequency projections can be highly correlated; order of nuisance removal is explicit'))
    signals=[s for s in read(source/'signals.json') if s['offset']==0]
    rows=[];arrays={}
    for model in ('bolt','timesfm'):
        worker=read(source/f'{model}_worker.json')
        for policy in ('native','locked'):
            name=f'{model}_{policy}_forecast.npy'
            if sha(source/name)!=worker['hashes'][name]: raise ValueError('Prior forecast binding changed')
            forecasts=np.load(source/name)
            for j,s in enumerate(signals):
                base=forecasts[s['base']].astype(float);effect=forecasts[s['index']].astype(float)-base
                signal=s['amplitude']*s['scale']*np.sin(2*np.pi*s['frequency']*np.arange(32)/128+s['phase'])
                metrics,proj=partition(effect,signal,base,s['frequency'],4956100+j)
                key=f'{model}_{policy}_{j}'
                arrays[key]=np.stack([effect,signal,base,*proj])
                rows.append(dict(model=model,policy=policy,source=s['source'],domain=s['domain'],center=s['center'],
                                 amplitude=s['amplitude'],key=key,**metrics))
    summary=[]
    fields=['off_tone_error','after_level_trend','after_background','level_trend_explained','background_explained',
            'background_over_permuted_median','background_exceeds_permuted95']
    for model in ('bolt','timesfm'):
        for policy in ('native','locked'):
            for center in sorted({s['center'] for s in signals}):
                for amp in (.1,.5,2.):
                    selected=[r for r in rows if (r['model'],r['policy'],r['center'],r['amplitude'])==(model,policy,center,amp)]
                    if len({r['source'] for r in selected})!=len(selected): raise ValueError('Source weighting not one row each')
                    summary.append(dict(model=model,policy=policy,center=center,amplitude=amp,n_sources=len(selected),
                                        **{key:float(np.mean([r[key] for r in selected])) for key in fields}))
    np.savez_compressed(out/'projections.npz',**arrays)
    write(out/'rows.json',rows);write(out/'summary.json',summary)
    write(out/'knowledge.json',dict(status='complete',question='Does off-tone distortion align with simple background forecast modulation?',
        exact_findings=summary,evidence='Descriptive response-fitted projections',controls='Nested projection and19 time-permutation templates, saved raw vectors',
        limitations='No forecasting or causal claim;8 existing backgrounds; projection order matters and32-step trends overlap sinusoids.',
        next_decision='Inspect whether genuine background templates beat arbitrary directions before testing this mechanism with new native data.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.out)
