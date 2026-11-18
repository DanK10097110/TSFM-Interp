"""Locate natural hybrid gains without inventing a true Fourier decomposition.

The diagnostic subspace uses only the frequency already fitted from history.
Future observations score errors but never determine the projection basis or
adapter choice. Exact orthogonal error identities separate tone-shaped and
other changes in the actual hybrid update.
"""
from pathlib import Path
import argparse
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha


def partition(native,hybrid,target,frequency):
    """Return exact MSE-gain and update-energy partitions in a fixed tone plane."""
    native,hybrid,target=[np.asarray(v,float) for v in (native,hybrid,target)]
    if not(native.shape==hybrid.shape==target.shape==(32,)):raise ValueError('Expected32-step vectors')
    t=np.arange(32);basis=np.column_stack([np.sin(2*np.pi*frequency*t/128),np.cos(2*np.pi*frequency*t/128)])
    q,_=np.linalg.qr(basis)
    project=lambda v:q@(q.T@v)
    e=native-target;h=hybrid-target;delta=hybrid-native
    ep,hp,dp=[project(v) for v in (e,h,delta)]
    mean_square=lambda v:float(np.mean(v*v))
    gain=mean_square(e)-mean_square(h)
    tone_gain=mean_square(ep)-mean_square(hp)
    other_gain=mean_square(e-ep)-mean_square(h-hp)
    update_energy=mean_square(delta);tone_update=mean_square(dp);other_update=mean_square(delta-dp)
    np.testing.assert_allclose(gain,tone_gain+other_gain,atol=2e-12,rtol=1e-10)
    np.testing.assert_allclose(update_energy,tone_update+other_update,atol=2e-12,rtol=1e-10)
    return dict(gain=gain,tone_gain=tone_gain,other_gain=other_gain,update_energy=update_energy,
                tone_update=tone_update,other_update=other_update,
                native_tone_error=mean_square(ep),hybrid_tone_error=mean_square(hp),
                native_other_error=mean_square(e-ep),hybrid_other_error=mean_square(h-hp)),np.stack([e,h,ep,hp,delta,dp])


def interval(values,seed=4956399):
    """Descriptive equal-source interval; no additional confirmation from reused data."""
    values=np.asarray(values,float);rng=np.random.default_rng(seed)
    samples=values[rng.integers(0,len(values),(4000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),n=len(values))


def run(source,out):
    """Preserve source-wise decompositions and abstention as an exact zero update."""
    out.mkdir(parents=True,exist_ok=False)
    names=['evaluated_forecasts.npz','main_targets.npz','rows.json','plans.json','analysis.json']
    write(out/'protocol.json',dict(question='Are natural Fourier-hybrid gains at the extracted frequency or elsewhere?',
        prior=str(source),bindings={name:sha(source/name) for name in names},code_sha256=sha(__file__),
        projection='Sine/cosine at first extracted past-only frequency, original128-step frequency units,32-step future grid.',
        inference='Post-hoc descriptive source bootstrap within model/domain; no TEST-derived selection or new confirmation.',
        abstention='Exact zero forecast update; frequency plane undefined, component gains assigned zero only because forecasts equal.',
        limits='Projected natural future is not ground-truth component decomposition; sin/cos may overlap trend over32 steps; effects do not identify training causes.'))
    bank=np.load(source/'evaluated_forecasts.npz');target=np.load(source/'main_targets.npz')
    rows=read(source/'rows.json');plans=read(source/'plans.json')
    if not np.array_equal(bank['source_ids'],target['source_ids']):raise ValueError('Source alignment')
    native_index=bank['methods'].tolist().index('native');hybrid_index=bank['methods'].tolist().index('hybrid1')
    recorded={(r['model'],r['index']):r['main_benefit'] for r in read(source/'analysis.json')['source_records']}
    records=[];arrays={}
    for model in ('bolt','timesfm'):
        for i,(row,plan) in enumerate(zip(rows,plans)):
            native=bank[model][i,native_index];hybrid=bank[model][i,hybrid_index];future=target['targets'][i]
            details=plan[2]['details'];eligible=bool(details)
            if eligible:
                freq=details[0]['frequency'];score,raw=partition(native,hybrid,future,freq);arrays[f'{model}_{i}']=raw
            else:
                if not np.array_equal(native,hybrid):raise ValueError('Abstention changes forecast')
                freq=None;score={k:0. for k in ('gain','tone_gain','other_gain','update_energy','tone_update','other_update')}
            np.testing.assert_allclose(score['gain'],recorded[model,i],atol=2e-12,rtol=1e-10)
            records.append(dict(model=model,index=i,source=row['source'],domain=row['domain'],eligible=eligible,frequency=freq,**score))
    summary=[]
    for model in ('bolt','timesfm'):
        for domain in sorted({r['domain'] for r in records}):
            for scope in ('all','eligible'):
                selected=[r for r in records if r['model']==model and r['domain']==domain and(scope=='all' or r['eligible'])]
                fields={name:interval([r[name] for r in selected]) for name in ('gain','tone_gain','other_gain','update_energy','tone_update','other_update')}
                total=fields['update_energy']['mean']
                summary.append(dict(model=model,domain=domain,scope=scope,n=len(selected),**fields,
                    off_tone_update_fraction=None if total==0 else fields['other_update']['mean']/total))
    np.savez_compressed(out/'partitions.npz',**arrays);write(out/'rows.json',records);write(out/'summary.json',summary)
    write(out/'knowledge.json',dict(status='complete',question='Where do natural hybrid gains occur?',exact_findings=summary,
        controls='Past-fitted basis only; exact orthogonal error/update partitions; original loss replay and source IDs; abstentions exactzero',
        limitations='Post-hoc descriptive decomposition; no true natural-component labels, no new forecasts, no causal training attribution',
        next_decision='Assess whether a fixed output-space projection is a worthwhile separate pilot; do not choose a TEST-fitted blend.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.source,a.out)
