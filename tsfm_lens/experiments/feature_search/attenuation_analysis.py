"""Can scalar conservatism explain native response errors, even optimistically?"""
from pathlib import Path
import argparse
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha


def project(effect,signal,frequency):
    """Exact squared-error partition; bounded attenuation and phase-plane controls."""
    effect,signal=np.asarray(effect,float),np.asarray(signal,float);energy=np.mean(signal**2)
    if energy<=1e-12:raise ValueError('Degenerate signal')
    q=float(effect@signal/(signal@signal));bounded=float(np.clip(q,0,1));residual=effect-q*signal
    phase=np.arange(len(signal))*2*np.pi*frequency/128;basis=np.column_stack([np.sin(phase),np.cos(phase)])
    waveform=basis@np.linalg.lstsq(basis,effect,rcond=None)[0]
    total=float(np.mean((effect-signal)**2)/energy);orth=float(np.mean(residual**2)/energy)
    np.testing.assert_allclose(total,(q-1)**2+orth,rtol=1e-10,atol=1e-10)
    return dict(native_signal_nmse=total,gain=q,bounded_gain=bounded,unrestricted_scalar_residual=orth,bounded_scalar_residual=float(np.mean((effect-bounded*signal)**2)/energy),off_frequency_residual=float(np.mean((effect-waveform)**2)/energy),response_energy=float(np.mean(effect**2)/energy),outside_conservative_range=bool(q<0 or q>1))


def run(source,out):
    """Save per-background evidence without treating dependent sweeps as independent."""
    out.mkdir(parents=True,exist_ok=False);p=read(source/'protocol.json');signals=[s for s in read(source/'signals.json') if s['offset']==0];rows=[]
    write(out/'protocol.json',dict(source=str(source),source_protocol_sha256=sha(source/'protocol.json'),code_sha256=sha(__file__),scope='Post-pilot explanatory follow-up; per-row projections optimistic, not predictive fitting; grouped heldout scalar additionally assessed',hypothesis='A native response is q times the actual added sinusoid with q in[0,1]'))
    for model in ['bolt','timesfm']:
        worker=read(source/f'{model}_worker.json')
        for policy in ['native','locked']:
            name=f'{model}_{policy}_forecast.npy';assert sha(source/name)==worker['hashes'][name];forecasts=np.load(source/name)
            for s in signals:
                effect=forecasts[s['index']].astype(float)-forecasts[s['base']].astype(float)
                signal=s['amplitude']*s['scale']*np.sin(2*np.pi*s['frequency']*np.arange(32)/128+s['phase'])
                row=dict(model=model,policy=policy,source=s['source'],domain=s['domain'],center=s['center'],amplitude=s['amplitude'],index=s['index'],**project(effect,signal,s['frequency']));rows.append(row)
    for r in rows:
        train=[a for a in rows if a['source']!=r['source'] and all(a[k]==r[k] for k in ['model','policy','center','amplitude'])]
        q=float(np.clip(np.mean([a['gain'] for a in train]),0,1));r['loso_gain']=q;r['loso_residual']=(q-r['gain'])**2+r['unrestricted_scalar_residual']
    summary=[]
    for model in ['bolt','timesfm']:
        for policy in ['native','locked']:
            for center in sorted({s['center'] for s in rows}):
                for amplitude in [.1,.5,2.]:
                    subset=[r for r in rows if r['model']==model and r['policy']==policy and r['center']==center and r['amplitude']==amplitude]
                    summary.append(dict(model=model,policy=policy,center=center,amplitude=amplitude,sources=len(subset),**{k:float(np.mean([r[k] for r in subset])) for k in ['native_signal_nmse','gain','bounded_scalar_residual','off_frequency_residual','response_energy','outside_conservative_range','loso_residual']}))
    write(out/'rows.json',rows);write(out/'summary.json',summary);write(out/'knowledge.json',dict(question='Is conservative amplitude attenuation a sufficient response explanation?',evidence='Descriptive projection and leave-background-out scalar calibration on existing instrumentation sources',exact_results=summary,controls='Exact Pythagorean error replay; same expected signal and separate normalization policies',limitations='Eight backgrounds, four reused real FIT sources; no training-data attribution; rowwise projection is an optimistic explanation ceiling',next_decision='Inspect bounded residual versus off-frequency residual to motivate phase and waveform controls'))
    for r in summary:
        if r['policy']=='native':print(r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source,a.out)
