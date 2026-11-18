"""Fresh paired behavioral replication of a possible period32 forecasting gap.

The independent unit is a generating seed bundle, reused over periods and
noise/trend regimes. No representation extraction or private data is used.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

PERIODS = (24, 28, 30, 32, 34, 36, 40, 48)
REGIMES = ('clean', 'noise', 'trend', 'both')


def corpus(n_seeds=32, base=882000):
    """Share latent draws across eight periods and four controlled regimes."""
    t = np.arange(160)
    contexts, truth, metadata = [], [], []
    for seed in range(base, base + n_seeds):
        rng = np.random.default_rng(seed)
        amplitude, phase = rng.uniform(.5, 2), rng.uniform(-np.pi, np.pi)
        sign = rng.choice([-1, 1])
        eps = rng.normal(0, .1 * amplitude, len(t))
        for period in PERIODS:
            sine = amplitude * np.sin(2 * np.pi * t / period + phase)
            for regime in REGIMES:
                slope = sign * amplitude if regime in ('trend', 'both') else 0.
                latent = sine + slope * (t - 64) / 128
                observed = latent + (eps if regime in ('noise', 'both') else 0)
                contexts.append(observed[:128])
                truth.append(latent[128:])
                metadata.append(dict(seed=seed, period=period, regime=regime,
                                     phase=phase, amplitude=amplitude, slope=slope))
    return np.asarray(contexts, np.float32), np.asarray(truth), metadata


def harmonic_fit(values, period, reference_amplitude=1.):
    """Fit known-frequency phase/amplitude separately from offset and trend."""
    t = np.arange(128, 160)
    design = np.column_stack((np.sin(2*np.pi*t/period), np.cos(2*np.pi*t/period),
                              np.ones(32), (t-64)/128))
    coef = np.linalg.lstsq(design, values, rcond=None)[0]
    amplitude=float(np.hypot(*coef[:2]))
    condition=float(np.linalg.cond(design))
    valid=amplitude > 1e-6*reference_amplitude and condition <= 1000
    return dict(amplitude=amplitude, phase=float(np.arctan2(coef[1], coef[0])) if valid else None,
                phase_defined=valid, offset=float(coef[2]),
                slope=float(coef[3]), condition_number=float(np.linalg.cond(design)),
                residual_mse=float(np.mean((values-design@coef)**2)))


def primary_contrasts(gaps):
    """Three frozen contrasts for [seed,period,regime] Bolt−TimesFM errors."""
    gaps = np.asarray(gaps)
    if gaps.ndim != 3 or gaps.shape[1:] != (8, 4):
        raise ValueError('gaps must have shape [seed,8periods,4regimes]')
    center = gaps[:, PERIODS.index(32)].mean(1)
    neighbors = gaps[:, [PERIODS.index(30), PERIODS.index(34)]].mean((1, 2))
    interaction = (gaps[:, :, [2, 3]] - gaps[:, :, [0, 1]]).mean((1, 2))
    return dict(gap32=center, excess32_over30_34=center-neighbors,
                model_by_trend=interaction)


def protocol():
    """Declare the complete analysis before loading either model."""
    return dict(base_seed=882000, n_independent_seed_bundles=32,
                periods=PERIODS, regimes=REGIMES, context=128, horizon=32,
                noise_sd='.1*amplitude', trend='random sign*amplitude*(t-64)/128',
                amplitude='uniform(.5,2)', phase='uniform(-pi,pi)',
                primary_metric='mean squared error vs clean latent future / amplitude²',
                primary_contrasts=list(primary_contrasts(np.zeros((32,8,4)))),
                gap_sign='Bolt minus TimesFM; positive means Bolt worse',
                inference='equal-seed bootstrap95%;two-sided paired signflip9999;Holm3',
                evidence='fresh synthetic behavioral replication; predictive not causal',
                harmonic='knownperiod sin/cos/offset/linear; report condition number',
                model_order=['amazon/chronos-bolt-small','google/timesfm-2.5-200m-pytorch'])


def analyze(root):
    """Score the frozen contrasts and a complete descriptive table."""
    from tsfm_lens.experiments.feature_search.statistics import compare_effects

    data = np.load(root/'corpus.npz')
    meta = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    amplitudes = np.array([m['amplitude'] for m in meta])
    errors, harmonics = {}, {}
    for name in ('bolt', 'timesfm'):
        pred = np.load(root/f'{name}.npz')['prediction']
        errors[name] = ((pred-data['clean_future'])**2).mean(1)/amplitudes**2
        fits = [harmonic_fit(row,m['period'],m['amplitude']) for row,m in zip(pred,meta)]
        harmonics[name] = [dict(phase_abs_error=float(abs(np.angle(np.exp(1j*(f['phase']-m['phase']))))) if f['phase_defined'] else None,
                                relative_amplitude_error=abs(f['amplitude']/m['amplitude']-1),
                                normalized_offset=abs(f['offset']/m['amplitude']),
                                normalized_slope_error=abs((f['slope']-m['slope'])/m['amplitude']),
                                **f) for f,m in zip(fits,meta)]
    gap = (errors['bolt']-errors['timesfm']).reshape(32,8,4)
    contrasts = primary_contrasts(gap)
    primary = compare_effects({name:(values,np.zeros_like(values)) for name,values in contrasts.items()},
                              np.arange(32), n_boot=9999,n_permutations=9999,seed=882000)
    table = {}
    for j,p in enumerate(PERIODS):
        for k,r in enumerate(REGIMES):
            table[f'{p}_{r}'] = {name:float(err.reshape(32,8,4)[:,j,k].mean())
                                  for name,err in errors.items()}
            for name,values in harmonics.items():
                selected = [v for v,m in zip(values,meta) if m['period']==p and m['regime']==r]
                table[f'{p}_{r}'][name+'_harmonic'] = {field:float(np.mean([v[field] for v in selected if v[field] is not None]))
                    if any(v[field] is not None for v in selected) else None for field in selected[0] if field!='phase'}
                table[f'{p}_{r}'][name+'_harmonic']['n_phase_undefined']=sum(not v['phase_defined'] for v in selected)
    result = dict(primary=primary,period_regime_table=table,
                  max_harmonic_condition=max(v['condition_number'] for values in harmonics.values() for v in values),
                  evidence=protocol()['evidence'])
    result['warnings']=['phase undefined when fitted amplitude <=1e-6*true amplitude or design condition>1000']
    result['descriptive_gap_noise_interaction']=float((gap[:,:,[1,3]]-gap[:,:,[0,2]]).mean())
    (root/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def run(root, prepare_only=False):
    """Write protocol first, then forecast with one model resident at a time."""
    root.mkdir(parents=True,exist_ok=False)
    x,truth,meta = corpus()
    (root/'protocol.json').write_text(json.dumps(protocol(),indent=2),encoding='utf-8')
    (root/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    np.savez_compressed(root/'corpus.npz',context=x,clean_future=truth)
    if prepare_only:
        return
    infer(root)


def infer(root):
    """Use offline cached model checkpoints and save forecasts only."""
    import torch
    from tsfm_lens.config import DataConfig,ModelConfig
    from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
    from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
    torch.set_num_threads(1)
    x = np.load(root/'corpus.npz')['context']
    for cls,name,checkpoint in [(ChronosBoltAdapter,'bolt','amazon/chronos-bolt-small'),
                                (TimesFMAdapter,'timesfm','google/timesfm-2.5-200m-pytorch')]:
        adapter=cls(ModelConfig(name=name,checkpoint=checkpoint),DataConfig(context_len=128,horizon=32),
                    torch.device('cuda:0'),torch.float32)
        adapter.ensure_loaded()
        prediction=np.concatenate([adapter.predict(x[i:i+8],32,[.5])['point'] for i in range(0,len(x),8)])
        repeat=adapter.predict(x[:8],32,[.5])['point']
        if not np.array_equal(prediction[:8],repeat):
            raise RuntimeError(f'{name} repeat prediction is not deterministic')
        np.savez_compressed(root/f'{name}.npz',prediction=prediction)
        print(name+' complete',flush=True)
        del adapter;gc.collect();torch.cuda.empty_cache()
    print(json.dumps(analyze(root)['primary'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--infer-existing',action='store_true')
    args=parser.parse_args()
    infer(args.out) if args.infer_existing else run(args.out,args.prepare_only)
