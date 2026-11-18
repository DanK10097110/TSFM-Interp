"""Baseline input-intervention interference in a two-frequency factorial.

Only one true component phase changes within each matched input pair. Forecast
changes in the untouched component measure endogenous model coupling, without
hidden-state edits or any change to the separate feature-selectivity gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tsfm_lens.analysis.stats import holm, mean_ci
from tsfm_lens.experiments.feature_search.statistics import paired_effect


def harmonic_components(values, *, start=128):
    """Fit orthogonal period32/16 harmonics and expose out-of-basis residuals."""
    values = np.asarray(values,dtype=float)
    if values.ndim != 2 or values.shape[1] != 32 or not np.isfinite(values).all():
        raise ValueError('Need finite32-point forecasts: complete32/16 periods')
    t = np.arange(start,start+32)
    design = np.column_stack((np.sin(2*np.pi*t/32),np.cos(2*np.pi*t/32),np.sin(2*np.pi*t/16),np.cos(2*np.pi*t/16),np.ones(32)))
    gram = design.T@design/32
    if not np.allclose(gram,np.diag([.5,.5,.5,.5,1.]),atol=1e-12):
        raise ValueError('Harmonic basis must be exactly orthogonal on the forecast window')
    coefficients = np.linalg.lstsq(design,values.T,rcond=None)[0].T
    return dict(coefficients=coefficients,residual=values-coefficients@design.T,design=design)


def forecast_error_decomposition(prediction, truth, *, forecast_start=128):
    """Exactly decompose full-window MSE into orthogonal frequency components.

    Each frequency has amplitude-only error, phase error at true amplitude,
    and a signed amplitude×phase interaction. That algebraic interaction is
    distinct from the separately measured cross-frequency input interference.
    Zero predicted amplitude has undefined phase; its entire error is assigned
    to amplitude. Out-of-basis residual error is retained, never discarded.
    """
    prediction, truth = np.asarray(prediction,dtype=float), np.asarray(truth,dtype=float)
    if prediction.shape != truth.shape:
        raise ValueError('Predictions and truth must have identical shapes')
    p,t = harmonic_components(prediction,start=forecast_start), harmonic_components(truth,start=forecast_start)
    result = dict(offset_mse=(p['coefficients'][:,4]-t['coefficients'][:,4])**2,
                  residual_mse=np.mean((p['residual']-t['residual'])**2,axis=1),frequencies={})
    total = result['offset_mse']+result['residual_mse']
    for phase, columns in [('phi',slice(0,2)),('psi',slice(2,4))]:
        pred,true = p['coefficients'][:,columns],t['coefficients'][:,columns]
        ap,at = np.linalg.norm(pred,axis=1),np.linalg.norm(true,axis=1)
        denom = ap*at
        cosine = np.divide(np.sum(pred*true,axis=1),denom,out=np.ones(len(pred)),where=denom>1e-24)
        cosine = np.clip(cosine,-1,1)
        amplitude = .5*(ap-at)**2
        phase_error = at**2*(1-cosine)
        interaction = at*(ap-at)*(1-cosine)
        component = .5*np.sum((pred-true)**2,axis=1)
        if not np.allclose(amplitude+phase_error+interaction,component,atol=1e-10,rtol=1e-9):
            raise ValueError('Amplitude/phase decomposition failed its exact identity')
        result['frequencies'][phase] = dict(coefficient_mse=component,amplitude_mse=amplitude,
                                          phase_mse=phase_error,amplitude_phase_interaction=interaction)
        total = total+component
    direct = np.mean((prediction-truth)**2,axis=1)
    if not np.allclose(total,direct,atol=1e-10,rtol=1e-9):
        raise ValueError('Orthogonal decomposition failed to reconstruct32-point MSE')
    result.update(total_mse=direct,reconstructed_mse=total)
    return result


def matched_pairs(metadata, target, split='test'):
    """Keep both directions and both untouched-phase cells inside each seed."""
    if target not in ('phi','psi'):
        raise ValueError('Target must be phi or psi')
    other = 'psi' if target == 'phi' else 'phi'
    lookup = {}
    for i,row in enumerate(metadata):
        if row['split'] != split:
            continue
        key = (row['seed'],row['cell_'+target],row['cell_'+other])
        if key in lookup:
            raise ValueError('Duplicate factorial cell')
        lookup[key] = i
    seeds = sorted({key[0] for key in lookup})
    receiver,donor,groups = [],[],[]
    for seed in seeds:
        if any((seed,a,b) not in lookup for a in (0,1) for b in (0,1)):
            raise ValueError('Incomplete four-cell generating bundle')
        for direction in (0,1):
            for nuisance in (0,1):
                receiver.append(lookup[seed,direction,nuisance])
                donor.append(lookup[seed,1-direction,nuisance])
                groups.append(seed)
    if not seeds:
        raise ValueError('No matching generating bundles')
    return np.array(receiver),np.array(donor),np.array(groups)


def input_interference(prediction, truth, metadata, target, *, split='test', phase_floor=1e-6, forecast_start=128):
    """Measure untouched-component changes, with truth and phase-identifiability gates.

    Phase is untestable when either forecast's untouched amplitude is below
    phase_floor times the positive true amplitude. Coefficient changes remain
    defined in that case. All four input comparisons are averaged per seed.
    """
    if not np.isfinite(phase_floor) or phase_floor <= 0:
        raise ValueError('Need a positive phase-identifiability floor')
    prediction,truth = np.asarray(prediction),np.asarray(truth)
    if len(prediction) != len(metadata) or truth.shape != prediction.shape:
        raise ValueError('Need forecasts/truth aligned to all metadata rows')
    for row in metadata:
        a,b,scale = row['a'],row['b'],row['scale']
        if not np.isfinite([a,b,scale,row['offset']]).all() or min(a,b,scale) <= 0 or not np.isclose(scale,np.sqrt((a*a+b*b)/2),rtol=1e-6):
            raise ValueError('Need positive amplitudes and correct shared RMS scale')
    receiver,donor,groups = matched_pairs(metadata,target,split)
    columns = slice(2,4) if target == 'phi' else slice(0,2)
    other_amp = 'b' if target == 'phi' else 'a'
    full_p,full_t = harmonic_components(prediction,start=forecast_start),harmonic_components(truth,start=forecast_start)
    pc,tc = full_p['coefficients'],full_t['coefficients']
    invariant = list(range(columns.start,columns.stop))+[4]
    if not np.allclose(tc[receiver][:,invariant],tc[donor][:,invariant],atol=1e-9,rtol=1e-7):
        raise ValueError('Truth untouched component or offset changed across input intervention')
    amplitudes = np.array([row[other_amp] for row in metadata],dtype=float)
    if not np.allclose(np.linalg.norm(tc[:,columns],axis=1),amplitudes,atol=1e-8,rtol=1e-7):
        raise ValueError('Truth component amplitude disagrees with metadata')
    r,d = pc[receiver][:,columns],pc[donor][:,columns]
    ar,ad = np.linalg.norm(r,axis=1),np.linalg.norm(d,axis=1)
    true_amp = amplitudes[receiver]
    valid = (ar > phase_floor*true_amp)&(ad > phase_floor*true_amp)
    angle = np.abs(np.angle(np.exp(1j*(np.arctan2(d[:,1],d[:,0])-np.arctan2(r[:,1],r[:,0])))))
    angle[~valid] = np.nan
    observations = dict(other_phase_change_radians=angle,other_amplitude_change_absolute=np.abs(ad-ar),
                        other_amplitude_change_relative=np.abs(ad-ar)/true_amp,
                        other_coefficient_change_mse=.5*np.sum((d-r)**2,axis=1),
                        other_coefficient_change_normalized=.5*np.sum((d-r)**2,axis=1)/true_amp**2,
                        phase_untestable_fraction=(~valid).astype(float))
    seeds = np.unique(groups)
    aggregated = {key:np.array([values[groups==seed].mean() for seed in seeds]) for key,values in observations.items()}
    return dict(seeds=seeds,metrics=aggregated,truth_untouched_verified=True,comparisons_per_seed=4)


def _mean_interval(values, *, n_boot, seed):
    """Report unavailable phase intervals without turning zero amplitude into zero effect."""
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return dict(untestable='fewer than3 phase-identifiable generating seeds',n_seeds=int(valid.sum()))
    result = mean_ci(values[valid],n_boot=n_boot,seed=seed,unit='generating_seed_bundle')
    result.update(n_seeds=int(valid.sum()),excluded_seeds=int((~valid).sum()),interval='pointwise95%;fixed native predictions')
    return result


def analyze(predictions, truth, metadata, *, n_boot=2000, n_permutations=1999, seed=113):
    """Audit four model/target groups and a fixed two-target paired model family."""
    if set(predictions) != {'bolt','timesfm'}:
        raise ValueError('Audit needs both supported saved model forecasts: bolt,timesfm')
    report = dict(evidence_class='behavioral_causal_input_intervention',models={},
                  caution='diagnoses native model coupling;does not rescue hidden-feature selectivity failures or establish uniqueness')
    measurements = {}
    test = np.array([row['split']=='test' for row in metadata])
    for model,prediction in predictions.items():
        entries = {}
        decomposition = forecast_error_decomposition(prediction[test],truth[test])
        for target in ('phi','psi'):
            result = input_interference(prediction,truth,metadata,target)
            measurements[model,target] = result
            entries[target] = {key:_mean_interval(values,n_boot=n_boot,seed=seed) for key,values in result['metrics'].items()}
        report['models'][model] = dict(input_interference=entries,forecast_error=dict(
            total_mse=float(decomposition['total_mse'].mean()),offset_mse=float(decomposition['offset_mse'].mean()),
            residual_mse=float(decomposition['residual_mse'].mean()),
            orthogonal_reconstruction_max_abs=float(np.max(np.abs(decomposition['total_mse']-decomposition['reconstructed_mse']))),
            frequencies={phase:{key:float(value.mean()) for key,value in values.items()} for phase,values in decomposition['frequencies'].items()}))
    contrasts = {}
    for target in ('phi','psi'):
        a,b = measurements['timesfm',target],measurements['bolt',target]
        if not np.array_equal(a['seeds'],b['seeds']):
            raise ValueError('Crossmodel contrast needs matched generating seeds')
        av,bv = a['metrics']['other_phase_change_radians'],b['metrics']['other_phase_change_radians']
        valid = np.isfinite(av)&np.isfinite(bv)
        contrasts[target] = dict(p=1.,untestable='fewer than3 jointly identifiable seeds') if valid.sum()<3 else paired_effect(av[valid],bv[valid],a['seeds'][valid],n_boot=n_boot,n_permutations=n_permutations,seed=seed)
    adjusted = holm({target:value['p'] for target,value in contrasts.items()})
    for target in contrasts:
        contrasts[target]['p_holm'] = adjusted[target]
    report['crossmodel_phase_change_timesfm_minus_bolt'] = contrasts
    report['inference'] = 'Two prespecified target-phase contrasts Holm2;descriptive mean intervals pointwise;seed averages4 comparisons'
    return report


def run(source,out):
    """Read baseline forecast artifacts only; no model loading or hidden interventions."""
    out.mkdir(parents=True,exist_ok=False)
    metadata = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    truth = np.load(source/'corpus.npz')['clean_future']
    predictions = {name:np.load(source/f'{name}_activations.npz')['prediction'] for name in ('bolt','timesfm')}
    result = analyze(predictions,truth,metadata)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    run(args.source,args.out)
