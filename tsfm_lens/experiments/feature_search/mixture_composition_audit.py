"""Descriptive composition defect using oracle-separated pure-component inputs.

The baseline adds forecasts from known pure components and subtracts the exact
oracle offset. It is not f(offset), a generic forecaster, or a deployable upgrade.
All scaled MSEs use the same mixed-signal variance for every condition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from experiments.feature_search.mixture_interference import harmonic_components
from experiments.feature_search.mixture_replication import _validate_compositions
from tsfm_lens.analysis.stats import mean_ci, bootstrap_ci


def composition_diagnostics(prediction,truth,metadata,*,n_boot=2000,seed=113,forecast_start=128):
    """Pair exact-phase pure forecasts with each of four mixed cells per seed."""
    prediction,truth = np.asarray(prediction,dtype=float),np.asarray(truth,dtype=float)
    seeds = _validate_compositions(truth,metadata,'condition',forecast_start=forecast_start)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError('Need finite native predictions aligned to factorial compositions')
    lookup = {(row['seed'],row['condition'],row['cell_phi'],row['cell_psi']):i for i,row in enumerate(metadata)}
    mixed = np.array([i for i,row in enumerate(metadata) if row['condition']=='mixed'])
    phi = np.array([lookup[metadata[i]['seed'],'pure_phi',metadata[i]['cell_phi'],0] for i in mixed])
    psi = np.array([lookup[metadata[i]['seed'],'pure_psi',0,metadata[i]['cell_psi']] for i in mixed])
    offset = np.array([metadata[i]['offset'] for i in mixed])[:,None]
    if not np.allclose(truth[phi]+truth[psi]-offset,truth[mixed],atol=1e-10):
        raise ValueError('True inputs are not additive under the oracle-offset decomposition')
    composed = prediction[phi]+prediction[psi]-offset
    energy = np.array([metadata[i]['scale']**2 for i in mixed])
    groups = np.array([metadata[i]['seed'] for i in mixed])
    native = np.mean((prediction[mixed]-truth[mixed])**2,axis=1)
    component = np.mean((composed-truth[mixed])**2,axis=1)
    defect = np.mean((prediction[mixed]-composed)**2,axis=1)
    row_metrics = dict(native_error_mse_raw=native,component_sum_error_mse_raw=component,composition_defect_mse_raw=defect,
                       native_error_mse_common_mixvariance=native/energy,component_sum_error_mse_common_mixvariance=component/energy,
                       composition_defect_mse_common_mixvariance=defect/energy,
                       error_reduction_mse_common_mixvariance=(native-component)/energy)
    fitted = harmonic_components(prediction,start=forecast_start)['coefficients']
    for target,columns,pure_key,amp_key in [('phi',slice(0,2),phi,'a'),('psi',slice(2,4),psi,'b')]:
        mp,pp = fitted[mixed][:,columns],fitted[pure_key][:,columns]
        ma,pa = np.linalg.norm(mp,axis=1),np.linalg.norm(pp,axis=1)
        true_amp = np.array([metadata[i][amp_key] for i in mixed])
        valid = (ma>1e-6*true_amp)&(pa>1e-6*true_amp)
        angle = np.abs(np.angle(np.exp(1j*(np.arctan2(mp[:,1],mp[:,0])-np.arctan2(pp[:,1],pp[:,0])))))
        angle[~valid] = np.nan
        row_metrics[target+'_amplitude_suppression_relative_to_true'] = (pa-ma)/true_amp
        row_metrics[target+'_phase_shift_radians'] = angle
        row_metrics[target+'_phase_unidentifiable_fraction'] = (~valid).astype(float)
    metrics = {key:np.array([value[groups==unit].mean() for unit in seeds]) for key,value in row_metrics.items()}
    intervals = {}
    for key,values in metrics.items():
        valid = np.isfinite(values)
        intervals[key] = dict(untestable='fewer than3 identifiable generating bundles') if valid.sum()<3 else mean_ci(values[valid],n_boot=n_boot,seed=seed,unit='generating_seed_bundle')
        intervals[key]['excluded_seeds'] = int((~valid).sum())
    raw_native,raw_composed = metrics['native_error_mse_raw'],metrics['component_sum_error_mse_raw']
    scaled_native,scaled_composed = metrics['native_error_mse_common_mixvariance'],metrics['component_sum_error_mse_common_mixvariance']
    fractions = {}
    for unit,native_values,composed_values in [('raw_mse',raw_native,raw_composed),('common_mixvariance_mse',scaled_native,scaled_composed)]:
        if np.any(native_values<=1e-20):
            fractions[unit] = dict(untestable='zero native error in one or more bundles;ratio bootstrap undefined')
        else:
            fractions[unit] = bootstrap_ci(lambda idx:1-float(composed_values[idx].mean()/native_values[idx].mean()),len(seeds),n_boot=n_boot,seed=seed,unit='generating_seed_bundle')
    return dict(n_seeds=len(seeds),n_mixed_cells=len(mixed),fitted_parameters=0,metrics=intervals,
                fraction_native_error_removed_by_oracle_component_forecast_sum=fractions,
                baseline='f(pure_phi)+f(pure_psi)-oracle_exact_offset;NOT f(offset)',
                units='raw MSE in original series units²;common_mixvariance MSE divided by SAME(a²+b²)/2',
                caution='Known input labels provide oracle component decomposition;secondary descriptive coverage diagnostic,not a generic forecast upgrade'),composed


def run(source,out):
    """Read saved fresh forecasts only, preserving the frozen primary analysis."""
    out.mkdir(parents=True,exist_ok=False)
    metadata = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    truth = np.load(source/'corpus.npz')['clean_future']
    report,saved = {},{}
    for model in ('bolt','timesfm'):
        prediction = np.load(source/f'{model}_predictions.npz')['prediction']
        report[model],saved[model+'_oracle_component_sum'] = composition_diagnostics(prediction,truth,metadata)
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(out/'composed_predictions.npz',**saved)
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.source,args.out)
