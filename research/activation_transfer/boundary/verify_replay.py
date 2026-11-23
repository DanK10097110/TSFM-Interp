"""Compare two independent captures and their held-out cross-model predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare_arrays(a,b):
    np.testing.assert_equal(a.shape,b.shape)
    if a.dtype.kind in 'OUS':
        np.testing.assert_array_equal(a,b)
        return 0.
    return float(np.max(np.abs(a.astype('float64')-b.astype('float64'))))


def verify(capture_a,capture_b,results_a,results_b):
    summary = {'captures':{},'predicted_displacements':{},'forecasts':{},'mean_state_cosines':{}}
    for dataset in ('m4','weekly','fast'):
        for model in ('timesfm','bolt'):
            k = dataset+'_'+model
            with np.load(capture_a/(k+'.npz')) as a, np.load(capture_b/(k+'.npz')) as b:
                np.testing.assert_equal(a.files,b.files)
                summary['captures'][k] = {v:compare_arrays(a[v],b[v]) for v in a.files}
            filename = f'{dataset}_smooth_{model}_forecasts.json'
            a,b = [json.loads((p/filename).read_text()) for p in [results_a,results_b]]
            np.testing.assert_array_equal(a['ids'],b['ids'])
            summary['forecasts'][k] = {name:compare_arrays(np.asarray(am),np.asarray(b['forecast_changes'][name]))
                                       for name,am in a['forecast_changes'].items()}
        filename = f'{dataset}_smooth_predictions.npz'
        with np.load(capture_a/filename) as a, np.load(capture_b/filename) as b:
            summary['predicted_displacements'][dataset] = {v:compare_arrays(a[v],b[v]) for v in a.files}
        a,b = [json.loads((p/(dataset+'_smooth.json')).read_text()) for p in [results_a,results_b]]
        diffs = [abs(c['cosine']['mean']-b['transfer'][pair][arm]['cosine']['mean'])
                 for pair,scores in a['transfer'].items() for arm,c in scores.items()
                 if isinstance(c,dict) and 'cosine' in c]
        summary['mean_state_cosines'][dataset] = max(diffs)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['capture_a','capture_b','results_a','results_b','out']:
        p.add_argument('--'+key.replace('_','-'),required=True,type=Path)
    a=p.parse_args()
    a.out.write_text(json.dumps(verify(a.capture_a,a.capture_b,a.results_a,a.results_b),indent=2)+'\n')
