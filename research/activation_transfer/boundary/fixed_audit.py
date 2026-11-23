"""Summarize each independently patched 32-direction random basis at its actual budget."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.boundary.analyze import BOOT, ci, metric_vectors
from research.activation_transfer.boundary.audit_forecasts import format_interval, pooled, sufficient_stats


def study(path):
    data = json.loads(path.read_text())
    raw = data['forecast_changes']
    reference = np.asarray(raw['native_dense'])
    draws = np.random.default_rng(20261120).integers(len(reference), size=(BOOT,len(reference)))
    means, pooled_stats, out = {}, {}, {}
    for arm in ('dense','code','shoulder','subtract','input','frequent_sae','clean_pca','random_sae','isotropic'):
        names = ([f'cross_{arm}_{i}' for i in range(8)] if arm in {'random_sae','isotropic'}
                 else [f'cross_{arm}'])
        preds = [np.asarray(raw[n]) for n in names]
        cosines = [metric_vectors(q, reference)['cosine'] for q in preds]
        mean_cosine = np.mean(cosines,axis=0)
        stats = np.mean([sufficient_stats(q, reference) for q in preds],axis=0)
        point, bootstrap = pooled(stats.sum(0)), pooled(stats[draws].sum(1))
        out[arm] = {'mean_item_cosine':ci(mean_cosine),
                    'replicate_mean_cosines':[float(c.mean()) for c in cosines],
                    **{k:format_interval(v, bootstrap[k]) for k,v in point.items()}}
        means[arm], pooled_stats[arm] = mean_cosine, (point, bootstrap)
    for a,b in [('frequent_sae','isotropic'),('random_sae','isotropic'),('frequent_sae','random_sae'),
                ('frequent_sae','clean_pca'),('frequent_sae','code'),('frequent_sae','input')]:
        p,bs = pooled_stats[a]
        p0,bs0 = pooled_stats[b]
        out[a+'_minus_'+b] = {'mean_item_cosine':ci(means[a]-means[b]),
                              **{k:format_interval(p[k]-p0[k],bs[k]-bs0[k]) for k in p}}
    return {'source':path.name,'n':len(reference),'native_mean_rms':ci(np.sqrt((reference**2).mean(axis=-1))),
            'zero_replay':data['self_patch_max'],'methods':out}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps([study(path) for path in args.reports], indent=2) + '\n')
