"""Original atom metrics and paired, common-feature contrasts for the two edit classes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from research.activation_transfer.atoms.materials import HERE as ATOMS, MODELS
from research.activation_transfer.atoms.matched_windows.prepare import HERE
from research.activation_transfer.atoms.scores import ci, metrics, resampling_counts, terms


def center(array,shape):
    return array-array.mean(-1,keepdims=True) if shape else array


def paired(active,reference,candidate,counts):
    point,boot={},{}
    for arm in ['atom','window']:
        parts=terms(active,candidate[arm],reference[arm])
        point[arm]=metrics(parts.sum(0))
        by_sample=(counts@parts.reshape(len(parts),-1)).reshape(len(counts),*parts.shape[1:])
        boot[arm]=metrics(by_sample)
    common=np.isfinite(point['atom']['cosine'])&np.isfinite(point['window']['cosine'])
    if not common.any():
        raise ValueError('no shared measurable source features')
    common_draw=(np.isfinite(boot['atom']['cosine']) & np.isfinite(boot['window']['cosine']) & common[None])
    count=common_draw.sum(1)
    if np.any(count==0):
        raise ValueError('no eligible feature in at least one series-bootstrap draw')
    arms={};diff={}; independent={}
    for key in point['atom']:
        values,draws={},{}
        for a in point:
            values[a]=float(point[a][key][common].mean())
            draws[a]=np.where(common_draw,boot[a][key],0.).sum(1)/count
            arms.setdefault(a,{})[key]=ci(values[a],draws[a]);independent.setdefault(a,{})[key]=float(np.nanmean(point[a][key]))
        diff[key]=ci(values['window']-values['atom'],draws['window']-draws['atom'])
    individual={a:{k:[float(z) if np.isfinite(z) else None for z in v] for k,v in row.items()}
                for a,row in point.items()}
    return {'common_indices':np.flatnonzero(common).tolist(),
            'eligible_indices':{a:np.flatnonzero(np.isfinite(row['cosine'])).tolist()for a,row in point.items()},
            'min_common_bootstrap_features':int(count.min()),'common':arms,'window_minus_atom':diff,
            'separate':independent,'per_feature':individual}


def study(results):
    out={}
    for src in MODELS:
        with np.load(results/(src+'_pairs.npz')) as f:
            data=dict(f)
        original=json.loads((ATOMS/'results'/(src+'_scores.json')).read_text())['daily']['source_native']
        counts=resampling_counts(len(data['ids']));scores={}
        for label in ['full','shape']:
            shaped=label=='shape'
            ref={a:center(data[a+'_source'],shaped) for a in ['atom','window']}
            est={a:center(data[a+'_target'],shaped) for a in ['atom','window']}
            calc=paired(data['active'],ref,est,counts)
            for k,v in calc['separate']['atom'].items():
                prior=original[label]['summary']['target_dense'][k]['value']
                np.testing.assert_allclose(v,prior,rtol=1e-8,atol=1e-10)
            calc['common_feature_ids']=data['feature_ids'][calc['common_indices']].tolist()
            scores[label]=calc
        out[src]={'source':src,'feature_ids':data['feature_ids'].tolist(),'firing_counts':data['active'].sum(0).tolist(),
                  'n_items':len(data['ids']),'n_active_cases':int(data['active'].sum()),'scores':scores}
    (results/'scores.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=HERE/'results')
    study(p.parse_args().results)
