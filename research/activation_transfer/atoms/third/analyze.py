"""Score recipient interventions against the active source feature's actual ablation."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, metrics, resampling_counts, terms

HERE=Path(__file__).resolve().parent


def score(active,predictions,reference,counts):
    measurable=np.isfinite(metrics(terms(active,reference,reference).sum(axis=0))['cosine'])
    if not measurable.any():
        return dict(measurable=measurable.tolist(),methods={},dense_minus_wrong=None)
    points, draws, methods = {},{},{}
    for key, pred in predictions.items():
        atom,point,boot=aggregate(active,pred,reference,counts)
        points[key]=point; draws[key]=boot
        methods[key]={metric:ci(point[metric],boot[metric]) for metric in ('cosine','skill','gain','source_rms','candidate_rms')}
        methods[key]['atom_cosine']=[float(v) if np.isfinite(v) else None for v in atom['cosine']]
        methods[key]['atom_skill']=[float(v) if np.isfinite(v) else None for v in atom['skill']]
    wrong=['wrong'+str(i) for i in range(4)]
    mean={metric:np.mean([points[w][metric] for w in wrong]) for metric in ('cosine','skill','gain')}
    draws['wrong_mean']={metric:np.mean([draws[w][metric] for w in wrong],axis=0) for metric in mean}
    methods['wrong_mean']={metric:ci(value,draws['wrong_mean'][metric]) for metric,value in mean.items()}
    contrast={}
    for metric in mean:
        diff=draws['dense'][metric]-draws['wrong_mean'][metric]
        contrast[metric]=ci(points['dense'][metric]-mean[metric],diff)
        contrast[metric]['lower_one_sided_98_75']=float(np.nanquantile(diff,.0125))
    return dict(measurable=measurable.tolist(),methods=methods,dense_minus_wrong=contrast)


def analyze(source):
    with np.load(HERE/(source+'_effects.npz')) as f:
        data=dict(f)
    out={}
    for domain in ('daily','electricity'):
        use=data['split']==domain; active=data['active'][use]; n=active.sum(axis=0)
        counts=resampling_counts(len(active))
        exact_null=active & (data['matched_target_codes'][use]==0)
        null_errors=np.abs(data['coactive'][use][exact_null])
        joint=((data['matched_target_codes'][use]>0)&active).sum(axis=0)
        group=dict(n_series=len(active),active_cases=n.tolist(),coactive_cases=joint.tolist(),
                   max_structural_zero_drift=float(null_errors.max()) if null_errors.size else 0.,
                   coactive_fraction_of_source_firings=float(joint.sum()/max(1,n.sum())),scores={})
        for minimum in (1,8):
            keep=n>=minimum
            record=dict(atom_ids=data['atoms'][keep].tolist()); group['scores'][str(minimum)]=record
            for centered in (False,True):
                def extract(key):
                    y=data[key][use][:,keep].astype(float)
                    if key=='coactive':
                        y[exact_null[:,keep]]=0.
                    return y-y.mean(-1,keepdims=True) if centered else y
                preds={key:extract(key) for key in ('dense','coactive','wrong0','wrong1','wrong2','wrong3')}
                record['shape' if centered else 'full']=score(active[:,keep],preds,extract('native'),counts)
        out[domain]=group
    (HERE/(source+'_scores.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',choices=['chronos2','timesfm']); a=p.parse_args(); analyze(a.source)
