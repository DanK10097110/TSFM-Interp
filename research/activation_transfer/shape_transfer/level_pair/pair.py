"""Compare the two fixed atoms' responses to the same controlled histories."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.level_pair.inputs import CUES,HERE
from research.activation_transfer.shape_transfer.level_pair.score import half_difference,mean_stat


def rank_correlation(x,y):
    if np.ptp(x)<1e-8 or np.ptp(y)<1e-8:return None
    return float(spearmanr(x,y).statistic)


def analyze(out):
    models={k:dict(np.load(out/(k+'_effects.npz'))) for k in ('bolt','timesfm')}
    b,t=models['bolt'],models['timesfm'];np.testing.assert_array_equal(b['ids'],t['ids'])
    results={}
    for domain in np.unique(b['split']):
        here=b['split']==domain;result={};results[str(domain)]=result
        for c,cue in enumerate(CUES):
            usable=here&((b['edited_z'][:,c].max(1)>0)|(t['edited_z'][:,c].max(1)>0))
            for population,keep in [('all',here),('either_model_edited_active',usable)]:
                arm={};result.setdefault(cue,{})[population]=arm;n=int(keep.sum());arm['n']=n
                if not n:continue
                draw=resampling_counts(n);z={k:half_difference(v['edited_z'][keep,c]) for k,v in models.items()}
                lost={k:half_difference(v['intact'][keep,c])-half_difference(v['named'][keep,c]) for k,v in models.items()}
                if cue!='translation':
                    arm['code_half_change_spearman']=rank_correlation(z['bolt'],z['timesfm'])
                    arm['both_half_changes_positive']=int(((z['bolt']>1e-5)&(z['timesfm']>1e-5)).sum())
                    arm['both_half_changes_negative']=int(((z['bolt']< -1e-5)&(z['timesfm']< -1e-5)).sum())
                level={k:v.mean(axis=1) for k,v in lost.items()}
                arm['timesfm_minus_bolt_removed_level']=mean_stat(level['timesfm']-level['bolt'],draw)
                if cue!='translation':arm['removed_level_spearman']=rank_correlation(level['bolt'],level['timesfm'])
                arm['both_clean_active']=int(((b['clean_z'][keep]>0)&(t['clean_z'][keep]>0)).sum())
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE/'results');a=p.parse_args()
    (a.out/'pair.json').write_text(json.dumps(analyze(a.out),indent=2,allow_nan=False)+'\n')
