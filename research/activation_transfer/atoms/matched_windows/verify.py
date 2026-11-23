"""Compare independent matched-window inputs, hidden responses and actual forecast effects."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare_npz(left,right):
    with np.load(left) as a,np.load(right) as b:
        assert set(a.files)==set(b.files)
        out={}
        for k in a.files:
            x,y=a[k],b[k]
            if x.dtype.kind in 'OUSb':
                np.testing.assert_array_equal(x,y);out[k]=0.
            else:
                np.testing.assert_allclose(x,y,atol=1e-4,rtol=1e-5)
                out[k]=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64))))
        return out


def run(original,replay,original_scratch,replay_scratch):
    result={'arrays':{},'score_max_abs':{}}
    for f in ['inputs.npz','bolt_shift.npz','timesfm_shift.npz']:
        result['arrays'][f]=compare_npz(original_scratch/f,replay_scratch/f)
    a=json.loads((original/'scores.json').read_text());b=json.loads((replay/'scores.json').read_text())
    for src in ['timesfm','bolt']:
        f=src+'_pairs.npz';result['arrays'][f]=compare_npz(original/f,replay/f)
        d=[]
        for shape in ['full','shape']:
            for stat in ['cosine','skill','gain','source_rms','candidate_rms']:
                av=a[src]['scores'][shape]['window_minus_atom'][stat]
                bv=b[src]['scores'][shape]['window_minus_atom'][stat]
                dv=np.abs(np.array([av['value'],*av['ci95']])-np.array([bv['value'],*bv['ci95']]))
                assert (dv<1e-5).all();d.extend(dv.tolist())
        result['score_max_abs'][src]=max(d)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['original','replay','original-scratch','replay-scratch','out']:
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.write_text(json.dumps(run(a.original,a.replay,a.original_scratch,a.replay_scratch),indent=2)+'\n')
