"""Does the small output-aware pilot span unseen calibration shape responses?"""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.shape_transfer.forward_only.probe import HERE as FORWARD
from research.activation_transfer.shape_transfer.calibration_budget.study import HERE,SEEDS


def energy_fraction(full, projected):
    return float(np.square(projected).sum()/np.square(full).sum())


def profile(target,out):
    with np.load(out/(target+'_libraries.npz')) as f,np.load(FORWARD/'results'/(target+'_probes.npz')) as z:
        basis=f['basis'];jac=z['pc_jac'].astype(float);perms=f['permutations'];rows=[]
        np.testing.assert_array_equal(f['cal_ids'],z['cal_ids'])
        for i in range(len(SEEDS)):
            unseen=perms[i,48:]
            assert len(unseen)==48 and set(unseen).isdisjoint(perms[i,:48])
            gradients=jac[unseen];centered=gradients-gradients.mean(axis=1,keepdims=True)
            row={'unused_ids':f['cal_ids'][unseen].tolist()}
            for k,axes in [('pc8',basis[:8]),('pilot',f[f'pilot_axes_{i}'])]:
                rotation=basis@axes.T
                projected=centered@rotation
                row[k]=dict(pooled_centered_energy=energy_fraction(centered,projected),
                    shared_centered_energy=energy_fraction(centered.mean(axis=0),projected.mean(axis=0)),
                    pooled_full_energy=energy_fraction(gradients,gradients@rotation))
            rows.append(row)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE/'results');a=p.parse_args()
    (a.out/'calibration_geometry.json').write_text(json.dumps({m:profile(m,a.out) for m in ('bolt','timesfm')},indent=2)+'\n')
